#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ase.io import read
from ase import units
from mace.calculators.mace import MACECalculator

# ============================================================
# CONFIG
# ============================================================
xyz_file = "../R-3B69.xyz"
model_base_path = "../mace_off/MACE-MDP.model"

# Search paths for ORCA Hessian files
hess_patterns = [
    "../bucket_0/{name}/input.hess",
    "../{name}/input.hess",
]

output_dir = "spectra_from_hessian_modes"
os.makedirs(output_dir, exist_ok=True)

device = "cuda"
dtype = "float64"

fd_step = 1.0e-3         # displacement amplitude along normal mode
low_freq_cutoff = 5.0    # cm^-1
fwhm = 30.0
freq_warn_tol = 0.1      # cm^-1

# ============================================================
# SINGLE PROPERTY CALCULATOR FOR BOTH IR + RAMAN
# ============================================================
prop_calc = MACECalculator(
    model_paths=model_base_path,
    model_type="DipolePolarizabilityMACE",
    device=device,
    default_dtype=dtype,
)

# ============================================================
# HELPERS
# ============================================================
def parse_name_from_comment(comment):
    match = re.search(r'name="([^"]+)"', comment)
    return match.group(1) if match else "unknown"

def read_xyz_comments(xyz_path):
    comments = []
    with open(xyz_path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        natoms = int(lines[i].strip())
        comments.append(lines[i + 1].strip())
        i += natoms + 2
    return comments

def find_hess_file(name):
    for pat in hess_patterns:
        matches = glob.glob(pat.format(name=name))
        if matches:
            return matches[0]
    return None

def broaden_spectrum_lorentzian(freqs, intens, fwhm=10.0, x_min=0, x_max=4000, step=1.0):
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
    if len(freqs) == 0:
        return x, np.zeros_like(x)
    gamma = fwhm / 2.0
    diff = x[:, None] - freqs[None, :]
    spec = np.sum(intens[None, :] * (gamma / (diff**2 + gamma**2)), axis=1)
    return x, spec

def safe_normalize(arr):
    m = np.max(arr)
    if m > 0:
        return arr / m
    return arr.copy()

def _is_int(tok):
    return re.fullmatch(r"[+-]?\d+", tok) is not None

def _is_float(tok):
    return re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?", tok) is not None

# ============================================================
# PARSERS FOR input.hess
# ============================================================
def parse_hessian_block(hess_file):
    """
    Parse the $hessian block from ORCA .hess file.
    Returns H as (3N, 3N) numpy array.
    """
    with open(hess_file, "r") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip() == "$hessian":
            start = i
            break
    if start is None:
        raise ValueError(f"$hessian block not found in {hess_file}")

    n_cart = int(lines[start + 1].strip())
    H = np.zeros((n_cart, n_cart), dtype=np.float64)

    i = start + 2
    current_cols = None
    seen_cols = set()

    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("$") and i > start + 2:
            break

        if stripped == "":
            i += 1
            continue

        tokens = stripped.split()

        # column header line
        if len(tokens) >= 1 and all(_is_int(t) for t in tokens):
            current_cols = [int(t) for t in tokens]
            seen_cols.update(current_cols)
            i += 1
            continue

        # matrix row line
        if current_cols is not None and len(tokens) >= 2 and _is_int(tokens[0]):
            row = int(tokens[0])
            vals = tokens[1:]
            if len(vals) < len(current_cols):
                raise ValueError(
                    f"Incomplete Hessian row in {hess_file}: row {row}, "
                    f"expected {len(current_cols)} values, got {len(vals)}"
                )
            for j, col in enumerate(current_cols):
                H[row, col] = float(vals[j])
            i += 1
            continue

        i += 1

    if sorted(seen_cols) != list(range(n_cart)):
        raise ValueError(
            f"Did not recover all Hessian columns from {hess_file}. "
            f"Got {sorted(seen_cols)}, expected 0..{n_cart-1}"
        )

    return H

def parse_vibrational_frequencies_block(hess_file):
    """
    Parse the $vibrational_frequencies block from ORCA .hess file.
    Returns frequencies in cm^-1 as shape (3N,).
    """
    with open(hess_file, "r") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip() == "$vibrational_frequencies":
            start = i
            break
    if start is None:
        raise ValueError(f"$vibrational_frequencies block not found in {hess_file}")

    n_cart = int(lines[start + 1].strip())
    freqs = np.zeros(n_cart, dtype=np.float64)

    for i in range(start + 2, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("$"):
            break
        if stripped == "":
            continue

        tokens = stripped.split()
        if len(tokens) >= 2 and _is_int(tokens[0]) and _is_float(tokens[1]):
            idx = int(tokens[0])
            val = float(tokens[1])
            freqs[idx] = val

    return freqs

# ============================================================
# HESSIAN -> FREQUENCIES + MODES
# ============================================================
EH_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
HESS_AU_TO_EV_A2 = EH_TO_EV / (BOHR_TO_ANG ** 2)

def get_modes_from_hessian(H_au, masses):
    """
    H_au is ORCA Hessian in Hartree/Bohr^2
    masses are in amu
    """
    n_atoms = len(masses)
    n_cart = 3 * n_atoms

    # convert Hessian from Hartree/Bohr^2 -> eV/Å^2
    H = H_au.reshape(n_cart, n_cart) * HESS_AU_TO_EV_A2

    # symmetrize
    H = 0.5 * (H + H.T)

    # mass-weight
    mw = np.repeat(masses**-0.5, 3)
    Hmw = (H * mw).T * mw

    # diagonalize
    omega2, vecs = np.linalg.eigh(Hmw)

    # same ASE conversion you already use
    conv = units._hbar * units.m / np.sqrt(units._e * units._amu)
    energies = conv * np.sqrt(np.abs(omega2))
    freq = np.real(energies) / units.invcm

    # Cartesian modes
    modes = vecs.T.reshape(n_cart, n_atoms, 3)
    modes *= masses[np.newaxis, :, np.newaxis]**-0.5

    return freq, modes, omega2

# ============================================================
# MAIN
# ============================================================
all_atoms = read(xyz_file, index=":")
all_comments = read_xyz_comments(xyz_file)

summary = []
failed_log = os.path.join(output_dir, "failed_molecules.txt")
warning_log = os.path.join(output_dir, "frequency_warnings.txt")

for p in [failed_log, warning_log]:
    if os.path.exists(p):
        os.remove(p)

for atoms, comment in zip(all_atoms, all_comments):
    name = parse_name_from_comment(comment).split("/")[-1]
    print(f"Processing {name}")

    try:
        hess_file = find_hess_file(name)
        if hess_file is None:
            raise FileNotFoundError(f"No input.hess found for {name}")

        n_atoms = len(atoms)
        n_modes = 3 * n_atoms
        masses = atoms.get_masses()

        # ---------------- parse Hessian + reference frequencies ----------------
        H = parse_hessian_block(hess_file)
        freq_ref = parse_vibrational_frequencies_block(hess_file)

        if H.shape != (n_modes, n_modes):
            raise ValueError(
                f"Hessian shape mismatch for {name}: got {H.shape}, expected {(n_modes, n_modes)}"
            )
        if len(freq_ref) != n_modes:
            raise ValueError(
                f"Frequency count mismatch for {name}: got {len(freq_ref)}, expected {n_modes}"
            )

        # ---------------- diagonalize our own mass-weighted Hessian ----------------
        freq, modes, omega2 = get_modes_from_hessian(H, masses)

        # compare against ORCA reference frequencies
        diff = np.abs(freq - freq_ref)
        max_diff = float(np.max(diff))
        if max_diff > freq_warn_tol:
            msg = (
                f"WARNING: {name}: max |Δfreq| = {max_diff:.6f} cm^-1 "
                f"(tolerance {freq_warn_tol:.3f} cm^-1)\n"
            )
            print(msg.strip())
            with open(warning_log, "a") as f:
                f.write(msg)

        # save diagnostic comparison
        pd.DataFrame({
            "mode_index": np.arange(n_modes),
            "freq_from_hessian_cm1": freq,
            "freq_reference_cm1": freq_ref,
            "abs_diff_cm1": diff,
            "omega2_raw": omega2,
        }).to_csv(os.path.join(output_dir, f"{name}_frequency_check.csv"), index=False)

        # ---------------- compute IR + Raman ----------------
        pos0 = atoms.get_positions().copy()

        ir_intens = []
        raman_tot = []
        raman_iso = []
        raman_ani = []

        for mode_idx, mode in enumerate(modes):
            f_cm1 = freq[mode_idx]

            if f_cm1 < low_freq_cutoff:
                ir_intens.append(0.0)
                raman_tot.append(0.0)
                raman_iso.append(0.0)
                raman_ani.append(0.0)
                continue

            dpos = mode * fd_step

            # ---------- IR ----------
            atoms.set_positions(pos0 + dpos)
            mu_p = prop_calc.get_property("dipole", atoms)

            atoms.set_positions(pos0 - dpos)
            mu_m = prop_calc.get_property("dipole", atoms)

            atoms.set_positions(pos0)
            dmu_dQ = (mu_p - mu_m) / (2.0 * fd_step)
            I_ir = float(np.dot(dmu_dQ, dmu_dQ))
            ir_intens.append(I_ir)

            # ---------- Raman ----------
            atoms.set_positions(pos0 + dpos)
            alpha_p = prop_calc.get_property("polarizability", atoms)

            atoms.set_positions(pos0 - dpos)
            alpha_m = prop_calc.get_property("polarizability", atoms)

            atoms.set_positions(pos0)
            dalpha_dQ = (alpha_p - alpha_m) / (2.0 * fd_step)

            a_iso = np.trace(dalpha_dQ) / 3.0
            a_ani_sq = 0.5 * (
                (dalpha_dQ[0, 0] - dalpha_dQ[1, 1])**2
                + (dalpha_dQ[1, 1] - dalpha_dQ[2, 2])**2
                + (dalpha_dQ[2, 2] - dalpha_dQ[0, 0])**2
                + 6.0 * (
                    dalpha_dQ[0, 1]**2
                    + dalpha_dQ[1, 2]**2
                    + dalpha_dQ[2, 0]**2
                )
            )

            I_iso = 45.0 * a_iso**2
            I_ani = 7.0 * a_ani_sq
            I_tot = I_iso + I_ani

            raman_iso.append(float(I_iso))
            raman_ani.append(float(I_ani))
            raman_tot.append(float(I_tot))

        atoms.set_positions(pos0)

        ir_intens = np.array(ir_intens, dtype=np.float64)
        raman_tot = np.array(raman_tot, dtype=np.float64)
        raman_iso = np.array(raman_iso, dtype=np.float64)
        raman_ani = np.array(raman_ani, dtype=np.float64)

        # ---------------- broaden ----------------
        x_ir, spec_ir = broaden_spectrum_lorentzian(freq, ir_intens, fwhm=fwhm)
        x_ra, spec_ra_tot = broaden_spectrum_lorentzian(freq, raman_tot, fwhm=fwhm)
        _, spec_ra_iso = broaden_spectrum_lorentzian(freq, raman_iso, fwhm=fwhm)
        _, spec_ra_ani = broaden_spectrum_lorentzian(freq, raman_ani, fwhm=fwhm)

        spec_ir_norm = safe_normalize(spec_ir)

        ra_max = np.max(spec_ra_tot)
        if ra_max > 0:
            spec_ra_tot_norm = spec_ra_tot / ra_max
            spec_ra_iso_norm = spec_ra_iso / ra_max
            spec_ra_ani_norm = spec_ra_ani / ra_max
        else:
            spec_ra_tot_norm = spec_ra_tot.copy()
            spec_ra_iso_norm = spec_ra_iso.copy()
            spec_ra_ani_norm = spec_ra_ani.copy()

        # ---------------- save raw data ----------------
        pd.DataFrame({
            "mode_index": np.arange(n_modes),
            "frequency_cm1": freq,
            "frequency_ref_cm1": freq_ref,
            "ir_intensity": ir_intens,
            "raman_total_intensity": raman_tot,
            "raman_isotropic_intensity": raman_iso,
            "raman_anisotropic_intensity": raman_ani,
        }).to_csv(os.path.join(output_dir, f"{name}_modes_raw.csv"), index=False)

        pd.DataFrame({
            "frequency_cm1": freq,
            "intensity": ir_intens,
        }).to_csv(os.path.join(output_dir, f"{name}_ir_raw.csv"), index=False)

        pd.DataFrame({
            "frequency_cm1": freq,
            "total_intensity": raman_tot,
            "isotropic_intensity": raman_iso,
            "anisotropic_intensity": raman_ani,
        }).to_csv(os.path.join(output_dir, f"{name}_raman_raw.csv"), index=False)

        # ---------------- save broadened normalized spectra ----------------
        pd.DataFrame({
            "cm1": x_ir,
            "intensity": spec_ir_norm,
        }).to_csv(os.path.join(output_dir, f"{name}_ir_norm.csv"), index=False)

        pd.DataFrame({
            "cm1": x_ra,
            "total": spec_ra_tot_norm,
            "isotropic": spec_ra_iso_norm,
            "anisotropic": spec_ra_ani_norm,
        }).to_csv(os.path.join(output_dir, f"{name}_raman_norm.csv"), index=False)

        # ---------------- plots ----------------
        plt.figure(figsize=(8, 6))
        plt.plot(x_ir, spec_ir_norm, label="IR")
        plt.xlim(0, 4000)
        plt.xlabel("IR shift (cm$^{-1}$)")
        plt.ylabel("Intensity (arb. units)")
        plt.title(f"{name} IR spectrum (from input.hess)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_ir.png"), dpi=400)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(x_ra, spec_ra_tot_norm, label="Total")
        plt.plot(x_ra, spec_ra_iso_norm, label="Isotropic", alpha=0.8)
        plt.plot(x_ra, spec_ra_ani_norm, label="Anisotropic", alpha=0.8)
        plt.xlim(0, 4000)
        plt.xlabel("Raman shift (cm$^{-1}$)")
        plt.ylabel("Intensity (arb. units)")
        plt.title(f"{name} Raman spectrum (from input.hess)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_raman.png"), dpi=400)
        plt.close()

        summary.append({
            "name": name,
            "n_atoms": n_atoms,
            "hess_file": hess_file,
            "max_freq_diff_cm1": max_diff,
        })

    except Exception as e:
        with open(failed_log, "a") as f:
            f.write(f"Molecule: {name}\n")
            f.write(f"{type(e).__name__}: {e}\n\n")
        print(f"  Error in {name}: {e}")

pd.DataFrame(summary).to_csv(
    os.path.join(output_dir, "successful_molecules.csv"),
    index=False
)
print(f"Done. Results written to: {output_dir}")