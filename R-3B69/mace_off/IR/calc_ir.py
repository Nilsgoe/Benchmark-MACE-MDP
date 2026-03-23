#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from ase import units
from ase.optimize import BFGS
from mace.calculators.mace import MACECalculator
from mace.calculators import mace_off

# ============================================================
# CONFIG
# ============================================================
xyz_file = "../../R-3B69.xyz"   # multi-structure XYZ
model_base_path = "../MACE-MDP.model"
model_sizes = ["small", "medium", "large"]

device = "cuda"
dtype = "float64"

raman_fwhm = 30.0
ir_fwhm = 30.0
x_min = 0.0
x_max = 4000.0
x_step = 1.0

fd_step = 1e-3          # displacement along mode in Å-like ASE coordinate units
fmax_opt = 0.002
max_opt_steps = 5000
low_freq_cutoff = 5.0   # cm^-1; suppress translation/rotation-like modes

# ============================================================
# PROPERTY CALCULATORS
# ============================================================
polar_calc = MACECalculator(
    model_paths=model_base_path,
    model_type="DipolePolarizabilityMACE",
    device=device,
    default_dtype=dtype
)


# ============================================================
# HELPERS
# ============================================================
def parse_name_from_comment(comment):
    match = re.search(r'name="([^"]+)"', comment)
    return match.group(1) if match else "unknown"

def broaden_spectrum_lorentzian(freqs, intens, fwhm=10.0, x_min=0, x_max=4000, step=1.0):
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
    gamma = fwhm / 2.0
    spec = np.zeros_like(x)
    for f0, I in zip(freqs, intens):
        spec += I * (gamma / ((x - f0)**2 + gamma**2))
    return x, spec

def safe_normalize(spec):
    max_val = np.max(spec)
    if max_val > 0:
        return spec / max_val
    return spec.copy()

class BFGSWithCount(BFGS):
    def run(self, fmax=None, steps=None):
        self.step_count = 0
        converged = False
        for _ in range(steps if steps is not None else 1000000):
            converged = super().run(fmax=fmax, steps=1)
            self.step_count += 1
            if converged:
                break
        return converged

def get_comments_from_xyz(xyz_path):
    comments = []
    with open(xyz_path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        natoms = int(lines[i].strip())
        comments.append(lines[i + 1].strip())
        i += natoms + 2
    return comments

# ============================================================
# READ INPUT
# ============================================================
all_atoms = read(xyz_file, index=":")
all_comments = get_comments_from_xyz(xyz_file)

# ============================================================
# MAIN LOOP
# ============================================================
for size in model_sizes:
    print(f"\nProcessing model size: {size}")

    output_dir = f"ir_results_{size}_SPICE"
    os.makedirs(output_dir, exist_ok=True)

    off_calc = mace_off(model=size, default_dtype=dtype, device=device)
    success_records = []

    failed_log = os.path.join(output_dir, "failed_molecules.txt")
    if os.path.exists(failed_log):
        os.remove(failed_log)

    for mol_idx, (atoms, comment) in enumerate(zip(all_atoms, all_comments), start=1):
        name = parse_name_from_comment(comment).split("/")[-1]

        try:
            print(f"Processing {name} ({mol_idx}/{len(all_atoms)})")

            n_atoms = len(atoms)
            masses = atoms.get_masses()

            # ------------------------------------------------
            # Geometry optimization with MACE-OFF
            # ------------------------------------------------
            atoms.calc = off_calc
            dyn = BFGSWithCount(atoms)
            dyn.run(fmax=fmax_opt, steps=max_opt_steps)

            # ------------------------------------------------
            # Hessian -> frequencies + modes
            # ------------------------------------------------
            H = off_calc.get_hessian(atoms).reshape(3 * n_atoms, 3 * n_atoms)
            H = 0.5 * (H + H.T)  # symmetrize

            mw = np.repeat(masses**-0.5, 3)
            Hmw = (H * mw).T * mw

            omega2, vecs = np.linalg.eigh(Hmw)

            conv = units._hbar * units.m / np.sqrt(units._e * units._amu)
            energies = conv * np.sqrt(np.abs(omega2))
            freq = np.real(energies) / units.invcm

            # vecs columns are eigenvectors; transpose to mode index first
            modes = vecs.T.reshape(3 * n_atoms, n_atoms, 3)
            modes *= masses[np.newaxis, :, np.newaxis] ** -0.5

            # Save optimized geometry
            opt_xyz_path = os.path.join(output_dir, f"{name}_optimized.xyz")
            atoms.write(opt_xyz_path)

            # ------------------------------------------------
            # Raman intensities
            # ------------------------------------------------
            raman_tot = []
            raman_iso = []
            raman_ani = []

            # ------------------------------------------------
            # IR intensities
            # ------------------------------------------------
            ir_intens = []

            pos0 = atoms.get_positions().copy()

            for mode_idx, mode in enumerate(modes):
                f_cm1 = freq[mode_idx]

                if f_cm1 < low_freq_cutoff:
                    raman_tot.append(0.0)
                    raman_iso.append(0.0)
                    raman_ani.append(0.0)
                    ir_intens.append(0.0)
                    continue

                dpos = mode * fd_step

                # ---------- Raman ----------
                atoms.set_positions(pos0 + dpos)
                alpha_p = polar_calc.get_property("polarizability", atoms)

                atoms.set_positions(pos0 - dpos)
                alpha_m = polar_calc.get_property("polarizability", atoms)

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

                raman_iso.append(I_iso)
                raman_ani.append(I_ani)
                raman_tot.append(I_tot)

                # ---------- IR ----------
                atoms.set_positions(pos0 + dpos)
                mu_p = polar_calc.get_property("dipole", atoms)

                atoms.set_positions(pos0 - dpos)
                mu_m = polar_calc.get_property("dipole", atoms)

                atoms.set_positions(pos0)
                dmu_dQ = (mu_p - mu_m) / (2.0 * fd_step)

                I_ir = np.dot(dmu_dQ, dmu_dQ)
                ir_intens.append(I_ir)

            raman_tot = np.array(raman_tot, dtype=np.float64)
            raman_iso = np.array(raman_iso, dtype=np.float64)
            raman_ani = np.array(raman_ani, dtype=np.float64)
            ir_intens = np.array(ir_intens, dtype=np.float64)

            # ------------------------------------------------
            # Broaden spectra
            # ------------------------------------------------
            x_raman, spec_raman_tot = broaden_spectrum_lorentzian(
                freq, raman_tot, fwhm=raman_fwhm, x_min=x_min, x_max=x_max, step=x_step
            )
            _, spec_raman_iso = broaden_spectrum_lorentzian(
                freq, raman_iso, fwhm=raman_fwhm, x_min=x_min, x_max=x_max, step=x_step
            )
            _, spec_raman_ani = broaden_spectrum_lorentzian(
                freq, raman_ani, fwhm=raman_fwhm, x_min=x_min, x_max=x_max, step=x_step
            )
            x_ir, spec_ir = broaden_spectrum_lorentzian(
                freq, ir_intens, fwhm=ir_fwhm, x_min=x_min, x_max=x_max, step=x_step
            )

            # ------------------------------------------------
            # Normalize
            # Raman iso/ani normalized by total max
            # ------------------------------------------------
            raman_max = np.max(spec_raman_tot)
            if raman_max > 0:
                spec_raman_tot_norm = spec_raman_tot / raman_max
                spec_raman_iso_norm = spec_raman_iso / raman_max
                spec_raman_ani_norm = spec_raman_ani / raman_max
            else:
                spec_raman_tot_norm = spec_raman_tot.copy()
                spec_raman_iso_norm = spec_raman_iso.copy()
                spec_raman_ani_norm = spec_raman_ani.copy()

            spec_ir_norm = safe_normalize(spec_ir)

            # ------------------------------------------------
            # Save raw mode data
            # ------------------------------------------------
            #raw_modes_path = os.path.join(output_dir, f"{name}_modes_raw.csv")
            #pd.DataFrame({
            #    "mode_index": np.arange(len(freq)),
            #    "frequency_cm1": freq,
            #    "raman_total_intensity": raman_tot,
            #    "raman_isotropic_intensity": raman_iso,
            #    "raman_anisotropic_intensity": raman_ani,
            #    "ir_intensity": ir_intens,
            #}).to_csv(raw_modes_path, index=False)

            # Separate raw CSVs if you want the previous naming scheme
            #raw_raman_csv_path = os.path.join(output_dir, f"{name}_raman_raw.csv")
            #pd.DataFrame({
            #   "frequency_cm1": freq,
            #    "total_intensity": raman_tot,
            #    "isotropic_intensity": raman_iso,
            #    "anisotropic_intensity": raman_ani,
            #}).to_csv(raw_raman_csv_path, index=False)

            raw_ir_csv_path = os.path.join(output_dir, f"{name}_ir_raw.csv")
            pd.DataFrame({
                "frequency_cm1": freq,
                "intensity": ir_intens,
            }).to_csv(raw_ir_csv_path, index=False)

            # ------------------------------------------------
            # Save broadened normalized spectra
            # ------------------------------------------------
            #raman_csv_path = os.path.join(output_dir, f"{name}_raman_norm.csv")
            #pd.DataFrame({
            #    "cm1": x_raman,
            #    "total": spec_raman_tot_norm,
            #    "isotropic": spec_raman_iso_norm,
            #    "anisotropic": spec_raman_ani_norm,
            #}).to_csv(raman_csv_path, index=False)

            ir_csv_path = os.path.join(output_dir, f"{name}_ir_norm.csv")
            pd.DataFrame({
                "cm1": x_ir,
                "intensity": spec_ir_norm,
            }).to_csv(ir_csv_path, index=False)

            # ------------------------------------------------
            # Plot Raman
            # ------------------------------------------------
            #plt.figure(figsize=(8, 6))
            #plt.plot(x_raman, spec_raman_tot_norm, label="Total")
            #plt.plot(x_raman, spec_raman_iso_norm, label="Isotropic", alpha=0.8)
            #plt.plot(x_raman, spec_raman_ani_norm, label="Anisotropic", alpha=0.8)
            #plt.xlim(x_min, x_max)
            #plt.xlabel("Raman shift (cm$^{-1}$)")
            #plt.ylabel("Intensity (arb. units)")
            #plt.title(f"{name} Raman spectrum ({size})")
            #plt.legend()
            #plt.tight_layout()
            #plt.savefig(os.path.join(output_dir, f"{name}_raman.png"), dpi=400)
            #plt.close()

            # ------------------------------------------------
            # Plot IR
            # ------------------------------------------------
            plt.figure(figsize=(8, 6))
            plt.plot(x_ir, spec_ir_norm, label="IR")
            plt.xlim(x_min, x_max)
            plt.xlabel("IR shift (cm$^{-1}$)")
            plt.ylabel("Intensity (arb. units)")
            plt.title(f"{name} IR spectrum ({size})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_ir.png"), dpi=400)
            plt.close()

            # ------------------------------------------------
            # Plot combined Raman + IR
            # ------------------------------------------------
            #plt.figure(figsize=(8, 6))
            #plt.plot(x_raman, spec_raman_tot_norm, label="Raman (total)")
            #plt.plot(x_ir, spec_ir_norm, label="IR", alpha=0.9)
            #plt.xlim(x_min, x_max)
            #plt.xlabel("Wavenumber (cm$^{-1}$)")
            #plt.ylabel("Normalized intensity")
            #plt.title(f"{name} Raman + IR spectra ({size})")
            #plt.legend()
            #plt.tight_layout()
            #plt.savefig(os.path.join(output_dir, f"{name}_raman_ir_combined.png"), dpi=400)
            #plt.close()

            success_records.append({
                "name": name,
                "steps": dyn.step_count,
                "n_atoms": n_atoms,
            })

        except Exception as e:
            with open(failed_log, "a") as f:
                f.write(f"Molecule: {name}\n")
                f.write(f"{type(e).__name__}: {e}\n\n")
            print(f"Error in {name}, recorded.")

    pd.DataFrame(success_records).to_csv(
        os.path.join(output_dir, f"successful_molecules_{size}.csv"),
        index=False
    )

    print(f"Finished processing model size: {size}")
