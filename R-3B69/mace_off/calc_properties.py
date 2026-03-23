#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
from ase import Atoms
from mace.calculators import mace_off
from mace.calculators.mace import MACECalculator

# ============================================================
# CONFIG
# ============================================================
xyz_file = "../R-3B69.xyz"   # extxyz with reference properties
model_base_path = "MACE-MDP.model"
model_sizes = ["small", "medium", "large"]
device = "cuda"
default_dtype = "float64"

output_prefix = "properties_vs_xyzref"  # -> properties_vs_xyzref_small.csv etc.

# Dipole/polarizability model (prediction)
polar_calc = MACECalculator(
    model_paths=model_base_path,
    model_type="DipolePolarizabilityMACE",
    device=device,
    default_dtype=default_dtype,
)

# ============================================================
# Parsing helpers (robust to your exact header format)
# ============================================================
re_name = re.compile(r'name="([^"]+)"')
re_energy = re.compile(r'\benergy_eV=([-\d\.eE+]+)\b')
re_dipole = re.compile(r'dipole_eA="([^"]+)"')
re_alpha6 = re.compile(r'polarizability_eA2V="([^"]+)"')

def parse_header(comment: str):
    """Return (name, energy_eV, dipole(3,), alpha6(6,)) from comment line."""
    m = re_name.search(comment)
    name = m.group(1) if m else "unknown"

    m = re_energy.search(comment)
    if not m:
        raise ValueError("Missing energy_eV in header")
    energy = float(m.group(1))

    m = re_dipole.search(comment)
    if not m:
        raise ValueError("Missing dipole_eA in header")
    dip_parts = m.group(1).replace(",", " ").split()
    if len(dip_parts) != 3:
        raise ValueError(f"Bad dipole_eA format: {m.group(1)}")
    dip = np.array([float(x) for x in dip_parts], dtype=float)

    m = re_alpha6.search(comment)
    if not m:
        raise ValueError("Missing polarizability_eA2V in header")
    a_parts = m.group(1).replace(",", " ").split()
    if len(a_parts) != 6:
        raise ValueError(f"Bad polarizability_eA2V format (expected 6): {m.group(1)}")
    a6 = np.array([float(x) for x in a_parts], dtype=float)  # xx xy xz yy yz zz

    return name, energy, dip, a6

def forces_to_string(F: np.ndarray) -> str:
    """(N,3) -> 'fx,fy,fz;fx,fy,fz;...'"""
    F = np.asarray(F, float)
    return ";".join([f"{fx:.10f},{fy:.10f},{fz:.10f}" for fx, fy, fz in F])

def parse_extxyz_frames(path: str):
    """
    Manual parser for extxyz-like XYZ:
    - Reads N
    - Reads comment
    - Reads N atom lines: symbol x y z fx fy fz  (forces optional but expected here)
    Returns list of dicts with symbols, positions, ref_forces, ref_energy, ref_dipole, ref_alpha6, name.
    """
    frames = []
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    frame_index = 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue

        N = int(lines[i].strip())
        comment = lines[i + 1].strip()

        name, ref_energy, ref_dip, ref_a6 = parse_header(comment)

        symbols = []
        pos = np.zeros((N, 3), dtype=float)
        ref_forces = np.zeros((N, 3), dtype=float)

        for j in range(N):
            parts = lines[i + 2 + j].split()
            if len(parts) < 4:
                raise ValueError(f"Bad atom line (too few columns): {lines[i+2+j]}")
            symbols.append(parts[0])
            pos[j, 0] = float(parts[1])
            pos[j, 1] = float(parts[2])
            pos[j, 2] = float(parts[3])

            # Forces expected (last 3 columns) given Properties includes forces
            if len(parts) >= 7:
                ref_forces[j, 0] = float(parts[4])
                ref_forces[j, 1] = float(parts[5])
                ref_forces[j, 2] = float(parts[6])
            else:
                raise ValueError("Forces missing from atom lines, but expected 7 columns.")

        frames.append({
            "frame_index": frame_index,
            "name": name.split("/")[-1],
            "n_atoms": N,
            "symbols": symbols,
            "positions": pos,
            "ref_energy_eV": ref_energy,
            "ref_dipole": ref_dip,
            "ref_alpha6": ref_a6,      # xx xy xz yy yz zz
            "ref_forces": ref_forces,  # (N,3)
        })

        frame_index += 1
        i += N + 2

    return frames

def alpha6_from_tensor(alpha3x3: np.ndarray) -> np.ndarray:
    """(3,3) -> xx,xy,xz,yy,yz,zz"""
    a = np.asarray(alpha3x3, float).reshape(3, 3)
    return np.array([a[0,0], a[0,1], a[0,2], a[1,1], a[1,2], a[2,2]], dtype=float)

# ============================================================
# Load reference frames from XYZ
# ============================================================
ref_frames = parse_extxyz_frames(xyz_file)
print(f"Loaded {len(ref_frames)} frames from {xyz_file}")

# ============================================================
# Compute predictions per model size; write CSVs in cwd
# ============================================================
for size in model_sizes:
    print(f"\nProcessing model size: {size}")
    off_calc = mace_off(model=size, default_dtype=default_dtype, device=device)

    rows = []
    failed = []

    for fr in ref_frames:
        idx = fr["frame_index"]
        name = fr["name"]
        N = fr["n_atoms"]

        try:
            # Build ASE atoms from symbols/positions for prediction
            at = Atoms(symbols=fr["symbols"], positions=fr["positions"])

            # Predict energy + forces
            at.calc = off_calc
            pred_energy = float(at.get_potential_energy())
            pred_forces = np.array(at.get_forces(), dtype=float)

            # Predict dipole + polarizability
            pred_dip = np.array(polar_calc.get_property("dipole", at), dtype=float).reshape(3)
            pred_alpha = np.array(polar_calc.get_property("polarizability", at), dtype=float).reshape(3, 3)
            pred_a6 = alpha6_from_tensor(pred_alpha)

            # Store full forces as strings
            ref_forces_str = forces_to_string(fr["ref_forces"])
            pred_forces_str = forces_to_string(pred_forces)

            # Force quick summary
            ref_fn = np.linalg.norm(fr["ref_forces"], axis=1)
            pred_fn = np.linalg.norm(pred_forces, axis=1)

            row = {
                "name": name,
                "frame_index": idx,
                "n_atoms": N,

                # Reference from XYZ
                "ref_energy_eV": fr["ref_energy_eV"],
                "ref_forces_eVA_full": ref_forces_str,
                "ref_dipole_x_eA": float(fr["ref_dipole"][0]),
                "ref_dipole_y_eA": float(fr["ref_dipole"][1]),
                "ref_dipole_z_eA": float(fr["ref_dipole"][2]),
                "ref_alpha_xx_eA2V": float(fr["ref_alpha6"][0]),
                "ref_alpha_xy_eA2V": float(fr["ref_alpha6"][1]),
                "ref_alpha_xz_eA2V": float(fr["ref_alpha6"][2]),
                "ref_alpha_yy_eA2V": float(fr["ref_alpha6"][3]),
                "ref_alpha_yz_eA2V": float(fr["ref_alpha6"][4]),
                "ref_alpha_zz_eA2V": float(fr["ref_alpha6"][5]),
                "ref_fmax_eVA": float(np.max(ref_fn)),

                # Predicted
                "pred_energy_eV": pred_energy,
                "pred_forces_eVA_full": pred_forces_str,
                "pred_dipole_x_eA": float(pred_dip[0]),
                "pred_dipole_y_eA": float(pred_dip[1]),
                "pred_dipole_z_eA": float(pred_dip[2]),
                "pred_alpha_xx_eA2V": float(pred_a6[0]),
                "pred_alpha_xy_eA2V": float(pred_a6[1]),
                "pred_alpha_xz_eA2V": float(pred_a6[2]),
                "pred_alpha_yy_eA2V": float(pred_a6[3]),
                "pred_alpha_yz_eA2V": float(pred_a6[4]),
                "pred_alpha_zz_eA2V": float(pred_a6[5]),
                "pred_fmax_eVA": float(np.max(pred_fn)),
            }
            rows.append(row)

        except Exception as e:
            failed.append((name, idx, type(e).__name__, str(e)))
            print(f"Error in {name} (frame {idx}): {type(e).__name__}: {e}")

    out_csv = f"{output_prefix}_{size}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(rows)} records)")

    if failed:
        fail_csv = f"{output_prefix}_{size}_failed.csv"
        pd.DataFrame(failed, columns=["name", "frame_index", "error_type", "error_msg"]).to_csv(fail_csv, index=False)
        print(f"Wrote: {fail_csv} ({len(failed)} records)")

print("\nDone.")
