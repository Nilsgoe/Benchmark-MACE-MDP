#!/usr/bin/env python3
import os
import re
import glob
import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ============================================================
# SETTINGS
# ============================================================
RESULTS_DIR = "./from_dft/spectra_from_hessian_modes"
FWHM = 30.0

# where the original ORCA outputs live
DFT_PATTERNS = [
    "./bucket_0/{compound_name}/input.out",
    "./{compound_name}/input.out",
]

# ============================================================
# BROADENING
# ============================================================
def broaden_spectrum_lorentzian(freqs, intens, fwhm=10.0, x_min=0, x_max=4000, step=1.0):
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
    gamma = fwhm / 2.0
    if len(freqs) == 0:
        return x, np.zeros_like(x)
    diff = x[:, None] - freqs[None, :]
    spec = np.sum(intens[None, :] * (gamma / (diff**2 + gamma**2)), axis=1)
    return x, spec

# ============================================================
# SIMILARITY METRICS
# ============================================================
def r_msc(u, v):
    du = np.dot(u, u)
    dv = np.dot(v, v)
    if du == 0 or dv == 0:
        return np.nan
    return (np.dot(u, v) ** 2) / (du * dv)

def r_euc(u, v):
    dv = np.sum(v ** 2)
    if dv == 0:
        return np.nan
    return 1.0 / (1.0 + np.sum((u - v) ** 2) / dv)

def r_pcc(u, v):
    if np.allclose(u, u[0]) or np.allclose(v, v[0]):
        return np.nan
    return pearsonr(u, v)[0]

def r_scc(u, v):
    if np.allclose(u, u[0]) or np.allclose(v, v[0]):
        return np.nan
    return spearmanr(u, v)[0]

# ============================================================
# HELPERS
# ============================================================
def find_dft_file(compound_name: str):
    for pattern in DFT_PATTERNS:
        matches = glob.glob(pattern.format(compound_name=compound_name))
        if matches:
            return matches[0]
    return None

def read_pred_raw_csv(path: str):
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        if arr.size < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    freq = arr[:, 0].astype(float)
    inten = arr[:, 1].astype(float)

    if inten.size == 0:
        return freq, inten

    m = np.nanmax(inten)
    if m > 0:
        inten = inten / m
    return freq, inten

def read_dft_ir(out_file: str):
    freq_dft, inten_dft = [], []
    with open(out_file, "r") as f:
        in_ir = False
        for line in f:
            if "IR SPECTRUM" in line:
                in_ir = True
                continue
            if "first frequency considered" in line and in_ir:
                break
            if not in_ir:
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    freq_dft.append(float(parts[1]))
                    inten_dft.append(float(parts[3]))
                except ValueError:
                    pass

    freq_dft = np.array(freq_dft, dtype=float)
    inten_dft = np.array(inten_dft, dtype=float)

    if inten_dft.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    m = np.max(inten_dft)
    if m > 0:
        inten_dft /= m
    return freq_dft, inten_dft

def read_dft_raman(out_file: str):
    freq_dft, inten_dft = [], []
    with open(out_file, "r") as f:
        in_raman = False
        for line in f:
            if "RAMAN SPECTRUM" in line:
                in_raman = True
                continue
            if "first frequency considered" in line and in_raman:
                break
            if not in_raman:
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    freq_dft.append(float(parts[1]))
                    inten_dft.append(float(parts[2]))
                except ValueError:
                    pass

    freq_dft = np.array(freq_dft, dtype=float)
    inten_dft = np.array(inten_dft, dtype=float)

    if inten_dft.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    m = np.max(inten_dft)
    if m > 0:
        inten_dft /= m
    return freq_dft, inten_dft

def compare_one_spectrum(compound_name: str, mode: str, results_dir: str, fwhm: float = 30.0):
    """
    mode = 'ir' or 'raman'
    """
    pred_file = os.path.join(results_dir, f"{compound_name}_{mode}_raw.csv")
    if not os.path.exists(pred_file):
        return (compound_name, None, None, None, None)

    dft_file = find_dft_file(compound_name)
    if dft_file is None:
        return (compound_name, None, None, None, None)

    freq_pred, inten_pred = read_pred_raw_csv(pred_file)
    if inten_pred.size == 0:
        return (compound_name, None, None, None, None)

    if mode == "ir":
        freq_dft, inten_dft = read_dft_ir(dft_file)
    elif mode == "raman":
        freq_dft, inten_dft = read_dft_raman(dft_file)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if inten_dft.size == 0:
        return (compound_name, None, None, None, None)

    x, spec_dft = broaden_spectrum_lorentzian(freq_dft, inten_dft, fwhm=fwhm)
    _, spec_pred = broaden_spectrum_lorentzian(freq_pred, inten_pred, fwhm=fwhm)

    # save broadened spectra
    np.savetxt(
        os.path.join(results_dir, f"{compound_name}_{mode}_broad_dft.csv"),
        np.column_stack([x, spec_dft]),
        delimiter=",",
        header="freq,intensity",
        comments=""
    )
    np.savetxt(
        os.path.join(results_dir, f"{compound_name}_{mode}_broad_pred.csv"),
        np.column_stack([x, spec_pred]),
        delimiter=",",
        header="freq,intensity",
        comments=""
    )

    nd = np.linalg.norm(spec_dft)
    npred = np.linalg.norm(spec_pred)
    if nd == 0 or npred == 0:
        return (compound_name, None, None, None, None)

    spec_dft /= nd
    spec_pred /= npred

    return (
        compound_name,
        r_msc(spec_dft, spec_pred),
        r_euc(spec_dft, spec_pred),
        r_pcc(spec_dft, spec_pred),
        r_scc(spec_dft, spec_pred),
    )

# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.isdir(RESULTS_DIR):
        raise RuntimeError(f"Results directory not found: {RESULTS_DIR}")

    ir_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_ir_raw.csv")))
    raman_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_raman_raw.csv")))

    ir_compounds = [os.path.basename(f).replace("_ir_raw.csv", "") for f in ir_files]
    raman_compounds = [os.path.basename(f).replace("_raman_raw.csv", "") for f in raman_files]

    # ---------------- IR ----------------
    ir_results = []
    print("\n=== Comparing IR spectra ===")
    for compound in ir_compounds:
        print(f" -> {compound}")
        ir_results.append(compare_one_spectrum(compound, "ir", RESULTS_DIR, fwhm=FWHM))

    ir_csv = os.path.join(RESULTS_DIR, "ir_spectra_matchscores.csv")
    with open(ir_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound", "rmsc", "reuc", "rpcc", "rscc"])
        for r in ir_results:
            if r[1] is not None and not np.isnan(r[1]):
                w.writerow(r)

    valid_ir = [r[1] for r in ir_results if r[1] is not None and not np.isnan(r[1])]
    if valid_ir:
        mean = float(np.mean(valid_ir))
        std = float(np.std(valid_ir, ddof=1)) if len(valid_ir) > 1 else 0.0
        med = float(np.median(valid_ir))
        print(f"IR average match score = {mean:.2f} ± {std:.2f} (median = {med:.2f})")
    else:
        print("No valid IR spectra.")

    # ---------------- Raman ----------------
    raman_results = []
    print("\n=== Comparing Raman spectra ===")
    for compound in raman_compounds:
        print(f" -> {compound}")
        raman_results.append(compare_one_spectrum(compound, "raman", RESULTS_DIR, fwhm=FWHM))

    raman_csv = os.path.join(RESULTS_DIR, "raman_spectra_matchscores.csv")
    with open(raman_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound", "rmsc", "reuc", "rpcc", "rscc"])
        for r in raman_results:
            if r[1] is not None and not np.isnan(r[1]):
                w.writerow(r)

    valid_raman = [r[1] for r in raman_results if r[1] is not None and not np.isnan(r[1])]
    if valid_raman:
        mean = float(np.mean(valid_raman))
        std = float(np.std(valid_raman, ddof=1)) if len(valid_raman) > 1 else 0.0
        med = float(np.median(valid_raman))
        print(f"Raman average match score = {mean:.2f} ± {std:.2f} (median = {med:.2f})")
    else:
        print("No valid Raman spectra.")

if __name__ == "__main__":
    main()
