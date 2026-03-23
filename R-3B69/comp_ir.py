#!/usr/bin/env python3
import numpy as np
import glob, os, csv
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# Fast Lorentzian broadening (vectorized)
# ---------------------------------------------------------
def broaden_spectrum_lorentzian(freqs, intens, fwhm=30.0, x_min=0, x_max=4000, step=1.0):
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
    if len(freqs) == 0:
        return x, np.zeros_like(x)
    gamma = fwhm / 2.0
    diff = x[:, None] - freqs[None, :]
    spec = np.sum(intens[None, :] * (gamma / (diff**2 + gamma**2)), axis=1)
    return x, spec

# ---------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Read IR spectrum from DFT input.out
# ---------------------------------------------------------
def read_dft_ir(dft_file: str):
    freq_dft, inten_dft = [], []

    with open(dft_file, "r") as f:
        in_ir = False
        for line in f:
            if "IR SPECTRUM" in line:
                in_ir = True
                continue
            if "first frequency considered" in line and in_ir:
                break
            if not in_ir:
                continue

            s = line.strip()
            if not s:
                continue
            if s.startswith("Mode"):
                continue

            parts = s.split()
            # Expected layout from your old script:
            # freq = parts[1], intensity = parts[3]
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

    max_i = np.nanmax(inten_dft)
    if max_i > 0:
        inten_dft /= max_i

    return freq_dft, inten_dft

# ---------------------------------------------------------
# Read MACE IR raw CSV
# ---------------------------------------------------------
def read_mace_ir_raw(mace_file: str):
    arr = np.genfromtxt(mace_file, delimiter=",", skip_header=1)

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

    max_i = np.nanmax(inten)
    if max_i > 0:
        inten /= max_i

    return freq, inten

# ---------------------------------------------------------
# Compare one compound
# ---------------------------------------------------------
def compare_ir_spectrum(mace_file: str, dft_pattern: str, output_dir: str, fwhm: float = 30.0):
    compound_name = os.path.basename(mace_file).replace("_ir_raw.csv", "")

    dft_files = glob.glob(dft_pattern.format(compound_name=compound_name))
    if not dft_files:
        return (compound_name, None, None, None, None)

    dft_file = dft_files[0]

    try:
        freq_dft, inten_dft = read_dft_ir(dft_file)
        if inten_dft.size == 0:
            return (compound_name, None, None, None, None)
    except Exception:
        return (compound_name, None, None, None, None)

    try:
        freq_mace, inten_mace = read_mace_ir_raw(mace_file)
        if inten_mace.size == 0:
            return (compound_name, None, None, None, None)
    except Exception:
        return (compound_name, None, None, None, None)

    x, spec_dft = broaden_spectrum_lorentzian(freq_dft, inten_dft, fwhm=fwhm)
    _, spec_mace = broaden_spectrum_lorentzian(freq_mace, inten_mace, fwhm=fwhm)

    # Save broadened spectra
    np.savetxt(
        os.path.join(output_dir, f"{compound_name}_broad_dft.csv"),
        np.column_stack([x, spec_dft]),
        delimiter=",",
        header="freq,intensity",
        comments=""
    )
    np.savetxt(
        os.path.join(output_dir, f"{compound_name}_broad_mace.csv"),
        np.column_stack([x, spec_mace]),
        delimiter=",",
        header="freq,intensity",
        comments=""
    )

    nd = np.linalg.norm(spec_dft)
    nm = np.linalg.norm(spec_mace)
    if nd == 0 or nm == 0:
        return (compound_name, None, None, None, None)

    spec_dft /= nd
    spec_mace /= nm

    return (
        compound_name,
        r_msc(spec_dft, spec_mace),
        r_euc(spec_dft, spec_mace),
        r_pcc(spec_dft, spec_mace),
        r_scc(spec_dft, spec_mace),
    )

# ---------------------------------------------------------
# MAIN (3b60 layout; no seeds)
# ---------------------------------------------------------
def main():
    # Adjust this path if your DFT outputs live somewhere else
    dft_pattern = "./bucket*/{compound_name}/input.out"

    mode_to_sizes = {
        "mace_off": ["small", "medium", "large"],
        "mace_omol": ["omol"],
    }

    for mode, sizes in mode_to_sizes.items():
        mode_root = os.path.join(".", mode,"IR")
        if not os.path.isdir(mode_root):
            print(f"Skipping missing mode dir: {mode_root}")
            continue

        for size in sizes:
            results_dir = os.path.join(mode_root, f"ir_results_{size}_SPICE")
            print(f"\nProcessing directory: {results_dir}")

            if not os.path.isdir(results_dir):
                print(f"  Skipping missing: {results_dir}")
                continue

            mace_files = sorted(glob.glob(os.path.join(results_dir, "*_ir_raw.csv")))
            if not mace_files:
                print("  No *_ir_raw.csv files found.")
                continue

            results = [
                compare_ir_spectrum(mf, dft_pattern, results_dir, fwhm=30.0)
                for mf in mace_files
            ]

            match_csv = os.path.join(results_dir, "spectra_matchscores.csv")
            with open(match_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["compound", "rmsc", "reuc", "rpcc", "rscc"])
                for r in results:
                    if r[1] is not None:
                        w.writerow(r)

            valid = [r[1] for r in results if r[1] is not None and not np.isnan(r[1])]
            if valid:
                mean = float(np.mean(valid))
                std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                med = float(np.median(valid))
                print(f"  → Average match score = {mean:.2f} ± {std:.2f} (median = {med:.2f})")
            else:
                print("  → No valid spectra.")

if __name__ == "__main__":
    main()
