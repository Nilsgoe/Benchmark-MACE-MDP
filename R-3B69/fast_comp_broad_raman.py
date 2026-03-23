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
def r_msc(u, v): return (np.dot(u, v)**2) / (np.dot(u, u) * np.dot(v, v))
def r_euc(u, v): return 1.0 / (1.0 + np.sum((u - v)**2) / np.sum(v**2))
def r_pcc(u, v): return pearsonr(u, v)[0]
def r_scc(u, v): return spearmanr(u, v)[0]

# ---------------------------------------------------------
# Read Raman spectrum from DFT input.out (robust-ish)
# ---------------------------------------------------------
def read_dft_raman(dft_file: str):
    freq_dft, act_dft = [], []
    with open(dft_file, "r") as f:
        in_raman = False
        for line in f:
            if "RAMAN SPECTRUM" in line:
                in_raman = True
                continue
            if "first frequency considered" in line and in_raman:
                break
            if not in_raman:
                continue

            s = line.strip()
            if not s:
                continue
            if s.startswith("Mode"):
                continue

            parts = s.split()
            if len(parts) >= 3:
                try:
                    freq_dft.append(float(parts[1]))
                    act_dft.append(float(parts[2]))
                except ValueError:
                    pass

    freq_dft = np.array(freq_dft, dtype=float)
    act_dft = np.array(act_dft, dtype=float)
    if act_dft.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    act_dft /= np.max(act_dft)
    return freq_dft, act_dft

# ---------------------------------------------------------
# Read MACE Raman raw CSV
# ---------------------------------------------------------
def read_mace_raman_raw(mace_file: str):
    arr = np.genfromtxt(mace_file, delimiter=",", skip_header=1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    freq = arr[:, 0].astype(float)
    inten = arr[:, 1].astype(float)
    if inten.size == 0 or np.nanmax(inten) == 0:
        return freq, inten
    inten /= np.nanmax(inten)
    return freq, inten

# ---------------------------------------------------------
# Compare one compound
# ---------------------------------------------------------
def compare_spectrum(mace_file: str, dft_pattern: str, output_dir: str, fwhm: float = 30.0):
    compound_name = os.path.basename(mace_file).replace("_raman_raw.csv", "")

    dft_files = glob.glob(dft_pattern.format(compound_name=compound_name))
    if not dft_files:
        return (compound_name, None, None, None, None)

    dft_file = dft_files[0]

    try:
        freq_dft, act_dft = read_dft_raman(dft_file)
        if act_dft.size == 0:
            return (compound_name, None, None, None, None)
    except Exception:
        return (compound_name, None, None, None, None)

    try:
        freq_mace, inten_mace = read_mace_raman_raw(mace_file)
        if inten_mace.size == 0:
            return (compound_name, None, None, None, None)
    except Exception:
        return (compound_name, None, None, None, None)

    x, spec_dft = broaden_spectrum_lorentzian(freq_dft, act_dft, fwhm=fwhm)
    _, spec_mace = broaden_spectrum_lorentzian(freq_mace, inten_mace, fwhm=fwhm)

    # Save broadened spectra (keeps your previous behavior)
    np.savetxt(
        os.path.join(output_dir, f"{compound_name}_broad_dft.csv"),
        np.column_stack([x, spec_dft]),
        delimiter=",", header="freq,intensity", comments=""
    )
    np.savetxt(
        os.path.join(output_dir, f"{compound_name}_broad_mace.csv"),
        np.column_stack([x, spec_mace]),
        delimiter=",", header="freq,intensity", comments=""
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
# MAIN (no seeds; mace_off and mace_omol are in ./)
# ---------------------------------------------------------
def main():
    # DFT lookup
    dft_pattern = "/ptmp/ngoen/Documents/3b69_Raman/bucket*/{compound_name}/input.out"

    mode_to_sizes = {
        "mace_off": ["small", "medium", "large"],
        "mace_omol": ["omol"],
    }

    for mode, sizes in mode_to_sizes.items():
        mode_root = os.path.join(".", mode)
        if not os.path.isdir(mode_root):
            print(f"Skipping missing mode dir: {mode_root}")
            continue

        for size in sizes:
            results_dir = os.path.join(mode_root, f"raman_results_{size}_SPICE")
            print(f"\nProcessing directory: {results_dir}")

            if not os.path.isdir(results_dir):
                print(f"  Skipping missing: {results_dir}")
                continue

            mace_files = sorted(glob.glob(os.path.join(results_dir, "*_raman_raw.csv")))
            if not mace_files:
                print("  No *_raman_raw.csv files found.")
                continue

            results = [compare_spectrum(mf, dft_pattern, results_dir, fwhm=30.0) for mf in mace_files]

            match_csv = os.path.join(results_dir, "spectra_matchscores.csv")
            with open(match_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["compound", "rmsc", "reuc", "rpcc", "rscc"])
                for r in results:
                    if r[1] is not None:
                        w.writerow(r)

            # ------------------- Print mean + std + median (RMSC)
            valid = [r[1] for r in results if r[1] is not None]
            if valid:
                mean = float(np.mean(valid))
                std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                med = float(np.median(valid))
                print(f"  → Average match score = {mean:.3f} ± {std:.3f} (median = {med:.3f})")
            else:
                print("  → No valid spectra.")

if __name__ == "__main__":
    main()
