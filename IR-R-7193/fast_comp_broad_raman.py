#!/usr/bin/env python3
import numpy as np
import glob, os, csv
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# Fast Lorentzian broadening (vectorized)
# ---------------------------------------------------------
def broaden_spectrum_lorentzian(freqs, intens, fwhm=30.0, x_min=0, x_max=4000, step=1.0):
    if len(freqs) == 0:
        x = np.arange(x_min, x_max + step, step, dtype=np.float64)
        return x, np.zeros_like(x)
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
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
# Compare a single spectrum and save broadened spectra
# ---------------------------------------------------------
def compare_spectrum(mace_file, dft_patterns, output_dir):
    compound_name = os.path.basename(mace_file).replace("_raman_raw.csv", "")

    # ------------------------- Locate matching DFT file
    dft_file = None
    for pat in dft_patterns:
        files = glob.glob(pat.format(compound_name=compound_name))
        if files:
            dft_file = files[0]
            break
    if dft_file is None:
        return (compound_name, None, None, None, None)

    # ------------------------- Read DFT data
    freq_dft, act_dft = [], []
    try:
        with open(dft_file, "r") as f:
            in_raman = False
            for line in f:
                if "RAMAN SPECTRUM" in line:
                    in_raman = True
                    continue
                if "first frequency considered" in line and in_raman:
                    break
                if in_raman:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            freq_dft.append(float(parts[1]))
                            act_dft.append(float(parts[2]))
                        except:
                            pass
        freq_dft = np.array(freq_dft)
        act_dft = np.array(act_dft)
        if len(act_dft) == 0:
            return (compound_name, None, None, None, None)
        act_dft /= np.max(act_dft)
    except:
        return (compound_name, None, None, None, None)

    # ------------------------- Read MACE data
    try:
        data = np.genfromtxt(mace_file, delimiter=",", skip_header=1)
        freq_mace = data[:, 0]
        inten_mace = data[:, 1]
        inten_mace /= np.max(inten_mace)
    except:
        return (compound_name, None, None, None, None)

    # ------------------------- Broaden both
    x, spec_dft = broaden_spectrum_lorentzian(freq_dft, act_dft, fwhm=30.0)
    _, spec_mace = broaden_spectrum_lorentzian(freq_mace, inten_mace, fwhm=30.0)

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

    # Normalize for similarity
    spec_dft /= np.linalg.norm(spec_dft)
    spec_mace /= np.linalg.norm(spec_mace)

    # ------------------------- Return metrics
    return (
        compound_name,
        r_msc(spec_dft, spec_mace),
        r_euc(spec_dft, spec_mace),
        r_pcc(spec_dft, spec_mace),
        r_scc(spec_dft, spec_mace),
    )

# ---------------------------------------------------------
# MAIN driver for seed_*/mace_off and seed_*/mace_omol
# ---------------------------------------------------------
def main():

    # DFT lookup
    dft_patterns = [
        "./bucket*/HCNOFClBr_{compound_name}/input.out",
        "./bucket*/S_{compound_name}/input.out",
        "./bucket*/P_{compound_name}/input.out",
    ]


   
    no_cueq_dir = "./"
    for mode in ["mace_off", "mace_omol"]:
        mode_dir = os.path.join(no_cueq_dir, mode)
        if not os.path.isdir(mode_dir):
            print(f"Skipping missing: {mode_dir}")
            continue
        for size in ["small", "medium", "large", "omol_actvity_full"]:
            if mode != "mace_omol" and size == "omol_actvity_full":
                continue
            elif mode == "mace_omol" and size != "omol_actvity_full":
                continue

            mode_dir = os.path.join(no_cueq_dir, mode, f"raman_results_{size}_SPICE")
            print(f"\nProcessing directory: {mode_dir}")

            mace_files = sorted(glob.glob(os.path.join(mode_dir, "*_raman_raw.csv")))
            if not mace_files:
                print("  No *_raman_raw.csv files found.")
                continue

            results = []
            match_csv = os.path.join(mode_dir, "spectra_matchscores.csv")

            for mf in mace_files:
                res = compare_spectrum(mf, dft_patterns, mode_dir)
                results.append(res)

            # ------------------- Write match scores
            with open(match_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["compound", "rmsc", "reuc", "rpcc", "rscc"])
                for r in results:
                    if r[1] is not None:
                        writer.writerow(r)

            # ------------------- Print mean + std (of rmsc)
            valid = [r[1] for r in results if r[1] is not None]
            if valid:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
                print(f"  → Average match score (RMSC) = {mean:.3f} ± {std:.3f}")
            else:
                print("  → No valid spectra.")

if __name__ == "__main__":
    main()
