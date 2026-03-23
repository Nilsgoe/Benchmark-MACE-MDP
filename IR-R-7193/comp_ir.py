#!/usr/bin/env python3
import numpy as np
import glob, os, csv
from scipy.stats import pearsonr, spearmanr

# ============================================================
# Vectorized Lorentzian Broadening
# ============================================================
def broaden_spectrum_lorentzian(freqs, intens, fwhm=30.0, x_min=0, x_max=4000, step=1.0):
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
    gamma = fwhm / 2.0
    if len(freqs) == 0:
        return x, np.zeros_like(x)
    diff = x[:, None] - freqs[None, :]
    spec = np.sum(intens[None, :] * (gamma / (diff**2 + gamma**2)), axis=1)
    return x, spec

# ============================================================
# Similarity Metrics
# ============================================================
def r_msc(u, v): return (np.dot(u, v)**2) / (np.dot(u, u) * np.dot(v, v))
def r_euc(u, v): return 1.0 / (1.0 + np.sum((u - v)**2) / np.sum(v**2))
def r_pcc(u, v): return pearsonr(u, v)[0]
def r_scc(u, v): return spearmanr(u, v)[0]

# ============================================================
# Compare a single IR spectrum and save broadened spectra
# ============================================================
def compare_ir_spectrum(mace_file, dft_patterns, output_dir):
    compound = os.path.basename(mace_file).replace("_ir_raw.csv", "")

    # ------------------------- Locate matching DFT file
    dft_file = None
    for pat in dft_patterns:
        files = glob.glob(pat.format(compound_name=compound))
        if files:
            dft_file = files[0]
            break
    if dft_file is None:
        return (compound, None, None, None, None)

    # ------------------------- Read DFT data
    freq_dft, inten_dft = [], []
    try:
        with open(dft_file, "r") as f:
            in_ir = False
            for line in f:
                if "IR SPECTRUM" in line:
                    in_ir = True
                    continue
                if "first frequency considered" in line and in_ir:
                    break
                if in_ir:
                    parts = line.split()
                    # In your original: freq=parts[1], intensity=parts[3]
                    if len(parts) >= 4:
                        try:
                            freq_dft.append(float(parts[1]))
                            inten_dft.append(float(parts[3]))
                        except:
                            pass

        freq_dft = np.array(freq_dft)
        inten_dft = np.array(inten_dft)
        if len(inten_dft) == 0:
            return (compound, None, None, None, None)
        inten_dft /= np.max(inten_dft)
    except:
        return (compound, None, None, None, None)

    # ------------------------- Read MACE data
    try:
        arr = np.genfromtxt(mace_file, delimiter=",", skip_header=1)
        freq_mace, inten_mace = arr[:, 0], arr[:, 1]
        inten_mace /= np.max(inten_mace)
    except:
        return (compound, None, None, None, None)

    # ------------------------- Broaden both
    x, spec_dft = broaden_spectrum_lorentzian(freq_dft, inten_dft, fwhm=30.0)
    _, spec_mace = broaden_spectrum_lorentzian(freq_mace, inten_mace, fwhm=30.0)

    # Save broadened spectra (same directory, like Raman script)
    np.savetxt(
        os.path.join(output_dir, f"{compound}_broad_dft.csv"),
        np.column_stack([x, spec_dft]),
        delimiter=",", header="freq,intensity", comments=""
    )

    np.savetxt(
        os.path.join(output_dir, f"{compound}_broad_mace.csv"),
        np.column_stack([x, spec_mace]),
        delimiter=",", header="freq,intensity", comments=""
    )

    # Normalize for similarity
    spec_dft /= np.linalg.norm(spec_dft)
    spec_mace /= np.linalg.norm(spec_mace)

    # ------------------------- Return metrics
    return (
        compound,
        r_msc(spec_dft, spec_mace),
        r_euc(spec_dft, spec_mace),
        r_pcc(spec_dft, spec_mace),
        r_scc(spec_dft, spec_mace),
    )

# ============================================================
# MAIN driver for seed_*/no_cueq/{mace_off,mace_omol}/IR
# ============================================================
def main():

    # DFT lookup (as in your IR script)
    dft_patterns = [
        "./bucket*/HCNOFClBr_{compound_name}/input.out",
        "./bucket*/S_{compound_name}/input.out",
        "./bucket*/P_{compound_name}/input.out",
    ]


    no_cueq_dir = "./"
    for mode in ["mace_off", "mace_omol"]:
        ir_root = os.path.join(no_cueq_dir, mode, "IR")
        if not os.path.isdir(ir_root):
            print(f"Skipping missing: {ir_root}")
            continue

        # Match your Raman logic for size handling
        for size in ["small", "medium", "large", "omol"]:
            if mode != "mace_omol" and size == "omol":
                continue
            elif mode == "mace_omol" and size != "omol":
                continue

            ir_dir = os.path.join(ir_root, f"ir_results_{size}_SPICE")
            print(f"\nProcessing directory: {ir_dir}")

            if not os.path.isdir(ir_dir):
                print(f"  Skipping missing: {ir_dir}")
                continue

            mace_files = sorted(glob.glob(os.path.join(ir_dir, "*_ir_raw.csv")))
            if not mace_files:
                print("  No *_ir_raw.csv files found.")
                continue

            results = []
            match_csv = os.path.join(ir_dir, "spectra_matchscores.csv")

            for mf in mace_files:
                res = compare_ir_spectrum(mf, dft_patterns, ir_dir)
                results.append(res)

            # ------------------- Write match scores
            with open(match_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["compound", "rmsc", "reuc", "rpcc", "rscc"])
                for r in results:
                    if r[1] is not None:
                        writer.writerow(r)

            # ------------------- Print mean + std (RMSC)
            valid = [r[1] for r in results if r[1] is not None]
            if valid:
                mean = float(np.mean(valid))
                std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                print(f"  → Average match score = {mean:.3f} ± {std:.3f}")
            else:
                print("  → No valid spectra.")

if __name__ == "__main__":
    main()
