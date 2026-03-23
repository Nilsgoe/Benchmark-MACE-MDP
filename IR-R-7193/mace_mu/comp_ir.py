#!/usr/bin/env python3
import numpy as np
import glob, os, csv
from scipy.stats import pearsonr, spearmanr

# ============================================================
# Vectorized Lorentzian Broadening
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
# Similarity Metrics
# ============================================================
def r_msc(u, v): return (np.dot(u, v)**2) / (np.dot(u, u) * np.dot(v, v))
def r_euc(u, v): return 1.0 / (1.0 + np.sum((u - v)**2) / np.sum(v**2))
def r_pcc(u, v): return pearsonr(u, v)[0]
def r_scc(u, v): return spearmanr(u, v)[0]

# ============================================================
# Discover IR result directories (mace_mu layout)
# ============================================================
search_patterns = [
    "ir_results*",
    "IR_results*",
    "*ir_results*",
    "*IR_results*"
]

mode_dirs = ["mace_off", "mace_omol"]
mace_sets = []

for mode in mode_dirs:
    if not os.path.isdir(mode):
        continue

    found = []
    for pat in search_patterns:
        found.extend(glob.glob(os.path.join(mode, pat)))

    found = sorted(set(found))

    for rd in found:
        if os.path.isdir(rd):
            mace_sets.append((mode, rd))

print(f"Discovered {len(mace_sets)} IR result sets:")
for m in mace_sets:
    print("  ", m)

# ============================================================
# DFT reference lookup
# ============================================================
dft_base = [
    "/ptmp/ngoen/Documents/Raman_7193_IR/bucket*/HCNOFClBr_{compound_name}/input.out",
    "/ptmp/ngoen/Documents/Raman_7193_IR/bucket*/S_{compound_name}/input.out",
    "/ptmp/ngoen/Documents/Raman_7193_IR/bucket*/P_{compound_name}/input.out",
]

# ============================================================
# Main IR Comparison Routine (mace_mu version)
# ============================================================
for mode, mace_dir in mace_sets:

    print(f"\n=== Processing: {mace_dir} ===")

    # Write results directly into mace_dir
    csv_file = os.path.join(mace_dir, "spectra_matchscores.csv")

    mace_files = sorted(glob.glob(os.path.join(mace_dir, "*_ir_raw.csv")))
    print(f"Found {len(mace_files)} IR raw spectra")

    rows = []
    rmscs = []

    for idx, mace_file in enumerate(mace_files, start=1):
        compound = os.path.basename(mace_file).replace("_ir_raw.csv", "")
        print(f"[{idx}/{len(mace_files)}] {compound}")

        # -----------------------------------------------
        # Find DFT file
        # -----------------------------------------------
        dft_file = None
        for pattern in dft_base:
            matches = glob.glob(pattern.format(compound_name=compound))
            if matches:
                dft_file = matches[0]
                break

        if dft_file is None:
            print("  No DFT file found.")
            continue

        # -----------------------------------------------
        # Read DFT spectrum
        # -----------------------------------------------
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
                        if len(parts) >= 4:
                            try:
                                freq_dft.append(float(parts[1]))
                                inten_dft.append(float(parts[3]))
                            except:
                                pass
        except:
            continue

        freq_dft = np.array(freq_dft)
        inten_dft = np.array(inten_dft)
        if len(inten_dft) == 0:
            continue
        inten_dft /= np.max(inten_dft)

        # -----------------------------------------------
        # Read MACE IR spectrum
        # -----------------------------------------------
        try:
            arr = np.genfromtxt(mace_file, delimiter=",", skip_header=1)
            freq_mace, inten_mace = arr[:, 0], arr[:, 1]
            inten_mace /= np.max(inten_mace)
        except:
            continue

        # -----------------------------------------------
        # Broaden spectra
        # -----------------------------------------------
        x, bdft = broaden_spectrum_lorentzian(freq_dft, inten_dft, fwhm=30.0)
        _, bmace = broaden_spectrum_lorentzian(freq_mace, inten_mace, fwhm=30.0)

        # Save broadened spectra in SAME directory
        np.savetxt(
            os.path.join(mace_dir, f"{compound}_dft_broad.csv"),
            np.column_stack([x, bdft]),
            delimiter=",", header="freq,intensity", comments=""
        )

        np.savetxt(
            os.path.join(mace_dir, f"{compound}_mace_broad.csv"),
            np.column_stack([x, bmace]),
            delimiter=",", header="freq,intensity", comments=""
        )

        # Normalize
        bdft /= np.linalg.norm(bdft)
        bmace /= np.linalg.norm(bmace)

        # -----------------------------------------------
        # Metrics
        # -----------------------------------------------
        msc = r_msc(bdft, bmace)
        euc = r_euc(bdft, bmace)
        pcc = r_pcc(bdft, bmace)
        scc = r_scc(bdft, bmace)

        rmscs.append(msc)
        rows.append([compound, msc, euc, pcc, scc])

    # -----------------------------------------------
    # Write match score CSV
    # -----------------------------------------------
    if rows:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["compound", "rmsc", "reuc", "rpcc", "rscc"])
            writer.writerows(rows)

    # -----------------------------------------------
    # Print summary line (mean + std of RMSC)
    # Keep the prefix EXACT, add std after it.
    # -----------------------------------------------
    if rmscs:
        mean = float(np.mean(rmscs))
        std = float(np.std(rmscs, ddof=1)) if len(rmscs) > 1 else 0.0
        print(f"Average match score in this directory = {mean:.3f} (std = {std:.3f})")
    else:
        print("No valid comparisons.")
