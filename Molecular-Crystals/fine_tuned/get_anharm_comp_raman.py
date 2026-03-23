#!/usr/bin/env python3
"""
Compare anharmonic Raman spectra (computed) with experimental spectra.

USAGE:
    python compare_anharmonic_vs_experimental.py

DESCRIPTION:
    - Reads all *_aharm_raman.csv files in sim_dir.
    - Finds matching experimental spectra (same base name) in exp_dir.
    - Plots and saves:
        • individual comparison plots per structure
        • one combined comparison plot with vertical offsets

ASSUMPTIONS:
    - Anharmonic CSVs have columns: freq_cm-1, I_paper, I_placzek
    - Experimental CSVs have columns: Raman_shift, Intensity
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# CONFIGURATION
# ----------------------------
sim_dir = "./anharm"                # directory with anharmonic spectra CSVs
exp_dir = "../exp_spectra"      # directory with experimental spectra CSVs
output_dir = "comparison_plots_anharmonic"

x_min = 0
x_max = 220
intensity_col = "I_paper"    # choose "I_paper" or "I_placzek"

os.makedirs(output_dir, exist_ok=True)
plt.rcParams['axes.linewidth'] = 1.5

# ----------------------------
# COLLECT DATA
# ----------------------------
sim_csvs = sorted(glob.glob(os.path.join(sim_dir, "*_aharm_raman.csv")))

if not sim_csvs:
    raise FileNotFoundError("No *_aharm_raman.csv files found in the directory!")

data_store = {}

for sim_path in sim_csvs:
    base_name = os.path.basename(sim_path).split("_aharm_raman")[0]

    # look for matching experimental CSV
    exp_candidates = glob.glob(os.path.join(exp_dir, f"{base_name}.csv"))
    if not exp_candidates:
        print(f"No experimental CSV found for {base_name}")
        continue
    exp_path = exp_candidates[0]

    print(f"Processing {base_name}...")
    print(f"Found experimental data: {exp_path}")

    # Load anharmonic simulated data
    sim_df = pd.read_csv(sim_path)
    if intensity_col not in sim_df.columns:
        raise KeyError(f"{sim_path} does not contain column '{intensity_col}'")

    sim_df = sim_df[(sim_df["freq_cm-1"] >= x_min) & (sim_df["freq_cm-1"] <= x_max)]
    sim_x = sim_df["freq_cm-1"].values
    sim_y = sim_df[intensity_col].values
    sim_y = sim_y / sim_y.max()  # normalize to 1

    # Load experimental data
    exp_df = pd.read_csv(exp_path)
    exp_df = exp_df[(exp_df.iloc[:, 0] >= x_min) & (exp_df.iloc[:, 0] <= x_max)]
    exp_x = exp_df.iloc[:, 0].values
    exp_y = exp_df.iloc[:, 1].values
    exp_y = exp_y - np.min(exp_y)  # shift to zero
    exp_y = exp_y / np.max(exp_y)  # normalize to 1

    # ----------------------------
    # INDIVIDUAL PLOT
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(exp_x, exp_y, label="Experimental", linewidth=2, color="black")
    plt.plot(sim_x, sim_y, label="Anharmonic (this work)", linewidth=2, color="red")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized intensity")
    plt.title(f"{base_name} – Experimental vs Anharmonic Raman spectra")
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{base_name}_exp_vs_anharmonic.png")
    plt.savefig(out_path, dpi=400)
    plt.close()
    print(f"Saved: {out_path}")

    # ----------------------------
    # STORE for combined plotting
    # ----------------------------
    if "_" in base_name:
        root = "_".join(base_name.split("_")[:-1])
    else:
        root = base_name

    data_store.setdefault(root, []).append({
        "name": base_name,
        "exp_x": exp_x, "exp_y": exp_y,
        "sim_x": sim_x, "sim_y": sim_y
    })

# ----------------------------
# COMBINED PLOT (always generated)
# ----------------------------
import math

if not data_store:
    raise RuntimeError("No data found for combined plot.")

style = ["-", ":", "--"]
n_groups = len(data_store)

# --- Arrange plots in 2 columns ---
ncols = 2
nrows = math.ceil(n_groups / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)
axes = axes.flatten()  # Flatten for easy iteration

for ax, (root, entries) in zip(axes, data_store.items()):
    for idx, entry in enumerate(entries):
        shift = idx * 1.2  # vertical shift factor
        ax.plot(entry["exp_x"], entry["exp_y"] + shift,
                label=f"{entry['name']} Exp", linewidth=2, color='black',
                linestyle=style[idx % len(style)])
        ax.plot(entry["sim_x"], entry["sim_y"] + shift,
                label=f"{entry['name']} Anharm", linewidth=2, color='red',
                linestyle=style[idx % len(style)])

    ax.set_title(root)
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("Intensity (a.u.) + offset")
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.legend(fontsize=8)

# --- Hide empty subplot boxes if odd number of groups ---
for ax in axes[n_groups:]:
    ax.axis("off")

plt.tight_layout()
comb_path = os.path.join(output_dir, "combined_comparison_anharmonic.png")
plt.savefig(comb_path, dpi=400)
plt.close()
print(f"Saved combined plot: {comb_path}")

print("All anharmonic comparison plots generated successfully.")
