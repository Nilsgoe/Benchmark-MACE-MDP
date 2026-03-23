#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Helper functions
# ============================================================
def load_rmsc(csv_path: str) -> np.ndarray:
    if not os.path.exists(csv_path):
        print(f"⚠️ Missing file: {csv_path}")
        return np.array([], dtype=float)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Failed to read {csv_path}: {e}")
        return np.array([], dtype=float)
    if "rmsc" not in df.columns:
        print(f"⚠️ No 'rmsc' column in: {csv_path}")
        return np.array([], dtype=float)
    return df["rmsc"].dropna().to_numpy(dtype=float)

# ============================================================
# Configuration
# ============================================================
RAMAN_CSV_OFF = "./mace_off/raman_results_large_SPICE/spectra_matchscores.csv"
IR_CSV_OFF = "./mace_off/IR/ir_results_large_SPICE/spectra_matchscores.csv"

RAMAN_CSV_DFT = "./from_dft/spectra_from_hessian_modes/raman_spectra_matchscores.csv"
IR_CSV_DFT = "./from_dft/spectra_from_hessian_modes/ir_spectra_matchscores.csv"

OUT_PNG = "RMSC_histograms_IR_Raman_OFF_and_DFT_2x2.png"
OUT_PDF = "RMSC_histograms_IR_Raman_OFF_and_DFT_2x2.pdf"

COLOR_RAMAN = "royalblue"
COLOR_IR = "indianred"

# ============================================================
# Load data
# ============================================================
raman_vals_off = load_rmsc(RAMAN_CSV_OFF)
ir_vals_off = load_rmsc(IR_CSV_OFF)
raman_vals_dft = load_rmsc(RAMAN_CSV_DFT)
ir_vals_dft = load_rmsc(IR_CSV_DFT)

all_arrays = [raman_vals_off, ir_vals_off, raman_vals_dft, ir_vals_dft]
non_empty = [v for v in all_arrays if v.size > 0]

if not non_empty:
    raise RuntimeError("No valid RMSC data found in any input CSV.")

# shared x-range / bins
all_vals = np.concatenate(non_empty)
xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
if np.isclose(xmin, xmax):
    xmin = max(0.0, xmin - 0.05)
    xmax = min(1.0, xmax + 0.05)
bins = np.linspace(xmin, xmax, 25)

# ============================================================
# Plot
# ============================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.3,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 13.5,
})

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=False)

panels = [
    (axes[0, 0], raman_vals_off, COLOR_RAMAN, "Raman", "MACE-MDP@MACE-OFF23(L)"),
    (axes[0, 1], ir_vals_off, COLOR_IR, "IR", "MACE-MDP@MACE-OFF23(L)"),
    (axes[1, 0], raman_vals_dft, COLOR_RAMAN, "Raman", r"MACE-MDP@$\,\omega$B97M"),
    (axes[1, 1], ir_vals_dft, COLOR_IR, "IR", r"MACE-MDP@$\,\omega$B97M"),
]
count=0
for ax, vals, color, title, subtitle in panels:
    if vals.size == 0:
        ax.set_visible(False)
        continue

    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))

    ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.7)
    ax.axvline(mean_val, color="orange", linestyle="--", linewidth=2.4, label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle="-", linewidth=2.4, label=f"Median = {median_val:.2f}")

    if count <2:
        ax.set_title(title, fontsize=15, fontweight="bold")
        count+=1
    ax.text(
        0.028, 0.815, subtitle,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, color="black"
    )

    ax.set_xlabel(r"Match Score ($r_{\mathrm{msc}}$)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.4, linestyle="--")
    ax.tick_params(labelsize=11, width=1.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    ax.legend(frameon=False, loc="upper left", fontsize=13)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=400)
plt.savefig(OUT_PDF)
plt.close()

print(f"✅ Saved figure: {OUT_PNG}")
print(f"✅ Saved figure: {OUT_PDF}")

# ============================================================
# Save summary CSV
# ============================================================
summary_df = pd.DataFrame({
    "source": ["Raman", "IR", "Raman", "IR"],
    "model": ["OFF-L", "OFF-L", "MACE-MDP@omegaB97M", "MACE-MDP@omegaB97M"],
    "mean_rmsc": [
        float(np.mean(raman_vals_off)) if raman_vals_off.size > 0 else np.nan,
        float(np.mean(ir_vals_off)) if ir_vals_off.size > 0 else np.nan,
        float(np.mean(raman_vals_dft)) if raman_vals_dft.size > 0 else np.nan,
        float(np.mean(ir_vals_dft)) if ir_vals_dft.size > 0 else np.nan,
    ],
    "median_rmsc": [
        float(np.median(raman_vals_off)) if raman_vals_off.size > 0 else np.nan,
        float(np.median(ir_vals_off)) if ir_vals_off.size > 0 else np.nan,
        float(np.median(raman_vals_dft)) if raman_vals_dft.size > 0 else np.nan,
        float(np.median(ir_vals_dft)) if ir_vals_dft.size > 0 else np.nan,
    ],
    "n": [
        int(raman_vals_off.size),
        int(ir_vals_off.size),
        int(raman_vals_dft.size),
        int(ir_vals_dft.size),
    ],
})

summary_df.to_csv("RMSC_summary_IR_Raman_OFF_and_DFT.csv", index=False, float_format="%.6f")
print("✅ Saved RMSC_summary_IR_Raman_OFF_and_DFT.csv")