#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Utility
# ============================================================
def load_rmsc_from_csv(csv_path: str) -> np.ndarray:
    """Load rmsc column from one spectra_matchscores.csv."""
    try:
        df = pd.read_csv(csv_path)
        if "rmsc" not in df.columns:
            return np.array([], dtype=float)
        return df["rmsc"].dropna().to_numpy(dtype=float)
    except Exception:
        return np.array([], dtype=float)

def collect_rmsc_for_pattern(pattern: str) -> np.ndarray:
    """
    Collect rmsc values from all spectra_matchscores.csv matching a glob pattern.
    Returns concatenated numpy array.
    """
    files = sorted(glob.glob(pattern))
    vals = []
    for fp in files:
        v = load_rmsc_from_csv(fp)
        if v.size > 0:
            vals.append(v)
    if not vals:
        return np.array([], dtype=float)
    return np.concatenate(vals)

def nice_mlip_title(mlip: str) -> str:
    if mlip == "mace_omol":
        return "OMOL"
    if mlip == "mace_off_small":
        return "OFF23(S)"
    if mlip == "mace_off_medium":
        return "OFF23(M)"
    if mlip == "mace_off_large":
        return "OFF23(L)"
    return mlip

# ============================================================
# Configuration
# ============================================================
mlips = ["mace_omol", "mace_off_small", "mace_off_medium", "mace_off_large"]

MU_ROOT = "./mace_mu"

def mlip_to_mode_size(mlip: str):
    if mlip == "mace_omol":
        return "mace_omol", "omol"
    parts = mlip.split("_")
    return "mace_off", parts[-1]

source_groups = {
    "Raman (MACE-MLIP / μ_α)": {
        "pattern_fn": lambda mode, size: (
            f"{mode}/raman_results_{size}_SPICE/spectra_matchscores.csv"
            if mode == "mace_off"
            else f"{mode}/raman_results_omol*_SPICE/spectra_matchscores.csv"
        ),
        "color": "royalblue",
        "outfile": "RMSC_histograms_Raman_MACE_MLIP.png",
    },
    "IR (MACE-MLIP / μ_α)": {
        "pattern_fn": lambda mode, size: (
            f"{mode}/IR/ir_results_{size}_SPICE/spectra_matchscores.csv"
            if mode == "mace_off"
            else f"{mode}/IR/ir_results_omol_SPICE/spectra_matchscores.csv"
        ),
        "color": "indianred",
        "outfile": "RMSC_histograms_IR_MACE_MLIP.png",
    },
    "IR (MACE-μ)": {
        "pattern_fn": lambda mode, size: (
            os.path.join(MU_ROOT, f"{mode}/ir_results_{size}_SPICE/spectra_matchscores.csv")
            if mode == "mace_off"
            else os.path.join(MU_ROOT, f"{mode}/ir_results_*_SPICE/spectra_matchscores.csv")
        ),
        "color": "#DAA520",
        "outfile": "RMSC_histograms_IR_MACE_mu.png",
    },
}

# ============================================================
# Plot Function
# ============================================================
def make_combined_histograms(source_label, info, mlips):
    color = info["color"]
    outfile = info["outfile"]
    pattern_fn = info["pattern_fn"]

    data = {}
    for mlip in mlips:
        mode, size = mlip_to_mode_size(mlip)
        pattern = pattern_fn(mode, size)
        vals = collect_rmsc_for_pattern(pattern)
        if vals.size == 0:
            print(f"⚠️ No data for {source_label} / {mlip} (pattern: {pattern})")
        data[mlip] = vals

    non_empty = [v for v in data.values() if v.size > 0]
    if not non_empty:
        print(f"⚠️ No valid data found for {source_label}; skipping plot.")
        return

    all_vals = np.concatenate(non_empty)
    xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
    if np.isclose(xmin, xmax):
        xmin = max(0.0, xmin - 0.05)
        xmax = min(1.0, xmax + 0.05)
    bins = np.linspace(xmin, xmax, 25)

    # match the newer style/sizes
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.3,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 13.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, mlip in enumerate(mlips):
        ax = axes[i]
        vals = data.get(mlip, np.array([], dtype=float))
        if vals.size == 0:
            ax.set_visible(False)
            continue

        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))

        ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.7)
        ax.axvline(mean_val, color="orange", linestyle="--", linewidth=2.4, label=f"Mean = {mean_val:.2f}")
        ax.axvline(median_val, color="green", linestyle="-", linewidth=2.4, label=f"Median = {median_val:.2f}")

        ax.set_title(nice_mlip_title(mlip), fontsize=15, fontweight="bold")
        ax.set_xlabel(r"Match Score ($r_{\mathrm{msc}}$)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.4, linestyle="--")
        ax.tick_params(labelsize=11, width=1.3)

        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

        ax.legend(frameon=False, fontsize=13.5, loc="upper left")

    for j in range(len(mlips), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(outfile, dpi=400)
    plt.savefig(os.path.splitext(outfile)[0] + ".pdf")
    plt.close()
    print(f"✅ Saved combined histogram: {outfile}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    for source_label, info in source_groups.items():
        make_combined_histograms(source_label, info, mlips)