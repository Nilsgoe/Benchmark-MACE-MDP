#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# ============================================================
# SETTINGS
# ============================================================
model_dir = os.path.expanduser("./mace_off/raman_results_large_SPICE/")
csv_file = "./mace_off/raman_results_large_SPICE/spectra_matchscores.csv"
output_dir = "./paper_spectra/"
os.makedirs(output_dir, exist_ok=True)

color_ref = "royalblue"   # DFT reference
color_pred = "indianred"  # MACE prediction
fwhm = 30.0

# ============================================================
# Vectorized Lorentzian broadening (match script #2)
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
# DFT search paths
# ============================================================
dft_base = [
    os.path.expanduser("./bucket_0/{compound_name}/input.out"),
]

# ============================================================
# Label formatting
#   - prepend "#"
#   - no underscores
#   - no capitalization (lowercase)
# Example: "07_AbC_C123" -> "#07-abc-c123"
# ============================================================
def format_compound_label(compound: str) -> str:
    return compound.replace("_", "-").capitalize()

# ============================================================
# Load compound list and RMSC lookup
# ============================================================
df = pd.read_csv(csv_file)
compounds = df["compound"].dropna().astype(str).unique().tolist()
rmsc_lookup = dict(zip(df["compound"].astype(str), df["rmsc"]))

# ============================================================
# Function to make a figure for one prefix group (e.g., "03" or "05")
# ============================================================
def make_group_figure(group_prefix: str):
    print(f"\n===== Processing group {group_prefix} =====")

    # NOTE: preserve your original selection behavior (contains substring match)
    compounds_group = sorted([c for c in compounds if group_prefix in c])[:3]
    if len(compounds_group) == 0:
        print(f"⚠️ No compounds found for prefix {group_prefix}, skipping.")
        return

    # Match script #2 style exactly
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.3,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharey=False)
    export_data = {"freq": None}

    for row_idx, compound in enumerate(compounds_group):
        print(f" -> {compound}")

        mace_file = os.path.join(model_dir, f"{compound}_raman_raw.csv")
        if not os.path.exists(mace_file):
            print(f"⚠️ Missing MACE file for {compound}, skipping...")
            continue

        mace_data = np.genfromtxt(mace_file, delimiter=",", skip_header=1)
        if mace_data.ndim != 2 or mace_data.shape[1] < 2:
            print(f"⚠️ Bad MACE CSV format for {compound}, skipping...")
            continue

        freq_mace, mace_intensity = mace_data[:, 0].astype(float), mace_data[:, 1].astype(float)
        if mace_intensity.size == 0 or np.nanmax(mace_intensity) == 0:
            print(f"⚠️ Empty/zero MACE intensity for {compound}, skipping...")
            continue
        mace_intensity = mace_intensity / np.nanmax(mace_intensity)

        # --- Find DFT file ---
        dft_file = None
        for pattern in dft_base:
            matches = glob.glob(pattern.format(compound_name=compound))
            if matches:
                dft_file = matches[0]
                break
        if not dft_file:
            print(f"⚠️ DFT not found for {compound}, skipping...")
            continue

        # --- Read DFT Raman data ---
        freq_dft, act_dft = [], []
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
                        except ValueError:
                            pass

        freq_dft = np.array(freq_dft, dtype=float)
        act_dft = np.array(act_dft, dtype=float)
        if act_dft.size == 0:
            print(f"⚠️ Empty DFT data for {compound}, skipping...")
            continue
        act_dft = act_dft / np.max(act_dft)

        # --- Broaden both spectra (vectorized) ---
        x, spec_dft = broaden_spectrum_lorentzian(freq_dft, act_dft, fwhm=fwhm)
        _, spec_mace = broaden_spectrum_lorentzian(freq_mace, mace_intensity, fwhm=fwhm)

        # Normalize broadened spectra (match script #2 guards)
        if np.max(spec_dft) > 0:
            spec_dft /= np.max(spec_dft)
        if np.max(spec_mace) > 0:
            spec_mace /= np.max(spec_mace)

        # --- Plot split: low (0–2000) and high (2000–4000) ---
        for col_idx, (x_min, x_max, y_max) in enumerate([(0, 2000, 0.25), (2000, 4000, 1.0)]):
            ax = axes[row_idx, col_idx]

            ax.plot(x, spec_dft, color=color_ref, lw=2, label="DFT")
            ax.fill_between(x, 0, spec_dft, color=color_ref, alpha=0.35)

            ax.plot(x, spec_mace, color=color_pred, lw=2, label="MACE")
            ax.fill_between(x, 0, spec_mace, color=color_pred, alpha=0.35)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max * 1.08)
            ax.grid(alpha=0.3)

            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
            ax.tick_params(axis="y", width=1.2, labelsize=10)

            if row_idx == 2:
                ax.set_xlabel("Frequency (cm$^{-1}$)")
            if col_idx == 0:
                ax.set_ylabel("Normalized\nIntensity")
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, loc="upper left")

        # --- In-plot title text (no box), and formatted compound name ---
        rmsc = rmsc_lookup.get(compound, np.nan)
        label_compound = format_compound_label(compound)
        label_text = f"{label_compound} — $r_{{\\mathrm{{msc}}}}$ = {rmsc:.3f}"

        axes[row_idx, 0].text(
            0.02, 0.965,
            label_text,
            transform=axes[row_idx, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )

        # --- Export data ---
        if export_data["freq"] is None:
            export_data["freq"] = x
        export_data[f"{compound}_DFT"] = spec_dft
        export_data[f"{compound}_MACE"] = spec_mace

    plt.tight_layout(h_pad=1.5, w_pad=0.8)
    fig_name = os.path.join(output_dir, f"Raman_{group_prefix}_three_examples_split_scaled")
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.pdf")
    plt.savefig(f"{fig_name}.svg")
    plt.close()
    print(f"✅ Saved figure -> {fig_name}.png / .pdf / .svg")

    df_out = pd.DataFrame(export_data)
    csv_path = os.path.join(output_dir, f"Raman_{group_prefix}_three_examples_split_scaled.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"✅ Saved data -> {csv_path}")

# ============================================================
# Generate figures for groups
# ============================================================
for prefix in ["01","02","03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15","16","17","18","19","20","21","22","23"]:  # ,"08","10","11"
    make_group_figure(prefix)
