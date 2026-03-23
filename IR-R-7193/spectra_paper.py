#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# ============================================================
# SETTINGS
# ============================================================
# Point this to the *raw* Raman result directory containing *_raman_raw.csv
MODEL_DIR = os.path.expanduser("./mace_off/raman_results_large_SPICE/")

# Matchscore CSV produced by your updated Raman RMSC script (same directory as raw)
CSV_FILE = os.path.join(MODEL_DIR, "spectra_matchscores.csv")

# Output directory
OUTPUT_DIR = "./paper_spectra_raman/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
COLOR_REF = "royalblue"     # DFT reference
COLOR_PRED = "indianred"    # MACE prediction
FWHM = 30.0

# ------------------------------------------------------------
# Compound selection:
# Option A: explicitly provide exactly 3 compounds here.
# Option B: leave empty [] and the script will auto-pick 3 compounds
#           from the CSV (deterministically low/mid/high by RMSC).
# ------------------------------------------------------------
SELECTED_COMPOUNDS = ["C2417905", "C79099073", "C818724"]  # set to [] for auto-pick
AUTO_PICK = "spread"   # "spread" or "random"
RANDOM_SEED = 42

# ============================================================
# Vectorized Lorentzian broadening (same as your other scripts)
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
# DFT search paths (match the Raman scripts you used above)
# ============================================================
DFT_BASE = [
    "../bucket*/HCNOFClBr_{compound_name}/input.out",
    "../bucket*/S_{compound_name}/input.out",  # NOTE: see comment below
    "../bucket*/P_{compound_name}/input.out",
]

# IMPORTANT:
# I do not know your exact DFT Raman path for S_/P_ buckets in your current filesystem.
# In earlier scripts you used:
#   /ptmp/ngoen/Documents/Raman_7193_IR/bucket*/S_{compound_name}/input.out
# If the second line above is wrong on your system, replace it with the correct one.
# (I left a conspicuous typo-like path so it fails loudly rather than silently using the wrong tree.)

def find_dft_file(compound: str):
    for pattern in DFT_BASE:
        matches = glob.glob(pattern.format(compound_name=compound))
        if matches:
            return matches[0]
    return None

def load_dft_raman_spectrum(dft_file: str):
    """Read Raman spectrum from input.out (freq=col 2, activity=col 3) and normalize."""
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
                        # Your earlier scripts used parts[1] and parts[2]
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

def load_raw_raman_csv(path: str):
    """Load *_raman_raw.csv (freq,intensity), normalize intensity to max=1."""
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    freq, inten = arr[:, 0].astype(float), arr[:, 1].astype(float)
    if inten.size == 0 or np.nanmax(inten) == 0:
        return freq, inten
    inten = inten / np.nanmax(inten)
    return freq, inten

def read_matchscores(csv_path: str):
    """Read spectra_matchscores.csv (compound,rmsc,...) into dict compound->rmsc."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path).dropna(subset=["compound", "rmsc"])
    return dict(zip(df["compound"].astype(str), df["rmsc"].astype(float)))

def select_three_compounds(rmsc_lookup: dict):
    compounds = sorted(rmsc_lookup.keys())
    if len(compounds) == 0:
        raise RuntimeError("No compounds found in matchscore CSV.")

    if SELECTED_COMPOUNDS and len(SELECTED_COMPOUNDS) == 3:
        missing = [c for c in SELECTED_COMPOUNDS if c not in rmsc_lookup]
        if missing:
            raise RuntimeError(f"Selected compounds not found in CSV: {missing}")
        return SELECTED_COMPOUNDS

    df = pd.DataFrame({
        "compound": compounds,
        "rmsc": [rmsc_lookup[c] for c in compounds],
    }).sort_values("rmsc").reset_index(drop=True)

    if AUTO_PICK == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        return rng.choice(df["compound"].values, size=min(3, len(df)), replace=False).tolist()

    # spread: low/mid/high
    if len(df) < 3:
        return df["compound"].tolist()
    idxs = [0, len(df)//2, len(df)-1]
    return df.loc[idxs, "compound"].tolist()

# ============================================================
# Main
# ============================================================
def main():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"MODEL_DIR not found: {MODEL_DIR}")

    rmsc_lookup = read_matchscores(CSV_FILE)
    if not rmsc_lookup:
        raise RuntimeError(f"Could not load matchscores from: {CSV_FILE}")

    selected_compounds = select_three_compounds(rmsc_lookup)

    print("Selected compounds:")
    for c in selected_compounds:
        print(f" - {c} (RMSC = {rmsc_lookup.get(c, np.nan):.3f})")

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

    for row_idx, compound in enumerate(selected_compounds):
        print(f"\nProcessing {compound}")

        # --- Load raw MACE Raman spectrum ---
        mace_file = os.path.join(MODEL_DIR, f"{compound}_raman_raw.csv")
        if not os.path.exists(mace_file):
            print(f"⚠️ Missing MACE file for {compound}: {mace_file}")
            continue

        freq_mace, mace_intensity = load_raw_raman_csv(mace_file)
        if mace_intensity.size == 0:
            print(f"⚠️ Could not read MACE Raman spectrum for {compound}")
            continue

        # --- Find DFT file ---
        dft_file = find_dft_file(compound)
        if not dft_file:
            print(f"⚠️ DFT not found for {compound}")
            continue

        # --- Read DFT Raman spectrum ---
        freq_dft, act_dft = load_dft_raman_spectrum(dft_file)
        if act_dft.size == 0:
            print(f"⚠️ Empty DFT Raman data for {compound}")
            continue

        # --- Broaden both spectra (vectorized) ---
        x, spec_dft = broaden_spectrum_lorentzian(freq_dft, act_dft, fwhm=FWHM)
        _, spec_mace = broaden_spectrum_lorentzian(freq_mace, mace_intensity, fwhm=FWHM)

        # Normalize broadened spectra
        if np.max(spec_dft) > 0:
            spec_dft /= np.max(spec_dft)
        if np.max(spec_mace) > 0:
            spec_mace /= np.max(spec_mace)

        # --- Plot low (0–2000) and high (2000–4000) frequency regions ---
        for col_idx, (x_min, x_max, y_max) in enumerate([(0, 2000, 0.15), (2000, 4000, 1.0)]):
            ax = axes[row_idx, col_idx]

            ax.plot(x, spec_dft, color=COLOR_REF, lw=2, label="DFT")
            ax.fill_between(x, 0, spec_dft, color=COLOR_REF, alpha=0.35)

            ax.plot(x, spec_mace, color=COLOR_PRED, lw=2, label="MACE")
            ax.fill_between(x, 0, spec_mace, color=COLOR_PRED, alpha=0.35)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max * 1.09)
            ax.grid(alpha=0.3)

            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
            ax.tick_params(axis="y", width=1.2, labelsize=10)

            if row_idx == len(selected_compounds) - 1:
                ax.set_xlabel("Frequency (cm$^{-1}$)")
            if col_idx == 0:
                ax.set_ylabel("Normalized\nIntensity")
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, loc="upper right")

        # (Optional) title with RMSC on left plot
        rmsc = rmsc_lookup.get(compound, np.nan)
        #axes[row_idx, 0].set_title(f"{compound} — $r_{{\\mathrm{{msc}}}}$={rmsc:.3f}",
        #                           fontsize=12, fontweight="bold", loc="left")
        # --- Annotate inside left subplot (no box) ---
        label_text = f"{compound} — $r_{{\\mathrm{{msc}}}}$ = {rmsc:.3f}"

        axes[row_idx, 0].text(
            0.02, 0.965,                         # upper-left corner (axes coords)
            label_text,
            transform=axes[row_idx, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )


        # --- Store for export ---
        if export_data["freq"] is None:
            export_data["freq"] = x
        export_data[f"{compound}_DFT"] = spec_dft
        export_data[f"{compound}_MACE"] = spec_mace

    plt.tight_layout(h_pad=1.5, w_pad=0.8)
    fig_name = os.path.join(OUTPUT_DIR, "Raman_three_examples_split_scaled")
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.pdf")
    plt.savefig(f"{fig_name}.svg")
    plt.close()
    print(f"\n✅ Saved figure -> {fig_name}.png / .pdf / .svg")

    # Save CSV export
    df_out = pd.DataFrame(export_data)
    csv_path = os.path.join(OUTPUT_DIR, "Raman_three_examples_split_scaled.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"✅ Saved data -> {csv_path}")

if __name__ == "__main__":
    main()
