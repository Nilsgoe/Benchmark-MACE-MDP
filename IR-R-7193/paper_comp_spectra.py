#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# ============================================================
# SETTINGS
# ============================================================
# Point these to the *raw* result directories (the ones containing *_ir_raw.csv)
# Examples (edit to your case):
MODEL_DIR_MU_ALPHA = os.path.expanduser(
    "./mace_off/IR/ir_results_large_SPICE/"
)ff
MODEL_DIR_MU = os.path.expanduser(
    "./mace_mu/mace_off/IR/ir_results_large_SPICE/"
)

# The per-directory matchscore CSVs produced by your updated scripts (same dirs as raw)
CSV_MU_ALPHA = os.path.join(MODEL_DIR_MU_ALPHA, "spectra_matchscores.csv")
CSV_MU = os.path.join(MODEL_DIR_MU, "spectra_matchscores.csv")

OUTPUT_DIR = "./paper_spectra_ir_combined_mu_mu_alpha/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot colors
COLOR_REF = "royalblue"     # DFT reference
COLOR_MU_ALPHA = "indianred"  # MACE-μ_α
COLOR_MU = "#DAA520"        # MACE-μ
FWHM = 30.0

# ------------------------------------------------------------
# Compound selection:
# Option A: explicitly provide exactly 3 compounds here.f
# Option B: leave empty [] and the script will select 3 compounds
#           deterministically from the overlap between both CSVs
#           (lowest/middle/highest RMSC in mu_alpha by default).
# ------------------------------------------------------------
SELECTED_COMPOUNDS = ["C583788", "C7493632", "C3937562"]  # set to [] for auto-pick

# Auto-pick behavior when SELECTED_COMPOUNDS == []
AUTO_PICK = "spread"   # "spread" -> low/mid/high by mu_alpha RMSC; "random" -> random 3
RANDOM_SEED = 42

# ============================================================
# Lorentzian broadening (same as your other scripts)
# ============================================================
def broaden_spectrum_lorentzian(freqs, intens, fwhm=10.0, x_min=0, x_max=4000, step=1.0):
    x = np.arange(x_min, x_max + step, step, dtype=np.float64)
    gamma = fwhm / 2.0
    if len(freqs) == 0:
        return x, np.zeros_like(x)
    diff = x[:, None] - freqs[None, :]
    spec = np.sum(intens[None, :] * (gamma / (diff ** 2 + gamma ** 2)), axis=1)
    return x, spec

# ============================================================
# DFT search paths (match the ones used in your scripts above)
# ============================================================
DFT_BASE = [
    "/ptmp/ngoen/Documents/Raman_7193_IR/IR/dft/bucket*/HCNOFClBr_{compound_name}/input.out",
    "/ptmp/ngoen/Documents/Raman_7193_IR/IR/dft/bucket*/S_{compound_name}/input.out",
    "/ptmp/ngoen/Documents/Raman_7193_IR/IR/dft/bucket*/P_{compound_name}/input.out",
]

def find_dft_file(compound: str):
    for pattern in DFT_BASE:
        matches = glob.glob(pattern.format(compound_name=compound))
        if matches:
            return matches[0]
    return None

def load_dft_ir_spectrum(dft_file: str):
    """Read IR spectrum from input.out (freq=col 2, intensity=col 4) and normalize."""
    freq_dft, inten_dft = [], []
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
                    except ValueError:
                        pass
    freq_dft = np.array(freq_dft, dtype=float)
    inten_dft = np.array(inten_dft, dtype=float)
    if inten_dft.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    inten_dft /= np.max(inten_dft)
    return freq_dft, inten_dft

def load_raw_ir_csv(path: str):
    """Load *_ir_raw.csv (freq,intensity), normalize intensity to max=1."""
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

def select_three_compounds(rmsc_alpha: dict, rmsc_mu: dict):
    """Choose 3 compounds either from user list or from overlap of both CSVs."""
    overlap = sorted(set(rmsc_alpha.keys()) & set(rmsc_mu.keys()))
    if len(overlap) == 0:
        raise RuntimeError("No overlap between mu_alpha and mu matchscore CSVs.")

    if SELECTED_COMPOUNDS and len(SELECTED_COMPOUNDS) == 3:
        missing = [c for c in SELECTED_COMPOUNDS if c not in overlap]
        if missing:
            raise RuntimeError(f"Selected compounds not found in BOTH CSVs: {missing}")
        return SELECTED_COMPOUNDS

    # Auto-pick
    df = pd.DataFrame({
        "compound": overlap,
        "rmsc_alpha": [rmsc_alpha[c] for c in overlap],
        "rmsc_mu": [rmsc_mu[c] for c in overlap],
    }).sort_values("rmsc_alpha").reset_index(drop=True)

    if AUTO_PICK == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        picks = rng.choice(df["compound"].values, size=min(3, len(df)), replace=False).tolist()
        return picks

    # "spread": low / mid / high by rmsc_alpha
    if len(df) < 3:
        return df["compound"].tolist()
    idxs = [0, len(df)//2, len(df)-1]
    return df.loc[idxs, "compound"].tolist()

# ============================================================
# Main
# ============================================================
def main():
    # Read matchscore lookups
    rmsc_lookup_mu_alpha = read_matchscores(CSV_MU_ALPHA)
    rmsc_lookup_mu = read_matchscores(CSV_MU)

    if not rmsc_lookup_mu_alpha:
        raise RuntimeError(f"Could not load mu_alpha matchscores from: {CSV_MU_ALPHA}")
    if not rmsc_lookup_mu:
        raise RuntimeError(f"Could not load mu matchscores from: {CSV_MU}")

    # Select compounds
    selected = select_three_compounds(rmsc_lookup_mu_alpha, rmsc_lookup_mu)

    print("Selected compounds:")
    for c in selected:
        print(f" - {c} (RMSC μₐ = {rmsc_lookup_mu_alpha.get(c, np.nan):.3f}, "
              f"RMSC μ = {rmsc_lookup_mu.get(c, np.nan):.3f})")

    # Matplotlib setup
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.3,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    export_data = {"freq": None}

    for row_idx, compound in enumerate(selected):
        print(f"\nProcessing {compound}")

        # --- Load raw MACE μ_α spectrum ---
        mace_file_mu_alpha = os.path.join(MODEL_DIR_MU_ALPHA, f"{compound}_ir_raw.csv")
        if not os.path.exists(mace_file_mu_alpha):
            print(f"⚠️ Missing MACE-μ_α file for {compound}: {mace_file_mu_alpha}")
            continue
        freq_mu_alpha, intens_mu_alpha = load_raw_ir_csv(mace_file_mu_alpha)
        if intens_mu_alpha.size == 0:
            print(f"⚠️ Could not read MACE-μ_α spectrum for {compound}")
            continue

        # --- Load raw MACE μ spectrum ---
        mace_file_mu = os.path.join(MODEL_DIR_MU, f"{compound}_ir_raw.csv")
        if not os.path.exists(mace_file_mu):
            print(f"⚠️ Missing MACE-μ file for {compound}: {mace_file_mu}")
            continue
        freq_mu, intens_mu = load_raw_ir_csv(mace_file_mu)
        if intens_mu.size == 0:
            print(f"⚠️ Could not read MACE-μ spectrum for {compound}")
            continue

        # --- Find DFT file ---
        dft_file = find_dft_file(compound)
        if not dft_file:
            print(f"⚠️ DFT IR not found for {compound}")
            continue

        # --- Read DFT IR spectrum ---
        freq_dft, intensity_dft = load_dft_ir_spectrum(dft_file)
        if intensity_dft.size == 0:
            print(f"⚠️ Empty DFT IR data for {compound}")
            continue

        # --- Broaden all spectra (same broadening for DFT and models) ---
        x, spec_dft = broaden_spectrum_lorentzian(freq_dft, intensity_dft, fwhm=FWHM)
        _, spec_mu_alpha = broaden_spectrum_lorentzian(freq_mu_alpha, intens_mu_alpha, fwhm=FWHM)
        _, spec_mu = broaden_spectrum_lorentzian(freq_mu, intens_mu, fwhm=FWHM)

        # Normalize broadened spectra to max=1 for plotting
        if np.max(spec_dft) > 0:
            spec_dft /= np.max(spec_dft)
        if np.max(spec_mu_alpha) > 0:
            spec_mu_alpha /= np.max(spec_mu_alpha)
        if np.max(spec_mu) > 0:
            spec_mu /= np.max(spec_mu)

        # --- Plot ---
        ax = axes[row_idx]
        ax.plot(x, spec_dft, color=COLOR_REF, lw=2, label="DFT (Reference)")
        ax.fill_between(x, 0, spec_dft, color=COLOR_REF, alpha=0.35)

        ax.plot(x, spec_mu_alpha, color=COLOR_MU_ALPHA, lw=2, label="MACE-μₐ")
        ax.fill_between(x, 0, spec_mu_alpha, color=COLOR_MU_ALPHA, alpha=0.3)

        ax.plot(x, spec_mu, color=COLOR_MU, lw=2, label="MACE-μ")
        ax.fill_between(x, 0, spec_mu, color=COLOR_MU, alpha=0.3)

        ax.set_xlim(0, 4000)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        if row_idx == len(selected) - 1:
            ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.set_ylabel("Normalized\nIntensity")

        if row_idx == 0:
            ax.legend(frameon=False, loc="upper right")

        rmsc_alpha = rmsc_lookup_mu_alpha.get(compound, np.nan)
        rmsc_mu = rmsc_lookup_mu.get(compound, np.nan)
        ax.set_title(
            f"{compound} — RMSC(μₐ)={rmsc_alpha:.3f}, RMSC(μ)={rmsc_mu:.3f}",
            fontsize=12, fontweight="bold", loc="left"
        )

        # --- Store for CSV export ---
        if export_data["freq"] is None:
            export_data["freq"] = x
        export_data[f"{compound}_DFT"] = spec_dft
        export_data[f"{compound}_MACE-mu-alpha"] = spec_mu_alpha
        export_data[f"{compound}_MACE-mu"] = spec_mu

    plt.tight_layout(h_pad=1.5)
    fig_name = os.path.join(OUTPUT_DIR, "IR_three_selected_examples_mu_vs_mu_alpha")
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.pdf")
    plt.close()
    print(f"\n✅ Saved figure -> {fig_name}.png / .pdf")

    # Save CSV export
    df_out = pd.DataFrame(export_data)
    csv_path = os.path.join(OUTPUT_DIR, "IR_three_selected_examples_mu_vs_mu_alpha.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"✅ Saved data -> {csv_path}")

if __name__ == "__main__":
    main()
