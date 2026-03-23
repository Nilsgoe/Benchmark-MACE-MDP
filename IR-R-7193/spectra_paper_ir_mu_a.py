#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# ============================================================
# SETTINGS
# ============================================================
MODEL_DIR = os.path.expanduser("./mace_off/IR/ir_results_large_SPICE/")
CSV_FILE = os.path.join(MODEL_DIR, "spectra_matchscores.csv")

OUTPUT_DIR = "./paper_spectra_ir_mace_mu_alpha/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_REF = "royalblue"
COLOR_PRED = "indianred"
FWHM = 30.0

# ------------------------------------------------------------
# Compound selection
# ------------------------------------------------------------
SELECTED_COMPOUNDS = ["C583788", "C7493632", "C33903503"]  # or [] for auto
AUTO_PICK = "spread"     # "spread" or "random"
RANDOM_SEED = 42

# ============================================================
# Vectorized Lorentzian broadening
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
# DFT search paths (same as your IR scripts above)
# ============================================================
DFT_BASE = [
    "../bucket*/HCNOFClBr_{compound_name}/input.out",
    "../bucket*/S_{compound_name}/input.out",  # adjust if needed
    "../bucket*/P_{compound_name}/input.out",
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

def select_three_compounds(rmsc_lookup: dict):
    compounds = sorted(rmsc_lookup.keys())
    if len(compounds) == 0:
        raise RuntimeError("No compounds found in matchscore CSV.")

    if SELECTED_COMPOUNDS and len(SELECTED_COMPOUNDS) == 3:
        missing = [c for c in SELECTED_COMPOUNDS if c not in rmsc_lookup]
        if missing:
            raise RuntimeError(f"Selected compounds not found in CSV: {missing}")
        return SELECTED_COMPOUNDS

    df = pd.DataFrame({"compound": compounds, "rmsc": [rmsc_lookup[c] for c in compounds]})
    df = df.sort_values("rmsc").reset_index(drop=True)

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

    selected = select_three_compounds(rmsc_lookup)

    print("Selected compounds:")
    for c in selected:
        print(f" - {c} (RMSC = {rmsc_lookup.get(c, np.nan):.3f})")

    # Match the Raman aesthetics as closely as possible
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.3,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # One single plot per compound (0–4000), stacked 3x1
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    export_data = {"freq": None}

    for row_idx, compound in enumerate(selected):
        print(f"\nProcessing {compound}")

        mace_file = os.path.join(MODEL_DIR, f"{compound}_ir_raw.csv")
        if not os.path.exists(mace_file):
            print(f"⚠️ Missing IR raw file: {mace_file}")
            continue

        freq_mace, inten_mace = load_raw_ir_csv(mace_file)
        if inten_mace.size == 0:
            print(f"⚠️ Could not read IR raw spectrum for {compound}")
            continue

        dft_file = find_dft_file(compound)
        if not dft_file:
            print(f"⚠️ DFT IR not found for {compound}")
            continue

        freq_dft, inten_dft = load_dft_ir_spectrum(dft_file)
        if inten_dft.size == 0:
            print(f"⚠️ Empty DFT IR data for {compound}")
            continue

        x, spec_dft = broaden_spectrum_lorentzian(freq_dft, inten_dft, fwhm=FWHM)
        _, spec_mace = broaden_spectrum_lorentzian(freq_mace, inten_mace, fwhm=FWHM)

        # Normalize broadened spectra
        if np.max(spec_dft) > 0:
            spec_dft /= np.max(spec_dft)
        if np.max(spec_mace) > 0:
            spec_mace /= np.max(spec_mace)

        ax = axes[row_idx]

        ax.plot(x, spec_dft, color=COLOR_REF, lw=2, label="DFT")
        ax.fill_between(x, 0, spec_dft, color=COLOR_REF, alpha=0.35)

        ax.plot(x, spec_mace, color=COLOR_PRED, lw=2, label="MACE-MDP")
        ax.fill_between(x, 0, spec_mace, color=COLOR_PRED, alpha=0.35)

        ax.set_xlim(0, 4000)
        ax.set_ylim(0, 1.09)
        ax.grid(alpha=0.3)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.tick_params(axis="y", width=1.2, labelsize=11)

        if row_idx == len(selected) - 1:
            ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.set_ylabel("Normalized\nIntensity")

        if row_idx == 0:
            ax.legend(frameon=False, loc="upper right")

        # In-plot label (no box), matching the Raman approach
        rmsc = rmsc_lookup.get(compound, np.nan)
        label_text = f"{compound} — $r_{{\\mathrm{{msc}}}}$ = {rmsc:.3f}"
        ax.text(
            0.02, 0.965,
            label_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )

        # Export
        if export_data["freq"] is None:
            export_data["freq"] = x
        export_data[f"{compound}_DFT"] = spec_dft
        export_data[f"{compound}_MACE-mu-alpha"] = spec_mace

    plt.tight_layout(h_pad=1.5)
    fig_name = os.path.join(OUTPUT_DIR, "IR_three_examples_mu_alpha_fullrange")
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.pdf")
    plt.savefig(f"{fig_name}.svg")
    plt.close()
    print(f"\n✅ Saved figure -> {fig_name}.png / .pdf / .svg")

    df_out = pd.DataFrame(export_data)
    csv_path = os.path.join(OUTPUT_DIR, "IR_three_examples_mu_alpha_fullrange.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"✅ Saved data -> {csv_path}")

if __name__ == "__main__":
    main()
