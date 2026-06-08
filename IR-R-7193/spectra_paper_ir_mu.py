#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# ============================================================
# SETTINGS
# ============================================================

# ---------------- IR (MACE-MDP / mu+alpha) ----------------
MODEL_DIR_MU_ALPHA = os.path.expanduser("./mace_off/IR/ir_results_large_SPICE/")
CSV_MU_ALPHA = os.path.join(MODEL_DIR_MU_ALPHA, "spectra_matchscores.csv")

# ---------------- IR (MACE-mu) ----------------
MODEL_DIR_MU = os.path.expanduser("./mace_mu/mace_off/ir_results_large_SPICE/")
CSV_MU = os.path.join(MODEL_DIR_MU, "spectra_matchscores.csv")

# ---------------- Output ----------------
OUTPUT_DIR = "./paper_spectra_ir_combined_mu_mu_alpha/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Colors ----------------
COLOR_REF = "royalblue"
COLOR_MU_ALPHA = "indianred"
COLOR_MU = "#DAA520"

FWHM = 30.0

# ============================================================
# COMPOUND SELECTION
# ============================================================
SELECTED_IR_COMPOUNDS = ["C583788", "C7493632", "C33903503"]   # or [] for auto

AUTO_PICK = "spread"   # "spread" or "random"
RANDOM_SEED = 42

# ============================================================
# DFT SEARCH PATHS
# ============================================================
DFT_BASE = [
    "../bucket*/HCNOFClBr_{compound_name}/input.out",
    "../bucket*/S_{compound_name}/input.out",   # adjust if needed
    "../bucket*/P_{compound_name}/input.out",
]

# ============================================================
# VECTORIZE LORENTZIAN BROADENING
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
# HELPERS
# ============================================================
def find_dft_file(compound: str):
    for pattern in DFT_BASE:
        matches = glob.glob(pattern.format(compound_name=compound))
        if matches:
            return matches[0]
    return None

def read_matchscores(csv_path: str):
    """Read spectra_matchscores.csv (compound,rmsc,...) into dict compound->rmsc."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path).dropna(subset=["compound", "rmsc"])
    return dict(zip(df["compound"].astype(str), df["rmsc"].astype(float)))

def select_three_compounds(rmsc_mu_alpha: dict, rmsc_mu: dict, selected_compounds: list):
    overlap = sorted(set(rmsc_mu_alpha.keys()) & set(rmsc_mu.keys()))
    if len(overlap) == 0:
        raise RuntimeError("No overlapping compounds found between mu+alpha and mu matchscore CSVs.")

    if selected_compounds and len(selected_compounds) == 3:
        missing = [c for c in selected_compounds if c not in overlap]
        if missing:
            raise RuntimeError(f"Selected IR compounds not found in BOTH CSVs: {missing}")
        return selected_compounds

    df = pd.DataFrame({
        "compound": overlap,
        "rmsc_mu_alpha": [rmsc_mu_alpha[c] for c in overlap],
        "rmsc_mu": [rmsc_mu[c] for c in overlap],
    }).sort_values("rmsc_mu_alpha").reset_index(drop=True)

    if AUTO_PICK == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        return rng.choice(df["compound"].values, size=min(3, len(df)), replace=False).tolist()

    if len(df) < 3:
        return df["compound"].tolist()

    idxs = [0, len(df) // 2, len(df) - 1]
    return df.loc[idxs, "compound"].tolist()

# ============================================================
# DFT LOADERS
# ============================================================
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

# ============================================================
# RAW LOADERS
# ============================================================
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

# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.isdir(MODEL_DIR_MU_ALPHA):
        raise RuntimeError(f"MODEL_DIR_MU_ALPHA not found: {MODEL_DIR_MU_ALPHA}")
    if not os.path.isdir(MODEL_DIR_MU):
        raise RuntimeError(f"MODEL_DIR_MU not found: {MODEL_DIR_MU}")

    rmsc_mu_alpha = read_matchscores(CSV_MU_ALPHA)
    rmsc_mu = read_matchscores(CSV_MU)

    if not rmsc_mu_alpha:
        raise RuntimeError(f"Could not load mu+alpha matchscores from: {CSV_MU_ALPHA}")
    if not rmsc_mu:
        raise RuntimeError(f"Could not load mu matchscores from: {CSV_MU}")

    selected_ir = select_three_compounds(rmsc_mu_alpha, rmsc_mu, SELECTED_IR_COMPOUNDS)

    print("Selected IR compounds:")
    for c in selected_ir:
        print(
            f" - {c} (RMSC MDP = {rmsc_mu_alpha.get(c, np.nan):.2f}, "
            f"RMSC μ = {rmsc_mu.get(c, np.nan):.2f})"
        )

    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.3,
        "axes.labelsize": 13,
        "axes.titlesize": 17,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # 3 rows x 1 column
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False, sharey=False)

    export_data_ir = {"freq": None}

    # ========================================================
    # IR PANELS
    # ========================================================
    for row_idx, compound in enumerate(selected_ir):
        print(f"\nProcessing IR: {compound}")
        ax = axes[row_idx]

        mu_alpha_file = os.path.join(MODEL_DIR_MU_ALPHA, f"{compound}_ir_raw.csv")
        mu_file = os.path.join(MODEL_DIR_MU, f"{compound}_ir_raw.csv")

        if not os.path.exists(mu_alpha_file):
            print(f"⚠️ Missing mu+alpha IR raw file: {mu_alpha_file}")
            continue

        if not os.path.exists(mu_file):
            print(f"⚠️ Missing mu IR raw file: {mu_file}")
            continue

        freq_mu_alpha, inten_mu_alpha = load_raw_ir_csv(mu_alpha_file)
        freq_mu, inten_mu = load_raw_ir_csv(mu_file)

        if inten_mu_alpha.size == 0:
            print(f"⚠️ Could not read mu+alpha IR raw spectrum for {compound}")
            continue

        if inten_mu.size == 0:
            print(f"⚠️ Could not read mu IR raw spectrum for {compound}")
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
        _, spec_mu_alpha = broaden_spectrum_lorentzian(freq_mu_alpha, inten_mu_alpha, fwhm=FWHM)
        _, spec_mu = broaden_spectrum_lorentzian(freq_mu, inten_mu, fwhm=FWHM)

        if np.max(spec_dft) > 0:
            spec_dft /= np.max(spec_dft)
        if np.max(spec_mu_alpha) > 0:
            spec_mu_alpha /= np.max(spec_mu_alpha)
        if np.max(spec_mu) > 0:
            spec_mu /= np.max(spec_mu)

        ax.plot(x, spec_dft, color=COLOR_REF, lw=2, label=r"$\omega$B97M-D3(BJ)")
        ax.fill_between(x, 0, spec_dft, color=COLOR_REF, alpha=0.35)

        ax.plot(x, spec_mu_alpha, color=COLOR_MU_ALPHA, lw=2, label="MACE-MDP@MACE-OFF23(L)")
        ax.fill_between(x, 0, spec_mu_alpha, color=COLOR_MU_ALPHA, alpha=0.30)

        ax.plot(x, spec_mu, color=COLOR_MU, lw=2, label="MACE-μ@MACE-OFF23(L)")
        ax.fill_between(x, 0, spec_mu, color=COLOR_MU, alpha=0.30)

        ax.set_xlim(0, 4000)
        ax.set_ylim(0, 1.12)
        ax.grid(alpha=0.3)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.tick_params(axis="both", width=1.2, labelsize=11)

        if row_idx == len(selected_ir) - 1:
            ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.set_ylabel("Normalized\nIntensity")

        if row_idx == 0:
            ax.legend(frameon=False, loc="upper right")
            # ax.set_title("IR", fontweight="bold")

        r_a = rmsc_mu_alpha.get(compound, np.nan)
        r_m = rmsc_mu.get(compound, np.nan)
        label_text = (
            f"{compound}\n"
            f"$\\mathbf{{r_{{\\mathrm{{msc}}}}(MDP)}}$ = {r_a:.2f}, "
            f"$\\mathbf{{r_{{\\mathrm{{msc}}}}(\\mu)}}$ = {r_m:.2f}"
        )
        ax.text(
            0.02, 0.965,
            label_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )

        if export_data_ir["freq"] is None:
            export_data_ir["freq"] = x
        export_data_ir[f"{compound}_DFT"] = spec_dft
        export_data_ir[f"{compound}_MACE-mu-alpha"] = spec_mu_alpha
        export_data_ir[f"{compound}_MACE-mu"] = spec_mu

    plt.tight_layout(h_pad=1.5)

    fig_name = os.path.join(OUTPUT_DIR, "IR_three_examples_mu_vs_mu_alpha_fullrange")
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.pdf")
    plt.savefig(f"{fig_name}.svg")
    plt.close()
    print(f"\n✅ Saved figure -> {fig_name}.png / .pdf / .svg")

    df_ir = pd.DataFrame(export_data_ir)
    csv_ir = os.path.join(OUTPUT_DIR, "IR_three_examples_mu_vs_mu_alpha_fullrange.csv")
    df_ir.to_csv(csv_ir, index=False, float_format="%.6f")

    print(f"✅ Saved IR data -> {csv_ir}")


if __name__ == "__main__":
    main()