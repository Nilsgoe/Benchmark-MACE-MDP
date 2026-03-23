#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# ============================================================
# SETTINGS
# ============================================================
RESULTS_DIR = "./from_dft/spectra_from_hessian_modes"
RAMAN_CSV_FILE = os.path.join(RESULTS_DIR, "raman_spectra_matchscores.csv")
IR_CSV_FILE = os.path.join(RESULTS_DIR, "ir_spectra_matchscores.csv")

OUTPUT_DIR = "./paper_dft_modes_spectra"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_REF = "royalblue"    # DFT reference
COLOR_PRED = "indianred"   # spectra recomputed from DFT modes
FWHM = 30.0

DFT_BASE = [
    os.path.expanduser("./bucket_0/{compound_name}/input.out"),
    os.path.expanduser("./{compound_name}/input.out"),
]

GROUP_PREFIXES = [
    "01","02","03","04","05","06","07","08","09","10","11","12",
    "13","14","15","16","17","18","19","20","21","22","23"
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
def format_compound_label(compound: str) -> str:
    return compound.replace("_", "-").lower()

def read_matchscores(csv_path: str):
    if not os.path.exists(csv_path):
        return {}, []
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["compound", "rmsc"])
    compounds = df["compound"].astype(str).unique().tolist()
    rmsc_lookup = dict(zip(df["compound"].astype(str), df["rmsc"].astype(float)))
    return rmsc_lookup, compounds

def find_dft_file(compound: str):
    for pattern in DFT_BASE:
        matches = glob.glob(pattern.format(compound_name=compound))
        if matches:
            return matches[0]
    return None

def load_raw_two_column_csv(path: str):
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        if arr.size < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    freq = arr[:, 0].astype(float)
    inten = arr[:, 1].astype(float)

    if inten.size == 0:
        return freq, inten

    max_i = np.nanmax(inten)
    if max_i > 0:
        inten = inten / max_i
    return freq, inten

def load_dft_ir_spectrum(dft_file: str):
    freq_dft, inten_dft = [], []
    with open(dft_file, "r") as f:
        in_ir = False
        for line in f:
            if "IR SPECTRUM" in line:
                in_ir = True
                continue
            if "first frequency considered" in line and in_ir:
                break
            if not in_ir:
                continue

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

    max_i = np.max(inten_dft)
    if max_i > 0:
        inten_dft /= max_i
    return freq_dft, inten_dft

def load_dft_raman_spectrum(dft_file: str):
    freq_dft, inten_dft = [], []
    with open(dft_file, "r") as f:
        in_raman = False
        for line in f:
            if "RAMAN SPECTRUM" in line:
                in_raman = True
                continue
            if "first frequency considered" in line and in_raman:
                break
            if not in_raman:
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    freq_dft.append(float(parts[1]))
                    inten_dft.append(float(parts[2]))
                except ValueError:
                    pass

    freq_dft = np.array(freq_dft, dtype=float)
    inten_dft = np.array(inten_dft, dtype=float)
    if inten_dft.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    max_i = np.max(inten_dft)
    if max_i > 0:
        inten_dft /= max_i
    return freq_dft, inten_dft

# ============================================================
# PLOTTING
# ============================================================
def make_group_figure(group_prefix: str, mode: str, compounds, rmsc_lookup):
    print(f"\n===== Processing {mode.upper()} group {group_prefix} =====")

    compounds_group = sorted([c for c in compounds if group_prefix in c])[:3]
    if len(compounds_group) == 0:
        print(f"⚠️ No compounds found for prefix {group_prefix} in {mode}, skipping.")
        return

    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.3,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=False)
    if len(compounds_group) == 1:
        axes = [axes]

    export_data = {"freq": None}

    for row_idx, compound in enumerate(compounds_group):
        print(f" -> {compound}")

        pred_file = os.path.join(RESULTS_DIR, f"{compound}_{mode}_raw.csv")
        if not os.path.exists(pred_file):
            print(f"⚠️ Missing predicted {mode} file for {compound}, skipping...")
            continue

        freq_pred, pred_intensity = load_raw_two_column_csv(pred_file)
        if pred_intensity.size == 0 or np.nanmax(pred_intensity) == 0:
            print(f"⚠️ Empty/zero predicted {mode} intensity for {compound}, skipping...")
            continue

        dft_file = find_dft_file(compound)
        if not dft_file:
            print(f"⚠️ DFT file not found for {compound}, skipping...")
            continue

        if mode == "ir":
            freq_dft, inten_dft = load_dft_ir_spectrum(dft_file)
        elif mode == "raman":
            freq_dft, inten_dft = load_dft_raman_spectrum(dft_file)
        else:
            print(f"⚠️ Unknown mode {mode}, skipping...")
            continue

        if inten_dft.size == 0:
            print(f"⚠️ Empty DFT {mode} data for {compound}, skipping...")
            continue

        x, spec_dft = broaden_spectrum_lorentzian(freq_dft, inten_dft, fwhm=FWHM)
        _, spec_pred = broaden_spectrum_lorentzian(freq_pred, pred_intensity, fwhm=FWHM)

        if np.max(spec_dft) > 0:
            spec_dft /= np.max(spec_dft)
        if np.max(spec_pred) > 0:
            spec_pred /= np.max(spec_pred)

        ax = axes[row_idx]
        ax.plot(x, spec_dft, color=COLOR_REF, lw=2, label="DFT")
        ax.fill_between(x, 0, spec_dft, color=COLOR_REF, alpha=0.35)

        ax.plot(x, spec_pred, color=COLOR_PRED, lw=2, label="From DFT modes")
        ax.fill_between(x, 0, spec_pred, color=COLOR_PRED, alpha=0.35)

        ax.set_xlim(0, 4000)
        ax.set_ylim(0, 1.08)
        ax.grid(alpha=0.3)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.tick_params(axis="y", width=1.2, labelsize=10)
        ax.set_ylabel("Normalized\nIntensity")

        if row_idx == len(compounds_group) - 1:
            ax.set_xlabel("Frequency (cm$^{-1}$)")
        if row_idx == 0:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92))

        rmsc = rmsc_lookup.get(compound, np.nan)
        label_compound = format_compound_label(compound)
        label_text = f"{label_compound} — $r_{{\\mathrm{{msc}}}}$ = {rmsc:.3f}"

        ax.text(
            0.02, 0.965,
            label_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )

        if export_data["freq"] is None:
            export_data["freq"] = x
        export_data[f"{compound}_DFT"] = spec_dft
        export_data[f"{compound}_FROM_DFT_MODES"] = spec_pred

    mode_upper = mode.upper()
    plt.tight_layout(h_pad=1.5)
    fig_name = os.path.join(OUTPUT_DIR, f"{mode_upper}_{group_prefix}_three_examples")
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.pdf")
    plt.savefig(f"{fig_name}.svg")
    plt.close()
    print(f"✅ Saved figure -> {fig_name}.png / .pdf / .svg")

    df_out = pd.DataFrame(export_data)
    csv_path = os.path.join(OUTPUT_DIR, f"{mode_upper}_{group_prefix}_three_examples.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"✅ Saved data -> {csv_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    raman_rmsc_lookup, raman_compounds = read_matchscores(RAMAN_CSV_FILE)
    ir_rmsc_lookup, ir_compounds = read_matchscores(IR_CSV_FILE)

    if not raman_compounds:
        print(f"⚠️ No Raman compounds found from {RAMAN_CSV_FILE}")
    if not ir_compounds:
        print(f"⚠️ No IR compounds found from {IR_CSV_FILE}")

    for prefix in GROUP_PREFIXES:
        make_group_figure(prefix, "raman", raman_compounds, raman_rmsc_lookup)
        make_group_figure(prefix, "ir", ir_compounds, ir_rmsc_lookup)

if __name__ == "__main__":
    main()