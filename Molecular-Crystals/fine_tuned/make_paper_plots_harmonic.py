#!/usr/bin/env python3
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================
molecules = [
    "aspirin_I", "aspirin_II",
    "paracetamol_I", "paracetamol_II",
    "acetaminophen_I", "acetaminophen_II",
]

harm_exp_folder = "harmonic_ref"
model_folder = "spectra_data_harm"

SKIP_OFF = False
BASE_OUTPUT_FOLDER = "paper_spectra"
output_folder = BASE_OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

ystep = 1.0
NORMALIZE = True
XMIN, XMAX = 0, 330

LABEL_X = 245
LABEL_Y_BASE = 0.18

# ==========================================================
# PLOT STYLE
# ==========================================================
plt.rcParams.update({
    "font.size": 17,
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "axes.linewidth": 2.3,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "lines.linewidth": 2.7,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
})

# ==========================================================
# COLORS
# ==========================================================
COLOR_HARM = "#DAA520"      # PBE+MBD
COLOR_EXP = "royalblue"
COLOR_FINETUNED = "indianred"
COLOR_LARGE = "#FF8C00"
COLOR_MEDIUM = "#8A2BE2"
COLOR_SMALL = "#32CD32"
COLOR_OMOL = "#505050"
COLOR_DEFAULT = "#808080"

# ==========================================================
# COLOR + LABEL MAPPERS
# ==========================================================
def color_for_model(model_key):
    key = model_key.lower()
    if "omol" in key:
        return COLOR_OMOL
    elif "finetuned" in key or "fine" in key:
        return COLOR_FINETUNED
    elif "extra_large" in key or "xlarge" in key or "xl" in key or "large" in key:
        return COLOR_LARGE
    elif "medium" in key:
        return COLOR_MEDIUM
    elif "small" in key:
        return COLOR_SMALL
    else:
        return COLOR_DEFAULT

def label_for_model(model_key):
    """
    Inline label rules:
      - OMOL -> "OMOL"
      - finetuned -> "FT"
      - OFF variants -> "OFF small/medium/large" (when detectable)
      - otherwise: no label
    """
    k = model_key.lower()
    if "omol" in k:
        return "OMOL"
    if "finetuned" in k or "fine" in k:
        return "FT"

    if "off" in k:
        if "small" in k:
            return "OFF small"
        if "medium" in k:
            return "OFF medium"
        if any(t in k for t in ("extra_large", "xlarge", "xl", "large")):
            return "OFF large"
        return "OFF"
    return None

def exp_label_xshift(name_or_base: str) -> int:
    s = name_or_base.lower()
    return 80 if ("para" in s or "aceta" in s) else 0

# ==========================================================
# NAME HELPERS (FOR CORRECT HEADERS)
# ==========================================================
def split_name(raw_name):
    raw = raw_name.strip()
    if "_" not in raw:
        return raw.lower(), ""
    base, phase = raw.rsplit("_", 1)
    base = base.lower()
    p = phase.upper()
    if p in ("I", "1"):
        p = "I"
    elif p in ("II", "2"):
        p = "II"
    return base, p

def pretty_base(base: str) -> str:
    return {
        "aspirin": "Aspirin",
        "paracetamol": "Paracetamol",
        "acetaminophen": "Acetaminophen",
    }.get(base.lower(), base.capitalize())

def pretty_molecule_title(base: str, phase: str) -> str:
    # "Aspirin I", "Paracetamol II", ...
    return f"{pretty_base(base)} {phase}"

def pretty_species_title(base: str) -> str:
    # used for combined figure suptitle / filename readability
    return pretty_base(base)

# ==========================================================
# FILE HELPERS
# ==========================================================
def reference_base_for(base):
    return "paracetamol" if base in ("paracetamol", "acetaminophen") else base

def model_bases_for(base):
    return ["paracetamol", "acetaminophen"] if base in ("paracetamol", "acetaminophen") else [base]

def find_model_csvs(model_folder, bases, phase):
    files = []
    for b in bases:
        files.extend(glob.glob(os.path.join(model_folder, f"{b}_{phase}_broadened_*.csv")))
        files.extend(glob.glob(os.path.join(model_folder, f"{b}_{phase}_*_broadened_*.csv")))
    return sorted(set(files))

def load_model_series(fpath):
    fname = os.path.basename(fpath)
    if "broadened_" not in fname:
        return None
    model_key = fname.split("broadened_")[-1].replace(".csv", "")
    df = pd.read_csv(fpath)
    col = f"intensity_{model_key}"
    if col not in df.columns:
        print(f"WARNING: missing {col} in {fname}")
        return None
    return model_key, df["cm1"].values, df[col].values

def max_in_plot_range(x, y, xmin=XMIN, xmax=XMAX):
    mask = (x >= xmin) & (x <= xmax)
    return float(np.max(y[mask])) if np.any(mask) else 0.0

# ==========================================================
# LOAD + NORMALIZE PHASE
# ==========================================================
def load_phase_data(raw_name):
    base, phase = split_name(raw_name)
    ref_base = reference_base_for(base)
    ref_harm = os.path.join(harm_exp_folder, f"{ref_base}_{phase}_harmonic_norm.csv")
    ref_exp  = os.path.join(harm_exp_folder, f"{ref_base}_{phase}_exp_norm.csv")

    if not (os.path.exists(ref_harm) and os.path.exists(ref_exp)):
        print(f"⚠️ Missing reference for {raw_name}")
        return None

    harm_df, exp_df = pd.read_csv(ref_harm), pd.read_csv(ref_exp)

    model_files = find_model_csvs(model_folder, model_bases_for(base), phase)
    series_raw = []
    for f in model_files:
        item = load_model_series(f)
        if item is not None:
            series_raw.append(item)

    if not series_raw:
        print(f"⚠️ No model series found for {raw_name}")
        return None

    if NORMALIZE:
        maxima = [
            max_in_plot_range(harm_df["cm1"], harm_df["intensity"]),
            max_in_plot_range(exp_df["cm1"], exp_df["intensity"]),
        ] + [max_in_plot_range(x, y) for _, x, y in series_raw]

        # guard
        maxima = [m if m > 0 else 1.0 for m in maxima]

        harm_df["intensity"] /= maxima[0]
        exp_df["intensity"]  /= maxima[1]
        model_series = [(mk, x, y / maxima[i + 2]) for i, (mk, x, y) in enumerate(series_raw)]
    else:
        model_series = series_raw

    return harm_df, exp_df, model_series

# ==========================================================
# SINGLE STACKED PLOTS
# ==========================================================
for raw_name in molecules:
    print(f"\nPlotting stacked spectra for {raw_name}")
    data = load_phase_data(raw_name)
    if data is None:
        print("  Skipping...")
        continue

    base, phase = split_name(raw_name)
    harm_df, exp_df, model_series = data
    fig, ax = plt.subplots(figsize=(8, 10))

    # PBE+MBD
    ax.plot(harm_df["cm1"], harm_df["intensity"], "--", color=COLOR_HARM)
    ax.text(LABEL_X, LABEL_Y_BASE, "PBE+MBD", color=COLOR_HARM, fontsize=17, va="center")

    plot_idx = 1
    for model_key, x, y in model_series:
        if SKIP_OFF and "off" in model_key.lower():
            continue

        color = color_for_model(model_key)
        yoff = plot_idx * ystep
        ax.plot(x, y + yoff, "-", color=color)

        lbl = label_for_model(model_key)
        if lbl is not None:
            ax.text(LABEL_X, LABEL_Y_BASE + yoff, lbl, color=color, fontsize=17, va="center")

        plot_idx += 1

    top = plot_idx
    ax.plot(exp_df["cm1"], exp_df["intensity"] + top * ystep, ":", color=COLOR_EXP, lw=2.5)
    ax.text(
        LABEL_X - exp_label_xshift(raw_name),
        LABEL_Y_BASE + top * ystep,
        "Exp",
        color=COLOR_EXP,
        fontsize=17,
        va="center",
    )

    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Normalized Inentsity")
    ax.set_yticks([])

    # ✅ correct header here too (optional but consistent)
    #ax.set_title(pretty_molecule_title(base, phase))

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()
    out_png = os.path.join(output_folder, f"{raw_name}_stacked_from_csv.png")
    plt.savefig(out_png)
    plt.savefig(out_png.replace(".png", ".pdf"))
    plt.close()
    print(f"  ✅ Saved {out_png}")

# ==========================================================
# COMBINED I vs II PLOTS (ONLY CHANGE: CORRECT HEADERS)
# ==========================================================
print("\n=== Creating publication-level combined plots (I vs II) ===")
species = sorted(set(base for base, _ in map(split_name, molecules)))

for base in species:
    data_I  = load_phase_data(f"{base}_I")
    data_II = load_phase_data(f"{base}_II")
    if data_I is None or data_II is None:
        print(f"⚠️ Skipping {base} (missing one phase)")
        continue

    harm_I, exp_I, models_I = data_I
    harm_II, exp_II, models_II = data_II

    fig, axes = plt.subplots(1, 2, figsize=(14, 11), sharey=True)

    # ✅ overall header (molecule type) - optional, but usually what people mean by "correct header"
    #fig.suptitle(pretty_species_title(base), fontsize=19)

    for ax, (harm_df, exp_df, model_series, phase) in zip(
        axes, [(harm_I, exp_I, models_I, "I"), (harm_II, exp_II, models_II, "II")]
    ):
        ax.plot(harm_df["cm1"], harm_df["intensity"], "--", color=COLOR_HARM)
        ax.text(LABEL_X, LABEL_Y_BASE, "PBE+MBD", color=COLOR_HARM, fontsize=17, va="center")

        plot_idx = 1
        for model_key, x, y in model_series:
            if SKIP_OFF and "off" in model_key.lower():
                continue

            color = color_for_model(model_key)
            yoff = plot_idx * ystep
            ax.plot(x, y + yoff, "-", color=color)

            lbl = label_for_model(model_key)
            if lbl is not None:
                ax.text(LABEL_X, LABEL_Y_BASE + yoff, lbl, color=color, fontsize=17, va="center")

            plot_idx += 1

        top = plot_idx
        ax.plot(exp_df["cm1"], exp_df["intensity"] + top * ystep, ":", color=COLOR_EXP, lw=2.5)
        ax.text(
            LABEL_X - exp_label_xshift(base),
            LABEL_Y_BASE + 0.04 + top * ystep,
            "Exp",
            color=COLOR_EXP,
            fontsize=17,
            va="center",
        )

        ax.set_xlim(XMIN, XMAX)
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_yticks([])
        ax.grid(True, which="both", axis="both")

        # ✅ "correct header for molecule type and polymorph number"
        ax.set_title(pretty_molecule_title(base, phase), fontweight="bold")

        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    axes[0].set_ylabel("Normalized Inentsity")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    out_comb = os.path.join(output_folder, f"{base}_I_II_combined")
    plt.savefig(out_comb + ".png")
    plt.savefig(out_comb + ".pdf")
    plt.close()
    print(f"  ✅ Saved publication-level combined → {out_comb}.png / .pdf")

print("\n✅ Done. (No 1×4 figure; 1×2 now has correct molecule+polymorph headers.)")
