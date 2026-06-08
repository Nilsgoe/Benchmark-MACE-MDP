#!/usr/bin/env python3
# use: python paper_plot_anharm.py --sim_dirs anharm_asp anharm_para --exp_dir ../exp_spectra/ --dft_dir dft_dir  --mlag 6000 --ftpad 14096 --ftwin cosine-hanning

import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from compare_anharm_approches_1 import (
    backend_direct,
    backend_ipi,
    HAS_IPI
)

# ==========================================================
# COLOR SCHEME (MATCHING HARMONIC SCRIPT)
# ==========================================================
COLOR_PBE = "#DAA520"   # gold (PBE+MBD)
COLOR_FT  = "indianred"   # bright green (FT)
COLOR_EXP = "royalblue"

# ==========================================================
# Utilities
# ==========================================================
def detect_time(df):
    if "time_fs" in df.columns:
        return df["time_fs"].to_numpy(), 1e-15, "fs"
    elif "time_ps" in df.columns:
        return df["time_ps"].to_numpy(), 1e-12, "ps"
    else:
        raise ValueError("No time column found")

def read_pol(df):
    a_xx = df["a00"].to_numpy()
    a_xy = 0.5*(df["a01"]+df["a10"]).to_numpy()
    a_xz = 0.5*(df["a02"]+df["a20"]).to_numpy()
    a_yy = df["a11"].to_numpy()
    a_yz = 0.5*(df["a12"]+df["a21"]).to_numpy()
    a_zz = df["a22"].to_numpy()
    return a_xx, a_xy, a_xz, a_yy, a_yz, a_zz

def normalize(y):
    y = y - y.min()
    if y.max() > 0:
        y /= y.max()
    return y

def find_alias_path(base, directory, suffix):
    direct = os.path.join(directory, f"{base}{suffix}")
    if os.path.isfile(direct):
        return direct

    root, poly = base.rsplit("_", 1)
    poly = "_" + poly
    aliases = ["para", "paracetamol", "acetaminophen"]

    for a in aliases:
        p = os.path.join(directory, f"{a}{poly}{suffix}")
        if os.path.isfile(p):
            return p
    return None

# ==========================================================
# Main
# ==========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dirs", nargs="+", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--dft_dir", required=True)
    ap.add_argument("--backend", choices=["direct","ipi"], default="direct")
    ap.add_argument("--mlag", type=int, default=2048)
    ap.add_argument("--ftpad", type=int, default=2048)
    ap.add_argument("--ftwin", type=str, default="cosine-hanning")
    ap.add_argument("--wn_min", type=float, default=5)
    ap.add_argument("--wn_max", type=float, default=330)
    ap.add_argument("--temperature", type=float, default=300)
    ap.add_argument("--outfile", default="anharm_para_asp_combined.pdf")
    args = ap.parse_args()

    if args.backend == "ipi" and not HAS_IPI:
        raise RuntimeError("i-PI not installed")

    # ---- OUTPUT DIRECTORY (ANHARM) ----
    BASE_OUTPUT_FOLDER = "paper_spectra_anharm"
    os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)
    outpath = os.path.join(BASE_OUTPUT_FOLDER, args.outfile)

    # ---- STYLE ----
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 19,
        "axes.titlesize": 19,
        "axes.linewidth": 2.0,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2.0,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
    })

    LABEL_X = 141
    LABEL_Y_BASE = 0.27

    # ---- FIXED 1x4 LAYOUT (TIGHT SPACING) ----
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    fig.subplots_adjust(wspace=-0.04)   # ✅ tighter horizontal spacing

    panel_order = [
        ("aspirin", "I"),
        ("aspirin", "II"),
        ("para", "I"),
        ("para", "II"),
    ]

    panel_map = {f"{c}_{p}": i for i, (c, p) in enumerate(panel_order)}
    stored = {}

    # ---- Collect all spectra first ----
    for d in args.sim_dirs:
        files = sorted(glob.glob(os.path.join(d, "*_nve_polar.csv")))

        for f in files:
            base = os.path.basename(f).replace("_nve_polar.csv","")
            
            compound = "aspirin" if "aspirin" in base.lower() else "para"
            poly = "I" if ("_I" in base and not "_II" in base) else "II"

            key = f"{compound}_{poly}"
            df = pd.read_csv(f)

            t, to_s, timeunit = detect_time(df)
            dt_s = (t[1]-t[0]) * to_s
            pol = read_pol(df)

            # ---- FT (Simulation) ----
            if args.backend == "direct":
                wn_m, Im = backend_direct(
                    t, dt_s, *pol,
                    args.mlag, args.ftpad, args.ftwin, args.temperature
                )
            else:
                print("Using i-PI backend...")
                wn_m, Im = backend_ipi(
                    t, timeunit, *pol,
                    args.mlag, args.ftpad, args.ftwin, args.temperature
                )

            mask = (wn_m>=args.wn_min)&(wn_m<=args.wn_max)
            wn_m, Im = wn_m[mask], normalize(Im[mask])

            # ---- Exp ----
            exp_path = find_alias_path(base, args.exp_dir, ".csv")
            exp = pd.read_csv(exp_path).iloc[:,0:2]
            exp = exp[(exp.iloc[:,0]>=args.wn_min)&(exp.iloc[:,0]<=args.wn_max)]
            we, Ie = exp.iloc[:,0].to_numpy(), normalize(exp.iloc[:,1].to_numpy())

            # ---- PBE+MBD (DFT reference) ----
            dft_path = find_alias_path(base, args.dft_dir, "_anharm.csv")
            dft = pd.read_csv(dft_path).iloc[:,0:2]
            wd, Id = dft.iloc[:,0].to_numpy(), normalize(dft.iloc[:,1].to_numpy())

            stored[key] = (we, Ie, wn_m, Im, wd, Id)

    # ---- Plot 1x4 panels ----
    for key, idx in panel_map.items():
        ax = axs[idx]
        ax.grid(True, which="both", axis="both")

        we, Ie, wn_m, Im, wd, Id = stored[key]

        # ---- PBE+MBD (offset 0) ----
        ax.plot(wd, Id + 0, "--", color=COLOR_PBE)
        ax.text(LABEL_X, LABEL_Y_BASE+0.47, r"LDA@PBE+MBD", color=COLOR_PBE, fontsize=13.4, va="center")

        # ---- FT (offset 1) ----
        ax.plot(wn_m, Im + 1.1, color=COLOR_FT)
        ax.text(LABEL_X, LABEL_Y_BASE + 1.57, "MACE-MDP@MACE-ft", color=COLOR_FT, fontsize=13.4, va="center")

        # ---- Exp (offset 2) ----
        ax.plot(we, Ie + 2.2, ":",color=COLOR_EXP)
        ax.text(LABEL_X, LABEL_Y_BASE + 2.67, "Experiment", color=COLOR_EXP, fontsize=13.4, va="center")

        ax.set_xlim(0, args.wn_max)
        ax.set_yticks([])
        if "asp" in key:
            text = f"Aspirin {key.split('_')[-1]}"
            ax.set_title(text)
        if "para" in key :
            text = f"Paracetamol {key.split('_')[-1]}"
            ax.set_title(text)

        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    axs[0].set_ylabel("Normalized Intensity", fontsize=17)
    for ax in axs:
        ax.set_xlabel("Frequency (cm$^{-1}$)",fontsize=17)

    plt.tight_layout()
    plt.savefig(f"{outpath.split('.')[0]}.pdf", dpi=400)
    plt.savefig(f"{outpath.split('.')[0]}.png", dpi=400)
    plt.savefig(f"{outpath.split('.')[0]}.svg")
    plt.close()

    print("Saved:", outpath)

if __name__ == "__main__":
    main()
