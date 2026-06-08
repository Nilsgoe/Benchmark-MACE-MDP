#!/usr/bin/env python3
"""
Raman (anharmonic) spectra: DIRECT FFT vs i-PI, with experimental comparison
---------------------------------------------------------------------------

This script:
  1) Reads all *_nve_polar.csv files from --sim_dir
  2) For each file, computes Raman spectra via:
       - DIRECT backend (NumPy FFT)
       - i-PI backend (if installed), using the validated XYZ packing
  3) Applies DETAILED BALANCE(?) :
     
  4) Finds matching experimental CSV in --exp_dir (same base name).
  5) Saves per-structure figure with TWO SUBPLOTS:
       Left : Experimental vs DIRECT
       Right: Experimental vs i-PI  (if available)
  6) Saves a COMBINED figure (two subplots with vertical offsets):
       Left : All Experimental vs DIRECT
       Right: All Experimental vs i-PI (if available)
  7) Writes *_paper_direct.csv and *_paper_ipi.csv with (freq_cm-1, I_paper).

Assumptions:
  - *_nve_polar.csv has columns: time_fs or time_ps, and a00..a22 (9 components)
  - Experimental CSV has two columns: Raman_shift (cm-1), Intensity
"""

import argparse
import os
import glob
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional i-PI import
try:
    from ipi.utils.tools import compute_acf_xyz
    HAS_IPI = True
except ImportError:
    HAS_IPI = False

# ---- constants ----
C_CM_S  = 2.99792458e10            # cm/s
T_AU_S  = 2.4188843265857e-17      # s (atomic unit of time)
HBAR    = 1.054_571_817e-34        # J·s
KB      = 1.380_649e-23            # J/K

# ==========================================================
# Helpers
# ==========================================================
def autocorr_fft(x):
    """Unbiased autocorrelation via FFT"""
    x = np.asarray(x, float) - np.mean(x)
    N = len(x)
    F = np.fft.fft(x, n=2 * N)
    ps = F * np.conjugate(F)
    acf = np.fft.ifft(ps).real[:N]
    #acf /= np.arange(N, 0, -1)        # unbiased
    #if acf[0] != 0:
    #    acf /= acf[0]
    return acf

def window_from_name(name, N):
    n = name.lower()
    if n in ("none", "rect", "rectangular"):
        return np.ones(N)
    if n in ("cosine-hanning", "hann", "hanning"):
        return np.hanning(N)
    if n in ("cosine-hamming", "hamming"):
        return np.hamming(N)
    if n in ("cosine-blackman", "blackman"):
        return np.blackman(N)
    if n in ("triangle-bartlett", "bartlett"):
        return np.bartlett(N)
    raise ValueError(f"Unknown window '{name}'")

def soft_floor(y, rel=1e-12):
    """Prevent large unphysical negative dips due to numerical noise."""
    if y.size == 0:
        return y
    m = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    y = y.copy()
    y[y < -rel * m] = -rel * m
    return y

def bose_detailed_balance(omega_si, beta, eps=1e-15):
    """Detailed-balance factor 1/(1 - exp(-βħω)) with small ε for ω≈0 stability."""
    x = beta * HBAR * omega_si
    # clip extremely small x to avoid division-by-zero at ω→0
    return x / (1.0 - np.exp(-np.clip(x, eps, None)))

# ==========================================================
# DIRECT backend (NumPy)
# ==========================================================
def backend_direct(t, dt_s, a_xx, a_xy, a_xz, a_yy, a_yz, a_zz,
                   mlag, ftpad, ftwin, temperature_K):
    # Isotropic + anisotropic decomposition
    alpha_bar = (a_xx + a_yy + a_zz) / 3.0
    b_xx = a_xx - alpha_bar
    b_yy = a_yy - alpha_bar
    b_zz = a_zz - alpha_bar
    b_xy, b_xz, b_yz = a_xy, a_xz, a_yz

    # Autocorrelations
    C_iso   = autocorr_fft(alpha_bar)
    C_aniso = (autocorr_fft(b_xx) + autocorr_fft(b_yy) + autocorr_fft(b_zz)
               + 2.0 * (autocorr_fft(b_xy) + autocorr_fft(b_xz) + autocorr_fft(b_yz)))

    N = len(C_iso)
    if mlag is None or mlag <= 0 or mlag >= N:
        mlag = N - 1

    C_iso   = C_iso[:mlag + 1]
    C_aniso = C_aniso[:mlag + 1]

    # Half-correlation → half-spectrum via rFFT (hfft of even extension)
    win = window_from_name(ftwin, 2 * mlag + 1)
    C_iso_p   = np.append(C_iso   * win[mlag:], np.zeros(int(ftpad)))
    C_aniso_p = np.append(C_aniso * win[mlag:], np.zeros(int(ftpad)))

    S_iso   = np.fft.hfft(C_iso_p).real
    S_aniso = np.fft.hfft(C_aniso_p).real

    L = mlag + int(ftpad)

    # Frequency grid in rad/s
    dft_factor=1#0.955
    omega = np.arange(2 * L) / float(2 * L) * (2 * np.pi / dt_s)*dft_factor
    # Convert to cm^-1 for output axis
    wn = omega / (2 * np.pi * C_CM_S)

    # Classical "paper" intensity (Placzek combination)
    I_iso   = S_iso
    I_aniso = S_aniso *0.1
    I_class = I_iso + (7.0 / 3.0) * I_aniso #soft_floor(I_iso + (7.0 / 3.0) * I_aniso)
    #I_class = I_iso + 7/45 * S_aniso  # Placzek combination
    # Detailed balance (Option C): I(ω) = S(ω)/(1 - exp(-βħω))
    beta = 1.0 / (KB * temperature_K)
    DB = bose_detailed_balance(omega, beta)
    I_paper = I_class * DB #*beta*omega
    nu_L = 1.0 / (647 * 1e-7)  # nm -> cm, then invert to cm^-1
    factor = 1#np.maximum(nu_L - wn, 1e-15) ** 4

    return wn, I_paper*factor

# ==========================================================
# i-PI backend (use validated packing)
# ==========================================================
def write_iso_xyz(path, times, alpha_bar):
    s = np.asarray(alpha_bar, float)
    with open(path, "w") as f:
        for t, v in zip(times, s):
            f.write("1\n")
            f.write(f"Properties=species:S:1:pos:R:3 Time={t:.10f}\n")
            f.write(f"H {v:.16e} 0.0 0.0\n")

def write_aniso_twoatom_xyz(path, times, b_xx, b_yy, b_zz, b_xy, b_xz, b_yz):
    b_xx = np.asarray(b_xx, float)
    b_yy = np.asarray(b_yy, float)
    b_zz = np.asarray(b_zz, float)
    b_xy = np.asarray(b_xy, float)
    b_xz = np.asarray(b_xz, float)
    b_yz = np.asarray(b_yz, float)
    rt2 = np.sqrt(2.0)
    with open(path, "w") as f:
        for i, t in enumerate(times):
            f.write("2\n")
            f.write(f"Properties=species:S:1:pos:R:3 Time={t:.10f}\n")
            f.write(f"H {b_xx[i]:.16e} {b_yy[i]:.16e} {b_zz[i]:.16e}\n")
            f.write(f"H {rt2*b_xy[i]:.16e} {rt2*b_xz[i]:.16e} {rt2*b_yz[i]:.16e}\n")

def run_ipi_facf(xyz_path, timestep, timeunit, mlag, ftpad, ftwin):
    unit_map = {"fs": "femtosecond", "ps": "picosecond"}
    ipi_timeunit = unit_map.get(timeunit, timeunit)
    block_len = 2 * mlag + 1
    time, acf, acf_err, omega_au, facf, facf_err = compute_acf_xyz(
        input_file=xyz_path,
        maximum_lag=mlag,
        block_length=block_len,
        length_zeropadding=ftpad,
        spectral_windowing=ftwin,
        labels=["*"],
        timestep=float(timestep),
        time_units=ipi_timeunit,
        skip=0,
        compute_derivative=True,
    )
    return omega_au, facf

def backend_ipi(t, timeunit, a_xx, a_xy, a_xz, a_yy, a_yz, a_zz,
                mlag, ftpad, ftwin, temperature_K):
    if not HAS_IPI:
        raise RuntimeError("i-PI not installed.")

    # Decomposition
    alpha_bar = (a_xx + a_yy + a_zz) / 3.0
    b_xx = a_xx - alpha_bar
    b_yy = a_yy - alpha_bar
    b_zz = a_zz - alpha_bar
    b_xy, b_xz, b_yz = a_xy, a_xz, a_yz

    # Write temporary packed XYZs
    dt = np.median(np.diff(t))
    tmp_iso = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz").name
    tmp_an  = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz").name
    write_iso_xyz(tmp_iso, t, alpha_bar)
    write_aniso_twoatom_xyz(tmp_an, t, b_xx, b_yy, b_zz, b_xy, b_xz, b_yz)

    try:
        omega_iso_au, S_iso = run_ipi_facf(tmp_iso, dt, timeunit, mlag, ftpad, ftwin) #should be s~
        omega_an_au,  S_aniso  = run_ipi_facf(tmp_an,  dt, timeunit, mlag, ftpad, ftwin)
    finally:
        for p in (tmp_iso, tmp_an):
            try:
                os.remove(p)
            except Exception:
                pass

    # i-PI returns ω in a.u.^(-1); convert to SI rad/s and to cm^-1 for x-axis
    dft_factor=1#0.955
    omega_si = omega_an_au / T_AU_S *dft_factor
    wn = (omega_an_au / (2 * np.pi)) * (1.0 / (C_CM_S * T_AU_S)) *dft_factor

    # Classical combined (same as direct)
    I_iso = S_iso
 
    I_aniso = S_aniso #*0.1
    I_class = I_iso + (7.0 / 3.0) * I_aniso#soft_floor(I_iso + (7.0 / 3.0) * I_aniso)
    #I_class = I_iso+ (7/45) * S_aniso  # Placzek combination
    
    # Apply detailed balance (Option C)
    beta = 1.0 / (KB * temperature_K)
    DB = bose_detailed_balance(omega_si, beta)  
    I_paper = I_class * DB /(omega_si+1e-4)**2 #*(1.0 / (1.0 * np.exp(-np.clip(omega_si, 1e-15, None))))

    return wn, I_paper

# ==========================================================
# Single file processing
# ==========================================================
def process_file(fpath, timeunit_cli, mlag, ftpad, ftwin, normalize,
                 wn_min, wn_max, temperature_K, exp_dir, output_dir,
                 do_plot=True):
    df = pd.read_csv(fpath)

    # time
    if "time_fs" in df.columns:
        t = df["time_fs"].to_numpy(); timeunit = "fs"; to_seconds = 1e-15
    elif "time_ps" in df.columns:
        t = df["time_ps"].to_numpy(); timeunit = "ps"; to_seconds = 1e-12
    else:
        raise KeyError("Need time_fs or time_ps in CSV")
    # allow overriding via CLI if wanted
    if timeunit_cli is not None:
        timeunit = timeunit_cli

    # components (symmetrized off-diagonals)
    req = ['a00','a01','a02','a10','a11','a12','a20','a21','a22']
    for c in req:
        if c not in df.columns:
            raise KeyError(f"Missing column {c}")
    a_xx = df["a00"].to_numpy()
    a_xy = 0.5 * (df["a01"] + df["a10"]).to_numpy()
    a_xz = 0.5 * (df["a02"] + df["a20"]).to_numpy()
    a_yy = df["a11"].to_numpy()
    a_yz = 0.5 * (df["a12"] + df["a21"]).to_numpy()
    a_zz = df["a22"].to_numpy()

    dt_s = (t[1] - t[0]) * to_seconds

    # DIRECT backend
    wn_d, Ipaper_d = backend_direct(
        t, dt_s, a_xx, a_xy, a_xz, a_yy, a_yz, a_zz,
        mlag, ftpad, ftwin, temperature_K
    )
    mask_d = (wn_d >= wn_min) & (wn_d <= wn_max)
    wn_d, Ipaper_d = wn_d[mask_d], Ipaper_d[mask_d]
    if normalize and Ipaper_d.max() > 0:
        Ipaper_d = Ipaper_d / Ipaper_d.max()

    base = os.path.basename(fpath).replace("_nve_polar.csv", "")
    out_direct = os.path.join(output_dir, f"{base}_paper_direct.csv")
    pd.DataFrame({"freq_cm-1": wn_d, "I_paper": Ipaper_d}).to_csv(out_direct, index=False)

    # i-PI backend (optional)
    have_ipi = False
    wn_i = Ipaper_i = None
    if HAS_IPI:
        print(f"  -> Running i-PI backend for {base}...")
        have_ipi = True
        wn_i, Ipaper_i = backend_ipi(
            t, timeunit, a_xx, a_xy, a_xz, a_yy, a_yz, a_zz,
            mlag, ftpad, ftwin, temperature_K
        )
        mask_i = (wn_i >= wn_min) & (wn_i <= wn_max)
        wn_i, Ipaper_i = wn_i[mask_i], Ipaper_i[mask_i]
        if normalize and Ipaper_i.max() > 0:
            Ipaper_i = Ipaper_i / Ipaper_i.max()
        out_ipi = os.path.join(output_dir, f"{base}_paper_ipi.csv")
        pd.DataFrame({"freq_cm-1": wn_i, "I_paper": Ipaper_i}).to_csv(out_ipi, index=False)

    # Experimental match
    exp_path = os.path.join(exp_dir, f"{base}.csv")
    exp_x = exp_y = None
    if os.path.isfile(exp_path):
        exp_df = pd.read_csv(exp_path)
        exp_df = exp_df[(exp_df.iloc[:, 0] >= wn_min) & (exp_df.iloc[:, 0] <= wn_max)]
        exp_x = exp_df.iloc[:, 0].to_numpy()
        exp_y = exp_df.iloc[:, 1].to_numpy()
        # normalize exp to [0,1]
        exp_y = exp_y - np.min(exp_y)
        m = np.max(exp_y)
        if m > 0:
            exp_y = exp_y / m

    # Per-structure figure: TWO SUBPLOTS + headers
    if do_plot and exp_x is not None and exp_y is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        # Left: Experimental vs DIRECT
        axes[0].plot(exp_x, np.maximum(exp_y, 0), label="Experimental", lw=2, color="black")
        axes[0].plot(wn_d, np.maximum(Ipaper_d, 0), label="Direct (FFT)", lw=2)
        axes[0].set_title(f"{base} — Experimental vs Direct", fontsize=12)
        axes[0].set_xlabel("Raman shift (cm$^{-1}$)")
        axes[0].set_ylabel("Normalized intensity")
        axes[0].set_xlim(wn_min, wn_max)
        axes[0].legend()

        # Right: Experimental vs i-PI (if available)
        if have_ipi and wn_i is not None:
            axes[1].plot(exp_x, np.maximum(exp_y, 0), label="Experimental", lw=2, color="black")
            axes[1].plot(wn_i, np.maximum(Ipaper_i, 0), label="i-PI", lw=2)
            axes[1].set_title(f"{base} — Experimental vs i-PI", fontsize=12)
            axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
            axes[1].set_xlim(wn_min, wn_max)
            axes[1].legend()
        else:
            axes[1].axis("off")

        plt.tight_layout()
        out_png = os.path.join(output_dir, f"{base}_exp_vs_direct_and_ipi.png")
        plt.savefig(out_png, dpi=350)
        plt.close()

    # Return data for combined plot
    return {
        "name": base,
        "exp": (exp_x, exp_y),
        "direct": (wn_d, Ipaper_d),
        "ipi": (wn_i, Ipaper_i) if have_ipi else None,
    }

# ==========================================================
# Combined plot: two subplots with offsets
# ==========================================================
def make_combined_plot(all_entries, wn_min, wn_max, output_dir,entry):
    base=entry['name']
    # Filter entries with experimental + at least one backend
    entries = [e for e in all_entries if e["exp"][0] is not None]
    if not entries:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axL, axR = axes

    # vertical offset helper
    def add_with_offset(ax, x, y, shift, label, color=None, style="-", lw=1.8):
        if x is None or y is None:
            return
        ax.plot(x, y + shift, label=label, linestyle=style, linewidth=lw, color=color)

    styles = ["-", "--", ":"]
    for idx, e in enumerate(entries):
        shift = idx * 1.2
        ex, ey = e["exp"]
        dx, dy = e["direct"]
        ipi = e["ipi"]

        # Left: Experimental vs Direct
        add_with_offset(axL, ex, np.maximum(ey, 0), shift, f"{e['name']} Exp", color="black", style=styles[idx % 3])
        add_with_offset(axL, dx, np.maximum(dy, 0), shift, f"{e['name']} Direct")

        # Right: Experimental vs i-PI (if available)
        if ipi is not None:
            ix, iy = ipi
            add_with_offset(axR, ex, np.maximum(ey, 0), shift, f"{e['name']} Exp", color="black", style=styles[idx % 3])
            add_with_offset(axR, ix, np.maximum(iy, 0), shift, f"{e['name']} i-PI")
        else:
            # If no i-PI anywhere, hide right panel after loop
            pass

    # Format axes
    axL.set_title("Experimental vs Direct (FFT)", fontsize=13)
    axL.set_xlim(wn_min, wn_max)
    axL.set_xlabel("Raman shift (cm$^{-1}$)")
    axL.set_ylabel("Intensity (a.u.) + offset")
    axL.legend(fontsize=8)

    # If nobody had i-PI, hide right axis
    if not any(e["ipi"] is not None for e in entries):
        axR.axis("off")
    else:
        axR.set_title("Experimental vs i-PI", fontsize=13)
        axR.set_xlim(wn_min, wn_max)
        axR.set_xlabel("Raman shift (cm$^{-1}$)")
        axR.legend(fontsize=8)

    plt.tight_layout()
    comb_path = os.path.join(output_dir, f"combined_{base}_exp_vs_direct_and_ipi.png")
    plt.savefig(comb_path, dpi=350)
    plt.close()

# ==========================================================
# CLI
# ==========================================================
def main():
    ap = argparse.ArgumentParser(description="Direct/i-PI anharmonic Raman with detailed balance and experimental comparison.")
    ap.add_argument("--sim_dir", required=True, help="Directory with *_nve_polar.csv")
    ap.add_argument("--exp_dir", required=True, help="Directory with experimental CSVs (base-name match).")
    ap.add_argument("--output_dir", default="raman_outputs", help="Where to write CSVs and plots")
    ap.add_argument("--timeunit", choices=["fs", "ps"], default=None, help="Override time unit of input (default: autodetect from CSV)")
    ap.add_argument("--mlag",  type=int, default=2048)
    ap.add_argument("--ftpad", type=int, default=2048)
    ap.add_argument("--ftwin", type=str, default="cosine-hanning")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--wn_min", type=float, default=5.0)
    ap.add_argument("--wn_max", type=float, default=350.0)
    ap.add_argument("--temperature", type=float, default=300.0, help="Temperature in K for detailed balance")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.sim_dir, "*_nve_polar.csv")))
    if not files:
        raise FileNotFoundError(f"No *_nve_polar.csv found in {args.sim_dir}")

    collected = []
    for f in files:
        print(f"Processing {os.path.basename(f)} ...")
        entry = process_file(
            f, args.timeunit, args.mlag, args.ftpad, args.ftwin, args.normalize,
            args.wn_min, args.wn_max, args.temperature,
            args.exp_dir, args.output_dir, do_plot=True
        )
        collected.append(entry)

    # Combined figure (two subplots with headers)
    make_combined_plot(collected, args.wn_min, args.wn_max, args.output_dir,entry)
    print("Done.")

if __name__ == "__main__":
    main()
