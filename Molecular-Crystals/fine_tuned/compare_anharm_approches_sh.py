#!/usr/bin/env python3
"""
Compare anharmonic Raman spectra: direct FFT vs i-PI ACF (Paper only)
---------------------------------------------------------------------
Implements exact Placzek combination but exports/plots only I_paper:

  I_paper(ω) = I_iso(ω) + (7/3) * I_aniso(ω)
  I_aniso(ω) = (1/10) * [ S(b_xx)+S(b_yy)+S(b_zz) + 2(S(b_xy)+S(b_xz)+S(b_yz)) ]

Backends:
  - direct : NumPy FFT (i-PI-like pipeline in pure Python)
  - ipi    : i-PI compute_acf_xyz using the *working* XYZ packing you provided:
      * iso: 1 atom/frame  -> H  alpha_bar  0  0
      * aniso: 2 atoms/frame:
          atom1: H  b_xx  b_yy  b_zz
          atom2: H  sqrt(2)*b_xy  sqrt(2)*b_xz  sqrt(2)*b_yz

Outputs per input:
  - *_paper_direct.csv      (freq_cm-1, I_paper)
  - *_paper_ipi.csv         (freq_cm-1, I_paper)  [only if i-PI present]
  - *_comparison_paper_direct_vs_ipi.png
"""

import argparse, os, glob, tempfile
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
HBAR = 1.054_571_817e-34      # J·s
KB   = 1.380_649e-23          # J/K
# ==========================================================
# Helpers (shared)
# ==========================================================
def ensure_uniform_grid(t, series_list):
    t = np.asarray(t, float)
    dt = np.diff(t)
    if len(dt) == 0:
        raise ValueError("Time array too short.")
    dt_med = np.median(dt)
    irr = np.max(np.abs(dt - dt_med)) / dt_med
    if irr <= 1e-3:
        return t, series_list, dt_med
    t_uniform = np.arange(t[0], t[-1] + 0.5 * dt_med, dt_med)
    new_series = [np.interp(t_uniform, t, s) for s in series_list]
    return t_uniform, new_series, dt_med

def autocorr_fft(x):
    x = np.asarray(x, float) - np.mean(x)
    N = len(x)
    F = np.fft.fft(x, n=2*N)
    ps = F * np.conjugate(F)
    acf = np.fft.ifft(ps).real[:N]
    acf /= np.arange(N, 0, -1)        # unbiased
    if acf[0] != 0: acf /= acf[0]
    return acf

def window_from_name(name, N):
    n = name.lower()
    if n in ("none","rect","rectangular"): return np.ones(N)
    if n in ("cosine-hanning","hann","hanning"): return np.hanning(N)
    if n in ("cosine-hamming","hamming"): return np.hamming(N)
    if n in ("cosine-blackman","blackman"): return np.blackman(N)
    if n in ("triangle-bartlett","bartlett"): return np.bartlett(N)
    raise ValueError(f"Unknown window '{name}'")

def soft_floor(y, rel=1e-10):
    if y.size == 0: return y
    m = np.max(np.abs(y))
    y = y.copy()
    y[y < -rel*m] = -rel*m
    return y

# ==========================================================
# DIRECT backend (NumPy)
# ==========================================================
def backend_direct(t, dt_s, a_xx,a_xy,a_xz,a_yy,a_yz,a_zz, mlag, ftpad, ftwin):
    alpha_bar = (a_xx + a_yy + a_zz)/3.0
    b_xx = a_xx - alpha_bar
    b_yy = a_yy - alpha_bar
    b_zz = a_zz - alpha_bar
    b_xy,b_xz,b_yz = a_xy,a_xz,a_yz

    C_iso   = autocorr_fft(alpha_bar)
    C_aniso = (autocorr_fft(b_xx) + autocorr_fft(b_yy) + autocorr_fft(b_zz)
               + 2*(autocorr_fft(b_xy)+autocorr_fft(b_xz)+autocorr_fft(b_yz)))

    N = len(C_iso)
    if mlag is None or mlag <= 0 or mlag >= N: mlag = N-1
    C_iso, C_aniso = C_iso[:mlag+1], C_aniso[:mlag+1]

    win = window_from_name(ftwin, 2*mlag+1)
    C_iso_p   = np.append(C_iso   * win[mlag:], np.zeros(int(ftpad)))
    C_aniso_p = np.append(C_aniso * win[mlag:], np.zeros(int(ftpad)))

    S_iso   = np.fft.hfft(C_iso_p)
    S_aniso = np.fft.hfft(C_aniso_p)

    L = mlag + int(ftpad)
    omega = np.arange(2*L) / float(2*L) * (2*np.pi / dt_s)
    wn = omega / (2*np.pi*C_CM_S)

    I_iso   = np.real(S_iso)
    I_aniso = 0.1 * np.real(S_aniso)
    I_paper = soft_floor(I_iso + (7.0/3.0)*I_aniso)
    return wn, I_paper

# ==========================================================
# i-PI backend (use your working packing)
# ==========================================================
def write_iso_xyz(path, times, alpha_bar):
    s = np.asarray(alpha_bar, float) - np.mean(alpha_bar)
    with open(path, "w") as f:
        for t, v in zip(times, s):
            f.write("1\n")
            f.write(f"Properties=species:S:1:pos:R:3 Time={t:.10f}\n")
            f.write(f"H {v:.16e} 0.0 0.0\n")

def write_aniso_twoatom_xyz(path, times, b_xx,b_yy,b_zz,b_xy,b_xz,b_yz):
    b_xx = np.asarray(b_xx, float) #- np.mean(b_xx)
    b_yy = np.asarray(b_yy, float) #- np.mean(b_yy)
    b_zz = np.asarray(b_zz, float) #- np.mean(b_zz)
    b_xy = np.asarray(b_xy, float) #- np.mean(b_xy)
    b_xz = np.asarray(b_xz, float) #- np.mean(b_xz)
    b_yz = np.asarray(b_yz, float) #- np.mean(b_yz)
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
    block_len = 2*mlag + 1
    time, acf, acf_err, omega, facf, facf_err = compute_acf_xyz(
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

    return omega, np.real(facf)

def backend_ipi(t, timeunit, a_xx,a_xy,a_xz,a_yy,a_yz,a_zz, mlag, ftpad, ftwin):
    if not HAS_IPI:
        raise RuntimeError("i-PI not installed.")

    alpha_bar = (a_xx + a_yy + a_zz)/3.0
    b_xx = a_xx - alpha_bar
    b_yy = a_yy - alpha_bar
    b_zz = a_zz - alpha_bar
    b_xy,b_xz,b_yz = a_xy,a_xz,a_yz

    dt = np.median(np.diff(t))
    tmp_iso = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz").name
    tmp_an  = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz").name
    write_iso_xyz(tmp_iso, t, alpha_bar)
    write_aniso_twoatom_xyz(tmp_an, t, b_xx,b_yy,b_zz,b_xy,b_xz,b_yz)

    try:
        omega_iso, S_iso   = run_ipi_facf(tmp_iso, dt, timeunit, mlag, ftpad, ftwin)
        omega_an,  S_aniso = run_ipi_facf(tmp_an,  dt, timeunit, mlag, ftpad, ftwin)
    finally:
        for p in (tmp_iso, tmp_an):
            try: os.remove(p)
            except: pass
    #omega_an[:3]+=1e-5
    omega_si = omega_an / T_AU_S

    beta = 1.0 / (KB * 300)
    """ std
    num = 1 - np.exp(-beta * HBAR * omega_si)
    denom =np.pi * omega_si* (1 + np.exp(-beta * HBAR * omega_si))**2
    F= num / denom
    #print(num,denom,F)"""
    #Kubo 
    num = beta * HBAR
    denom = 2*np.pi *omega_si**0 *(1+ np.exp(-beta * HBAR * omega_si)) 
    F= num / denom
    print("F min, max:", float(F.min()), float(F.max()))
    #exit()
    S_iso = F*S_iso
    S_aniso = F*S_aniso
    # Convert ω (au^-1) → cm^-1
    freq_cm1 = (omega_an / (2*np.pi)) * (1.0 / (C_CM_S * T_AU_S))

    # Paper intensity
    I_iso   = S_iso
    I_aniso = 0.1 * S_aniso
    I_paper = soft_floor(I_iso + (7.0/3.0)*I_aniso)

    return freq_cm1, I_paper

# ==========================================================
# Driver
# ==========================================================
def process_file(fpath, timeunit, mlag, ftpad, ftwin, normalize, wn_min, wn_max, do_plot):
    df = pd.read_csv(fpath)

    # time
    if "time_fs" in df.columns:
        t = df["time_fs"].to_numpy(); timeunit = "fs"; to_seconds = 1e-15
    elif "time_ps" in df.columns:
        t = df["time_ps"].to_numpy(); timeunit = "ps"; to_seconds = 1e-12
    else:
        raise KeyError("Need time_fs or time_ps in CSV")

    # components
    req = ['a00','a01','a02','a10','a11','a12','a20','a21','a22']
    for c in req:
        if c not in df.columns: raise KeyError(f"Missing column {c}")
    a_xx = df["a00"].to_numpy()
    a_xy = 0.5*(df["a01"] + df["a10"]).to_numpy()
    a_xz = 0.5*(df["a02"] + df["a20"]).to_numpy()
    a_yy = df["a11"].to_numpy()
    a_yz = 0.5*(df["a12"] + df["a21"]).to_numpy()
    a_zz = df["a22"].to_numpy()

    ## sort & uniform time grid
    #order = np.argsort(t); t = t[order]
    #a_xx,a_xy,a_xz,a_yy,a_yz,a_zz = [arr[order] for arr in (a_xx,a_xy,a_xz,a_yy,a_yz,a_zz)]
    #t, [a_xx,a_xy,a_xz,a_yy,a_yz,a_zz], dt = ensure_uniform_grid(t, [a_xx,a_xy,a_xz,a_yy,a_yz,a_zz])
    dt=1e-15 if timeunit=="fs" else 1e-12
    dt_s = dt * to_seconds

    # direct backend
    wn_d, Ipaper_d = backend_direct(t, dt_s, a_xx,a_xy,a_xz,a_yy,a_yz,a_zz, mlag, ftpad, ftwin)
    mask_d = (wn_d >= wn_min) & (wn_d <= wn_max)
    wn_d, Ipaper_d = wn_d[mask_d], Ipaper_d[mask_d]
    if normalize and Ipaper_d.max()>0: Ipaper_d = Ipaper_d / Ipaper_d.max()
    base = fpath.replace("_nve_polar.csv", "")
    out_direct = f"{base}_paper_direct.csv"
    pd.DataFrame({"freq_cm-1": wn_d, "I_paper": Ipaper_d}).to_csv(out_direct, index=False)
    print(f"  → saved {out_direct}")

    # i-PI backend (if available)
    wn_i = Ipaper_i = None
    have_ipi = False
    if HAS_IPI:
        have_ipi = True
        wn_i, Ipaper_i = backend_ipi(t, timeunit, a_xx,a_xy,a_xz,a_yy,a_yz,a_zz, mlag, ftpad, ftwin)
        mask_i = (wn_i >= wn_min) & (wn_i <= wn_max)
        wn_i, Ipaper_i = wn_i[mask_i], Ipaper_i[mask_i]
        if normalize and Ipaper_i.max()>0: Ipaper_i = Ipaper_i / Ipaper_i.max()
        out_ipi = f"{base}_paper_ipi.csv"
        pd.DataFrame({"freq_cm-1": wn_i, "I_paper": Ipaper_i}).to_csv(out_ipi, index=False)
        print(f"  → saved {out_ipi}")
    else:
        print("  (i-PI not found; skipping ipi curve)")

    # comparison plot (Paper only)
    if do_plot:
        plt.figure(figsize=(7,4.5))
        plt.plot(wn_d, np.maximum(Ipaper_d,0), label="Paper (direct)", lw=1.8)
        if have_ipi:
            plt.plot(wn_i, np.maximum(Ipaper_i,0), label="Paper (i-PI)", lw=1.8)
        plt.xlabel("Raman shift / cm$^{-1}$"); plt.ylabel("Intensity (a.u.)")
        plt.xlim(wn_min, wn_max); plt.legend(); plt.tight_layout()
        out_png = f"{base}_comparison_paper_direct_vs_ipi.png"
        plt.savefig(out_png, dpi=300); plt.close()
        print(f"  → saved {out_png}")

# ==========================================================
# CLI
# ==========================================================
def main():
    ap = argparse.ArgumentParser(description="Direct vs i-PI anharmonic Raman (Paper only).")
    ap.add_argument("--indir", required=True, help="Directory with *_nve_polar.csv")
    ap.add_argument("--timeunit", choices=["fs","ps"], default="fs")
    ap.add_argument("--mlag",  type=int, default=2048)
    ap.add_argument("--ftpad", type=int, default=2048)
    ap.add_argument("--ftwin", type=str, default="cosine-hanning")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--wn_min", type=float, default=10.0)
    ap.add_argument("--wn_max", type=float, default=200.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*_nve_polar.csv")))
    if not files:
        raise FileNotFoundError(f"No *_nve_polar.csv found in {args.indir}")

    for f in files:
        print(f"\nProcessing {os.path.basename(f)} ...")
        process_file(f, args.timeunit, args.mlag, args.ftpad, args.ftwin,
                     args.normalize, args.wn_min, args.wn_max, args.plot)

if __name__ == "__main__":
    main()
