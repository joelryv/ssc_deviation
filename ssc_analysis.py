"""Measure the SSC (spread-spectrum clocking) deviation of a PCIe REFCLK capture.

Input: CSV with column 1 = time [s], column 2 = differential voltage [V].

Example:
    python ssc_analysis.py refclk.csv --fnom 100e6 --lpf 2e6 --plot
"""
import argparse
import sys

import numpy as np
from scipy import optimize, signal


def load_waveform(path):
    """Load (t, v) from a 2-column CSV; header / non-numeric rows are skipped."""
    data = np.genfromtxt(path, delimiter=",", usecols=(0, 1), invalid_raise=False)
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] < 100:
        raise ValueError(f"{path}: not enough numeric samples")
    order = np.argsort(data[:, 0], kind="stable")
    return data[order, 0], data[order, 1]


def rising_edges(t, v, threshold=0.0, hysteresis=None):
    """Rising threshold-crossing times, linearly interpolated, with hysteresis.

    A crossing is accepted only after the signal has been below
    threshold - hysteresis/2 and then reaches threshold + hysteresis/2, so noise
    around the threshold does not create extra edges.
    """
    if hysteresis is None:
        lo, hi = np.percentile(v, [5, 95])
        hysteresis = 0.2 * (hi - lo)
    upper, lower = threshold + hysteresis / 2, threshold - hysteresis / 2

    # Forward-filled hysteresis state: +1 above upper, -1 below lower
    state = np.where(v >= upper, 1, np.where(v <= lower, -1, 0))
    nz = np.flatnonzero(state)
    filled = np.zeros_like(state)
    filled[nz] = state[nz]
    filled = filled[np.maximum.accumulate(np.where(filled != 0, np.arange(filled.size), 0))]
    armed_rise = np.flatnonzero((filled[1:] == 1) & (filled[:-1] == -1)) + 1

    # Raw crossings of the threshold; keep the last one before each armed transition
    raw = np.flatnonzero((v[:-1] < threshold) & (v[1:] >= threshold))
    pos = np.searchsorted(raw, armed_rise, side="right") - 1
    k = np.unique(raw[pos[pos >= 0]])

    return t[k] + (threshold - v[k]) * (t[k + 1] - t[k]) / (v[k + 1] - v[k])


def frequency_vs_time(edges):
    """Per-cycle frequency from consecutive rising edges, placed at the period midpoint."""
    periods = np.diff(edges)
    return 0.5 * (edges[1:] + edges[:-1]), 1.0 / periods


def lowpass_profile(t_f, f, cutoff, order=2):
    """Resample to a uniform grid and apply a zero-phase Butterworth low-pass.

    The low-pass removes cycle-to-cycle jitter while keeping the slow SSC
    modulation. Filter start-up transients are trimmed from both ends.
    """
    dt = np.median(np.diff(t_f))
    tu = np.arange(t_f[0], t_f[-1], dt)
    fu = np.interp(tu, t_f, f)
    fs = 1.0 / dt
    if cutoff is None or cutoff <= 0 or cutoff >= fs / 2:
        return tu, fu
    sos = signal.butter(order, cutoff, fs=fs, output="sos")
    ff = signal.sosfiltfilt(sos, fu - fu.mean()) + fu.mean()
    trim = min(int(np.ceil(5.0 / cutoff / dt)), tu.size // 10)
    return tu[trim:tu.size - trim], ff[trim:ff.size - trim]


def modulation_cycles(tu, ff):
    """Return SSC modulation frequency and the rising mid-crossings of the profile."""
    x = ff - 0.5 * (ff.max() + ff.min())
    h = 0.1 * (ff.max() - ff.min())
    state = np.where(x >= h, 1, np.where(x <= -h, -1, 0))
    nz = np.flatnonzero(state)
    if nz.size == 0:
        return np.nan, np.array([], dtype=int)
    filled = state[nz]
    trans = nz[1:][(filled[1:] == 1) & (filled[:-1] == -1)]
    if trans.size >= 2:
        fm = (trans.size - 1) / (tu[trans[-1]] - tu[trans[0]])
    else:
        # Fewer than 2 full cycles: fall back to the FFT peak
        n = 1 << int(np.ceil(np.log2(ff.size * 8)))
        spec = np.abs(np.fft.rfft((ff - ff.mean()) * np.hanning(ff.size), n))
        freqs = np.fft.rfftfreq(n, tu[1] - tu[0])
        fm = freqs[1 + np.argmax(spec[1:])]
    return fm, trans


def max_slew_ppm_per_us(tu, ff, f_ref, window=0.5e-6):
    """Largest |df/dt| over the given window, in ppm/us of f_ref."""
    n = max(1, int(round(window / (tu[1] - tu[0]))))
    if n >= ff.size:
        return np.nan
    slope = (ff[n:] - ff[:-n]) / (tu[n:] - tu[:-n])
    return np.max(np.abs(slope)) / f_ref * 1e6 * 1e-6


def triangle(x):
    """Unit triangle wave: 0 at integer x, 1 at half-integer x."""
    return 1.0 - np.abs(2.0 * np.mod(x, 1.0) - 1.0)


def fit_triangle(t_f, f_raw, f_nom, fm0, t_peak0, hi0, lo0):
    """Least-squares fit of a down-going triangle to the raw per-cycle frequency.

    Model: f(t) = f_hi - (f_hi - f_lo) * triangle(fm * (t - t_peak)).
    Fitting the unfiltered data avoids the corner rounding a low-pass introduces.
    Returns (f_hi, f_lo, fm, rms_residual_ppm).
    """
    tt = (t_f - t_f[0]) * 1e6                      # us
    y = (f_raw / f_nom - 1.0) * 1e6                # ppm

    def resid(p):
        hi, span, fm_mhz, tp = p
        return y - (hi - span * triangle(fm_mhz * (tt - tp)))

    p0 = [(hi0 / f_nom - 1) * 1e6, (hi0 - lo0) / f_nom * 1e6, fm0 * 1e-6, (t_peak0 - t_f[0]) * 1e6]
    sol = optimize.least_squares(resid, p0, x_scale=[100.0, 100.0, 1e-3, 1.0])
    hi, span, fm_mhz, _ = sol.x
    f_hi = f_nom * (1 + hi * 1e-6)
    f_lo = f_nom * (1 + (hi - span) * 1e-6)
    return f_hi, f_lo, fm_mhz * 1e6, np.std(sol.fun)


def analyze(t, v, f_nom=100e6, lpf=2e6, lpf_order=2, threshold=0.0, hysteresis=None,
            min_ssc_ppm=300.0):
    edges = rising_edges(t, v, threshold, hysteresis)
    if edges.size < 10:
        raise ValueError("too few rising edges found; check threshold / hysteresis")
    t_f, f_raw = frequency_vs_time(edges)
    tu, ff = lowpass_profile(t_f, f_raw, lpf, lpf_order)
    fm, trans = modulation_cycles(tu, ff)

    if trans.size >= 2:
        # Average the max/min of each complete SSC cycle (reduces residual-noise bias)
        maxima = np.array([ff[a:b].max() for a, b in zip(trans[:-1], trans[1:])])
        minima = np.array([ff[a:b].min() for a, b in zip(trans[:-1], trans[1:])])
        n_cycles = maxima.size
    else:
        maxima, minima = np.array([ff.max()]), np.array([ff.min()])
        n_cycles = 0

    f_max, f_min = maxima.mean(), minima.mean()
    if n_cycles:
        f_avg = ff[trans[0]:trans[-1]].mean()      # whole cycles only, unbiased by partial cycle
    else:
        f_avg = ff.mean()
    inside = (t_f >= tu[0]) & (t_f <= tu[-1])
    residual = f_raw[inside] - np.interp(t_f[inside], tu, ff)

    ssc_detected = (ff.max() - ff.min()) / f_nom * 1e6 >= min_ssc_ppm
    if not ssc_detected:
        fm, trans, n_cycles = np.nan, np.array([], dtype=int), 0

    fit = None
    if ssc_detected and np.isfinite(fm):
        t_peak = tu[np.argmax(ff[:max(2, int(round(1.0 / fm / (tu[1] - tu[0]))))])]
        try:
            fit = fit_triangle(t_f[inside], f_raw[inside], f_nom, fm, t_peak, f_max, f_min)
        except (ValueError, np.linalg.LinAlgError):
            fit = None

    return {
        "ssc_detected": ssc_detected,
        "fit": fit,
        "n_edges": edges.size,
        "n_ssc_cycles": n_cycles,
        "ssc_freq_hz": fm,
        "f_max_hz": f_max,
        "f_min_hz": f_min,
        "f_avg_hz": f_avg,
        "spread_ppm": (f_max - f_min) / f_max * 1e6,
        "spread_pct": (f_max - f_min) / f_max * 100,
        "f_max_vs_nom_ppm": (f_max - f_nom) / f_nom * 1e6,
        "f_min_vs_nom_ppm": (f_min - f_nom) / f_nom * 1e6,
        "f_avg_vs_nom_ppm": (f_avg - f_nom) / f_nom * 1e6,
        "spread_cycle_std_ppm": np.std((maxima - minima) / maxima * 1e6) if n_cycles > 1 else np.nan,
        "max_slew_ppm_per_us": max_slew_ppm_per_us(tu, ff, f_nom),
        "raw_freq_std_ppm": np.std(residual) / f_nom * 1e6,
        "_t_f": t_f, "_f_raw": f_raw, "_tu": tu, "_ff": ff, "_trans": trans,
    }


def report(r, f_nom):
    print(f"Rising edges found       : {r['n_edges']}")
    if not r["ssc_detected"]:
        print("SSC                      : NOT detected (profile p-p below --min-ssc-ppm)")
    print(f"Complete SSC cycles      : {r['n_ssc_cycles']}")
    print(f"SSC modulation frequency : {r['ssc_freq_hz'] / 1e3:.3f} kHz")
    print(f"f_max                    : {r['f_max_hz'] / 1e6:.6f} MHz ({r['f_max_vs_nom_ppm']:+.0f} ppm vs {f_nom / 1e6:g} MHz)")
    print(f"f_min                    : {r['f_min_hz'] / 1e6:.6f} MHz ({r['f_min_vs_nom_ppm']:+.0f} ppm vs {f_nom / 1e6:g} MHz)")
    print(f"f_avg                    : {r['f_avg_hz'] / 1e6:.6f} MHz ({r['f_avg_vs_nom_ppm']:+.0f} ppm vs {f_nom / 1e6:g} MHz)")
    print(f"SSC deviation (p-p)      : {r['spread_ppm']:.0f} ppm ({r['spread_pct']:.4f} %)")
    if np.isfinite(r["spread_cycle_std_ppm"]):
        print(f"  cycle-to-cycle std     : {r['spread_cycle_std_ppm']:.1f} ppm")
    print(f"Max |df/dt| (0.5 us)     : {r['max_slew_ppm_per_us']:.0f} ppm/us")
    print(f"Raw per-cycle freq noise : {r['raw_freq_std_ppm']:.0f} ppm rms (removed by the low-pass)")
    if r["fit"] is not None:
        f_hi, f_lo, fm, rms = r["fit"]
        print("Triangle fit (raw data, no filter):")
        print(f"  f_max / f_min          : {(f_hi - f_nom) / f_nom * 1e6:+.0f} / {(f_lo - f_nom) / f_nom * 1e6:+.0f} ppm vs nominal")
        print(f"  SSC deviation (p-p)    : {(f_hi - f_lo) / f_hi * 1e6:.0f} ppm ({(f_hi - f_lo) / f_hi * 100:.4f} %)")
        print(f"  modulation frequency   : {fm / 1e3:.3f} kHz")
        print(f"  fit residual           : {rms:.0f} ppm rms")
        if rms > 1.1 * r["raw_freq_std_ppm"]:
            print("  NOTE: residual >> raw noise, profile is not triangular; use the filtered result above.")
    if r["ssc_detected"] and r["n_ssc_cycles"] < 2:
        print("WARNING: fewer than 2 complete SSC cycles captured; result uses global max/min "
              "and is less reliable. Capture >= 3 SSC periods (~100 us).", file=sys.stderr)


def plot(r, f_nom, out=None):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7))
    to_ppm = lambda f: (f - f_nom) / f_nom * 1e6
    a1.plot(r["_t_f"] * 1e6, to_ppm(r["_f_raw"]), ".", ms=1, alpha=0.3, label="per-cycle (raw)")
    a1.plot(r["_tu"] * 1e6, to_ppm(r["_ff"]), "r", lw=1.5, label="low-pass filtered")
    a1.axhline(r["f_max_vs_nom_ppm"], color="g", ls="--", label="mean cycle max")
    a1.axhline(r["f_min_vs_nom_ppm"], color="m", ls="--", label="mean cycle min")
    for i in r["_trans"]:
        a1.axvline(r["_tu"][i] * 1e6, color="k", lw=0.5, alpha=0.3)
    a1.set_xlabel("time [us]")
    a1.set_ylabel(f"frequency offset vs {f_nom / 1e6:g} MHz [ppm]")
    a1.set_title(f"SSC deviation = {r['spread_ppm']:.0f} ppm, fm = {r['ssc_freq_hz'] / 1e3:.2f} kHz")
    a1.legend(loc="lower right", fontsize=8)
    a1.grid(True, alpha=0.3)

    tu, ff = r["_tu"], r["_ff"]
    freqs, psd = signal.welch(ff - ff.mean(), fs=1.0 / (tu[1] - tu[0]), nperseg=min(ff.size, 1 << 16))
    a2.semilogy(freqs / 1e3, psd)
    a2.set_xlim(0, 10 * r["ssc_freq_hz"] / 1e3 if np.isfinite(r["ssc_freq_hz"]) else 500)
    a2.set_xlabel("frequency [kHz]")
    a2.set_ylabel("PSD of filtered profile")
    a2.grid(True, alpha=0.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        print(f"Plot saved to {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="CSV file: time [s], differential voltage [V]")
    p.add_argument("--fnom", type=float, default=100e6, help="nominal frequency [Hz] (default 100e6)")
    p.add_argument("--lpf", type=float, default=2e6, help="low-pass cutoff for the frequency profile [Hz] (default 2e6, 0 = off)")
    p.add_argument("--lpf-order", type=int, default=2, help="Butterworth order (default 2)")
    p.add_argument("--threshold", type=float, default=0.0, help="crossing threshold [V] (default 0)")
    p.add_argument("--hysteresis", type=float, default=None, help="hysteresis [V] (default 20%% of swing)")
    p.add_argument("--min-ssc-ppm", type=float, default=300.0, help="minimum p-p deviation to declare SSC present [ppm]")
    p.add_argument("--plot", action="store_true", help="show plot")
    p.add_argument("--plot-file", default=None, help="save plot to this file instead of showing it")
    a = p.parse_args()

    t, v = load_waveform(a.csv)
    r = analyze(t, v, a.fnom, a.lpf, a.lpf_order, a.threshold, a.hysteresis, a.min_ssc_ppm)
    report(r, a.fnom)
    if a.plot or a.plot_file:
        plot(r, a.fnom, a.plot_file)


if __name__ == "__main__":
    main()
