"""Generate a synthetic PCIe REFCLK differential waveform with SSC and jitter.

Output is a CSV with two columns: time [s], differential voltage [V].

Example:
    python generate_waveform.py refclk.csv --ssc-ppm 5000 --ssc-freq 31.5e3 \
        --rj-ps 2 --duration 100e-6 --fs 20e9
"""
import argparse

import numpy as np
from scipy.special import erf, erfinv


def frequency_profile(t, f_nom, ssc_ppm, ssc_freq, spread="down", shape="triangle",
                      offset_ppm=0.0, ssc_phase=0.0):
    """Instantaneous clock frequency [Hz] at times t."""
    delta = ssc_ppm * 1e-6
    x = np.mod(ssc_freq * t + ssc_phase, 1.0)
    if shape == "triangle":
        u = 1.0 - np.abs(2.0 * x - 1.0)          # 0 -> 1 -> 0 over one SSC period
    elif shape == "sine":
        u = 0.5 * (1.0 - np.cos(2.0 * np.pi * x))
    else:
        raise ValueError(f"unknown SSC shape: {shape}")

    if spread == "down":
        dev = -delta * u
    elif spread == "center":
        dev = delta * (0.5 - u)
    elif spread == "up":
        dev = delta * u
    else:
        raise ValueError(f"unknown spread type: {spread}")

    return f_nom * (1.0 + offset_ppm * 1e-6) * (1.0 + dev)


def edge_times(t_start, t_stop, freq_func, rng, rj_rms=0.0, pj_amp=0.0, pj_freq=1e6,
               grid_step=1e-9):
    """Return (edge_times, is_rising) for the clock between t_start and t_stop.

    Ideal edges are placed where the accumulated phase crosses a multiple of
    half a cycle; jitter is then added as a time displacement (TIE) per edge.
    """
    tg = np.arange(t_start, t_stop + grid_step, grid_step)
    f = freq_func(tg)
    cycles = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * grid_step)))
    cycles += rng.uniform(0.0, 1.0)

    k = np.arange(np.ceil(2.0 * cycles[0]), np.floor(2.0 * cycles[-1]) + 1)
    edges = np.interp(k / 2.0, cycles, tg)

    if rj_rms > 0:
        edges = edges + rng.normal(0.0, rj_rms, edges.size)
    if pj_amp > 0:
        edges = edges + pj_amp * np.sin(2.0 * np.pi * pj_freq * edges + rng.uniform(0, 2 * np.pi))

    return edges, (k % 2 == 0)


def synthesize(t, edges, rising, amplitude, rise_time):
    """Build the differential voltage from edge times using erf-shaped transitions."""
    tau = rise_time / (2.0 * erfinv(0.6))    # rise_time is the 20-80 % time
    idx = np.clip(np.searchsorted(edges, t), 1, edges.size - 1)
    level = np.where(rising[idx - 1], 1.0, -1.0)
    dist = np.minimum(t - edges[idx - 1], edges[idx] - t)
    return amplitude * level * erf(dist / tau)


def generate(f_nom=100e6, ssc_ppm=5000.0, ssc_freq=31.5e3, spread="down", shape="triangle",
             offset_ppm=0.0, rj_rms=1e-12, pj_amp=0.0, pj_freq=1e6, duration=100e-6, fs=20e9,
             amplitude=0.7, rise_time=150e-12, noise_rms=1e-3, seed=None):
    """Return (t, v) arrays for a differential REFCLK waveform."""
    rng = np.random.default_rng(seed)
    ssc_phase = rng.uniform(0.0, 1.0)

    def freq_func(tt):
        return frequency_profile(tt, f_nom, ssc_ppm, ssc_freq, spread, shape, offset_ppm, ssc_phase)

    margin = 10.0 / f_nom
    edges, rising = edge_times(-margin, duration + margin, freq_func, rng, rj_rms, pj_amp, pj_freq)

    t = np.arange(int(round(duration * fs))) / fs
    v = synthesize(t, edges, rising, amplitude, rise_time)
    if noise_rms > 0:
        v = v + rng.normal(0.0, noise_rms, t.size)
    return t, v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", help="output CSV file")
    p.add_argument("--fnom", type=float, default=100e6, help="nominal frequency [Hz] (default 100e6)")
    p.add_argument("--ssc-ppm", type=float, default=5000.0, help="SSC peak-to-peak deviation [ppm] (default 5000 = 0.5%%)")
    p.add_argument("--ssc-freq", type=float, default=31.5e3, help="SSC modulation frequency [Hz] (default 31.5e3)")
    p.add_argument("--spread", choices=["down", "center", "up"], default="down")
    p.add_argument("--shape", choices=["triangle", "sine"], default="triangle")
    p.add_argument("--offset-ppm", type=float, default=0.0, help="static frequency offset [ppm]")
    p.add_argument("--rj-ps", type=float, default=1.0, help="random jitter, rms per edge [ps]")
    p.add_argument("--pj-ps", type=float, default=0.0, help="periodic (sinusoidal) jitter, peak [ps]")
    p.add_argument("--pj-freq", type=float, default=1e6, help="periodic jitter frequency [Hz]")
    p.add_argument("--duration", type=float, default=100e-6, help="capture length [s] (default 100e-6)")
    p.add_argument("--fs", type=float, default=20e9, help="sample rate [Sa/s] (default 20e9)")
    p.add_argument("--amplitude", type=float, default=0.7, help="differential peak amplitude [V]")
    p.add_argument("--rise-time", type=float, default=150e-12, help="20-80%% rise time [s]")
    p.add_argument("--noise-mv", type=float, default=1.0, help="vertical noise, rms [mV]")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    a = p.parse_args()

    t, v = generate(f_nom=a.fnom, ssc_ppm=a.ssc_ppm, ssc_freq=a.ssc_freq, spread=a.spread,
                    shape=a.shape, offset_ppm=a.offset_ppm, rj_rms=a.rj_ps * 1e-12,
                    pj_amp=a.pj_ps * 1e-12, pj_freq=a.pj_freq, duration=a.duration, fs=a.fs,
                    amplitude=a.amplitude, rise_time=a.rise_time, noise_rms=a.noise_mv * 1e-3,
                    seed=a.seed)

    # time needs ~12 significant digits to keep sub-ps resolution over 100s of us
    np.savetxt(a.output, np.column_stack((t, v)), delimiter=",", fmt=["%.12e", "%.6e"],
               header="time_s,vdiff_V", comments="")
    print(f"Wrote {t.size} samples ({a.duration * 1e6:.1f} us @ {a.fs / 1e9:.1f} GSa/s) to {a.output}")
    print(f"SSC: {a.spread}-spread {a.shape}, {a.ssc_ppm:.0f} ppm p-p, {a.ssc_freq / 1e3:.2f} kHz; "
          f"RJ {a.rj_ps} ps rms, PJ {a.pj_ps} ps pk @ {a.pj_freq / 1e6:.3f} MHz")


if __name__ == "__main__":
    main()
