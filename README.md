# PCIe REFCLK SSC deviation measurement

Tools to measure the **Spread-Spectrum Clocking (SSC) deviation** of a PCIe 100 MHz reference clock from an oscilloscope capture, and to generate synthetic REFCLK waveforms with configurable SSC and jitter for testing.

| File | Purpose |
|---|---|
| `ssc_analysis.py` | Reads a CSV capture (time, differential voltage) and reports SSC deviation, modulation frequency, f_max / f_min / f_avg, df/dt, and a triangle fit |
| `generate_waveform.py` | Creates a synthetic differential REFCLK CSV with SSC (down/center/up, triangle/sine), random jitter, periodic jitter, vertical noise and finite rise time |

## Quick start

```powershell
pip install -r requirements.txt

# 1. Generate a test waveform: 0.5 % down-spread, 31.5 kHz, 1 ps rms RJ, 100 us @ 20 GSa/s
python generate_waveform.py refclk.csv --ssc-ppm 5000 --ssc-freq 31.5e3 --rj-ps 1 --seed 1

# 2. Measure it
python ssc_analysis.py refclk.csv --plot
```

Example output:

```
Rising edges found       : 9975
Complete SSC cycles      : 2
SSC modulation frequency : 31.503 kHz
f_max                    : 99.998291 MHz (-17 ppm vs 100 MHz)
f_min                    : 99.501721 MHz (-4983 ppm vs 100 MHz)
f_avg                    : 99.749998 MHz (-2500 ppm vs 100 MHz)
SSC deviation (p-p)      : 4966 ppm (0.4966 %)
  cycle-to-cycle std     : 2.4 ppm
Max |df/dt| (0.5 us)     : 334 ppm/us
Raw per-cycle freq noise : 144 ppm rms (removed by the low-pass)
Triangle fit (raw data, no filter):
  f_max / f_min          : +0 / -5000 ppm vs nominal
  SSC deviation (p-p)    : 5000 ppm (0.5000 %)
  modulation frequency   : 31.500 kHz
  fit residual           : 144 ppm rms
```

## Input format

A CSV with **column 1 = time [s]** and **column 2 = differential voltage [V]** (`V(REFCLK+) - V(REFCLK-)`). Header rows and non-numeric lines are skipped. Extra columns are ignored. If you captured the two legs single-ended, subtract them before you save the file.

## Theory

### What SSC is

PCIe REFCLK is usually spread-spectrum modulated to lower peak EMI. The clock frequency is slowly swept (typically with a triangular profile) between a maximum and a minimum value. This spreads the energy of each harmonic over a band instead of concentrating it in a single line. PCIe uses **down-spread**: the frequency is swept from the nominal 100 MHz *downward*.

Commonly cited PCIe REFCLK SSC limits (check the PCIe Base / CEM spec revision you are testing against):

| Parameter | Typical limit |
|---|---|
| Nominal frequency | 100 MHz ± 300 ppm |
| SSC type | down-spread |
| SSC deviation | 0 to −0.5 % (≤ 5000 ppm p-p) |
| SSC modulation frequency | 30 – 33 kHz |
| SSC df/dt | ≤ 1250 ppm/µs (as commonly specified) |

### Definition of SSC deviation

With $f_{max}$ and $f_{min}$ the extremes of the modulation profile:

$$
\delta_{SSC} = \frac{f_{max} - f_{min}}{f_{max}} \qquad [\text{ppm or \%}]
$$

For an ideal down-spread $f_{max} = f_{nom}$, so $f_{min} = f_{nom}(1-\delta)$. For a triangular profile the average frequency is $f_{avg} = f_{max}(1-\delta/2)$. For example, 0.5 % down-spread gives an average that is −2500 ppm from nominal.

The tool reports the deviation referenced to $f_{max}$ (the SSC definition). It also reports $f_{max}$, $f_{min}$ and $f_{avg}$ in ppm against the nominal frequency, so you can check the ±300 ppm tolerance separately.

### Measuring instantaneous frequency

The SSC rate (~30 kHz) is about 3000× slower than the clock, so the frequency is effectively constant over one clock cycle. Each clock period is therefore a valid sample of the instantaneous frequency:

$$
f_n = \frac{1}{t_{n+1} - t_n}, \qquad \text{placed at } \tfrac{1}{2}(t_n + t_{n+1})
$$

where $t_n$ are the **rising 0 V crossings of the differential signal**. Only rising edges are used, so duty-cycle distortion does not affect the result.

### Separating jitter from SSC

Each period measurement contains period jitter. With uncorrelated random edge jitter $\sigma_{RJ}$:

$$
\frac{\sigma_f}{f} \approx \frac{\sqrt{2}\,\sigma_{RJ}}{T_0} = \sqrt{2}\,\sigma_{RJ}\,f_0
$$

At 100 MHz, 1 ps rms RJ gives ≈ 141 ppm rms of noise per cycle. That is large compared with the accuracy you want on a 5000 ppm deviation. SSC energy is concentrated below a few MHz, while period jitter is broadband and rises with frequency. A **low-pass filter** on the frequency-vs-time profile therefore removes the jitter and keeps the SSC.

Two complementary estimators are used:

1. **Filtered profile + per-cycle extremes (primary, shape-independent)**
   * The period sequence is resampled onto a uniform grid (step = median period).
   * A zero-phase Butterworth low-pass is applied (`--lpf`, default 2 MHz, 2nd order, `sosfiltfilt`). Filter start-up transients are trimmed.
   * Complete SSC cycles are located from the rising mid-level crossings of the filtered profile (with hysteresis). The modulation frequency is `(N-1)/(t_last - t_first)` over those crossings.
   * The max and min of each complete cycle are averaged to get $f_{max}$ and $f_{min}$. Averaging reduces the bias that residual noise puts on a single global max/min.
   * Limitation: any low-pass also rounds the sharp corners of a triangle. A triangle's Fourier series has only odd harmonics with amplitude ∝ 1/n². Removing the harmonics above $N = f_c/f_m$ lowers the peak-to-peak by roughly

     $$\frac{\Delta\delta}{\delta} \approx \frac{8}{\pi^2}\sum_{n\ \text{odd} > N}\frac{1}{n^2} \approx \frac{4}{\pi^2 N}$$

     With $f_c$ = 2 MHz and $f_m$ = 31.5 kHz ($N$ ≈ 63), that is ≈ 0.6 % of the deviation (about 30 ppm out of 5000). A higher cutoff reduces this bias but lets more jitter through.

2. **Triangle least-squares fit (on the raw, unfiltered period data)**
   * Model: $f(t) = f_{hi} - (f_{hi}-f_{lo})\cdot\text{tri}\big(f_m (t - t_p)\big)$. This is solved with `scipy.optimize.least_squares` and initialized from estimator 1.
   * Because thousands of noisy points are fitted, the noise averages out and no filter bias is added. This gives the most accurate result **when the profile really is triangular**.
   * If the fit residual is clearly larger than the raw per-cycle noise, the profile is not triangular (for example sinusoidal or "Hershey-kiss"). The tool prints a note, and you should use estimator 1.

### df/dt (SSC slew)

The tool reports the maximum $|f(t+\Delta) - f(t)|/\Delta$ of the filtered profile over a 0.5 µs window, in ppm/µs. An ideal triangle has slope $2\,\delta\,f_m$, which is ≈ 315 ppm/µs for 0.5 % at 31.5 kHz. Residual jitter adds a little to this value.

### SSC detection

If the peak-to-peak of the filtered profile is below `--min-ssc-ppm` (default 300 ppm), the tool reports **SSC not detected**. The numbers it prints then represent only residual frequency wander.

## Assumptions and requirements

* **Differential signal, 0 V threshold.** Column 2 is the differential voltage. The crossing threshold is `--threshold` (default 0 V). Hysteresis (`--hysteresis`, default 20 % of the 5–95 % swing) rejects noise-induced double crossings.
* **Crossing interpolation.** Crossings are linearly interpolated between samples. This assumes the edge is nearly linear around 0 V and that there are at least ~3–5 samples on each edge (for example ≥ 10–20 GSa/s for a 100–300 ps edge). The timing error per crossing is about σ_v / slew rate.
* **Capture length.** At least **2 complete SSC cycles** are needed for the per-cycle estimator, so capture ≥ ~100 µs (≥ 3 periods at 30 kHz). With shorter captures the tool falls back to global max/min and prints a warning. The triangle fit still works reasonably well with about 1 cycle.
* **Memory/size.** 100 µs at 20 GSa/s is 2 M samples (tens of MB of CSV). Store time with ≥ 12 significant digits, otherwise the CSV rounding itself adds timing jitter.
* **SSC is slow compared with the clock** ($f_m \ll f_0$), so one period equals one instantaneous frequency sample.
* **Jitter is zero-mean and mostly above the low-pass cutoff.** Jitter or wander below `--lpf` cannot be separated from SSC and will appear in the profile. Periodic jitter near the SSC frequency will bias the result.
* **Timebase accuracy.** A scope timebase error of *x* ppm shifts all absolute frequencies by *x* ppm. The **deviation (a ratio) is essentially unaffected**, but the ±300 ppm nominal check is directly affected. Use a calibrated or reference-locked scope for absolute frequency.
* **Down-spread reference.** The ppm-vs-nominal values use `--fnom` (default 100 MHz).

## Analyzer options

```
python ssc_analysis.py CSV [--fnom 100e6] [--lpf 2e6] [--lpf-order 2]
                           [--threshold 0] [--hysteresis V] [--min-ssc-ppm 300]
                           [--plot] [--plot-file out.png]
```

The plot shows the raw per-cycle frequency, the filtered profile, the averaged max/min, the detected cycle boundaries, and the PSD of the profile.

You can also use it from Python:

```python
from ssc_analysis import load_waveform, analyze
t, v = load_waveform("refclk.csv")
r = analyze(t, v, f_nom=100e6, lpf=2e6)
print(r["spread_ppm"], r["ssc_freq_hz"], r["fit"])
```

## Waveform generator

### Model

1. The instantaneous frequency is $f(t) = f_{nom}(1 + \text{offset})(1 + d(t))$, where $d(t)$ is the SSC profile:
   * down: $-\delta\,u(t)$, center: $\delta(\tfrac12 - u(t))$, up: $+\delta\,u(t)$
   * $u(t)$ is a 0→1→0 triangle or raised-cosine (sine) at $f_m$, with a random start phase.
2. Phase is integrated numerically (1 ns grid): $\phi(t) = \int f\,dt$. Ideal edges are placed where $\phi$ crosses multiples of ½ cycle (rising on integers, falling on half-integers).
3. Jitter is added as a time displacement (TIE) of each edge:
   * RJ: independent Gaussian, σ = `--rj-ps`
   * PJ: $A\sin(2\pi f_{pj} t + \theta)$, peak = `--pj-ps`, frequency = `--pj-freq`
4. The voltage is built from the edges with erf-shaped transitions. The 20–80 % rise time is `--rise-time`, and the level is ± `--amplitude` (differential peak). Gaussian vertical noise (`--noise-mv`) is then added.

### Options

```
python generate_waveform.py OUT.csv
    [--fnom 100e6] [--ssc-ppm 5000] [--ssc-freq 31.5e3]
    [--spread down|center|up] [--shape triangle|sine] [--offset-ppm 0]
    [--rj-ps 1] [--pj-ps 0] [--pj-freq 1e6]
    [--duration 100e-6] [--fs 20e9] [--amplitude 0.7]
    [--rise-time 150e-12] [--noise-mv 1] [--seed N]
```

### Validation results (synthetic data)

| Generated | Filtered estimate | Triangle fit |
|---|---|---|
| 5000 ppm down, triangle, 31.5 kHz, RJ 1 ps | 4966 ppm, 31.503 kHz | 5000 ppm, 31.500 kHz |
| 3000 ppm down, 33 kHz, +200 ppm offset, RJ 5 ps, PJ 10 ps @ 2 MHz, 5 mV noise | 3093 ppm, 33.06 kHz | 3001 ppm, +201 ppm f_max, 32.999 kHz |
| 5000 ppm center, sine | 4990 ppm (f_max/f_min ±2501 ppm) | not applicable (non-triangular, residual flagged) |
| no SSC, RJ 2 ps | SSC not detected (8 ppm wander) | – |
| 5000 ppm, only 25 µs captured | 4962 ppm + warning | 5000 ppm, 31.502 kHz |
