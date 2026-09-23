#!/usr/bin/env python3
"""
simulate_rfi_chime.py
=======================

Simulates the negative ("not an FRB") class for a CHIME FRB-vs-noise CNN:
pure noise / RFI / instrumental / astrophysical-confusion images with NO
burst signal present. This is the companion file to simulate_frbs_fitburst.py.

CRITICAL DESIGN POINT: the noise-generation code below (every function from
build_bandpass_gain() through radiometer_noise(), plus choose_grid()) is
copied VERBATIM from simulate_frbs_fitburst.py. This is deliberate, not
laziness -- if the two classes' noise statistics differ even slightly (a
different bandpass model, different masking rates, a different window-size
distribution), a CNN will learn to separate "which generator produced this
image" instead of "is there a real dispersed burst here", since that
shortcut is far easier to find than actual burst physics. If you ever edit
the noise code in simulate_frbs_fitburst.py, port the same edit here (or
switch both files to import from one shared module) to keep them in sync.

The window-size distribution is matched the same way: a real catalog row's
(width_fitb, scat_time, dm_fitb) still gets sampled per image and fed
through the identical choose_grid() function -- even though no burst is
ever placed -- purely so the negative class's image dimensions follow the
same distribution as the positive class. Otherwise "window is unusually
wide/narrow" would itself become a shortcut feature.

Two probabilities are deliberately DIFFERENT from the FRB file:
RFI_STORM_PROB and PULSAR_PROB are raised substantially here, because in
the FRB file those represent a rare coincidental second signal alongside a
real burst, whereas here they're primary content -- the actual hard-
negative cases (a fake dispersion sweep, a periodic dispersed pulsar train)
that a classifier most needs practice telling apart from a genuine burst.

No fitburst dependency -- this file only needs numpy, pandas, matplotlib.

Folder setup
------------
    project/
    |-- chimefrbcat2.csv
    |-- simulate_frbs_fitburst.py
    |-- simulate_rfi_chime.py        <- this file
    `-- sim_output_rfi/                <- created automatically on first run

Usage
-----
    python3 simulate_rfi_chime.py --catalog chimefrbcat2.csv --n 200 --outdir sim_output_rfi --plot 16
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from sim_common import (
    FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ, MIN_NUM_TIME, MAX_NUM_TIME, NATIVE_TSAMP_S, K_DM,
    REF_FREQ_MHZ, NOISE_SIGMA, BANDPASS_RIPPLE_PERIOD_MHZ, BANDPASS_ROLLOFF_FRAC,
    BANDPASS_RIPPLE_FRAC, BANDPASS_JITTER_FRAC, NOISE_DOF_NPOL, MIN_WINDOW_NATIVE_SAMPLES,
    DISPLAY_PAD_FACTOR, MAX_WINDOW_S, BASELINE_OFFSET_STD, RFI_BANDS_MHZ, RFI_BAND_PROB,
    build_bandpass_gain, build_baseline_offset, build_rfi_mask, radiometer_noise, choose_grid,
    dispersion_delay_ms, verify_matched_window_distributions,
)

# NOTE (fixes error #1 from the shortcut-learning review): every constant
# and function above used to be a hand-copied duplicate of the same code in
# simulate_frbs_fitburst.py, with a comment warning you to keep them in
# sync manually. That's a real drift risk -- both files now import from
# sim_common.py instead, so there is no copy to forget to update.

# --- comprehensive noise/RFI/confusion-source knobs (new in this revision) ---
# Real narrowband RFI isn't confined to the 3 fixed slots above -- always
# masking the exact same 3 positions is itself a shortcut-learning risk.
# These are used as WEIGHTED CENTERS for randomized-position bands instead.
RFI_BAND_CENTERS_MHZ = [550.0, 625.0, 742.5]
AIRCRAFT_PROB = 0.10        # persistent-for-the-whole-window RFI with radar-PRF pulsing + Doppler drift
IMPULSIVE_RFI_PROB = 0.25   # short broadband hits (satellites, cell phones, local electronics)
PERIODIC_RFI_PROB = 0.10    # switching power supplies, radar side-lobes
RFI_STORM_PROB = 0.18       # patchy fake dispersion sweep at random DM -- the paper's own
                             # unsolved "RFI storm" false-positive class. Raised relative to the
                             # FRB file (0.06 there) since here it's primary hard-negative content,
                             # not a rare distractor alongside an actual burst.
PULSAR_PROB = 0.15           # a real Galactic pulsar caught via sidelobes -- raised relative to
                             # the FRB file since this is now primary hard-negative content, not a
                             # rare distractor alongside an actual burst
SCINTILLATION_PROB = 0.12
QUANTIZATION_PROB = 0.3
TIME_DROPOUT_PROB = 0.2
GAIN_DRIFT_SMOOTH_PROB = 0.35
GAIN_JUMP_PROB = 0.12
MAX_SEARCH_DM = 13000.0     # CHIME/FRB's incoherent-dedispersion search ceiling
PURE_NOISE_PROB = 0.08      # a fraction of images should be plain noise with essentially nothing
                             # else layered on, matching real quiet stretches of data


def add_randomized_narrowband_rfi(freqs_mhz: np.ndarray, rng: np.random.Generator,
                                   noise_scale_freq: np.ndarray, zero_mask: np.ndarray, applied: list,
                                   n_bands_range=(0, 2)) -> None:
    """EXTRA narrowband RFI bands beyond the 3 fixed documented slots
    (build_rfi_mask), at randomized positions -- so masking/elevated-noise
    bands show up in variable locations, not always the same 3 spots.
    70% of the time centered near a documented real band (with jitter),
    30% of the time fully random anywhere in the observed band."""
    n_bands = int(rng.integers(*n_bands_range))
    for _ in range(n_bands):
        if rng.random() < 0.7:
            center = rng.choice(RFI_BAND_CENTERS_MHZ) + rng.uniform(-15, 15)
        else:
            center = rng.uniform(freqs_mhz.min() + 10, freqs_mhz.max() - 10)
        width = rng.uniform(8.0, 40.0)
        lo, hi = center - width / 2, center + width / 2
        sel = (freqs_mhz >= lo) & (freqs_mhz <= hi)
        if not sel.any():
            continue
        if rng.random() < 0.4:
            zero_mask[sel] = True
            applied.append(f"narrowband_zero:{center:.0f}MHz")
        else:
            noise_scale_freq[sel] *= rng.uniform(4.0, 20.0)
            applied.append(f"narrowband_bright:{center:.0f}MHz")


def add_aircraft_rfi(freqs_mhz: np.ndarray, rng: np.random.Generator, extra_signal: np.ndarray,
                      sigma_base: float, time_axis_ms: np.ndarray, applied: list) -> None:
    """Aircraft-triggered RFI (radar/transponder reflections). Distinct
    signature from a burst or brief impulsive RFI: PERSISTENT across the
    ENTIRE visible window (a real transit lasts many seconds, far longer
    than any candidate crop), slow Doppler-like frequency drift, and often
    fast periodic structure from the radar's pulse-repetition frequency.
    The RFI paper notes aircraft cause thousands of subthreshold false
    positives per day."""
    if rng.random() >= AIRCRAFT_PROB:
        return
    f0 = rng.uniform(freqs_mhz.min() + 30, freqs_mhz.max() - 30)
    drift_rate = rng.uniform(-0.05, 0.05)  # MHz/ms
    bw = rng.uniform(3.0, 20.0)
    amp = sigma_base * rng.uniform(4.0, 18.0)
    prf_hz = rng.uniform(50.0, 400.0)
    prf_period_ms = 1000.0 / prf_hz
    f_center_t = f0 + drift_rate * time_axis_ms
    envelope = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * time_axis_ms / prf_period_ms))
    profile = amp * envelope
    extra_signal += profile[None, :] * np.exp(-0.5 * ((freqs_mhz[:, None] - f_center_t[None, :]) / bw) ** 2)
    applied.append(f"aircraft_rfi:f0={f0:.0f}MHz,prf={prf_hz:.0f}Hz")


def add_rfi_storm(freqs_mhz: np.ndarray, rng: np.random.Generator, extra_signal: np.ndarray,
                   sigma_base: float, time_axis_ms: np.ndarray, applied: list) -> None:
    """The RFI paper's own unsolved category: intense "RFI storms" that
    preferentially trigger false positives at HIGH trial DM -- i.e. RFI
    that happens to trace a real dispersion-like sweep. A genuine
    dispersion-shaped ridge at a random DM across the full CHIME search
    range, but with PATCHY/inconsistent amplitude along frequency (unlike
    a real burst's smooth spectral envelope) -- that inconsistency is the
    discriminating feature a classifier should learn to use, rather than
    "is there a sweep at all"."""
    if rng.random() >= RFI_STORM_PROB:
        return
    fake_dm = float(np.exp(rng.uniform(np.log(20.0), np.log(MAX_SEARCH_DM))))
    f_ref = freqs_mhz.max()
    delay_ms = dispersion_delay_ms(freqs_mhz, fake_dm, f_ref)
    width_ms = rng.uniform(NATIVE_TSAMP_S * 1e3, 3.0 * NATIVE_TSAMP_S * 1e3)
    amp = sigma_base * rng.uniform(5.0, 30.0)

    n_patches = int(rng.integers(3, 10))
    patch_centers = rng.uniform(freqs_mhz.min(), freqs_mhz.max(), size=n_patches)
    patch_width = rng.uniform(5.0, 40.0)
    chan_amp = np.zeros_like(freqs_mhz)
    for c in patch_centers:
        chan_amp += np.exp(-0.5 * ((freqs_mhz - c) / patch_width) ** 2)
    chan_amp = chan_amp / max(chan_amp.max(), 1e-9)
    if rng.random() < 0.5:
        chan_amp *= (rng.random(len(freqs_mhz)) > 0.3)

    for i in range(len(freqs_mhz)):
        if chan_amp[i] <= 0:
            continue
        profile = np.exp(-0.5 * ((time_axis_ms - delay_ms[i]) / width_ms) ** 2)
        extra_signal[i, :] += amp * chan_amp[i] * profile
    applied.append(f"rfi_storm:dm={fake_dm:.0f}")


def add_impulsive_rfi(rng: np.random.Generator, extra_signal: np.ndarray, sigma_base: float,
                       time_axis_ms: np.ndarray, applied: list) -> None:
    """Short broadband impulsive hits (satellites, cell phones, local
    electronics) -- undispersed, the classic case that must be told apart
    from a burst by the ABSENCE of a frequency-time sweep."""
    if rng.random() >= IMPULSIVE_RFI_PROB:
        return
    nchan = extra_signal.shape[0]
    n_events = int(rng.integers(1, 4))
    for _ in range(n_events):
        t0 = rng.uniform(time_axis_ms[0], time_axis_ms[-1])
        width_ms = rng.uniform(NATIVE_TSAMP_S * 1e3 * 0.5, NATIVE_TSAMP_S * 1e3 * 3.0)
        amp = sigma_base * rng.uniform(6.0, 40.0)
        chan_jitter = rng.uniform(0.4, 1.0, size=nchan)
        profile = np.exp(-0.5 * ((time_axis_ms - t0) / max(width_ms, NATIVE_TSAMP_S * 1e3 / 2)) ** 2)
        extra_signal += amp * np.outer(chan_jitter, profile)
    applied.append(f"impulsive_rfi:{n_events}")


def add_periodic_rfi(freqs_mhz: np.ndarray, rng: np.random.Generator, extra_signal: np.ndarray,
                      sigma_base: float, time_axis_ms: np.ndarray, applied: list) -> None:
    """Quasi-periodic narrowband pulse train (switching power supplies,
    radar side-lobes)."""
    if rng.random() >= PERIODIC_RFI_PROB:
        return
    period_ms = rng.uniform(3.0, 40.0)
    phase_ms = rng.uniform(0.0, period_ms)
    f0 = rng.uniform(freqs_mhz.min() + 20, freqs_mhz.max() - 20)
    bw = rng.uniform(1.0, 6.0)
    amp = sigma_base * rng.uniform(4.0, 25.0)
    pulse_width_ms = rng.uniform(NATIVE_TSAMP_S * 1e3, max(NATIVE_TSAMP_S * 1e3, period_ms * 0.3) + 1e-6)
    freq_gauss = np.exp(-0.5 * ((freqs_mhz - f0) / bw) ** 2)
    phase = np.mod(time_axis_ms - phase_ms, period_ms)
    time_gauss = np.exp(-0.5 * (np.minimum(phase, period_ms - phase) /
                                 max(pulse_width_ms, NATIVE_TSAMP_S * 1e3 / 2)) ** 2)
    extra_signal += amp * np.outer(freq_gauss, time_gauss)
    applied.append(f"periodic_rfi:{period_ms:.1f}ms")


def add_pulsar_periodic_signal(freqs_mhz: np.ndarray, rng: np.random.Generator, extra_signal: np.ndarray,
                                sigma_base: float, time_axis_ms: np.ndarray, applied: list) -> None:
    """A real Galactic pulsar, detected via the instrument's sidelobes (the
    RFI paper specifically notes PSR B0329+54 being visible for hours this
    way). Each individual pulse is dispersed just like a real burst -- the
    only thing distinguishing it WITHIN a single window is genuine
    PERIODICITY (multiple repeating pulses), so a fast-ish (but still
    physically plausible) period is used so several repeats show in the
    window. Sidelobe attenuation means these are usually fainter than a
    well-detected burst."""
    if rng.random() >= PULSAR_PROB:
        return
    period_ms = rng.uniform(8.0, 45.0)
    dm = float(np.exp(rng.uniform(np.log(10.0), np.log(300.0))))
    width_ms = rng.uniform(0.3, 2.0)
    amp = sigma_base * rng.uniform(3.0, 10.0)
    phase0_ms = rng.uniform(0.0, period_ms)

    delay_ms = dispersion_delay_ms(freqs_mhz, dm, freqs_mhz.max())
    pulse_times = np.arange(time_axis_ms[0] - delay_ms.max(), time_axis_ms[-1] + period_ms, period_ms) + phase0_ms
    for pt in pulse_times:
        for i, f in enumerate(freqs_mhz):
            t0 = pt + delay_ms[i]
            if t0 < time_axis_ms[0] - 5 * width_ms or t0 > time_axis_ms[-1] + 5 * width_ms:
                continue
            extra_signal[i, :] += amp * np.exp(-0.5 * ((time_axis_ms - t0) / width_ms) ** 2)
    applied.append(f"pulsar_periodic:P={period_ms:.1f}ms,dm={dm:.0f}")


def add_scintillation(freqs_mhz: np.ndarray, rng: np.random.Generator, time_axis_ms: np.ndarray,
                       applied: list) -> Optional[np.ndarray]:
    """Solar wind / interplanetary or ionospheric scintillation: a slow,
    broadband, frequency-correlated amplitude modulation. Real scintillation
    timescales (seconds to minutes) are much longer than one candidate
    window, so within a single image it mostly shows up as a smooth
    brightness gradient across time plus a mild frequency-dependent
    envelope (stronger at the low-frequency end for ionospheric effects)
    rather than fast structure. Returns a 2D (nchan, nsamp) multiplier, or
    None if not triggered this draw."""
    if rng.random() >= SCINTILLATION_PROB:
        return None
    is_ionospheric = rng.random() < 0.4
    n_knots = int(rng.integers(2, 4))
    knot_vals = rng.normal(1.0, 0.15, size=n_knots)
    time_envelope = np.interp(np.arange(len(time_axis_ms)), np.linspace(0, len(time_axis_ms) - 1, n_knots), knot_vals)
    if is_ionospheric:
        freq_weight = (freqs_mhz / freqs_mhz.max()) ** -1.2
        freq_weight /= freq_weight.max()
        applied.append("scintillation_ionospheric")
    else:
        freq_weight = np.ones_like(freqs_mhz)
        applied.append("scintillation_solar")
    combined = 1.0 + (freq_weight[:, None] - 0.5) * (time_envelope[None, :] - 1.0) * 0.6
    return np.clip(combined, 0.5, 1.8)


def add_gain_drift(rng: np.random.Generator, nsamp: int, applied: list) -> np.ndarray:
    """A discrete gain jump (recalibration boundary) combined with a smooth
    continuous drift, since real gain rarely moves in clean single steps
    only. Returns a per-TIME-sample multiplicative gain."""
    gain = np.ones(nsamp)
    if rng.random() < GAIN_DRIFT_SMOOTH_PROB:
        n_knots = int(rng.integers(2, 5))
        knots = rng.normal(1.0, 0.08, size=n_knots)
        gain = np.interp(np.arange(nsamp), np.linspace(0, nsamp - 1, n_knots), knots)
        applied.append("gain_drift_smooth")
    if rng.random() < GAIN_JUMP_PROB:
        jump_idx = int(rng.integers(nsamp // 4, 3 * nsamp // 4))
        factor = rng.uniform(0.6, 1.6)
        gain[jump_idx:] *= factor
        applied.append(f"gain_jump:{factor:.2f}")
    return gain


def add_time_dropout(nchan: int, rng: np.random.Generator, zero_mask_2d: np.ndarray, applied: list) -> None:
    """Blocks of literally-missing data (correlator/packet-loss issues)."""
    if rng.random() >= TIME_DROPOUT_PROB:
        return
    nsamp = zero_mask_2d.shape[1]
    block_len = int(rng.integers(max(1, nsamp // 25), max(2, nsamp // 6)))
    start = int(rng.integers(0, max(1, nsamp - block_len)))
    frac_chan = rng.uniform(0.3, 1.0)
    chan_idx = rng.choice(nchan, size=int(frac_chan * nchan), replace=False)
    zero_mask_2d[np.ix_(chan_idx, np.arange(start, start + block_len))] = True
    applied.append(f"time_dropout:{block_len}samp")


def apply_quantization(data: np.ndarray, rng: np.random.Generator, applied: list,
                        n_levels_range=(24, 64), dynamic_range_sigma: float = 8.0) -> np.ndarray:
    """Finite-bit-depth digitization (ADC quantization) -- a subtle
    "posterization" texture distinct from pure continuous noise, since real
    intensity data is digitized at a finite number of levels, not stored as
    infinite-precision floats.

    The step size is set from a ROBUST estimate of the typical noise scale
    (median absolute deviation), NOT the raw min/max or a high percentile
    of the data. Using a percentile instead was tried first and is a real
    trap: when a bright, spatially-extended component (aircraft RFI, a
    pulsar train, or the burst itself) is present, it drags a high-
    percentile-based range way up, making the step size far coarser than
    the normal noise fluctuations -- which then rounds nearly ALL typical
    pixels down to exactly 0, leaving only the brightest outliers nonzero.
    This silently zeroed out ~16% of a 200-image test batch before the fix;
    MAD is robust up to ~50% contamination so it stays sane even with a
    bright feature (or the burst) present.
    """
    if rng.random() >= QUANTIZATION_PROB:
        return data
    n_levels = int(rng.integers(*n_levels_range))
    med = np.median(data)
    mad = np.median(np.abs(data - med)) * 1.4826 + 1e-6
    step = (dynamic_range_sigma * mad) / n_levels
    quantized = np.round(data / step) * step
    applied.append(f"quantization:{n_levels}levels")
    return quantized


def compose_noise_environment(freqs_mhz: np.ndarray, rng: np.random.Generator, sigma_base: float,
                               time_axis_ms: np.ndarray, noise_scale_freq: np.ndarray,
                               zero_mask: np.ndarray, extra_signal: np.ndarray,
                               applied: list) -> Optional[np.ndarray]:
    """Randomly layers the RFI/instrumental/confusion-source components
    onto the given (already-allocated) arrays -- matching how real data
    shows multiple effects simultaneously, not just one at a time. Returns
    an optional 2D scintillation multiplier (time-AND-frequency varying,
    doesn't fit the 1D noise_scale_freq contract the rest share)."""
    add_randomized_narrowband_rfi(freqs_mhz, rng, noise_scale_freq, zero_mask, applied)
    add_impulsive_rfi(rng, extra_signal, sigma_base, time_axis_ms, applied)
    add_periodic_rfi(freqs_mhz, rng, extra_signal, sigma_base, time_axis_ms, applied)
    add_aircraft_rfi(freqs_mhz, rng, extra_signal, sigma_base, time_axis_ms, applied)
    add_rfi_storm(freqs_mhz, rng, extra_signal, sigma_base, time_axis_ms, applied)
    add_pulsar_periodic_signal(freqs_mhz, rng, extra_signal, sigma_base, time_axis_ms, applied)
    return add_scintillation(freqs_mhz, rng, time_axis_ms, applied)


# ----------------------------------------------------------------------
def load_catalog(csv_path: str) -> pd.DataFrame:
    """Minimal load of just the columns fitburst's parameters need."""
    df = pd.read_csv(csv_path)
    needed = ["dm_fitb", "width_fitb", "scat_time", "sp_idx", "sp_run", "fluence", "tns_name"]
    df = df.dropna(subset=needed)
    df = df[(df["dm_fitb"] > 0) & (df["width_fitb"] > 0) & (df["fluence"] > 0)]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
def generate_one(row: pd.Series, freqs_mhz: np.ndarray, rng: np.random.Generator, dispersed: bool):
    """Generate one negative-class (no burst) image. `row` is only used to
    match the window-size distribution to the FRB file (see module
    docstring) -- none of its burst-physics columns (dm_fitb aside, for the
    dispersed-window-sizing branch) ever place an actual signal."""
    num_time, tsamp_s = choose_grid(row, dispersed)
    time_axis_ms = (np.arange(num_time) - num_time // 2) * tsamp_s * 1e3
    applied: list = ["ambient"]  # bandpass_gain + baseline_offset are always present but weren't
                                  # individually tagged; this avoids an empty components list (which
                                  # pandas silently reads back as NaN) when no conditional RFI triggers

    bandpass = build_bandpass_gain(freqs_mhz, rng)
    baseline = build_baseline_offset(freqs_mhz, rng)
    rfi_mask = build_rfi_mask(freqs_mhz, rng)
    zero_mask = ~rfi_mask[:, None] * np.ones((1, num_time), dtype=bool)
    noise_scale_freq = bandpass.copy()
    extra_signal = np.zeros((len(freqs_mhz), num_time))

    chan_bw_mhz = freqs_mhz[1] - freqs_mhz[0]
    sigma_base = NOISE_SIGMA

    is_pure = rng.random() < PURE_NOISE_PROB
    scint_factor = None
    if not is_pure:
        scint_factor = compose_noise_environment(freqs_mhz, rng, sigma_base, time_axis_ms,
                                                   noise_scale_freq, zero_mask, extra_signal, applied)
        add_time_dropout(len(freqs_mhz), rng, zero_mask, applied)
    else:
        applied.append("pure_noise")

    dof = NOISE_DOF_NPOL * (chan_bw_mhz * 1e6) * tsamp_s
    sigma_per_chan = NOISE_SIGMA * bandpass
    noise = radiometer_noise(sigma_per_chan, dof, num_time, rng)

    time_gain = add_gain_drift(rng, num_time, applied)

    baseline_2d = baseline[:, None] * np.ones((1, num_time))
    if scint_factor is not None:
        extra_signal = extra_signal * scint_factor
        noise = noise * scint_factor

    noisy = (extra_signal + baseline_2d) * time_gain[None, :] + noise
    noisy[zero_mask] = 0.0
    noisy = apply_quantization(noisy, rng, applied)

    return noisy, time_axis_ms, tsamp_s, ~zero_mask.all(axis=1), applied


def plot_examples(outdir: str, n: int, seed: int, savepath: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = pd.read_csv(os.path.join(outdir, "labels.csv"))
    pick = labels.sample(min(n, len(labels)), random_state=seed).reset_index(drop=True)

    ncols = 4
    nrows = int(np.ceil(len(pick) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, row) in zip(axes, pick.iterrows()):
        npz = np.load(os.path.join(outdir, row["file"]), allow_pickle=True)
        d = npz["data_full"]
        meta = npz["metadata"].item()
        t_ms = (np.arange(meta["num_time"]) - meta["num_time"] // 2) * meta["res_time"] * 1e3
        vmin, vmax = np.percentile(d, [1.0, 99.5])
        ax.imshow(d, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                  extent=[t_ms[0], t_ms[-1], FREQ_LO_MHZ, FREQ_HI_MHZ])
        label = str(row["components"])[:55] + ("..." if len(str(row["components"])) > 55 else "")
        ax.set_title(label, fontsize=7)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")
    for ax in axes[len(pick):]:
        ax.axis("off")

    fig.suptitle("Simulated CHIME noise/RFI negative-class examples (no burst present)")
    plt.tight_layout()
    plt.savefig(savepath, dpi=120)
    print(f"Saved {savepath}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="chimefrbcat2.csv",
                    help="Used ONLY to match the window-size distribution to the FRB file -- "
                         "see module docstring. No burst is ever placed.")
    p.add_argument("--outdir", default="sim_output_rfi")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dispersed", action="store_true",
                    help="Match the FRB file's --dispersed window-sizing convention")
    p.add_argument("--plot", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_catalog(args.catalog)
    rng = np.random.default_rng(args.seed)
    freqs_mhz = np.linspace(FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ)

    n = len(df) if args.n is None else args.n
    n = min(n, len(df))
    rows = df.sample(n=n, random_state=args.seed, replace=(n > len(df))).reset_index(drop=True)

    summary_rows = []
    gen_num_time = []
    for i, row in rows.iterrows():
        noisy, time_axis_ms, tsamp_s, rfi_mask, applied = generate_one(row, freqs_mhz, rng, args.dispersed)
        gen_num_time.append(len(time_axis_ms))

        metadata = {
            "bad_chans": np.where(~rfi_mask)[0].tolist(),
            "freqs_bin0": float(freqs_mhz[0]),
            "is_dedispersed": not args.dispersed,
            "num_freq": NUM_FREQ,
            "num_time": len(time_axis_ms),
            "times_bin0": 0.0,
            "res_freq": float(freqs_mhz[1] - freqs_mhz[0]),
            "res_time": float(tsamp_s),
            "noise_components": applied,
            "label": 0,
        }
        fname = f"rfi_{i:06d}.npz"
        out_npz = os.path.join(args.outdir, fname)
        np.savez(out_npz, data_full=noisy, metadata=metadata)
        summary_rows.append(dict(file=fname, components=";".join(applied)))

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{n}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.outdir, "labels.csv"), index=False)
    print(f"Done: {n} negative-class images saved to {args.outdir}/ (.npz format)")
    if gen_num_time:
        arr = np.array(gen_num_time)
        print(f"[window-size check] RFI num_time: mean={arr.mean():.1f} median={np.median(arr):.0f} "
              f"std={arr.std():.1f}  (compare against simulate_frbs_fitburst.py's own printout -- "
              f"error #2 in chat; a large gap means the two classes' image sizes aren't actually "
              f"matched in practice, e.g. from different catalog subsets or --n row counts)")
    for tag in ["pure_noise", "aircraft_rfi", "rfi_storm", "pulsar_periodic", "scintillation",
                "narrowband_zero", "narrowband_bright", "impulsive_rfi", "periodic_rfi",
                "time_dropout", "quantization", "gain_drift", "gain_jump"]:
        count = summary["components"].str.contains(tag).sum()
        print(f"  {tag}: {count} ({100 * count / max(n, 1):.1f}%)")

    if args.plot > 0:
        plot_examples(args.outdir, args.plot, args.seed, os.path.join(args.outdir, "preview.png"))


if __name__ == "__main__":
    main()