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

# CHIME/FRB band / grid configuration
# ----------------------------------------------------------------------
FREQ_LO_MHZ = 400.0
FREQ_HI_MHZ = 800.0
NUM_FREQ = 256              # channels in the simulated image (native CHIME is 16384;
                             # 256 keeps images small and fast to generate)
MIN_NUM_TIME = 32            # keep images from becoming absurdly small...
MAX_NUM_TIME = 512           # ...or unnecessarily huge
NATIVE_TSAMP_S = 0.00098304  # CHIME/FRB's native sampling time
K_DM = 4.148808e6            # dispersion constant, ms * MHz^2 * pc^-1 * cm^3

# fitburst's `ref_freq` is a fixed, non-fittable reference frequency (NOT
# 600 MHz, NOT tied to a burst's own peak). This value is read directly out
# of a real CHIME/FRB Catalog 2 dynamic-spectrum file's own metadata (its
# `ref_freq` attribute) -- i.e. it's the real instrumental convention, not
# a guess.
REF_FREQ_MHZ = 400.1953125

NOISE_SIGMA = 0.2  # baseline noise stdev (before per-channel gain, see below)

# --- realism knobs added to fix "too clean" simulated output (see chat) ---
# Real CHIME data shows pronounced ROW-TO-ROW banding (each frequency
# channel has its own persistent brightness level) from real hardware gain
# variation, not flat noise. RIPPLE_PERIOD_MHZ=30 matches the documented
# feed-reflection ripple in CHIME/FRB (Rafiei-Ravandi & Smith 2023, Fig. 1
# caption: "period of ~30 MHz across the bandwidth").
BANDPASS_RIPPLE_PERIOD_MHZ = 30.0
BANDPASS_ROLLOFF_FRAC = 0.15     # smooth large-scale gain variation across the band
BANDPASS_RIPPLE_FRAC = 0.12      # ~30 MHz ripple amplitude
BANDPASS_JITTER_FRAC = 0.12      # independent per-channel gain jitter
NOISE_DOF_NPOL = 2.0             # for chi^2-shaped (not pure Gaussian) radiometer noise
MIN_WINDOW_NATIVE_SAMPLES = 4    # mild floor so a near-zero-width/unscattered burst still gets
                                  # a sensible minimum window (native samples)
DISPLAY_PAD_FACTOR = 5.0         # mild increase from the original 4.0 -- for a ~0.3ms burst this
                                  # gives ~a 17-20ms window, matching real CHIME quick-look plots.
                                  # (Note: the "single-pixel" look in the original bug report was
                                  # mostly from color-scale saturation and flat noise, not window
                                  # size -- an earlier, much larger value here was overcorrecting.)
MAX_WINDOW_S = 0.4                # sanity cap (400 ms) for the non-dispersed window, so a rare
                                  # extreme-scattering burst can't blow the display out to seconds

# The bandpass fix above only changes each channel's noise SPREAD, which is
# too subtle to produce the bold, obviously-different-color horizontal
# stripes real CHIME data shows. That look comes from a PERSISTENT PER-
# CHANNEL BASELINE -- a near-constant offset across the whole time axis
# within a channel (residual gain/calibration error, or a channel simply
# sitting at a different power level) -- which is an additive effect, not
# a variance effect. Added as its own knob so it's easy to tune separately.
BASELINE_OFFSET_STD = 0.28

# Real released dynamic spectra have several completely flat, zeroed-out
# bands from static masking + persistent RFI (see the RFI-mitigation paper
# discussed earlier in this chat: persistent RFI 600-650 MHz, an LTE band
# ~730-755 MHz; 500-600 MHz digital-TV contamination is also commonly
# reported). A simulated image with NO masked bands at all is one of the
# most obvious "too clean" tells.
RFI_BANDS_MHZ = [(500.0, 600.0), (600.0, 650.0), (730.0, 755.0)]
RFI_BAND_PROB = 0.6   # not every burst's data has all bands masked; keep it randomized

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


def build_bandpass_gain(freqs_mhz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative per-channel gain: smooth roll-off + ~30 MHz ripple +
    per-channel jitter. This -- not "add more noise" -- is what actually
    produces the banded look of real data; uniform noise everywhere is the
    main visual giveaway of a too-clean simulation."""
    n = len(freqs_mhz)
    x = (freqs_mhz - freqs_mhz.mean()) / (freqs_mhz.max() - freqs_mhz.min())
    smooth = 1.0 + BANDPASS_ROLLOFF_FRAC * np.sin(2 * np.pi * x * rng.uniform(0.5, 1.5) + rng.uniform(0, 2 * np.pi))
    ripple = 1.0 + BANDPASS_RIPPLE_FRAC * np.sin(2 * np.pi * freqs_mhz / BANDPASS_RIPPLE_PERIOD_MHZ
                                                  + rng.uniform(0, 2 * np.pi))
    jitter = np.exp(rng.normal(0.0, BANDPASS_JITTER_FRAC, size=n))
    return np.clip(smooth * ripple * jitter, 0.3, 2.5)


def build_baseline_offset(freqs_mhz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A persistent per-channel ADDITIVE offset -- constant across the
    whole time axis within a channel. This is what actually produces real
    data's bold horizontal banding (each row sitting at its own distinct
    brightness level), which a purely multiplicative noise-variance scaling
    (build_bandpass_gain) does NOT reproduce, since zero-mean noise stays
    zero-mean regardless of its spread."""
    n = len(freqs_mhz)
    # smooth, slowly-varying component (a handful of channels' worth of
    # correlation length) plus independent per-channel jitter
    smooth = np.convolve(rng.normal(0.0, BASELINE_OFFSET_STD, size=n),
                          np.ones(9) / 9, mode="same")
    jitter = rng.normal(0.0, BASELINE_OFFSET_STD * 0.5, size=n)
    return smooth + jitter


def build_rfi_mask(freqs_mhz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean per-channel mask, True = usable. Zeros out a random subset
    of the known persistent-RFI bands so most simulated images carry at
    least one flat masked gap, matching real released data."""
    mask = np.ones(len(freqs_mhz), dtype=bool)
    for lo, hi in RFI_BANDS_MHZ:
        if rng.random() < RFI_BAND_PROB:
            mask &= ~((freqs_mhz >= lo) & (freqs_mhz <= hi))
    return mask


def dispersion_delay_ms(freq_mhz: np.ndarray, dm: float, f_ref_mhz: float) -> np.ndarray:
    return K_DM * dm * (freq_mhz ** -2 - f_ref_mhz ** -2)


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


def radiometer_noise(sigma_per_chan: np.ndarray, dof: float, num_time: int,
                      rng: np.random.Generator) -> np.ndarray:
    """chi^2-shaped (not pure Gaussian) noise, per channel, with each
    channel's own sigma tied to its bandpass gain -- matching how real
    radiometer noise scales with instrumental gain. Real intensity data is
    a sum of squared voltages (chi^2 distributed), which has a slightly
    heavier positive tail than Gaussian noise; using it instead of flat
    `rng.normal` is a small but real step toward matching real statistics."""
    dof = max(dof, 3.0)
    raw = rng.chisquare(dof, size=(len(sigma_per_chan), num_time))
    standardized = (raw - dof) / np.sqrt(2.0 * dof)
    return standardized * sigma_per_chan[:, None]


# ----------------------------------------------------------------------
def load_catalog(csv_path: str) -> pd.DataFrame:
    """Minimal load of just the columns fitburst's parameters need."""
    df = pd.read_csv(csv_path)
    needed = ["dm_fitb", "width_fitb", "scat_time", "sp_idx", "sp_run", "fluence", "tns_name"]
    df = df.dropna(subset=needed)
    df = df[(df["dm_fitb"] > 0) & (df["width_fitb"] > 0) & (df["fluence"] > 0)]
    return df.reset_index(drop=True)


def choose_grid(row: pd.Series, dispersed: bool, pad_factor: float = DISPLAY_PAD_FACTOR):
    """Pick (num_time, tsamp_s) so the burst -- and, in dispersed mode, the
    full DM sweep across the band -- comfortably fits in the window.

    FIX #1: the original version of this function fixed num_time at a
    constant 256 and *derived* tsamp_s = window_s / 256, floored at
    NATIVE_TSAMP_S. For almost any realistic burst (width a few ms or
    less), the naturally desired tsamp comes out FINER than CHIME can
    actually sample at -- so the floor silently kicked in, forcing
    tsamp_s up to ~0.983 ms while num_time stayed at 256. The ACTUAL
    resulting window was then 256 * 0.983 ms =~ 252 ms regardless of what
    pad_factor/floor implied, ~15x wider than intended -- which is exactly
    why the burst rendered as a single narrow sliver in a huge, mostly
    empty field. The fix: always sample at the real native resolution
    (never invent a finer virtual one), and let the NUMBER of samples vary
    per burst to hit the intended window instead.

    FIX #2: cross-checked against real released quick-look windows for a
    handful of named bursts (FRB20190712C, FRB20190716B, ...) shown
    alongside their actual catalog scattering_timescale -- the original
    6x-scattering-timescale term badly overshot for scattered bursts
    (predicted ~1.5 SECONDS of window for a 49ms-scattering burst that
    real plots show in ~125ms). Cut to 2.5x and added a sanity cap.
    """
    width_s = float(row["width_fitb"])
    scat_s = float(row["scat_time"])
    burst_extent_s = max(6 * width_s + 2.5 * scat_s, MIN_WINDOW_NATIVE_SAMPLES * NATIVE_TSAMP_S)

    if dispersed:
        dm = float(row["dm_fitb"])
        sweep_s = (K_DM * dm * (FREQ_LO_MHZ ** -2 - FREQ_HI_MHZ ** -2)) / 1e3  # ms -> s
        window_s = sweep_s + pad_factor * burst_extent_s
    else:
        window_s = min(pad_factor * burst_extent_s, MAX_WINDOW_S)

    tsamp_s = NATIVE_TSAMP_S  # never sample finer than the real instrument can
    num_time = int(np.clip(round(window_s / tsamp_s), MIN_NUM_TIME, MAX_NUM_TIME))
    return num_time, tsamp_s


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
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dispersed", action="store_true",
                    help="Match the FRB file's --dispersed window-sizing convention")
    p.add_argument("--plot", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_catalog(args.catalog)
    rng = np.random.default_rng(args.seed)
    freqs_mhz = np.linspace(FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ)

    n = args.n
    rows = df.sample(n=n, random_state=args.seed, replace=(n > len(df))).reset_index(drop=True)

    summary_rows = []
    for i, row in rows.iterrows():
        noisy, time_axis_ms, tsamp_s, rfi_mask, applied = generate_one(row, freqs_mhz, rng, args.dispersed)

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
    for tag in ["pure_noise", "aircraft_rfi", "rfi_storm", "pulsar_periodic", "scintillation",
                "narrowband_zero", "narrowband_bright", "impulsive_rfi", "periodic_rfi",
                "time_dropout", "quantization", "gain_drift", "gain_jump"]:
        count = summary["components"].str.contains(tag).sum()
        print(f"  {tag}: {count} ({100 * count / max(n, 1):.1f}%)")

    if args.plot > 0:
        plot_examples(args.outdir, args.plot, args.seed, os.path.join(args.outdir, "preview.png"))


if __name__ == "__main__":
    main()