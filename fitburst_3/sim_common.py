#!/usr/bin/env python3
"""
sim_common.py
=============

Single source of truth for everything that MUST be byte-identical between
simulate_frbs_fitburst.py (positive class) and simulate_rfi.py (negative
class): the CHIME/FRB band/grid configuration, the instrumental-effects
model (bandpass gain, persistent baseline offset, RFI channel masking,
radiometer noise), and the window-sizing logic (`choose_grid`).

Why this file exists (fixes error #1 from the shortcut-learning review):
simulate_rfi.py used to keep its own COPY of this code with a comment
telling you to manually port any edit made in simulate_frbs_fitburst.py.
That's a real drift risk -- one missed edit and the two classes' noise
statistics diverge, handing a CNN a free "which generator produced this"
shortcut instead of making it learn real burst morphology. Both scripts
now `import` this module instead of duplicating its contents, so there is
no copy to forget to update -- editing this file changes both classes at
once, by construction.
"""

import numpy as np
import pandas as pd
from typing import Optional

# ----------------------------------------------------------------------
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

# --- realism knobs (identical for both classes -- see module docstring) ---
BANDPASS_RIPPLE_PERIOD_MHZ = 30.0
BANDPASS_ROLLOFF_FRAC = 0.15     # smooth large-scale gain variation across the band
BANDPASS_RIPPLE_FRAC = 0.12      # ~30 MHz ripple amplitude
BANDPASS_JITTER_FRAC = 0.12      # independent per-channel gain jitter
NOISE_DOF_NPOL = 2.0             # for chi^2-shaped (not pure Gaussian) radiometer noise
MIN_WINDOW_NATIVE_SAMPLES = 4    # mild floor so a near-zero-width/unscattered burst still gets
                                  # a sensible minimum window (native samples)
DISPLAY_PAD_FACTOR = 5.0         # for a ~0.3ms burst this gives ~a 17-20ms window,
                                  # matching real CHIME quick-look plots
MAX_WINDOW_S = 0.4                # sanity cap (400 ms) for the non-dispersed window

BASELINE_OFFSET_STD = 0.28       # persistent per-channel additive offset (real banding)

RFI_BANDS_MHZ = [(500.0, 600.0), (600.0, 650.0), (730.0, 755.0)]
RFI_BAND_PROB = 0.6


def build_bandpass_gain(freqs_mhz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative per-channel gain: smooth roll-off + ~30 MHz ripple +
    per-channel jitter. This -- not "add more noise" -- is what actually
    produces the banded look of real data."""
    n = len(freqs_mhz)
    x = (freqs_mhz - freqs_mhz.mean()) / (freqs_mhz.max() - freqs_mhz.min())
    smooth = 1.0 + BANDPASS_ROLLOFF_FRAC * np.sin(2 * np.pi * x * rng.uniform(0.5, 1.5) + rng.uniform(0, 2 * np.pi))
    ripple = 1.0 + BANDPASS_RIPPLE_FRAC * np.sin(2 * np.pi * freqs_mhz / BANDPASS_RIPPLE_PERIOD_MHZ
                                                  + rng.uniform(0, 2 * np.pi))
    jitter = np.exp(rng.normal(0.0, BANDPASS_JITTER_FRAC, size=n))
    return np.clip(smooth * ripple * jitter, 0.3, 2.5)


def build_baseline_offset(freqs_mhz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A persistent per-channel ADDITIVE offset -- constant across the
    whole time axis within a channel."""
    n = len(freqs_mhz)
    smooth = np.convolve(rng.normal(0.0, BASELINE_OFFSET_STD, size=n),
                          np.ones(9) / 9, mode="same")
    jitter = rng.normal(0.0, BASELINE_OFFSET_STD * 0.5, size=n)
    return smooth + jitter


def build_rfi_mask(freqs_mhz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean per-channel mask, True = usable."""
    mask = np.ones(len(freqs_mhz), dtype=bool)
    for lo, hi in RFI_BANDS_MHZ:
        if rng.random() < RFI_BAND_PROB:
            mask &= ~((freqs_mhz >= lo) & (freqs_mhz <= hi))
    return mask


def dispersion_delay_ms(freq_mhz: np.ndarray, dm: float, f_ref_mhz: float) -> np.ndarray:
    return K_DM * dm * (freq_mhz ** -2 - f_ref_mhz ** -2)


def radiometer_noise(sigma_per_chan: np.ndarray, dof: float, num_time: int,
                      rng: np.random.Generator) -> np.ndarray:
    """chi^2-shaped (not pure Gaussian) noise, per channel."""
    dof = max(dof, 3.0)
    raw = rng.chisquare(dof, size=(len(sigma_per_chan), num_time))
    standardized = (raw - dof) / np.sqrt(2.0 * dof)
    return standardized * sigma_per_chan[:, None]


def choose_grid(row: pd.Series, dispersed: bool, pad_factor: float = DISPLAY_PAD_FACTOR):
    """Pick (num_time, tsamp_s) so the burst -- and, in dispersed mode, the
    full DM sweep across the band -- comfortably fits in the window.

    Used IDENTICALLY by both classes (see module docstring, and error #2
    from the shortcut-learning review): the negative-class generator feeds
    real catalog rows through this exact function too, purely to match the
    positive class's image-size distribution, even though no burst is ever
    placed. If the two classes' num_time distributions differed, "how wide
    is this window" would itself become a free shortcut feature.
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


def verify_matched_window_distributions(frb_num_time: list, rfi_num_time: list, verbose: bool = True) -> dict:
    """Diagnostic for error #2: compares the two classes' realized num_time
    distributions after a generation run. They should be close (drawn
    through the same choose_grid() from real catalog rows); a large gap
    would mean the size-matching isn't actually working in practice (e.g.
    different catalog subsets, different --dispersed flags, or different
    n_pos/n_neg row-sampling behavior) and is itself a shortcut risk.
    """
    frb_arr, rfi_arr = np.asarray(frb_num_time), np.asarray(rfi_num_time)
    stats = {
        "frb_mean": float(frb_arr.mean()), "frb_median": float(np.median(frb_arr)),
        "frb_std": float(frb_arr.std()),
        "rfi_mean": float(rfi_arr.mean()), "rfi_median": float(np.median(rfi_arr)),
        "rfi_std": float(rfi_arr.std()),
    }
    stats["mean_diff_frac"] = abs(stats["frb_mean"] - stats["rfi_mean"]) / max(stats["frb_mean"], 1e-9)
    if verbose:
        print(f"[window-size check] FRB num_time: mean={stats['frb_mean']:.1f} "
              f"median={stats['frb_median']:.0f} std={stats['frb_std']:.1f}")
        print(f"[window-size check] RFI num_time: mean={stats['rfi_mean']:.1f} "
              f"median={stats['rfi_median']:.0f} std={stats['rfi_std']:.1f}")
        flag = "OK" if stats["mean_diff_frac"] < 0.05 else "WARNING -- distributions diverge, check catalog sampling"
        print(f"[window-size check] mean difference: {100 * stats['mean_diff_frac']:.1f}%  [{flag}]")
    return stats

# ======================================================================
# RFI / instrumental environment (MOVED here from simulate_rfi.py so that
# BOTH classes draw their background from the exact same code path)
# ======================================================================
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


# ======================================================================
# Catalog loading (shared: one catalog subset for BOTH classes)
# ======================================================================
def load_catalog(csv_path: str, verbose: bool = True) -> pd.DataFrame:
    """Keep one row per physical event (sub_num == 0) with sane fitburst
    parameters. Used by BOTH generators; the old simulate_rfi.py used a
    looser filter (4527 rows vs 4081), so the two classes' window-size
    distributions were drawn from different catalog subsets."""
    df = pd.read_csv(csv_path)
    n0 = len(df)
    df = df[df["sub_num"] == 0]
    n1 = len(df)
    needed = ["dm_fitb", "width_fitb", "scat_time", "sp_idx", "sp_run", "fluence", "tns_name"]
    df = df.dropna(subset=needed)
    n2 = len(df)
    df = df[(df["dm_fitb"] > 0) & (df["width_fitb"] > 0) & (df["fluence"] > 0)]
    if verbose:
        print(f"  rows: raw={n0} -> sub_num==0={n1} -> has fit params={n2} -> sane={len(df)}")
    return df.reset_index(drop=True)


# ======================================================================
# THE single rendering path (Step 0 fix)
# ======================================================================
def render_image(freqs_mhz: np.ndarray, num_time: int, tsamp_s: float,
                 rng: np.random.Generator, burst_model: Optional[np.ndarray] = None):
    """Render ONE image. Used for BOTH classes:
        negative: burst_model=None
        positive: burst_model=<fitburst model, already scaled to fluence>

    The two classes now differ ONLY by whether a burst is added. Every
    other ingredient (bandpass, baseline banding, masks, narrowband RFI,
    impulsive / aircraft / periodic RFI, RFI storms, pulsars, scintillation,
    gain drift, time dropouts, ADC quantization) is drawn from the same
    distribution with the same probabilities.

    Before this fix the positive generator applied none of the
    drift / dropout / scintillation / quantization / RFI steps, so
    "anything irregular => RFI" was a free shortcut.

    Returns (noisy, bandpass, chan_ok, applied)."""
    applied: list = ["ambient"]
    time_axis_ms = (np.arange(num_time) - num_time // 2) * tsamp_s * 1e3

    bandpass = build_bandpass_gain(freqs_mhz, rng)
    baseline = build_baseline_offset(freqs_mhz, rng)
    rfi_mask = build_rfi_mask(freqs_mhz, rng)
    zero_mask = ~rfi_mask[:, None] * np.ones((1, num_time), dtype=bool)
    noise_scale_freq = bandpass.copy()
    extra_signal = np.zeros((len(freqs_mhz), num_time))

    is_pure = rng.random() < PURE_NOISE_PROB
    scint_factor = None
    if not is_pure:
        scint_factor = compose_noise_environment(freqs_mhz, rng, NOISE_SIGMA, time_axis_ms,
                                                   noise_scale_freq, zero_mask, extra_signal, applied)
        add_time_dropout(len(freqs_mhz), rng, zero_mask, applied)
    else:
        applied.append("pure_noise")

    chan_bw_mhz = freqs_mhz[1] - freqs_mhz[0]
    dof = NOISE_DOF_NPOL * (chan_bw_mhz * 1e6) * tsamp_s
    # BUG FIX: the old code used NOISE_SIGMA * bandpass here, so the
    # "narrowband_bright" RFI (which edits noise_scale_freq) was tagged in
    # the metadata but never actually changed the image.
    sigma_per_chan = NOISE_SIGMA * noise_scale_freq
    noise = radiometer_noise(sigma_per_chan, dof, num_time, rng)

    time_gain = add_gain_drift(rng, num_time, applied)

    signal = extra_signal.copy()
    if burst_model is not None:
        signal = signal + burst_model * bandpass[:, None]
    if scint_factor is not None:
        signal = signal * scint_factor
        noise = noise * scint_factor

    baseline_2d = baseline[:, None] * np.ones((1, num_time))
    noisy = (signal + baseline_2d) * time_gain[None, :] + noise
    noisy[zero_mask] = 0.0
    noisy = apply_quantization(noisy, rng, applied)

    chan_ok = ~zero_mask.all(axis=1)
    return noisy, bandpass, chan_ok, applied
