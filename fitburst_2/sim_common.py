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
