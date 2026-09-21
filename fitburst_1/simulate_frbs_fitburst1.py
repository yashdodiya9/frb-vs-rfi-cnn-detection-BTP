#!/usr/bin/env python3
"""
simulate_frbs_fitburst.py
===========================

Simulates realistic FRB dynamic spectra using ONLY the official CHIME/FRB
`fitburst` package (https://github.com/CHIMEFRB/fitburst) as the physics
engine -- no dependency on any of my earlier sim_common.py / frb_simulator.py
code. The only other imports are numpy, pandas, and matplotlib (standard
scientific-Python packages, not anything custom).

Burst parameters (DM, width, scattering time, spectral index/running,
fluence) are bootstrapped from real values in your CHIME/FRB Catalog 2 CSV
(chimefrbcat2.csv), so each simulated burst is grounded in an actual
detected FRB's fitted physics, rendered through fitburst's own model.

I (Claude) do not have network access in this sandbox and could not
install or run fitburst myself, so this script is written carefully
against fitburst's documented API (github.com/CHIMEFRB/fitburst,
chimefrb.github.io/fitburst/usage/creating_models/) but has NOT been
executed end-to-end. Run it locally and send me any traceback.

Folder setup
------------
    project/
    |-- chimefrbcat2.csv
    |-- simulate_frbs_fitburst.py   <- this file
    `-- sim_output/                  <- created automatically on first run

fitburst is installed as a normal importable package -- it does NOT need
to live inside this project folder (see install instructions below).

Install fitburst (not on PyPI under its own name)
--------------------------------------------------
    git clone https://github.com/CHIMEFRB/fitburst.git
    cd fitburst
    pip install .
    cd ..    # back to your project folder

Usage
-----
    python3 simulate_frbs_fitburst.py --catalog chimefrbcat2.csv --n 50 --outdir sim_output --plot 12

    # simulate the DISPERSED (pre-dedispersion) sweep instead of a
    # dedispersed candidate cutout:
    python3 simulate_frbs_fitburst.py --catalog chimefrbcat2.csv --n 50 --dispersed

    # omit --n entirely to simulate every row in the catalog:
    python3 simulate_frbs_fitburst.py --catalog chimefrbcat2.csv --plot 12
"""

import argparse
import os

import numpy as np
import pandas as pd

from fitburst.analysis.model import SpectrumModeler

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

    IMPORTANT FIX: the original version of this function fixed num_time at
    a constant 256 and *derived* tsamp_s = window_s / 256, floored at
    NATIVE_TSAMP_S. For almost any realistic burst (width a few ms or
    less), the naturally desired tsamp comes out FINER than CHIME can
    actually sample at -- so the floor silently kicked in, forcing
    tsamp_s up to ~0.983 ms while num_time stayed at 256. The ACTUAL
    resulting window was then 256 * 0.983 ms =~ 252 ms regardless of what
    pad_factor/floor implied, ~15x wider than intended -- which is exactly
    why the burst rendered as a single narrow sliver in a huge, mostly
    empty field (see chat comparison). The fix: always sample at the real
    native resolution (never invent a finer virtual one), and let the
    NUMBER of samples vary per burst to hit the intended window instead.
    """
    width_s = float(row["width_fitb"])
    scat_s = float(row["scat_time"])
    burst_extent_s = max(6 * width_s + 6 * scat_s, MIN_WINDOW_NATIVE_SAMPLES * NATIVE_TSAMP_S)

    if dispersed:
        dm = float(row["dm_fitb"])
        sweep_s = (K_DM * dm * (FREQ_LO_MHZ ** -2 - FREQ_HI_MHZ ** -2)) / 1e3  # ms -> s
        window_s = sweep_s + pad_factor * burst_extent_s
    else:
        window_s = pad_factor * burst_extent_s

    tsamp_s = NATIVE_TSAMP_S  # never sample finer than the real instrument can
    num_time = int(np.clip(round(window_s / tsamp_s), MIN_NUM_TIME, MAX_NUM_TIME))
    return num_time, tsamp_s


def row_to_fitburst_params(row: pd.Series, arrival_time_s: float, dispersed: bool) -> dict:
    """Build a fitburst parameter dictionary for ONE burst component,
    reading directly from a chimefrbcat2.csv row. arrival_time/burst_width/
    scattering_timescale are in SECONDS, matching both fitburst's docs and
    the units the catalog already stores these columns in.

    `dm`: when dispersed=False (the usual case -- an already-dedispersed
    candidate cutout, matching how CHIME's own released waterfalls and
    your CNN's training images are built), fitburst treats `dm` as a DM
    OFFSET, so 0.0 = perfectly dedispersed. When dispersed=True, `dm` is
    the real catalog DM and produces a genuine dispersed sweep.
    """
    return {
        "amplitude": [1.0],                        # calibrated post-hoc, see calibrate_amplitude()
        "arrival_time": [arrival_time_s],
        "burst_width": [float(row["width_fitb"])],
        "dm": [float(row["dm_fitb"])] if dispersed else [0.0],
        "dm_index": [-2.0],
        "ref_freq": [REF_FREQ_MHZ],
        "scattering_index": [-4.0],
        "scattering_timescale": [float(row["scat_time"])],
        "spectral_index": [float(row["sp_idx"])],
        "spectral_running": [float(row["sp_run"])],
    }


def calibrate_amplitude(model_obj: SpectrumModeler, params: dict, target_fluence_jyms: float,
                         tsamp_s: float) -> np.ndarray:
    """fitburst's own docs/examples don't specify a physical unit for
    `amplitude` -- their example scripts just leave it at a placeholder
    value. Since we DO want brightness to reflect each burst's real
    catalog fluence, compute the model once at amplitude=1 and rescale the
    OUTPUT ARRAY linearly to hit the target fluence. The model is linear in
    amplitude, so this is equivalent to solving for the correct amplitude
    directly, without needing to know its internal units.
    """
    model_obj.update_parameters(params)
    model = model_obj.compute_model()  # (num_freq, num_time), amplitude=1

    # fluence [Jy ms] := (tsamp_ms / num_freq) * sum(model), i.e. the time
    # integral of the band-averaged flux density -- standard definition.
    tsamp_ms = tsamp_s * 1e3
    current_fluence = (tsamp_ms / model.shape[0]) * model.sum()
    if current_fluence <= 0:
        return model
    return model * (target_fluence_jyms / current_fluence)


def simulate_one(row: pd.Series, freqs_mhz: np.ndarray, rng: np.random.Generator, dispersed: bool):
    num_time, tsamp_s = choose_grid(row, dispersed)
    times_s = np.arange(num_time) * tsamp_s
    arrival_time_s = times_s[num_time // 2] if not dispersed else times_s[num_time // 4]

    params = row_to_fitburst_params(row, arrival_time_s, dispersed)
    model_obj = SpectrumModeler(freqs_mhz, times_s, is_dedispersed=not dispersed, num_components=1)
    model = calibrate_amplitude(model_obj, params, float(row["fluence"]), tsamp_s)

    # --- realistic instrumental effects layered on top of the pure fitburst
    # signal (the FRB physics itself is still 100% fitburst's model; this
    # only adds the receiver/telescope response around it) ---
    bandpass = build_bandpass_gain(freqs_mhz, rng)
    model_with_bandpass = model * bandpass[:, None]

    chan_bw_mhz = freqs_mhz[1] - freqs_mhz[0]
    dof = NOISE_DOF_NPOL * (chan_bw_mhz * 1e6) * tsamp_s
    sigma_per_chan = NOISE_SIGMA * bandpass  # noisier channels track higher-gain channels,
                                              # matching the radiometer equation
    noise = radiometer_noise(sigma_per_chan, dof, num_time, rng)

    noisy = model_with_bandpass + noise
    return noisy, model, times_s, params, bandpass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default="chimefrbcat2.csv")
    p.add_argument("--outdir", default="sim_output")
    p.add_argument("--n", type=int, default=None,
                    help="Number of FRBs to simulate (default: every row in the catalog)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dispersed", action="store_true",
                    help="Simulate the raw DISPERSED sweep instead of a dedispersed candidate cutout")
    p.add_argument("--plot", type=int, default=0, help="Also save this many PNG previews")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_catalog(args.catalog)
    rng = np.random.default_rng(args.seed)
    freqs_mhz = np.linspace(FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ)

    n = len(df) if args.n is None else args.n
    n = min(n, len(df))
    rows = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    plotted = 0
    for i, row in rows.iterrows():
        noisy, model, times_s, params, bandpass = simulate_one(row, freqs_mhz, rng, args.dispersed)

        metadata = {
            "bad_chans": [],
            "freqs_bin0": float(freqs_mhz[0]),
            "is_dedispersed": not args.dispersed,
            "num_freq": NUM_FREQ,
            "num_time": len(times_s),
            "times_bin0": 0.0,
            "res_freq": float(freqs_mhz[1] - freqs_mhz[0]),
            "res_time": float(times_s[1] - times_s[0]),
        }
        out_npz = os.path.join(args.outdir, f"{row['tns_name']}_sim.npz")
        # fitburst-generic format: same data_full / metadata / burst_parameters
        # keys used by fitburst's own pipelines/simulate_burst.py
        np.savez(out_npz, data_full=noisy, metadata=metadata, burst_parameters=params)

        if plotted < args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Percentile-based color scale instead of matplotlib's default
            # auto min/max: a single extreme bright pixel at the burst peak
            # was stretching the whole scale, crushing all the real noise
            # texture into a flat block -- the main reason the earlier
            # preview looked artificially "clean" even before the bandpass
            # fix above.
            vmin, vmax = np.percentile(noisy, [1.0, 99.5])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pcolormesh(times_s * 1e3, freqs_mhz, noisy, shading="auto",
                          vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (MHz)")
            ax.set_title(f"{row['tns_name']}  DM={row['dm_fitb']:.1f}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"{row['tns_name']}_sim.png"), dpi=120)
            plt.close(fig)
            plotted += 1

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{n}")

    print(f"Done: {n} simulated FRBs saved to {args.outdir}/ (.npz, fitburst-generic format)")


if __name__ == "__main__":
    main()