#!/usr/bin/env python3
"""
simulate_rfi_chime.py
=======================

Simulates the negative ("not an FRB") class for a CHIME FRB-vs-noise CNN:
pure noise / RFI / instrumental / astrophysical-confusion images with NO
burst signal present. This is the companion file to simulate_frbs_fitburst.py.

STEP 0 UPDATE: this file no longer contains any noise/RFI code of its own.
Both classes call sim_common.render_image(); the negative class is simply
render_image(burst_model=None). The text below is kept for history.

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

import numpy as np
import pandas as pd

from sim_common import (
    FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ, choose_grid, load_catalog, render_image,
    verify_matched_window_distributions,
)


def generate_one(row: pd.Series, freqs_mhz: np.ndarray, rng: np.random.Generator, dispersed: bool):
    """One negative-class (no burst) image. `row` only sets the window size
    (same choose_grid() as the positive class)."""
    num_time, tsamp_s = choose_grid(row, dispersed)
    noisy, _bandpass, chan_ok, applied = render_image(freqs_mhz, num_time, tsamp_s, rng, burst_model=None,
                                                     dm_candidate=float(row["dm_fitb"]))
    time_axis_ms = (np.arange(num_time) - num_time // 2) * tsamp_s * 1e3
    return noisy, time_axis_ms, tsamp_s, chan_ok, applied


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

    n = len(df) if args.n is None else args.n   # n > len(df) is allowed: rows are re-drawn with replacement
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