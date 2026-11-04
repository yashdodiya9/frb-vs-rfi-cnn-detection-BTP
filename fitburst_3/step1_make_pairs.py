#!/usr/bin/env python3
"""
step1_make_pairs.py -- PAIRED images: identical background, with / without burst.

For each catalog row we render the image twice from the SAME random seed:
    <name>_with.npz     background + burst
    <name>_without.npz  the same background, burst switched off

render_image() draws random numbers in exactly the same order whether or not
a burst is present, so everything (bandpass, banding, masks, RFI, drift,
dropouts, noise realisation) is identical and the ONLY difference is the
burst. A model that really detects bursts must score "with" > "without" in
(nearly) every pair; a model that reads the background cannot.

No change to simulate_frbs_fitburst.py is needed: the 'without' image is made
by zeroing the fitburst model through a thin wrapper.

Usage:
    python3 step1_make_pairs.py --catalog chimefrbcat2.csv --n 600 --outdir sim_paired
"""
import argparse
import os

import numpy as np

import simulate_frbs_fitburst as sfm
from sim_common import FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ, load_catalog

_orig_calibrate = sfm.calibrate_amplitude


def _zero_burst(*a, **k):
    return _orig_calibrate(*a, **k) * 0.0     # same shape / validity checks, no burst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="chimefrbcat2.csv")
    ap.add_argument("--outdir", default="sim_paired")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_catalog(args.catalog, verbose=True)
    freqs = np.linspace(FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ)
    rows = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    saved, skipped, max_edge_diff = 0, 0, 0.0
    for i, row in rows.iterrows():
        s = 1_000_000 + args.seed * 1000 + i
        try:
            sfm.calibrate_amplitude = _orig_calibrate
            with_img, *_r, pos_frac, faint_scale, applied = sfm.simulate_one(row, freqs, np.random.default_rng(s), False)
            sfm.calibrate_amplitude = _zero_burst
            without_img, *_ = sfm.simulate_one(row, freqs, np.random.default_rng(s), False)
        except ValueError:
            skipped += 1
            continue
        finally:
            sfm.calibrate_amplitude = _orig_calibrate
        if not (np.all(np.isfinite(with_img)) and np.all(np.isfinite(without_img))):
            skipped += 1
            continue
        nt = with_img.shape[1]; k = max(4, int(0.2 * nt))
        edge = np.abs(with_img[:, :k] - without_img[:, :k]).max()
        max_edge_diff = max(max_edge_diff, float(edge))
        meta = {"faint_scale": float(faint_scale), "burst_pos_frac": float(pos_frac),
                "noise_components": applied, "is_dedispersed": True}
        name = str(row["tns_name"])
        np.savez(os.path.join(args.outdir, f"{name}_with.npz"), data_full=with_img, metadata={**meta, "label": 1})
        np.savez(os.path.join(args.outdir, f"{name}_without.npz"), data_full=without_img, metadata={**meta, "label": 0})
        saved += 1
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(rows)}")

    print(f"saved {saved} pairs to {args.outdir}/  (skipped {skipped})")
    print(f"sanity: largest |with - without| inside burst-free edge columns = {max_edge_diff:.3g}  "
          f"(should be ~0; tiny values can come from the quantization step being re-estimated)")


if __name__ == "__main__":
    main()
