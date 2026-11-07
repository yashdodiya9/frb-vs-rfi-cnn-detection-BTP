#!/usr/bin/env python3
"""
step2_make_sweep.py -- brightness sweep: the SAME burst on the SAME background at many brightness levels.

For every catalog row we render the image from one fixed random seed, once per brightness level:
    scale 0      -> background only (no burst): the paired negative
    scale 0.02 .. 1.0 -> the burst at that fraction of its catalog fluence
Everything except the burst amplitude is identical across the 11 images (same trick as step1_make_pairs.py).

Each image also gets a measured matched-filter S/N stored in its metadata ('snr_mf'):
    S/N = sqrt( sum over channels c and samples t of ( template[c,t] / sigma_c )^2 )
    template = the injected burst model x bandpass,  sigma_c = robust per-channel noise
    (1.4826 x MAD over time) measured on the background-only image.
This is the best-case (template-aware) S/N of that burst in that image. The CNN is not template-aware,
so its own 50 %-recall threshold will sit at a HIGHER S/N; the point is the curve, not the absolute value.

Note: the training range of the 'faint' bursts was 15-60 % of fluence (35 % of bursts) plus 100 % (65 %),
so scales below 0.15 are outside the training distribution. The eval script marks them.

Usage:
    python3 step2_make_sweep.py --catalog chimefrbcat2.csv --n 300 --outdir sim_sweep
"""
import argparse
import os

import numpy as np

import simulate_frbs_fitburst as sfm
from sim_common import FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ, load_catalog

SCALES = [0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.45, 0.70, 1.00]

_orig_calibrate = sfm.calibrate_amplitude
CUR = {"fluence": 1.0, "scale": 1.0}


def _calibrate_override(model_obj, params, target_fluence, tsamp_s):
    """Ignore the randomly drawn faint factor; use row fluence x the requested scale (0 -> no burst)."""
    if CUR["scale"] <= 0:
        return _orig_calibrate(model_obj, params, CUR["fluence"], tsamp_s) * 0.0
    return _orig_calibrate(model_obj, params, CUR["fluence"] * CUR["scale"], tsamp_s)


def robust_sigma_per_channel(img: np.ndarray) -> np.ndarray:
    med = np.median(img, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(img - med), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="chimefrbcat2.csv")
    ap.add_argument("--outdir", default="sim_sweep")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_catalog(args.catalog, verbose=True)
    freqs = np.linspace(FREQ_LO_MHZ, FREQ_HI_MHZ, NUM_FREQ)
    rows = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    sfm.calibrate_amplitude = _calibrate_override

    saved_rows, skipped = 0, 0
    for i, row in rows.iterrows():
        seed = 5_000_000 + args.seed * 1000 + i
        CUR["fluence"] = float(row["fluence"])
        try:
            CUR["scale"] = 0.0
            bg, *_ = sfm.simulate_one(row, freqs, np.random.default_rng(seed), False)
            sigma_c = robust_sigma_per_channel(bg)
            ok = sigma_c > 0
            outs = []
            for s in SCALES:
                CUR["scale"] = s
                noisy, model, times_s, params, bandpass, chan_ok, pos_frac, _fs, applied = sfm.simulate_one(
                    row, freqs, np.random.default_rng(seed), False)
                template = model * bandpass[:, None]
                snr = float(np.sqrt(np.sum((template[ok] / sigma_c[ok, None]) ** 2)))
                outs.append((s, noisy, snr, pos_frac, applied))
        except ValueError:
            skipped += 1
            continue
        if not (np.all(np.isfinite(bg)) and all(np.all(np.isfinite(o[1])) for o in outs)):
            skipped += 1
            continue
        name = str(row["tns_name"])
        meta0 = {"label": 0, "scale": 0.0, "snr_mf": 0.0, "row": name, "is_dedispersed": True}
        np.savez(os.path.join(args.outdir, f"{name}_s0.000.npz"), data_full=bg, metadata=meta0)
        for s, noisy, snr, pos_frac, applied in outs:
            meta = {"label": 1, "scale": float(s), "snr_mf": snr, "row": name, "burst_pos_frac": float(pos_frac),
                    "noise_components": applied, "is_dedispersed": True}
            np.savez(os.path.join(args.outdir, f"{name}_s{s:.3f}.npz"), data_full=noisy, metadata=meta)
        saved_rows += 1
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(rows)}")
    sfm.calibrate_amplitude = _orig_calibrate
    print(f"saved {saved_rows} bursts x {len(SCALES) + 1} images to {args.outdir}/  (skipped {skipped})")


if __name__ == "__main__":
    main()
