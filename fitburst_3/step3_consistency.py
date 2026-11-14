#!/usr/bin/env python3
"""
step3_consistency.py -- is the burst actually visible in its own waterfall?  (independent of the CNN)

For every event we correlate the channel-averaged data with CHIME's fitted-model time profile (matched filter):
    S/N_mf = sum_t (series_t - trend_t) * p_t / ( sigma * sqrt(sum_t p_t^2) )
(trend and sigma from the burst-free samples; for the ~2 % of files without a model, p_t is a boxcar over
pulse_emission_region). A burst that CHIME detected but that is absent from the supplied waterfall gets S/N_mf ~ 0
(e.g. FRB20190208B: catalog S/N 96, S/N_mf -0.1).

Then recall of the CNN ensemble (mean P(FRB) > 0.5, from real_scores.csv) is tabulated by S/N_mf, for ALL bins:
nothing is cut using the CNN's output, so the table can be quoted as it is.

Usage:
    python step3_consistency.py --cache ~/Desktop/using-h5-files/real_cache.npz --scores runs/step3_real_peak/real_scores.csv
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

from step3_real_lib import burst_zone, load_cache

EDGES = [-np.inf, 2, 4, 6, 10, 20, np.inf]


def mf_snr(img, profile):
    t = img.shape[1]
    live = (img != 0).any(axis=1)
    if live.sum() < 8:
        return np.nan
    s = img[live].mean(axis=0).astype(float)
    p = np.where(np.isfinite(profile), profile, 0.0).astype(float)
    if p.max() <= 0 or len(p) != t:
        return np.nan
    zs, ze = burst_zone(p, margin=10)
    off = np.ones(t, bool); off[zs:ze + 1] = False
    if off.sum() < 20:                                  # window too short to leave burst-free samples: use samples where p is ~0
        off = p < 0.01 * p.max()
        if off.sum() < 10:
            return np.nan
    x = np.arange(t)
    coef = np.polyfit(x[off], s[off], 1)               # linear trend (slow baseline drifts)
    r = s - np.polyval(coef, x)
    sig = 1.4826 * np.median(np.abs(r[off] - np.median(r[off])))
    return float((r * p).sum() / (max(sig, 1e-12) * np.sqrt((p ** 2).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="real_cache.npz")
    ap.add_argument("--scores", default="runs/step3_real_peak/real_scores.csv")
    ap.add_argument("--catalog", default="chimefrbcat2.csv")
    ap.add_argument("--out", default="runs/step3_real_peak/consistency.csv")
    args = ap.parse_args()

    c = load_cache(args.cache)
    X, prof, meta = c["X"], c["model_prof"], pd.DataFrame(list(c["meta"]))
    cat = pd.read_csv(args.catalog)
    cat = cat[cat.sub_num == 0].drop_duplicates("tns_name").set_index("tns_name")
    meta["mf"] = [mf_snr(X[i], prof[i]) for i in range(len(meta))]
    meta["bonsai"] = meta.tns_name.map(cat["bonsai_snr"])
    meta["has_params"] = meta.get("has_params", True)
    print(f"{len(meta)} events | matched-filter S/N of the burst in its own waterfall: median {meta.mf.median():.1f}, "
          f"5th pct {meta.mf.quantile(.05):.1f}, 95th pct {meta.mf.quantile(.95):.1f}")
    print(f"events with S/N_mf < 3 (burst essentially invisible in the supplied waterfall): {int((meta.mf < 3).sum())} "
          f"({100 * (meta.mf < 3).mean():.1f}%); of those, without a CHIME fit: {int(((meta.mf < 3) & ~meta.has_params.astype(bool)).sum())}")

    sc = pd.read_csv(args.scores)
    pos = sc[sc.label == 1].merge(meta[["tns_name", "mf", "bonsai", "has_params"]], on="tns_name", how="left", suffixes=("", "_m"))
    pos["hit"] = pos.mean_prob > 0.5
    pos["bin"] = pd.cut(pos.mf, EDGES)
    print(f"\nRecall of the CNN ensemble on the {len(pos)} scored real bursts, by how visible the burst is (S/N_mf):")
    print(f"   {'S/N_mf bin':>14} {'n':>5} {'recall':>8} {'median catalog S/N':>20}")
    for b, g in pos.groupby("bin", observed=True):
        print(f"   {str(b):>14} {len(g):>5} {g.hit.mean():>8.3f} {g.bonsai.median():>20.1f}")
    print(f"   {'all':>14} {len(pos):>5} {pos.hit.mean():>8.3f}")
    vis = pos[pos.mf >= 4]
    print(f"\nrecall among events whose burst is visible (S/N_mf >= 4): {vis.hit.mean():.3f} (n={len(vis)}); "
          f"misses among them: {int((~vis.hit).sum())}")
    miss = pos[~pos.hit].sort_values("mf")
    print(f"\n{len(miss)} missed bursts; {int((miss.mf < 4).sum())} of them have S/N_mf < 4. The 10 most-visible misses:")
    for _, r in miss.sort_values("mf", ascending=False).head(10).iterrows():
        print(f"   {r.tns_name:14s} S/N_mf={r.mf:6.1f} catalog S/N={r.bonsai:5.1f} window={int(r.window):3d} has_fit={bool(r.has_params)}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta[["tns_name", "mf", "bonsai", "has_params"]].to_csv(args.out, index=False)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
