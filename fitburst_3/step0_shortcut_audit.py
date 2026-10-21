#!/usr/bin/env python3
"""
step0_shortcut_audit.py
=======================

Burst-free shortcut audit. Run it on the generated dataset BEFORE training.

Idea
----
In dedispersed images the burst lives in the middle of the window
(burst_pos_frac in 0.35-0.65, extent ~ +-10% of the window). So the FIRST 20%
and LAST 20% of the time axis contain no burst, only the background.
If a plain classifier can tell the FRB-class files from the RFI-class files
using ONLY those edge columns, then the two classes differ in something
other than the burst  ->  a shortcut a CNN can (and will) exploit.

PASS  : edge-only cross-validated ROC-AUC close to 0.5 (< 0.58)
FAIL  : AUC > 0.65, and the feature importances tell you which generator
        step is leaking (quantization, dropout, banding, window size ...).

Usage
-----
    python3 step0_shortcut_audit.py --frb-dir sim_output --rfi-dir sim_output_rfi
    python3 step0_shortcut_audit.py --frb-dir sim_output --rfi-dir sim_output_rfi --max-n 1500

Needs: numpy, scikit-learn.  (No torch.)
"""
import argparse
import glob
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

EDGE = 0.20  # fraction of columns kept at each end of the time axis

FEATURE_NAMES = [
    "num_time",
    "n_masked_chan", "longest_masked_band", "frac_zero_pix_in_good_chan",
    "n_unique_ratio", "lattice_score",
    "colmed_std", "colmed_max_jump", "colmed_trend",
    "chan_med_std", "chan_mad_std", "chan_mad_frac_hi", "mad_med_corr",
    "pix_kurtosis", "pix_skew", "p99_over_mad", "n_hot_pix",
]


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def edge_features(arr: np.ndarray) -> list:
    """Burst-free features computed from edge columns only."""
    nt = arr.shape[1]
    k = max(4, int(EDGE * nt))
    e = np.concatenate([arr[:, :k], arr[:, -k:]], axis=1)

    chan_all_zero = (arr == 0).all(axis=1)
    n_masked = int(chan_all_zero.sum())
    longest = _longest_run(chan_all_zero)

    good = e[~chan_all_zero]
    if good.size == 0:
        return [nt, n_masked, longest] + [0.0] * (len(FEATURE_NAMES) - 3)

    frac_zero = float((good == 0).mean())

    nz = good[good != 0]
    if nz.size < 50:
        return [nt, n_masked, longest, frac_zero] + [0.0] * (len(FEATURE_NAMES) - 4)

    # quantization: few distinct values, and values sitting on a regular lattice
    sub = nz.ravel()[:20000]
    n_unique_ratio = len(np.unique(np.round(sub, 6))) / len(sub)
    d = np.diff(np.sort(np.unique(np.round(sub, 6))))
    d = d[d > 1e-9]
    lattice = float(np.median(d) / (d.std() + 1e-12)) if d.size > 5 else 0.0
    lattice = min(lattice, 1e3)

    med = np.median(nz)
    mad = np.median(np.abs(nz - med)) * 1.4826 + 1e-9

    # time-gain structure (per-column median over good channels)
    gm = np.where(good == 0, np.nan, good)
    colmed = np.nanmedian(gm, axis=0)
    colmed = colmed[np.isfinite(colmed)]
    colmed_std = float(colmed.std() / mad)
    colmed_jump = float(np.abs(np.diff(colmed)).max() / mad) if colmed.size > 1 else 0.0
    half = len(colmed) // 2
    colmed_trend = float((colmed[half:].mean() - colmed[:half].mean()) / mad) if half > 0 else 0.0

    # per-channel banding / narrowband-bright structure
    chan_med = np.nanmedian(gm, axis=1)
    chan_mad = np.nanmedian(np.abs(gm - chan_med[:, None]), axis=1) * 1.4826
    chan_med = chan_med[np.isfinite(chan_med)]
    chan_mad = chan_mad[np.isfinite(chan_mad)]
    chan_med_std = float(chan_med.std() / mad)
    chan_mad_std = float(chan_mad.std() / (np.median(chan_mad) + 1e-9))
    chan_mad_hi = float((chan_mad > 3 * np.median(chan_mad)).mean())
    mad_med_corr = float(np.corrcoef(chan_med, chan_mad)[0, 1]) if chan_med.size == chan_mad.size and chan_med.size > 3 else 0.0

    z = (nz - med) / mad
    kurt = float(np.mean(z ** 4))
    skew = float(np.mean(z ** 3))
    p99 = float(np.percentile(z, 99.9))
    n_hot = float((z > 6).mean())

    return [nt, n_masked, longest, frac_zero, n_unique_ratio, lattice,
            colmed_std, colmed_jump, colmed_trend,
            chan_med_std, chan_mad_std, chan_mad_hi, np.nan_to_num(mad_med_corr),
            kurt, skew, p99, n_hot]


def load_dir(d, max_n, seed):
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    rng = np.random.default_rng(seed)
    if len(files) > max_n:
        files = list(rng.choice(files, size=max_n, replace=False))
    X = []
    for f in files:
        npz = np.load(f, allow_pickle=True)
        meta = npz["metadata"].item()
        if not meta.get("is_dedispersed", True):
            raise SystemExit("This audit assumes dedispersed images (burst in the middle of the window).")
        X.append(edge_features(npz["data_full"].astype(np.float64)))
    return np.nan_to_num(np.array(X, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frb-dir", default="sim_output")
    ap.add_argument("--rfi-dir", default="sim_output_rfi")
    ap.add_argument("--max-n", type=int, default=1500, help="max files per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xp = load_dir(args.frb_dir, args.max_n, args.seed)
    Xn = load_dir(args.rfi_dir, args.max_n, args.seed)
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    print(f"audit set: {len(Xp)} FRB-class, {len(Xn)} RFI-class files; "
          f"{len(FEATURE_NAMES)} burst-free edge features")

    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.08, random_state=args.seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)

    rf = RandomForestClassifier(300, max_depth=8, random_state=args.seed, n_jobs=-1).fit(X, y)
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])

    print(f"\nedge-only 5-fold CV ROC-AUC = {auc:.3f}   (0.50 = no shortcut)")
    verdict = "PASS" if auc < 0.58 else ("WARN" if auc < 0.65 else "FAIL")
    print(f"verdict: {verdict}")
    print("\ntop leaking features (random-forest importance):")
    for name, v in imp[:6]:
        print(f"  {name:28s} {v:.3f}   FRB-mean={Xp[:, FEATURE_NAMES.index(name)].mean():.4g}  "
              f"RFI-mean={Xn[:, FEATURE_NAMES.index(name)].mean():.4g}")


if __name__ == "__main__":
    main()
