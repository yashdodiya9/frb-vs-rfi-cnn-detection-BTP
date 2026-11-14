#!/usr/bin/env python3
"""
step3_show_errors.py -- look at the mistakes on real data, and quantify the 10-model ensemble.

Run AFTER step3_eval_real.py (it reads runs/step3_real/real_scores.csv) with the SAME --max-events and --seed,
so the very same crops are rebuilt (alignment with the CSV is asserted).

Shows
  top row    : real burst-free windows the ensemble is most sure are FRBs     ("false alarms")
  bottom row : real bursts the ensemble is most sure are NOT FRBs               ("misses")
Prints
  * ensemble ROC AUC on real data, recall at P>0.5, false-alarm rate at P>0.5,
    and the recall you would get if the threshold were set for 1 % / 5 % false alarms on real backgrounds
  * whether false alarms come disproportionately from REPEATING sources (other bursts may sit in the window)
  * the listed events, so you can open them in the catalog

Usage:
    python step3_show_errors.py --cache ~/Desktop/using-h5-files/real_cache.npz --catalog chimefrbcat2.csv
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

from step3_real_lib import build_crops, load_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="real_cache.npz")
    ap.add_argument("--catalog", default="chimefrbcat2.csv")
    ap.add_argument("--scores", default="runs/step3_real/real_scores.csv")
    ap.add_argument("--max-events", type=int, default=1500, help="must equal the value used in step3_eval_real.py")
    ap.add_argument("--seed", type=int, default=0, help="must equal the value used in step3_eval_real.py")
    ap.add_argument("--center", choices=["peak", "middle"], default="peak", help="must equal the value used in step3_eval_real.py")
    ap.add_argument("--hot-k", type=float, default=None, help="must equal the value used in step3_eval_real.py")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mode", choices=["interp", "area"], default="area")
    ap.add_argument("--out", default="runs/step3_real/real_errors.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from step1_lib import preprocess_array

    cache = load_cache(args.cache)
    X, prof, meta = cache["X"], cache["model_prof"], pd.DataFrame(list(cache["meta"]))
    cat = pd.read_csv(args.catalog)
    cat = cat[cat.sub_num == 0].drop_duplicates("tns_name").set_index("tns_name")
    meta["snr"] = meta.tns_name.map(cat["bonsai_snr"])
    rep = cat["repeater_name"].map(lambda v: isinstance(v, str) and v.strip() not in ("", "-9999"))
    meta["repeater"] = meta.tns_name.map(rep).fillna(False).astype(bool)

    # rebuild the SAME crops in the SAME order as step3_eval_real.py
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(meta))[:args.max_events]
    raws, y, ev, win, shift = build_crops(X, prof, meta.width_s.values, meta.scat_s.values, idx, rng, args.center, args.hot_k)

    rows = list(csv.DictReader(open(args.scores)))
    assert len(rows) == len(y), f"score file has {len(rows)} rows but {len(y)} crops were rebuilt: use the same --max-events/--seed"
    for k, r in enumerate(rows):
        assert int(r["label"]) == y[k] and r["tns_name"] == meta.tns_name.iloc[ev[k]], f"row {k} does not match: different --seed/--max-events?"
    p = np.array([float(r["mean_prob"]) for r in rows])

    # ------------------------------------------------------------ ensemble statistics
    from sklearn.metrics import roc_auc_score
    pos, neg = p[y == 1], p[y == 0]
    print(f"{len(pos)} real bursts, {len(neg)} real burst-free windows; scores are the mean P(FRB) over all checkpoints")
    print(f"ROC AUC on real data (burst vs real background): {roc_auc_score(y, p):.4f}")
    print(f"at P>0.5: recall {np.mean(pos > 0.5):.3f}, false-alarm rate {np.mean(neg > 0.5):.3f} ({int((neg > 0.5).sum())} of {len(neg)} windows)")
    for fpr in (0.05, 0.01):
        thr = np.quantile(neg, 1 - fpr)
        print(f"threshold chosen for {fpr:.0%} false alarms on real backgrounds: P>{thr:.3f} -> recall {np.mean(pos > thr):.3f}")
    fa = (y == 0) & (p > 0.5)
    rep_all = meta.repeater.values[ev[y == 0]].mean()
    rep_fa = meta.repeater.values[ev[fa]].mean() if fa.any() else float("nan")
    print(f"share of REPEATER sources: among all real backgrounds {rep_all:.1%}, among the false alarms {rep_fa:.1%} "
          f"(a big excess would mean other bursts from the same source sit in those windows)")

    # ------------------------------------------------------------ gallery
    fa_idx = np.where(y == 0)[0][np.argsort(-neg)[:args.n]]
    miss_idx = np.where(y == 1)[0][np.argsort(pos)[:args.n]]
    fig, axs = plt.subplots(2, args.n, figsize=(2.1 * args.n, 5.2))
    def draw(ax, k, title, col):
        img = preprocess_array(raws[k], args.mode)
        nz = img[img != 0]
        v = np.percentile(nz, [1, 99.7]) if nz.size else (0, 1)
        ax.imshow(img, aspect="auto", origin="lower", vmin=v[0], vmax=v[1], cmap="viridis")
        ax.set_title(title, fontsize=7.5, color=col); ax.axis("off")
    print("\nTop false alarms (real burst-free windows):")
    for j, k in enumerate(fa_idx):
        m = meta.iloc[ev[k]]
        print(f"  {m.tns_name:14s} P={p[k]:.2f} window={raws[k].shape[1]:3d}  repeater={bool(m.repeater)}")
        draw(axs[0, j], k, f"{m.tns_name}\nP={p[k]:.2f}", "#c0392b")
    print("\nMost-missed real bursts:")
    for j, k in enumerate(miss_idx):
        m = meta.iloc[ev[k]]
        print(f"  {m.tns_name:14s} P={p[k]:.2f} S/N={m.snr:.1f} window={raws[k].shape[1]:3d} DM={m.dm:.0f} repeater={bool(m.repeater)}")
        draw(axs[1, j], k, f"{m.tns_name}\nS/N {m.snr:.0f}, P={p[k]:.2f}", "#1f3a5f")
    fig.suptitle("Real data: highest-scoring burst-free windows (top) and lowest-scoring real bursts (bottom), as the CNN sees them",
                 fontsize=10.5, fontweight="bold", color="#1f3a5f")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
