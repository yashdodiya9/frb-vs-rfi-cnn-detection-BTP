#!/usr/bin/env python3
"""
step2_eval_sweep.py -- recall vs signal-to-noise for every trained model.

Scores the images made by step2_make_sweep.py with all checkpoints in the given run folders
(default: Step 1 models = trained on the old impulsive-RFI data, Step 2 models = trained on the new data) and reports
  * recall vs measured matched-filter S/N (binned + logistic fit)   -> S/N at 50 % and 90 % recall
  * recall vs brightness scale (fraction of catalog fluence)
  * false-positive rate on the burst-free copies of the same backgrounds (scale 0)

Usage:
  python3 step2_eval_sweep.py --sweep-dir sim_sweep --groups "Step 1 models=runs/step1" "Step 2 models=runs/step2"
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

TRAIN_MIN_FAINT = 0.15          # faint bursts in training were 15-60 % of fluence (35 % of bursts), the rest 100 %
SNR_EDGES = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 14, 20, 30, 50, 100, 300, 1e4])


# ------------------------------------------------------------------ pure numpy analysis
def logistic_snr(snr, hit, target):
    """S/N at which the fitted recall curve crosses `target` (logistic fit in log S/N). nan if undefined."""
    snr, hit = np.asarray(snr, float), np.asarray(hit, int)
    m = snr > 0
    x, y = np.log(snr[m]).reshape(-1, 1), hit[m]
    if y.min() == y.max():
        return float("nan")
    lr = LogisticRegression(C=1e4, max_iter=2000).fit(x, y)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    if b <= 0:
        return float("nan")
    return float(np.exp((np.log(target / (1 - target)) - a) / b))


def binned_recall(snr, hit, edges=SNR_EDGES, min_n=15):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (snr >= lo) & (snr < hi)
        out.append((np.sqrt(lo * hi), float(hit[m].mean()), int(m.sum())) if m.sum() >= min_n else (np.sqrt(lo * hi), np.nan, int(m.sum())))
    return out


def analyze(snr, scale, probs):
    """snr, scale: arrays for ALL sweep images (scale 0 = burst-free). probs: model P(FRB)."""
    pred = np.asarray(probs) > 0.5
    burst = scale > 0
    r = {"fp_rate_scale0": float(pred[~burst].mean()),
         "snr50": logistic_snr(snr[burst], pred[burst], 0.5), "snr90": logistic_snr(snr[burst], pred[burst], 0.9),
         "recall_by_scale": {float(s): float(pred[scale == s].mean()) for s in np.unique(scale[burst])},
         "binned": binned_recall(snr[burst], pred[burst])}
    return r


# ------------------------------------------------------------------ scoring (needs torch)
def score_group(root, arrays, device):
    import torch
    from step1_lib import FRBCNN, predict_arrays
    out = {}
    for ck in sorted(glob.glob(os.path.join(root, "seed_*", "best_model.pt"))):
        m = FRBCNN().to(device)
        m.load_state_dict(torch.load(ck, map_location=device)["model_state"])
        out[os.path.basename(os.path.dirname(ck))] = predict_arrays(m, arrays, device)
    return out


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty S/N bins are expected
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="sim_sweep")
    ap.add_argument("--groups", nargs="+", default=["Step 1 models=runs/step1", "Step 2 models=runs/step2"])
    ap.add_argument("--out", default="runs/step2_sweep")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from step1_lib import pick_device, preprocess_array

    files = sorted(glob.glob(os.path.join(args.sweep_dir, "*.npz")))
    if not files:
        raise SystemExit("no sweep images found -- run step2_make_sweep.py first")
    arrays, snr, scale = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = z["metadata"].item()
        arrays.append(preprocess_array(z["data_full"])); snr.append(meta["snr_mf"]); scale.append(meta["scale"])
    snr, scale = np.array(snr), np.array(scale)
    print(f"{len(files)} sweep images ({int((scale == 0).sum())} burst-free); measured S/N of bursts: "
          f"5th/50th/95th percentile = {np.percentile(snr[scale > 0], 5):.1f} / {np.percentile(snr[scale > 0], 50):.1f} / {np.percentile(snr[scale > 0], 95):.1f}")

    device = pick_device(args.force_cpu)
    results = {}
    for spec in args.groups:
        name, root = spec.split("=", 1)
        probs = score_group(root, arrays, device)
        if not probs:
            print(f"warning: no checkpoints under {root}, skipping {name}"); continue
        results[name] = {seed: analyze(snr, scale, p) for seed, p in probs.items()}

    # ------------------------------- table
    print(f"\n{'group / seed':<22} {'S/N @50%':>9} {'S/N @90%':>9} {'FP(no burst)':>13} {'recall@0.07':>12} {'@0.15':>7} {'@0.30':>7} {'@1.00':>7}")
    for name, res in results.items():
        for seed, r in res.items():
            rs = r["recall_by_scale"]
            print(f"{name + ' ' + seed:<22} {r['snr50']:9.2f} {r['snr90']:9.2f} {r['fp_rate_scale0']:13.3f} "
                  f"{rs.get(0.07, np.nan):12.3f} {rs.get(0.15, np.nan):7.3f} {rs.get(0.3, np.nan):7.3f} {rs.get(1.0, np.nan):7.3f}")
        mean = lambda k: np.nanmean([r[k] for r in res.values()])
        std = lambda k: np.nanstd([r[k] for r in res.values()], ddof=1) if len(res) > 1 else 0.0
        print(f"{name + ' MEAN':<22} {mean('snr50'):9.2f} {mean('snr90'):9.2f} {mean('fp_rate_scale0'):13.3f}   "
              f"(S/N@50% std {std('snr50'):.2f}, @90% std {std('snr90'):.2f})")
        print("-" * 98)

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "sweep_results.json"), "w"), indent=2, default=float)

    # ------------------------------- plot
    colors = ["#c0392b", "#1b998b", "#1f3a5f", "#e07a1f"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for (name, res), col in zip(results.items(), colors):
        edges_mid = np.array([b[0] for b in next(iter(res.values()))["binned"]])
        curves = np.array([[b[1] for b in r["binned"]] for r in res.values()])
        ok = ~np.all(np.isnan(curves), axis=0)
        axs[0].plot(edges_mid[ok], np.nanmean(curves, 0)[ok], "o-", color=col, label=name)
        axs[0].fill_between(edges_mid[ok], np.nanmin(curves, 0)[ok], np.nanmax(curves, 0)[ok], color=col, alpha=0.18)
        s50 = np.nanmean([r["snr50"] for r in res.values()])
        if np.isfinite(s50):
            axs[0].axvline(s50, color=col, ls="--", lw=1)
            axs[0].text(s50, 0.04, f" S/N50={s50:.1f}", color=col, fontsize=8.5, rotation=90, va="bottom")
        scales = sorted(next(iter(res.values()))["recall_by_scale"])
        sc_curves = np.array([[r["recall_by_scale"][s] for s in scales] for r in res.values()])
        axs[1].plot(scales, sc_curves.mean(0), "o-", color=col, label=name)
        axs[1].fill_between(scales, sc_curves.min(0), sc_curves.max(0), color=col, alpha=0.18)
    axs[0].set_xscale("log"); axs[0].set_xlim(0.7, max(30.0, float(np.percentile(snr[scale > 0], 98)))); axs[0].set_ylim(-0.02, 1.04); axs[0].axhline(0.5, color="#8a94a0", lw=0.8, ls=":")
    axs[0].set_xlabel("measured matched-filter S/N of the burst"); axs[0].set_ylabel("recall (fraction of bursts found)")
    axs[0].set_title("Recall vs S/N (band = min/max over seeds)", fontsize=10.5, color="#1f3a5f"); axs[0].legend(fontsize=9, loc="lower right")
    axs[1].set_xscale("log"); axs[1].set_ylim(-0.02, 1.04); axs[1].axvspan(0.015, TRAIN_MIN_FAINT, color="#8a94a0", alpha=0.15)
    axs[1].text(0.0165, 0.1, "below training\nrange", fontsize=8.5, color="#555")
    axs[1].set_xlabel("brightness: fraction of catalog fluence"); axs[1].set_ylabel("recall")
    axs[1].set_title("Recall vs brightness of the SAME bursts", fontsize=10.5, color="#1f3a5f")
    for a in axs:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "sweep_recall_vs_snr.png"), dpi=150)
    print(f"saved {args.out}/sweep_results.json and sweep_recall_vs_snr.png")


if __name__ == "__main__":
    main()
