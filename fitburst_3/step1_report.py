#!/usr/bin/env python3
"""
step1_report.py -- aggregate the multi-seed runs.   (numpy / sklearn / matplotlib only)

Prints + saves:
  * per-seed table and mean +- std of accuracy / precision / recall / F1 / AUC
  * convergence check (did any seed hit the epoch cap?)
  * ERROR BREAKDOWN pooled over all seeds' test sets:
        - false-positive rate for RFI images, split by what the image contains
          (impulsive, narrowband, pulsar, storm, ... read from each .npz's
          metadata['noise_components'])
        - recall for FRB images, split by brightness (faint_scale) and by
          what else is in the image
Files written to <rundir>/: step1_summary.json, step1_summary.txt, step1_summary.png

Usage:  python3 step1_report.py --rundir runs/step1
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

NAVY, TEAL, ORANGE, RED, GREY = "#1f3a5f", "#1b998b", "#e07a1f", "#c0392b", "#8a94a0"
META_CACHE = {}


def get_meta(path):
    if path not in META_CACHE:
        try:
            m = np.load(path, allow_pickle=True)["metadata"].item()
        except Exception:
            m = {}
        META_CACHE[path] = m
    return META_CACHE[path]


def components(meta):
    return sorted({str(c).split(":")[0] for c in meta.get("noise_components", [])} - {"ambient"})


def faint_bucket(fs):
    if fs is None:
        return "unknown"
    if fs >= 0.999:
        return "full fluence"
    if fs >= 0.30:
        return "faint (30-60%)"
    return "very faint (15-30%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/step1")
    ap.add_argument("--min-n", type=int, default=15, help="hide breakdown rows with fewer images")
    args = ap.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.rundir, "seed_*")))
    runs = []
    for d in seed_dirs:
        mp, pp = os.path.join(d, "test_metrics.json"), os.path.join(d, "test_predictions.csv")
        if os.path.exists(mp) and os.path.exists(pp):
            m = json.load(open(mp))
            with open(pp) as f:
                rows = [(r["path"], int(r["label"]), float(r["prob"])) for r in csv.DictReader(f)]
            runs.append((m, rows))
    if not runs:
        raise SystemExit(f"no finished seeds found in {args.rundir}")

    out = []
    def P(s=""):
        print(s); out.append(s)

    P(f"MULTI-SEED RESULTS  ({len(runs)} seeds, each with its own split + initialisation)")
    P("-" * 78)
    P(f"{'seed':>5} {'acc':>7} {'prec':>7} {'recall':>7} {'F1':>7} {'AUC':>7} {'errors':>7} {'best ep/ran':>12}")
    keys = ["acc", "precision", "recall", "f1", "auc"]
    for m, _ in runs:
        P(f"{m['seed']:>5} {m['acc']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['f1']:7.4f} "
          f"{m['auc']:7.4f} {m['fp'] + m['fn']:>3}/{m['n_test']:<4} {m['best_epoch']:>5}/{m['epochs_ran']:<5}")
    summary = {}
    P("-" * 78)
    for k in keys:
        v = np.array([m[k] for m, _ in runs])
        summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "min": float(v.min()), "max": float(v.max())}
        P(f"{k:>10}:  {v.mean():.4f} +- {summary[k]['std']:.4f}    (min {v.min():.4f}, max {v.max():.4f})")
    capped = [m["seed"] for m, _ in runs if m.get("hit_epoch_cap")]
    P("")
    P("convergence: " + ("all seeds early-stopped before the epoch cap."
                          if not capped else f"seeds {capped} hit the epoch cap while still improving -> raise --epochs."))

    # ---------------------------------------------------------- breakdown
    neg_all, pos_all = [], []
    for _, rows in runs:
        for path, lab, prob in rows:
            (pos_all if lab == 1 else neg_all).append((path, prob, get_meta(path)))
    fp_by, tp_by = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])   # [errors, n]
    for _, prob, meta in neg_all:
        wrong = prob > 0.5
        comps = components(meta) or ["(plain / pure noise)"]
        for c in comps:
            fp_by[c][0] += wrong; fp_by[c][1] += 1
        fp_by["ALL RFI"][0] += wrong; fp_by["ALL RFI"][1] += 1
    for _, prob, meta in pos_all:
        wrong = prob <= 0.5
        b = faint_bucket(meta.get("faint_scale"))
        tp_by["brightness: " + b][0] += wrong; tp_by["brightness: " + b][1] += 1
        for c in components(meta):
            tp_by["also contains: " + c][0] += wrong; tp_by["also contains: " + c][1] += 1
        tp_by["ALL FRB"][0] += wrong; tp_by["ALL FRB"][1] += 1

    P("")
    P("ERROR BREAKDOWN (all seeds pooled)")
    P("-" * 78)
    P("False-positive rate for RFI images, by what the image contains (image can have several):")
    fp_rows = sorted([(c, e, n) for c, (e, n) in fp_by.items() if n >= args.min_n], key=lambda t: -t[1] / t[2])
    for c, e, n in fp_rows:
        P(f"   {c:34s} {e:5d} / {n:<6d} = {100 * e / n:5.1f}% wrongly called FRB")
    P("")
    P("Miss rate (false negatives) for FRB images:")
    fn_rows = sorted([(c, e, n) for c, (e, n) in tp_by.items() if n >= args.min_n], key=lambda t: -t[1] / t[2])
    for c, e, n in fn_rows:
        P(f"   {c:34s} {e:5d} / {n:<6d} = {100 * e / n:5.1f}% missed")

    summary["fp_by_component"] = {c: {"errors": int(e), "n": int(n)} for c, e, n in fp_rows}
    summary["fn_by_group"] = {c: {"errors": int(e), "n": int(n)} for c, e, n in fn_rows}
    json.dump(summary, open(os.path.join(args.rundir, "step1_summary.json"), "w"), indent=2)
    open(os.path.join(args.rundir, "step1_summary.txt"), "w").write("\n".join(out))

    # ------------------------------------------------------------ figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    names = ["acc", "f1", "auc"]; xs = np.arange(len(names))
    for i, k in enumerate(names):
        v = [m[k] for m, _ in runs]
        ax.scatter([i] * len(v), v, color=GREY, zorder=3, s=22)
        ax.errorbar(i, np.mean(v), yerr=np.std(v, ddof=1) if len(v) > 1 else 0, fmt="o", color=TEAL, capsize=6, ms=9, lw=2, zorder=4)
        ax.text(i + 0.12, np.mean(v), f"{np.mean(v):.3f}\n±{np.std(v, ddof=1) if len(v) > 1 else 0:.3f}", fontsize=8.5, va="center")
    ax.set_xticks(xs); ax.set_xticklabels(["accuracy", "F1", "ROC AUC"]); ax.set_xlim(-.5, 2.8)
    ax.set_title(f"{len(runs)} seeds: each grey dot = one seed\n(teal = mean ± std)", fontsize=10, color=NAVY)
    ax = axs[1]
    rows = [r for r in fp_rows if r[0] != "ALL RFI"][:9][::-1]
    ax.barh([r[0] for r in rows], [100 * r[1] / r[2] for r in rows], color=ORANGE)
    all_fp = fp_by["ALL RFI"]; ax.axvline(100 * all_fp[0] / max(all_fp[1], 1), color=NAVY, ls="--", lw=1)
    ax.set_xlabel("% of RFI images wrongly called FRB"); ax.set_title("False positives by RFI content\n(dashed = overall)", fontsize=10, color=NAVY)
    ax.tick_params(axis="y", labelsize=8)
    ax = axs[2]
    order = ["brightness: full fluence", "brightness: faint (30-60%)", "brightness: very faint (15-30%)"]
    rows = [(c.replace("brightness: ", ""), tp_by[c]) for c in order if tp_by[c][1] >= 1]
    ax.bar([r[0] for r in rows], [100 * (1 - r[1][0] / r[1][1]) for r in rows], color=TEAL)
    ax.set_ylim(0, 105); ax.set_ylabel("recall (% of FRBs found)"); ax.set_title("Recall vs burst brightness", fontsize=10, color=NAVY)
    for i, r in enumerate(rows):
        ax.text(i, 100 * (1 - r[1][0] / r[1][1]) + 1.5, f"{100 * (1 - r[1][0] / r[1][1]):.0f}%\n(n={r[1][1]})", ha="center", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8.5)
    for a in axs:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(args.rundir, "step1_summary.png"), dpi=150)
    print(f"\nsaved: {args.rundir}/step1_summary.png, step1_summary.txt, step1_summary.json")


if __name__ == "__main__":
    main()
