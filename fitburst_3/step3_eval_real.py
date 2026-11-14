#!/usr/bin/env python3
"""
step3_eval_real.py -- your trained models on REAL CHIME/FRB Catalog-2 bursts, WITHOUT retraining.

Positives : real detected bursts (centred crop of the length the simulator's window rule gives each burst)
Negatives : burst-free windows of the same files = REAL telescope background (not RFI candidates!)

Reports, per model group (mean over seeds, min-max in brackets):
  * recall on real bursts overall, and vs the catalog detection S/N (bonsai_snr)
  * false-alarm rate on real burst-free windows
  * recall by window length
Every checkpoint is scored with the resize mode it was trained with (interp / area).

Caveats to quote with the result
  * The catalog only contains bursts CHIME DETECTED, so the sample is biased to S/N >~ 9; there are no real
    undetected faint bursts and no labelled real-RFI false alarms in these files.
  * bonsai_snr is the real-time pipeline's S/N, not the matched-filter S/N of the simulated sweep:
    the two axes are not interchangeable.
  * Real windows are fixed per file (40-162+ samples); where the file's window is shorter than the simulator's rule gives, the whole window is used.

Usage:
    python step3_eval_real.py --cache real_cache.npz --catalog chimefrbcat2.csv \
        --groups "Step 2 models (interp)=runs/step2" "Area-averaging models=runs/step2d_area" --out runs/step3_real
"""
import argparse
import csv
import json
import os
import warnings

import numpy as np
import pandas as pd

from step2_eval_sweep import score_group
from step3_real_lib import build_crops, load_cache, rule_length

SNR_EDGES = [0, 10, 12, 15, 20, 30, 50, 100, 1e9]
WIN_EDGES = [0, 64, 100, 130, 163]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - r, c + r


def binned(values, hit, edges, min_n=15):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi)
        n = int(m.sum())
        out.append({"lo": lo, "hi": hi, "n": n, "recall": float(hit[m].mean()) if n >= min_n else float("nan")})
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="real_cache.npz")
    ap.add_argument("--catalog", default="chimefrbcat2.csv")
    ap.add_argument("--groups", nargs="+", default=["Step 2 models (interp)=runs/step2", "Area-averaging models=runs/step2d_area"])
    ap.add_argument("--max-events", type=int, default=1500, help="events to score (memory: ~0.8 GB per 3000 images)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/step3_real")
    ap.add_argument("--center", choices=["peak", "middle"], default="peak",
                    help="crop around the brightest burst component (peak) or the file centre (middle = first version)")
    ap.add_argument("--hot-k", type=float, default=None, help="EXPLORATORY: zero channels with scatter > k x median (e.g. 4)")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from step1_lib import RawScorer, pick_device

    cache = load_cache(args.cache)
    X, prof, meta = cache["X"], cache["model_prof"], pd.DataFrame(list(cache["meta"]))
    cat = pd.read_csv(args.catalog)
    cat = cat[cat.sub_num == 0].drop_duplicates("tns_name").set_index("tns_name")
    meta["bonsai_snr"] = meta.tns_name.map(cat["bonsai_snr"])
    meta["snr_fitb"] = meta.tns_name.map(cat["snr_fitb"])
    print(f"{len(meta)} real bursts in cache; {int(meta.bonsai_snr.notna().sum())} matched to the catalog "
          f"(bonsai S/N: median {meta.bonsai_snr.median():.1f}, 5th-95th pct {meta.bonsai_snr.quantile(.05):.1f}-{meta.bonsai_snr.quantile(.95):.1f})")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(meta))[:args.max_events]
    raws, y, ev, win, shift = build_crops(X, prof, meta.width_s.values, meta.scat_s.values, idx, rng, args.center, args.hot_k)
    pos_win = win[y == 1]
    off = np.abs(shift[y == 1]) > np.maximum(pos_win // 2, 1)
    print(f"crop centre = {args.center}; {100 * off.mean():.1f}% of bursts have their brightest component more than half a crop away "
          f"from the file centre (those are cropped on empty sky if --center middle)" + (f"; hot-channel masking k={args.hot_k}" if args.hot_k else ""))
    t_file = np.array([X[i].shape[1] for i in idx]); w_rule = np.array([rule_length(meta.width_s.iloc[i], meta.scat_s.iloc[i]) for i in idx])
    print(f"window in file: median {int(np.median(t_file))} samples; for {100 * np.mean(w_rule > t_file):.0f}% of events the file window is SHORTER than "
          f"the simulator rule would give (they are used whole); negatives possible for {int(len(set(ev[y == 0])))} of {len(idx)} events")
    print(f"scoring {int(y.sum())} real bursts and {int((1 - y).sum())} burst-free windows "
          f"(window length median: bursts {int(np.median(win[y == 1]))}, backgrounds {int(np.median(win[y == 0]))} samples)")

    device = pick_device(args.force_cpu)
    scorer = RawScorer(raws)
    snr = meta.bonsai_snr.values[ev]
    results, pooled = {}, []
    for spec in args.groups:
        gname, root = spec.split("=", 1)
        probs = score_group(root, scorer, device)
        if not probs:
            print(f"warning: no checkpoints under {root}"); continue
        per_seed = {}
        for seed, p in probs.items():
            hit = p > 0.5
            fit = np.array([bool(meta.has_params.iloc[k]) if "has_params" in meta else True for k in ev])
            pos_fit, pos_nofit = (y == 1) & fit, (y == 1) & ~fit
            per_seed[seed] = {"recall": float(hit[y == 1].mean()), "fp_rate": float(hit[y == 0].mean()),
                              "recall_fit": float(hit[pos_fit].mean()) if pos_fit.any() else float("nan"),
                              "recall_nofit": float(hit[pos_nofit].mean()) if pos_nofit.any() else float("nan"),
                              "n_fit": int(pos_fit.sum()), "n_nofit": int(pos_nofit.sum()),
                              "snr_bins": binned(snr[y == 1], hit[y == 1], SNR_EDGES),
                              "win_bins": binned(win[y == 1], hit[y == 1], WIN_EDGES)}
        results[gname] = per_seed
        pooled.append(np.mean([p for p in probs.values()], axis=0))

    # ---------------------------------------------------------------- report
    out_lines = []
    def P(s=""):
        print(s); out_lines.append(s)
    P(f"\n{'group':<28} {'recall on real bursts':>26} {'false alarms on real backgrounds':>34}")
    for g, ps in results.items():
        r = [v["recall"] for v in ps.values()]; f = [v["fp_rate"] for v in ps.values()]
        P(f"{g:<28} {np.mean(r):>10.3f} [{min(r):.3f}-{max(r):.3f}] {np.mean(f):>18.3f} [{min(f):.3f}-{max(f):.3f}]")
    P("\nRecall on real bursts vs catalog detection S/N (bonsai_snr); mean over seeds:")
    P(f"   {'S/N bin':>12} {'n':>5} " + "".join(f"{g[:26]:>30}" for g in results))
    first = next(iter(results.values()))
    nb = len(next(iter(first.values()))["snr_bins"])
    for b in range(nb):
        row0 = next(iter(first.values()))["snr_bins"][b]
        vals = [np.nanmean([v["snr_bins"][b]["recall"] for v in ps.values()]) for ps in results.values()]
        label = f"{row0['lo']:g}-{row0['hi']:g}" if row0["hi"] < 1e8 else f">{row0['lo']:g}"
        P(f"   {label:>12} {row0['n']:>5} " + "".join(f"{v:>30.3f}" for v in vals))
    P("\nRecall by window length (samples), mean over seeds:")
    for b in range(len(next(iter(first.values()))["win_bins"])):
        row0 = next(iter(first.values()))["win_bins"][b]
        vals = [np.nanmean([v["win_bins"][b]["recall"] for v in ps.values()]) for ps in results.values()]
        P(f"   {row0['lo']:g}-{row0['hi']:g}".ljust(16) + f"{row0['n']:>5} " + "".join(f"{v:>30.3f}" for v in vals))
    P("\nRecall by CHIME fit status (bursts whose own fit failed have no parameters; their window falls back to 64 samples):")
    for g, ps in results.items():
        v = list(ps.values())
        P(f"   {g:<28} with fit (n={v[0]['n_fit']}): {np.nanmean([s['recall_fit'] for s in v]):.3f}    "
          f"without fit (n={v[0]['n_nofit']}): {np.nanmean([s['recall_nofit'] for s in v]):.3f}")
    P("\nCaveats: detected bursts only (biased to S/N >~ 9); negatives are real backgrounds, not real RFI candidates; "
      "bonsai S/N is not the simulated matched-filter S/N; real windows are shorter than the simulator's for some bursts.")

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "real_results.json"), "w"), indent=2, default=float)
    open(os.path.join(args.out, "real_results.txt"), "w").write("\n".join(out_lines))
    mean_p = np.mean(pooled, axis=0)
    with open(os.path.join(args.out, "real_scores.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tns_name", "label", "window", "bonsai_snr", "mean_prob", "has_params"])
        for k in range(len(y)):
            w.writerow([meta.tns_name.iloc[ev[k]], int(y[k]), int(win[k]), snr[k], f"{mean_p[k]:.5f}",
                        int(bool(meta.has_params.iloc[ev[k]])) if "has_params" in meta else 1])

    # ---------------------------------------------------------------- figure
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
    colors = ["#c0392b", "#1b998b", "#1f3a5f", "#e07a1f"]
    edges = SNR_EDGES
    plotted = False
    for (g, ps), col in zip(results.items(), colors):
        mids, mean_r, lo_r, hi_r = [], [], [], []
        for b in range(len(edges) - 1):
            vals = [v["snr_bins"][b]["recall"] for v in ps.values()]
            if np.all(np.isnan(vals)):
                continue
            lo, hi = edges[b], min(edges[b + 1], 300)
            mids.append(np.sqrt(max(lo, 6) * hi)); mean_r.append(np.nanmean(vals)); lo_r.append(np.nanmin(vals)); hi_r.append(np.nanmax(vals))
        if mids:
            axs[0].plot(mids, mean_r, "o-", color=col, label=g)
            axs[0].fill_between(mids, lo_r, hi_r, color=col, alpha=0.18)
            plotted = True
    if plotted:
        axs[0].set_xscale("log"); axs[0].legend(fontsize=9, loc="lower right")
    else:
        axs[0].text(0.5, 0.5, "too few events per S/N bin to plot", ha="center", transform=axs[0].transAxes)
    axs[0].set_ylim(0, 1.04)
    axs[0].set_xlabel("catalog detection S/N (bonsai_snr)"); axs[0].set_ylabel("recall on real bursts (no retraining)")
    axs[0].set_title("Real CHIME bursts: recall vs S/N", fontsize=10.5, color="#1f3a5f")
    axs[1].hist(mean_p[y == 1], bins=30, range=(0, 1), alpha=0.7, color="#1b998b", label="real bursts")
    axs[1].hist(mean_p[y == 0], bins=30, range=(0, 1), alpha=0.7, color="#c0392b", label="real burst-free windows")
    axs[1].set_yscale("log"); axs[1].set_xlabel("mean model P(FRB) over all checkpoints"); axs[1].set_ylabel("images")
    axs[1].set_title("Separation on real data", fontsize=10.5, color="#1f3a5f"); axs[1].legend(fontsize=9)
    for a in axs:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "real_recall.png"), dpi=150)
    print(f"\nsaved {args.out}/real_results.txt, real_results.json, real_scores.csv, real_recall.png")


if __name__ == "__main__":
    main()
