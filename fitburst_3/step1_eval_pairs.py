#!/usr/bin/env python3
"""
step1_eval_pairs.py -- the strict "does it need the burst?" test.

Loads every runs/step1/seed_*/best_model.pt and scores the paired images made
by step1_make_pairs.py (same background, with / without burst).

Reported per seed and averaged:
  pair-win rate   fraction of pairs where P(with burst) > P(without)
                  1.00 = always finds the burst, 0.50 = blind to it
  recall(with)    FRB images called FRB
  spec(without)   burst-free images called RFI
  shuffled        same numbers after shuffling the time columns of both
                  images of each pair. Pair-win should fall to ~0.50.
Broken out for pairs WITHOUT quantization (images identical outside the
burst) and WITH quantization (can differ by one quantization step).

Caveat: the burst parameters come from catalog rows that also appear in the
training set (with different backgrounds). This is a test of background
shortcuts, not a fresh-event generalization test (that is Step 4/5).

Usage:  python3 step1_eval_pairs.py --rundir runs/step1 --pairs-dir sim_paired
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step1_lib import FRBCNN, pick_device, predict_arrays, preprocess_array, require_interp

TEAL, ORANGE, NAVY, GREY = "#1b998b", "#e07a1f", "#1f3a5f", "#8a94a0"


def pair_win(pw, pn):
    return float(np.mean((pw > pn) + 0.5 * (pw == pn)))


def stats(pw, pn):
    return {"pair_win": pair_win(pw, pn), "recall_with": float(np.mean(pw > 0.5)),
            "spec_without": float(np.mean(pn <= 0.5)), "n": int(len(pw))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/step1")
    ap.add_argument("--pairs-dir", default="sim_paired")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()
    device = pick_device(args.force_cpu)

    with_files = sorted(glob.glob(os.path.join(args.pairs_dir, "*_with.npz")))
    if not with_files:
        raise SystemExit("no pairs found -- run step1_make_pairs.py first")
    A_w, A_n, quant, faint = [], [], [], []
    for f in with_files:
        g = f.replace("_with.npz", "_without.npz")
        A_w.append(preprocess_array(np.load(f, allow_pickle=True)["data_full"]))
        A_n.append(preprocess_array(np.load(g, allow_pickle=True)["data_full"]))
        m = np.load(f, allow_pickle=True)["metadata"].item()
        quant.append(any("quantization" in str(c) for c in m.get("noise_components", [])))
        faint.append(float(m.get("faint_scale", 1.0)))
    quant, faint = np.array(quant), np.array(faint)
    rng = np.random.default_rng(0)
    perm = lambda a: a[:, rng.permutation(a.shape[1])]
    S_w, S_n = [perm(a) for a in A_w], [perm(a) for a in A_n]
    print(f"{len(with_files)} pairs  ({int(quant.sum())} with quantization, {int((~quant).sum())} without)")

    ckpts = sorted(glob.glob(os.path.join(args.rundir, "seed_*", "best_model.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.rundir}")
    results, diffs = [], []
    for ck in ckpts:
        model = FRBCNN().to(device)
        _ck = torch.load(ck, map_location=device); require_interp(_ck, ck)
        model.load_state_dict(_ck["model_state"])
        pw, pn = predict_arrays(model, A_w, device), predict_arrays(model, A_n, device)
        sw, sn = predict_arrays(model, S_w, device), predict_arrays(model, S_n, device)
        r = {"seed": os.path.basename(os.path.dirname(ck)),
             "all": stats(pw, pn), "no_quant": stats(pw[~quant], pn[~quant]),
             "quant": stats(pw[quant], pn[quant]) if quant.any() else None,
             "shuffled": stats(sw, sn),
             "faint_full": stats(pw[faint >= .999], pn[faint >= .999]),
             "faint_low": stats(pw[faint < .999], pn[faint < .999]) if (faint < .999).any() else None}
        results.append(r); diffs.append(pw - pn)

    print(f"\n{'seed':>8} | {'pair-win':>8} {'recall':>7} {'spec':>6} | {'win(no-quant)':>13} {'win(quant)':>10} | {'win(shuffled)':>13}")
    for r in results:
        print(f"{r['seed']:>8} | {r['all']['pair_win']:8.3f} {r['all']['recall_with']:7.3f} {r['all']['spec_without']:6.3f} | "
              f"{r['no_quant']['pair_win']:13.3f} {(r['quant'] or {'pair_win': float('nan')})['pair_win']:10.3f} | {r['shuffled']['pair_win']:13.3f}")
    mean = lambda f: float(np.mean([f(r) for r in results]))
    print("-" * 86)
    print(f"{'MEAN':>8} | {mean(lambda r: r['all']['pair_win']):8.3f} {mean(lambda r: r['all']['recall_with']):7.3f} "
          f"{mean(lambda r: r['all']['spec_without']):6.3f} | {mean(lambda r: r['no_quant']['pair_win']):13.3f} "
          f"{mean(lambda r: (r['quant'] or {'pair_win': np.nan})['pair_win']):10.3f} | {mean(lambda r: r['shuffled']['pair_win']):13.3f}")
    print("\nREAD: pair-win near 1.0 and shuffled pair-win near 0.5  ->  the model needs the burst.")
    print("      if win(no-quant) is clearly lower than win(quant): the model leans on the quantization lattice.")

    json.dump(results, open(os.path.join(args.rundir, "step1_pairs.json"), "w"), indent=2)
    d = np.concatenate(diffs)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(d, bins=40, color=TEAL); ax.axvline(0, color=GREY, ls="--")
    ax.set_xlabel("P(with burst) − P(without burst)   [same background]"); ax.set_ylabel("pairs (all seeds)")
    ax.set_title("Paired test: shift towards +1 = model sees the burst", fontsize=10, color=NAVY)
    plt.tight_layout(); plt.savefig(os.path.join(args.rundir, "step1_pairs.png"), dpi=150)
    print(f"saved {args.rundir}/step1_pairs.json, step1_pairs.png")


if __name__ == "__main__":
    main()
