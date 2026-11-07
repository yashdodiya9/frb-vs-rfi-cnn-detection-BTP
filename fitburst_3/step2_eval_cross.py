#!/usr/bin/env python3
"""
step2_eval_cross.py -- score the 5 Step-1 models (trained on the OLD impulsive-RFI data)
on the NEW Step-2 data, WITHOUT retraining.

Question answered: how much of the 94.8% (and of the 13.9% false-positive rate on
impulsive RFI) depended on the unphysical vertical-streak impulses?

Reported per seed and mean:
  accuracy / recall / specificity / AUC on the new images
  false-positive rate on RFI images containing  (a) swept impulses (new, physical)
                                                (b) mimic impulses (old vertical streak, 15% of events)
                                                (c) no impulsive RFI
  miss rate on FRB images containing the same three groups

Usage:
  python3 step2_eval_cross.py --ckpt-root runs/step1 --frb-dir sim_output_v2 --rfi-dir sim_output_rfi_v2
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from step1_lib import FRBCNN, pick_device, predict_arrays, preprocess_array


def group(meta):
    comps = {str(c).split(":")[0] for c in meta.get("noise_components", [])}
    return "swept" if "impulsive_rfi" in comps else ("mimic" if "impulsive_mimic" in comps else "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="runs/step1")
    ap.add_argument("--frb-dir", default="sim_output_v2")
    ap.add_argument("--rfi-dir", default="sim_output_rfi_v2")
    ap.add_argument("--max-n", type=int, default=1500, help="max images per class")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = pick_device(args.force_cpu)
    rng = np.random.default_rng(0)

    arrays, labels, groups = [], [], []
    for d, lab in [(args.frb_dir, 1), (args.rfi_dir, 0)]:
        files = sorted(Path(d).glob("*.npz"))
        if len(files) > args.max_n:
            files = [files[i] for i in rng.choice(len(files), args.max_n, replace=False)]
        for f in files:
            z = np.load(f, allow_pickle=True)
            raw = z["data_full"]
            if not np.all(np.isfinite(raw)):
                continue
            arrays.append(preprocess_array(raw)); labels.append(lab); groups.append(group(z["metadata"].item()))
    y, g = np.array(labels), np.array(groups)
    print(f"{int(y.sum())} FRB + {int((1 - y).sum())} RFI images  | impulsive groups: " +
          ", ".join(f"{k}={int((g == k).sum())}" for k in ("swept", "mimic", "none")))

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_root, "seed_*", "best_model.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpt_root}")
    rows = []
    for ck in ckpts:
        m = FRBCNN().to(device); m.load_state_dict(torch.load(ck, map_location=device)["model_state"])
        p = predict_arrays(m, arrays, device); pred = p > 0.5
        r = {"seed": os.path.basename(os.path.dirname(ck)), "acc": float((pred == y).mean()),
             "recall": float(pred[y == 1].mean()), "spec": float((~pred[y == 0]).mean()), "auc": float(roc_auc_score(y, p))}
        for grp in ("swept", "mimic", "none"):
            neg, pos = (y == 0) & (g == grp), (y == 1) & (g == grp)
            r[f"fp_{grp}"] = float(pred[neg].mean()) if neg.any() else float("nan")
            r[f"fn_{grp}"] = float((~pred[pos]).mean()) if pos.any() else float("nan")
        rows.append(r)

    print(f"\n{'seed':>8} | {'acc':>6} {'recall':>6} {'spec':>6} {'AUC':>6} | FP rate: {'swept':>6} {'mimic':>6} {'none':>6} | miss rate: {'swept':>6} {'mimic':>6} {'none':>6}")
    for r in rows:
        print(f"{r['seed']:>8} | {r['acc']:6.3f} {r['recall']:6.3f} {r['spec']:6.3f} {r['auc']:6.3f} |          {r['fp_swept']:6.3f} {r['fp_mimic']:6.3f} {r['fp_none']:6.3f} |            {r['fn_swept']:6.3f} {r['fn_mimic']:6.3f} {r['fn_none']:6.3f}")
    mean = lambda k: float(np.nanmean([r[k] for r in rows]))
    print("-" * 118)
    print(f"{'MEAN':>8} | {mean('acc'):6.3f} {mean('recall'):6.3f} {mean('spec'):6.3f} {mean('auc'):6.3f} |          "
          f"{mean('fp_swept'):6.3f} {mean('fp_mimic'):6.3f} {mean('fp_none'):6.3f} |            {mean('fn_swept'):6.3f} {mean('fn_mimic'):6.3f} {mean('fn_none'):6.3f}")
    print("\nReference, same models on the OLD data (Step 1): acc 0.948, FP rate on impulsive-RFI images 0.139, "
          "miss rate on FRBs containing impulsive RFI 0.176, overall FP 0.040.")
    out = args.out or os.path.join(args.ckpt_root, "step2_cross_eval.json")
    json.dump(rows, open(out, "w"), indent=2); print(f"saved {out}")


if __name__ == "__main__":
    main()
