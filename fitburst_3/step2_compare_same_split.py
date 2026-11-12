#!/usr/bin/env python3
"""
step2_compare_same_split.py -- did RETRAINING on the new data help, or did the new data just get easier?

For every seed s: take the images in runs/step2/seed_s/test_predictions.csv (that seed's new test split),
score them with
    NEW model  = runs/step2/seed_s  (trained on the new data; its saved probabilities)
    OLD model  = runs/step1/seed_s  (trained on the old data, never saw any new-data image)
and compare on the SAME images: accuracy, F1, AUC, and an exact McNemar test on the disagreements.

b = images the OLD model gets right and the NEW wrong;  c = NEW right, OLD wrong.
If retraining helped, c >> b. If both are ~equal, retraining added nothing beyond the easier data.

Run from the folder that contains sim_output_v2/ (paths in the CSV are relative to it):
    python step2_compare_same_split.py --old-root runs/step1 --new-root runs/step2
"""
import argparse, csv, glob, json, os
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from step1_lib import FRBCNN, pick_device, predict_arrays, preprocess_array, require_interp


def f1(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", default="runs/step1")
    ap.add_argument("--new-root", default="runs/step2")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()
    device = pick_device(args.force_cpu)

    rows, B, C = [], 0, 0
    for sdir in sorted(glob.glob(os.path.join(args.new_root, "seed_*"))):
        s = os.path.basename(sdir)
        ck = os.path.join(args.old_root, s, "best_model.pt")
        if not os.path.exists(ck):
            print(f"skip {s}: no old checkpoint"); continue
        with open(os.path.join(sdir, "test_predictions.csv")) as f:
            recs = list(csv.DictReader(f))
        paths = [r["path"] for r in recs]; y = np.array([int(r["label"]) for r in recs])
        p_new = np.array([float(r["prob"]) for r in recs])
        arrays = [preprocess_array(np.load(p, allow_pickle=True)["data_full"]) for p in paths]
        _ck = torch.load(ck, map_location=device); require_interp(_ck, ck)
        m = FRBCNN().to(device); m.load_state_dict(_ck["model_state"])
        p_old = predict_arrays(m, arrays, device)
        pn, po = p_new > 0.5, p_old > 0.5
        b = int(((po == y) & (pn != y)).sum()); c = int(((pn == y) & (po != y)).sum())
        pval = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
        rows.append({"seed": s, "n": len(y), "acc_old": float((po == y).mean()), "acc_new": float((pn == y).mean()),
                     "f1_old": f1(y, po), "f1_new": f1(y, pn), "auc_old": float(roc_auc_score(y, p_old)),
                     "auc_new": float(roc_auc_score(y, p_new)), "old_right_new_wrong": b, "new_right_old_wrong": c, "mcnemar_p": float(pval)})
        B += b; C += c

    print(f"\n{'seed':>8} | {'acc OLD':>8} {'acc NEW':>8} | {'F1 OLD':>7} {'F1 NEW':>7} | {'AUC OLD':>8} {'AUC NEW':>8} | {'b':>4} {'c':>4} {'p':>6}")
    for r in rows:
        print(f"{r['seed']:>8} | {r['acc_old']:8.4f} {r['acc_new']:8.4f} | {r['f1_old']:7.4f} {r['f1_new']:7.4f} | {r['auc_old']:8.4f} {r['auc_new']:8.4f} | {r['old_right_new_wrong']:>4} {r['new_right_old_wrong']:>4} {r['mcnemar_p']:6.3f}")
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    print("-" * 92)
    print(f"{'MEAN':>8} | {mean('acc_old'):8.4f} {mean('acc_new'):8.4f} | {mean('f1_old'):7.4f} {mean('f1_new'):7.4f} | {mean('auc_old'):8.4f} {mean('auc_new'):8.4f} | {B:>4} {C:>4}")
    print("\nb = old right / new wrong, c = new right / old wrong (summed over seeds; test sets overlap between seeds, so the pooled counts are descriptive, not a formal test).")
    print("c >> b: retraining helped.   b ~ c: retraining added nothing beyond the easier data.   b >> c: retraining hurt.")
    out = os.path.join(args.new_root, "step2_same_split.json"); json.dump(rows, open(out, "w"), indent=2); print("saved", out)


if __name__ == "__main__":
    main()
