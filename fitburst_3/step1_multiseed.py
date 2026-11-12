#!/usr/bin/env python3
"""
step1_multiseed.py -- train the SAME CNN with several seeds / splits.

For every seed s:
  * stratified 70/15/15 split with split-seed = s   (different test set each time)
  * torch / numpy / python RNG seeded with s         (different init + sampling)
  * up to --epochs epochs (default 60; the Step-0 run was still improving at 25)
  * early stopping on validation F1 (patience --patience, default 10)
  * best checkpoint evaluated ONCE on that seed's test split

Outputs (per seed) in runs/step1/seed_<s>/ :
    best_model.pt  history.json  test_metrics.json  test_predictions.csv

Then run   python3 step1_report.py   for mean +- std and the error breakdown.

Examples
--------
    # 1-minute smoke test (do this first!)
    python3 step1_multiseed.py --seeds 42 --epochs 2 --outdir runs/step1_smoke

    # the real thing (resumable: finished seeds are skipped)
    python3 step1_multiseed.py --seeds 42 1 2 3 4

Add --force-cpu if the Apple GPU (MPS) misbehaves.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from step1_lib import (FRBCNN, CachedDataset, build_cache, pick_device, predict_probs,
                       run_epoch, set_all_seeds, stratified_split_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frb-dir", default="sim_output")
    ap.add_argument("--rfi-dir", default="sim_output_rfi")
    ap.add_argument("--outdir", default="runs/step1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--resize-mode", choices=["interp", "area"], default="interp",
                    help="how windows longer than 256 samples are shrunk: interp (Step 0-2) or area (block averaging)")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="retrain seeds even if finished")
    args = ap.parse_args()

    device = pick_device(args.force_cpu)
    print(f"device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    print(f"preprocessing all images once (cached in RAM), resize mode = {args.resize_mode} ...")
    cache = build_cache(args.frb_dir, args.rfi_dir, args.resize_mode)
    labels = cache["y"].astype(int)
    print(f"  {int(labels.sum())} FRB / {int((1 - labels).sum())} RFI   ({time.time() - t0:.0f}s)")

    for seed in args.seeds:
        sdir = os.path.join(args.outdir, f"seed_{seed}")
        if os.path.exists(os.path.join(sdir, "test_metrics.json")) and not args.no_resume:
            print(f"\n=== seed {seed}: already finished, skipping (use --no-resume to redo) ===")
            continue
        os.makedirs(sdir, exist_ok=True)
        print(f"\n=== seed {seed} ===")
        set_all_seeds(seed)

        tr_idx, va_idx, te_idx = stratified_split_idx(labels, 0.15, 0.15, seed=seed)
        print(f"split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
        train_ds = CachedDataset(cache, tr_idx, augment_on=True)
        val_ds = CachedDataset(cache, va_idx, augment_on=False)
        test_ds = CachedDataset(cache, te_idx, augment_on=False)

        tr_labels = labels[tr_idx]
        class_count = np.bincount(tr_labels, minlength=2)
        sampler = WeightedRandomSampler(1.0 / class_count[tr_labels], num_samples=len(tr_labels), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        n_pos, n_neg = int(class_count[1]), int(class_count[0])
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

        model = FRBCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_f1, no_improve, history = -1.0, 0, []
        best_path = os.path.join(sdir, "best_model.pt")
        ts = time.time()
        for epoch in range(1, args.epochs + 1):
            trm = run_epoch(model, train_loader, device, optimizer=optimizer, pos_weight=pos_weight)
            vam = run_epoch(model, val_loader, device, optimizer=None, pos_weight=pos_weight)
            scheduler.step(vam["f1"])
            print(f"  ep {epoch:03d} | train loss {trm['loss']:.4f} acc {trm['acc']:.3f} | "
                  f"val loss {vam['loss']:.4f} acc {vam['acc']:.3f} P {vam['precision']:.3f} "
                  f"R {vam['recall']:.3f} F1 {vam['f1']:.3f} | {time.time() - ts:.0f}s")
            history.append({"epoch": epoch, "train": trm, "val": vam})
            if vam["f1"] > best_f1:
                best_f1, no_improve = vam["f1"], 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1": best_f1,
                            "seed": seed, "resize_mode": args.resize_mode}, best_path)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  early stopping at epoch {epoch}")
                    break

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        tem = run_epoch(model, test_loader, device, optimizer=None, pos_weight=pos_weight)
        probs, labs = predict_probs(model, test_loader, device)
        tem["auc"] = float(roc_auc_score(labs, probs))
        tem.update({"seed": seed, "best_epoch": int(ckpt["epoch"]), "epochs_ran": len(history),
                    "val_f1_best": float(best_f1), "n_test": len(te_idx),
                    "hit_epoch_cap": bool(len(history) == args.epochs and ckpt["epoch"] >= args.epochs - 1)})
        print(f"  TEST  acc {tem['acc']:.4f}  P {tem['precision']:.4f}  R {tem['recall']:.4f}  "
              f"F1 {tem['f1']:.4f}  AUC {tem['auc']:.4f}  (best epoch {tem['best_epoch']}/{len(history)})")

        with open(os.path.join(sdir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(sdir, "test_metrics.json"), "w") as f:
            json.dump(tem, f, indent=2)
        with open(os.path.join(sdir, "test_predictions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "prob"])
            for j, p, l in zip(te_idx, probs, labs):
                w.writerow([cache["paths"][j], int(l), f"{p:.6f}"])

    print(f"\nall done in {(time.time() - t0) / 60:.1f} min.  Next: python3 step1_report.py --rundir {args.outdir}")


if __name__ == "__main__":
    main()
