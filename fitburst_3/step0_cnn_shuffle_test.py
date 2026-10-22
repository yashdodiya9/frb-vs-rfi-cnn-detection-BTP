#!/usr/bin/env python3
"""
step0_cnn_shuffle_test.py
=========================

Second half of the Step-0 audit: test the TRAINED CNN itself.

Shuffling the time columns of an image (a) destroys the burst's temporal
localisation / dispersion morphology, but (b) keeps every per-channel
statistic: banding, masks, quantization lattice, noise level.

  * morphology-driven model -> accuracy collapses to ~50% on shuffled images
  * shortcut-driven model   -> accuracy stays high on shuffled images

Also run with --erase-center, which replaces the middle 50% of the time axis
(where the burst lives) with a copy of the edge columns. Same logic.

NOTE: written against the notebook's FRBCNN / preprocessing, but NOT executed
by the author (no torch in the authoring sandbox). Run it once on your Mac
and tell me if anything needs adjusting.

Usage
-----
    python3 step0_cnn_shuffle_test.py --frb-dir sim_output --rfi-dir sim_output_rfi \
            --ckpt runs/exp4/best_model.pt
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FREQ_BINS, TIME_BINS = 256, 256


class FRBCNN(nn.Module):  # identical to the notebook
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def load_img(path):
    arr = np.load(path, allow_pickle=True)["data_full"].astype(np.float32)
    x_old, x_new = np.linspace(0, 1, arr.shape[1]), np.linspace(0, 1, TIME_BINS)
    arr = np.stack([np.interp(x_new, x_old, r) for r in arr]).astype(np.float32)
    nz = arr[arr != 0.0]
    med, mad = (np.median(nz), np.median(np.abs(nz - np.median(nz))) + 1e-6) if nz.size >= 10 else (0.0, 1.0)
    arr = (arr - med) / (1.4826 * mad)
    return np.nan_to_num(np.clip(arr, -8.0, 8.0), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def stratified_test_split(items, val_frac=0.15, test_frac=0.15, seed=42):  # same as notebook
    rng = random.Random(seed)
    pos = [x for x in items if x[1] == 1]; neg = [x for x in items if x[1] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    out = []
    for lst in (pos, neg):
        out += lst[:int(round(len(lst) * test_frac))]
    return out


@torch.no_grad()
def acc_recall(model, imgs, labels, device):
    p = []
    for i in range(0, len(imgs), 64):
        x = torch.from_numpy(np.stack(imgs[i:i + 64])[:, None]).to(device)
        p.append((torch.sigmoid(model(x)) > 0.5).cpu().numpy())
    p = np.concatenate(p).astype(int); y = np.array(labels)
    return (p == y).mean(), (p[y == 1] == 1).mean(), (p[y == 0] == 0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frb-dir", default="sim_output")
    ap.add_argument("--rfi-dir", default="sim_output_rfi")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = FRBCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state"])
    model.eval()

    items = [(str(p), 1) for p in sorted(Path(args.frb_dir).glob("*.npz"))] + \
            [(str(p), 0) for p in sorted(Path(args.rfi_dir).glob("*.npz"))]
    test = stratified_test_split(items, seed=args.seed)
    imgs = [load_img(p) for p, _ in test]; labels = [l for _, l in test]
    rng = np.random.default_rng(0)

    def shuffled(a):
        return a[:, rng.permutation(a.shape[1])]

    def erased(a):
        q = a.shape[1] // 4
        b = a.copy()
        b[:, q:3 * q] = np.concatenate([a[:, :q], a[:, 3 * q:]], axis=1)[:, rng.integers(0, 2 * q, 2 * q)]
        return b

    print(f"{len(imgs)} test images")
    for name, f in [("original", lambda a: a), ("time-columns shuffled", shuffled), ("burst region erased", erased)]:
        acc, rec_frb, rec_rfi = acc_recall(model, [f(a) for a in imgs], labels, device)
        print(f"{name:24s} acc={acc:.3f}  FRB-recall={rec_frb:.3f}  RFI-recall={rec_rfi:.3f}")
    print("\nPASS  = shuffled / erased accuracy near 0.50 (model needs the burst).")
    print("FAIL  = shuffled / erased accuracy stays well above 0.50 (model reads the background).")


if __name__ == "__main__":
    main()
