"""
step1_lib.py -- shared code for Step 1 (multi-seed training + paired test).

Everything here is copied from train_frb_cnn_step0.ipynb (same model, same
preprocessing, same augmentation, same split logic, same train/eval loop).
The ONLY addition is an in-memory cache of the preprocessed images, so that
5 seeds do not each re-read and re-interpolate 8000+ .npz files every epoch.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

FREQ_BINS = 256
TIME_BINS = 256

# augmentation constants (identical to the notebook)
AUG_TIME_SHIFT_FRAC = 0.08
AUG_CHANNEL_DROPOUT_FRAC = 0.05
AUG_NOISE_STD = 0.05
AUG_PROB = 0.5


# ----------------------------------------------------------------- model
class FRBCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(1)


# --------------------------------------------------------- preprocessing
def area_resample_time(arr: np.ndarray, target_time: int) -> np.ndarray:
    """Exact area (block) averaging of the time axis from T > target samples to `target` bins.
    Bin edges are fractional, so any T works; every native sample contributes to the output
    (unlike interpolation, which only reads ~2 samples per output bin when downsampling)."""
    n_f, t = arr.shape
    csum = np.concatenate([np.zeros((n_f, 1)), np.cumsum(arr, axis=1, dtype=np.float64)], axis=1)
    edges = np.linspace(0.0, float(t), target_time + 1)
    idx = np.arange(t + 1, dtype=np.float64)
    at_edges = np.stack([np.interp(edges, idx, csum[i]) for i in range(n_f)])
    return ((at_edges[:, 1:] - at_edges[:, :-1]) / (t / target_time)).astype(np.float32)


def resize_time_axis(arr: np.ndarray, target_time: int, mode: str = "interp") -> np.ndarray:
    """mode='interp' (Step 0-2 behaviour): linear interpolation for any length.
    mode='area': block-average when downsampling (T > target); upsampling still uses interpolation."""
    t_native = arr.shape[1]
    if t_native == target_time:
        return arr
    if mode == "area" and t_native > target_time:
        return area_resample_time(arr, target_time)
    x_old = np.linspace(0.0, 1.0, t_native)
    x_new = np.linspace(0.0, 1.0, target_time)
    out = np.empty((arr.shape[0], target_time), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.interp(x_new, x_old, arr[i])
    return out


def robust_normalize(arr: np.ndarray) -> np.ndarray:
    nonzero = arr[arr != 0.0]
    if nonzero.size < 10:
        med, mad = 0.0, 1.0
    else:
        med = np.median(nonzero)
        mad = np.median(np.abs(nonzero - med)) + 1e-6
    return (arr - med) / (1.4826 * mad)


def preprocess_array(arr: np.ndarray, mode: str = "interp") -> np.ndarray:
    """raw data_full -> resized + normalized + clipped (256x256 float32).
    Same operations, same order as DynSpecDataset.__getitem__ in the notebook
    (augmentation happens after this, clipping is re-applied after it)."""
    arr = arr.astype(np.float32)
    if arr.shape[0] != FREQ_BINS:
        raise ValueError(f"expected {FREQ_BINS} freq channels, got {arr.shape[0]}")
    arr = resize_time_axis(arr, TIME_BINS, mode)
    arr = robust_normalize(arr)
    return np.nan_to_num(np.clip(arr, -8.0, 8.0), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def load_preprocessed(path: str, mode: str = "interp"):
    """Returns preprocessed array, or None if the raw file is non-finite."""
    raw = np.load(path, allow_pickle=True)["data_full"]
    if not np.all(np.isfinite(raw)):
        return None
    return preprocess_array(raw, mode)


def augment(arr: np.ndarray) -> np.ndarray:
    if np.random.random() < AUG_PROB:
        max_shift = max(1, int(AUG_TIME_SHIFT_FRAC * arr.shape[1]))
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift != 0:
            arr = np.roll(arr, shift, axis=1)
    if np.random.random() < AUG_PROB:
        n_drop = int(AUG_CHANNEL_DROPOUT_FRAC * arr.shape[0])
        if n_drop > 0:
            drop_idx = np.random.choice(arr.shape[0], size=n_drop, replace=False)
            arr = arr.copy()
            arr[drop_idx, :] = 0.0
    if np.random.random() < AUG_PROB:
        arr = arr + np.random.normal(0.0, AUG_NOISE_STD, size=arr.shape).astype(np.float32)
    return arr


class CachedDataset(Dataset):
    """Same behaviour as the notebook's DynSpecDataset, but images are
    preprocessed once and kept in RAM as float16 (values are clipped to +-8,
    so float16 loses nothing that matters)."""

    def __init__(self, cache, indices, augment_on=False):
        self.X = cache["X"]            # (N, 256, 256) float16
        self.y = cache["y"]            # (N,) float32
        self.indices = list(indices)
        self.augment_on = augment_on

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        arr = self.X[j].astype(np.float32)
        if self.augment_on:
            arr = augment(arr)
        arr = np.clip(arr, -8.0, 8.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=8.0, neginf=-8.0)
        return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0), torch.tensor(self.y[j], dtype=torch.float32)


def build_cache(frb_dir: str, rfi_dir: str, mode: str = "interp"):
    """Preprocess every file ONCE. Returns dict with X, y, paths."""
    items = [(str(p), 1) for p in sorted(Path(frb_dir).glob("*.npz"))] + \
            [(str(p), 0) for p in sorted(Path(rfi_dir).glob("*.npz"))]
    if not items:
        raise SystemExit("no .npz files found")
    X, y, paths, bad = [], [], [], 0
    for k, (p, lab) in enumerate(items):
        arr = load_preprocessed(p, mode)
        if arr is None:
            bad += 1
            continue
        X.append(arr.astype(np.float16)); y.append(lab); paths.append(p)
        if (k + 1) % 1000 == 0:
            print(f"  preprocessed {k + 1}/{len(items)}")
    print(f"cache: {len(X)} clean files, {bad} rejected (NaN/Inf)")
    return {"X": np.stack(X), "y": np.array(y, dtype=np.float32), "paths": paths, "mode": mode}


# ----------------------------------------------------------------- split
def stratified_split_idx(labels, val_frac=0.15, test_frac=0.15, seed=42):
    """Same logic as the notebook's stratified_split, on index arrays."""
    rng = random.Random(seed)
    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_one(lst):
        n = len(lst)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        return lst[n_test + n_val:], lst[n_test:n_test + n_val], lst[:n_test]

    p_tr, p_va, p_te = split_one(pos)
    n_tr, n_va, n_te = split_one(neg)
    train, val, test = p_tr + n_tr, p_va + n_va, p_te + n_te
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def pick_device(force_cpu: bool):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ------------------------------------------------------------ train/eval
def run_epoch(model, loader, device, optimizer=None, pos_weight=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, all_logits, all_labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            if is_train:
                if not torch.isfinite(loss):
                    print("  WARNING: non-finite loss on a training batch -- skipping this update")
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"loss": total_loss / len(loader.dataset), "acc": acc, "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def predict_arrays(model, arrays, device, batch=64):
    """arrays: list of preprocessed (256,256) float32 arrays -> probabilities."""
    model.eval()
    out = []
    for i in range(0, len(arrays), batch):
        x = torch.from_numpy(np.stack(arrays[i:i + batch])[:, None]).float().to(device)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------ checkpoint-aware scoring (Step 2d)
def ckpt_resize_mode(ckpt: dict) -> str:
    """Checkpoints from Step 0-2 have no 'resize_mode' key: they were trained with interpolation."""
    return ckpt.get("resize_mode", "interp")


def require_interp(ckpt: dict, path: str):
    """Guard for scripts that preprocess with the default (interp) mode only."""
    if ckpt_resize_mode(ckpt) != "interp":
        raise SystemExit(f"{path} was trained with resize_mode='{ckpt_resize_mode(ckpt)}'; this script only supports "
                         f"interp checkpoints. Use step2_eval_sweep.py / step2_diagnose_sweep.py for those.")


class RawScorer:
    """Holds RAW images and preprocesses them once per resize mode, so models trained with
    different preprocessing are each scored on correctly preprocessed input."""

    def __init__(self, raws):
        self.raws = raws
        self._cache = {}

    def arrays(self, mode: str):
        if mode not in self._cache:
            self._cache[mode] = [preprocess_array(r, mode) for r in self.raws]
        return self._cache[mode]


def score_checkpoint(ck_path: str, scorer: "RawScorer", device):
    ck = torch.load(ck_path, map_location=device)
    model = FRBCNN().to(device)
    model.load_state_dict(ck["model_state"])
    return predict_arrays(model, scorer.arrays(ckpt_resize_mode(ck)), device)
