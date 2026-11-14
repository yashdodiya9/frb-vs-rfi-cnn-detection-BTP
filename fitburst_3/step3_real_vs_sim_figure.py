#!/usr/bin/env python3
"""
step3_real_vs_sim_figure.py -- real CHIME bursts next to simulated ones, EXACTLY as the CNN sees them
(256 x 256, after resizing and robust normalisation).

Top row   : 8 real bursts spread from faint to bright (catalog detection S/N)
Bottom row: 8 random simulated bursts from the training data

Usage:
    python step3_real_vs_sim_figure.py --cache real_cache.npz --catalog chimefrbcat2.csv --sim-dir sim_output_v2
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from step3_real_lib import burst_center, crop_positive, load_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="real_cache.npz")
    ap.add_argument("--catalog", default="chimefrbcat2.csv")
    ap.add_argument("--sim-dir", default="sim_output_v2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mode", choices=["interp", "area"], default="area")
    ap.add_argument("--out", default="runs/step3_real/real_vs_sim.png")
    ap.add_argument("--seed", type=int, default=1)
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
    ok = meta.snr.notna().values
    order = np.argsort(meta.snr.where(ok, np.inf).values)[:int(ok.sum())]
    pick = [order[int(q * (len(order) - 1))] for q in np.linspace(0.05, 0.97, args.n)]

    rng = np.random.default_rng(args.seed)
    sim_files = sorted(glob.glob(os.path.join(args.sim_dir, "*.npz")))
    sim_pick = list(rng.choice(sim_files, args.n, replace=False)) if sim_files else []

    fig, axs = plt.subplots(2, args.n, figsize=(2.1 * args.n, 4.9))
    for j, i in enumerate(pick):
        m = meta.iloc[i]
        img = preprocess_array(crop_positive(X[i], m.width_s, m.scat_s, burst_center(prof[i], X[i].shape[1])), args.mode)
        v = np.percentile(img[img != 0], [1, 99.7])
        axs[0, j].imshow(img, aspect="auto", origin="lower", vmin=v[0], vmax=v[1], cmap="viridis")
        axs[0, j].set_title(f"real, S/N {m.snr:.0f}", fontsize=8.5, color="#1f3a5f")
    for j, f in enumerate(sim_pick):
        z = np.load(f, allow_pickle=True)
        img = preprocess_array(z["data_full"], args.mode)
        v = np.percentile(img[img != 0], [1, 99.7])
        axs[1, j].imshow(img, aspect="auto", origin="lower", vmin=v[0], vmax=v[1], cmap="viridis")
        axs[1, j].set_title(f"simulated", fontsize=8.5, color="#c0392b")
    for a in axs.flat:
        a.axis("off")
    fig.suptitle("What the CNN sees: real CHIME bursts (top) vs simulated training bursts (bottom)",
                 fontsize=11, fontweight="bold", color="#1f3a5f")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
