#!/usr/bin/env python3
"""
step3_inspect_event.py -- look at individual real events in full: the whole 256-channel window (not only the crop),
the channel-averaged time series, the fitted-model profile, and where the crop was taken.

Use it for surprising misses (e.g. a burst with S/N 96 that every model calls "not an FRB").

Usage:
    python step3_inspect_event.py --cache ~/Desktop/using-h5-files/real_cache.npz --catalog chimefrbcat2.csv \
        --names FRB20190208B FRB20180912A FRB20190729B --out runs/step3_real_peak/inspect.png
"""
import argparse
import os

import numpy as np
import pandas as pd

from step3_real_lib import burst_center, rule_length, window_length, load_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="real_cache.npz")
    ap.add_argument("--catalog", default="chimefrbcat2.csv")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="runs/step3_real_peak/inspect.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = load_cache(args.cache)
    X, prof, meta = c["X"], c["model_prof"], pd.DataFrame(list(c["meta"]))
    cat = pd.read_csv(args.catalog)
    cat = cat[cat.sub_num == 0].drop_duplicates("tns_name").set_index("tns_name")
    fig, axs = plt.subplots(len(args.names), 3, figsize=(14, 3.2 * len(args.names)), squeeze=False,
                            gridspec_kw={"width_ratios": [2.2, 1.6, 1.6]})
    for r, name in enumerate(args.names):
        hit = np.where(meta.tns_name.values == name)[0]
        if not len(hit):
            print(f"{name}: not in cache"); continue
        i = int(hit[0]); m = meta.iloc[i]; img = X[i]; t = img.shape[1]
        bc = burst_center(prof[i], t); w = window_length(m.width_s, m.scat_s, t)
        s0 = int(np.clip(bc - w // 2, 0, t - w))
        live = (img != 0).any(axis=1)
        series = img[live].mean(axis=0)
        noise = 1.4826 * np.median(np.abs(series - np.median(series)))
        peak_snr = (series.max() - np.median(series)) / max(noise, 1e-9)
        row = cat.loc[name] if name in cat.index else None
        print(f"{name}: T={t} samples | has_params={bool(m.get('has_params', True))} has_model={bool(m.get('has_model', True))} | "
              f"width={m.width_s*1e3:.2f} ms scat={m.scat_s*1e3:.2f} ms | rule length {rule_length(m.width_s, m.scat_s)} -> crop {w} samples "
              f"[{s0}, {s0 + w}) around sample {bc} (file centre {t // 2}) | live channels {int(live.sum())}/256 | "
              f"peak of channel-averaged series {peak_snr:.1f} sigma" + (f" | catalog bonsai S/N {row.bonsai_snr:.1f}, fitburst S/N {row.snr_fitb:.1f}" if row is not None else ""))
        nz = img[img != 0]; v = np.percentile(nz, [1, 99.5]) if nz.size else (0, 1)
        ax = axs[r, 0]; ax.imshow(img, aspect="auto", origin="lower", vmin=v[0], vmax=v[1], cmap="viridis")
        ax.axvline(s0, color="w", lw=1, ls="--"); ax.axvline(s0 + w - 1, color="w", lw=1, ls="--"); ax.axvline(bc, color="r", lw=0.8)
        ax.set_title(f"{name}: whole window (white = crop, red = model peak)", fontsize=9)
        axs[r, 1].plot(series, color="#1f3a5f"); axs[r, 1].axvspan(s0, s0 + w, color="#1b998b", alpha=0.2)
        axs[r, 1].set_title(f"channel-averaged series (peak {peak_snr:.1f} sigma)", fontsize=9)
        axs[r, 2].plot(np.asarray(prof[i], float), color="#e07a1f"); axs[r, 2].set_title("fitted model profile", fontsize=9)
    plt.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True); plt.savefig(args.out, dpi=120)
    print("saved", args.out)


if __name__ == "__main__":
    main()
