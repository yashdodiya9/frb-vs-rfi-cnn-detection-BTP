#!/usr/bin/env python3
"""
step2_diagnose_sweep.py -- WHERE do the misses concentrate? (Step 2, part c)

The brightness sweep showed that all errors sit in the S/N 6-25 transition region. Here we hold S/N fixed
(default band 8-20) and ask which kinds of images fail, by splitting the burst images on
  * native window length (samples)        -> does resizing to 256 bins by interpolation hurt long windows?
  * fraction of masked (all-zero) channels
  * burst width / scattering (catalog)    -> broad bursts spread their power
  * which RFI / instrument effects are present in the image
and the false-positive rate on the burst-free copies, split by the same RFI effects.

All 10 checkpoints (Step 1 + Step 2 models) are pooled: they behave almost identically on the sweep, so pooling
reduces model noise. The remaining uncertainty is image-sampling noise; every row therefore shows n and a 95 %
bootstrap CI. With ~300 bursts the CIs are wide: treat differences smaller than the CI as unproven.

Per-image scores are saved to <out>/sweep_scores.npz so nothing has to be re-scored.

Usage:
  python3 step2_diagnose_sweep.py --sweep-dir sim_sweep --catalog chimefrbcat2.csv \
      --groups "Step 1 models=runs/step1" "Step 2 models=runs/step2"
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

from step2_eval_sweep import score_group  # torch is imported lazily inside score_group


def comp_names(meta):
    return {str(c).split(":")[0] for c in meta.get("noise_components", [])} - {"ambient"}


def build_table(files, preprocess, catalog=None):
    rows, arrays = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        raw, meta = z["data_full"], z["metadata"].item()
        arrays.append(preprocess(raw))
        rows.append({"path": f, "row": meta["row"], "scale": meta["scale"], "snr": meta["snr_mf"],
                     "num_time": raw.shape[1], "mask_frac": float((raw == 0).all(axis=1).mean()),
                     "comps": comp_names(meta)})
    t = pd.DataFrame(rows)
    # burst-free copies carry no component list: inherit it from the brightest image of the same row
    best = t[t.scale > 0].sort_values("scale").groupby("row").tail(1).set_index("row")["comps"].to_dict()
    t["comps"] = [best.get(r, set()) if s == 0 else c for r, s, c in zip(t.row, t.scale, t.comps)]
    if catalog is not None:
        cat = catalog.set_index("tns_name")
        t["width_ms"] = t.row.map(lambda r: float(cat.loc[r, "width_fitb"]) * 1e3 if r in cat.index else np.nan)
        t["scat_ms"] = t.row.map(lambda r: float(cat.loc[r, "scat_time"]) * 1e3 if r in cat.index else np.nan)
    return t, arrays


def boot_ci(x, rng, n=1000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(hit, mask, snr, rng):
    x = hit[mask]
    lo, hi = boot_ci(x, rng)
    return {"n": int(mask.sum()), "recall": float(x.mean()) if len(x) else np.nan, "lo": lo, "hi": hi,
            "mean_snr": float(np.exp(np.log(snr[mask]).mean())) if mask.any() else np.nan}


def group_table(t, hit, band, col, bins=None, q=None, rng=None, min_n=15):
    """Recall in the S/N band for groups of `col` (fixed bins or quantile bins)."""
    v = t[col].values
    if bins is not None:
        edges = np.array(bins, float)
    else:
        edges = np.unique(np.quantile(v[band], np.linspace(0, 1, q + 1)))
        edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = band & (v >= lo) & (v < hi)
        s = summarize(hit, m, t.snr.values, rng)
        s["label"] = f"{lo:.3g}-{hi:.3g}" if np.isfinite(hi) and hi < 1e8 else f">{lo:.3g}"
        out.append(s)
    return [o for o in out if o["n"] >= min_n]


def component_table(t, hit, mask_base, rng, min_n=30):
    comps = sorted({c for s in t.comps for c in s})
    out = []
    for c in comps:
        has = np.array([c in s for s in t.comps])
        a, b = mask_base & has, mask_base & ~has
        if a.sum() >= min_n and b.sum() >= min_n:
            sa, sb = summarize(hit, a, t.snr.values, rng), summarize(hit, b, t.snr.values, rng)
            out.append({"component": c, "with": sa, "without": sb, "diff": sa["recall"] - sb["recall"]})
    return sorted(out, key=lambda d: d["diff"])


def fp_table(t, hit, rng, min_n=30):
    base = t.scale.values == 0
    out = [{"component": "ALL burst-free images", **summarize(hit, base, np.where(base, 1.0, 1.0), rng)}]
    for c in sorted({c for s in t.comps for c in s}):
        has = np.array([c in s for s in t.comps])
        if (base & has).sum() >= min_n:
            out.append({"component": c, **summarize(hit, base & has, np.ones(len(t)), rng)})
    return sorted(out, key=lambda d: -d["recall"])  # 'recall' here = false-positive rate


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="sim_sweep")
    ap.add_argument("--catalog", default="chimefrbcat2.csv")
    ap.add_argument("--groups", nargs="+", default=["Step 1 models=runs/step1", "Step 2 models=runs/step2"])
    ap.add_argument("--snr-lo", type=float, default=8.0)
    ap.add_argument("--snr-hi", type=float, default=20.0)
    ap.add_argument("--out", default="runs/step2_sweep")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from step1_lib import RawScorer, pick_device

    files = sorted(glob.glob(os.path.join(args.sweep_dir, "*.npz")))
    if not files:
        raise SystemExit("no sweep images -- run step2_make_sweep.py first")
    from sim_common import load_catalog      # same one-row-per-event filter the simulators use
    catalog = load_catalog(args.catalog, verbose=False) if os.path.exists(args.catalog) else None
    t, arrays = build_table(files, lambda a: a, catalog)      # arrays = RAW images
    scorer = RawScorer(arrays)
    os.makedirs(args.out, exist_ok=True)

    score_path = os.path.join(args.out, "sweep_scores.npz")
    names, H = [], []
    device = pick_device(args.force_cpu)
    for spec in args.groups:
        gname, root = spec.split("=", 1)
        for seed, p in score_group(root, scorer, device).items():
            names.append(f"{gname} | {seed}"); H.append(p > 0.5)
    H = np.array(H)
    np.savez(score_path, names=np.array(names), hits=H)
    hit = H.mean(axis=0)  # fraction of the 10 models that detect each image
    rng = np.random.default_rng(0)
    snr, scale = t.snr.values, t.scale.values
    band = (scale > 0) & (snr >= args.snr_lo) & (snr < args.snr_hi)

    out = []
    def P(s=""):
        print(s); out.append(s)

    P(f"{len(names)} models pooled | {int((scale > 0).sum())} burst images, {int((scale == 0).sum())} burst-free")
    P(f"S/N band {args.snr_lo:g}-{args.snr_hi:g}: {int(band.sum())} burst images, overall recall {hit[band].mean():.3f}"
      f"  (95% CI {boot_ci(hit[band], rng)[0]:.3f}-{boot_ci(hit[band], rng)[1]:.3f})")

    def show(title, rows, extra=""):
        if len(rows) < 2:
            P(f"\n{title}\n   (fewer than 2 groups with enough images -- skipped)"); return
        P(f"\n{title}")
        P(f"   {'group':>14} {'n':>5} {'recall':>7} {'95% CI':>15} {'geo-mean S/N':>13}")
        for r in rows:
            P(f"   {r['label']:>14} {r['n']:>5} {r['recall']:>7.3f}  [{r['lo']:.3f}, {r['hi']:.3f}] {r['mean_snr']:>13.1f}")
        if extra:
            P("   " + extra)

    res = {}
    res["window"] = group_table(t, hit, band, "num_time", bins=[0, 64, 128, 256, 512, 1e9], rng=rng)
    show("Recall by NATIVE WINDOW LENGTH (samples; the CNN sees 256 bins -> long windows are downsampled):", res["window"],
         "check: 'geo-mean S/N' should be similar across rows, otherwise S/N is confounding the comparison")
    res["mask"] = group_table(t, hit, band, "mask_frac", q=3, rng=rng)
    show("Recall by FRACTION OF MASKED CHANNELS (terciles):", res["mask"])
    if "width_ms" in t:
        res["width"] = group_table(t, hit, band, "width_ms", q=3, rng=rng)
        show("Recall by BURST WIDTH in ms (catalog, terciles):", res["width"])
        res["scat"] = group_table(t, hit, band, "scat_ms", q=3, rng=rng)
        show("Recall by SCATTERING TIME in ms (catalog, terciles):", res["scat"])
    res["comp"] = component_table(t, hit, band, rng)
    P("\nRecall WITH vs WITHOUT each effect present in the image (same S/N band), sorted by difference:")
    P(f"   {'effect':<26} {'with (n)':>16} {'without (n)':>16} {'diff':>7}")
    for r in res["comp"]:
        P(f"   {r['component']:<26} {r['with']['recall']:>7.3f} ({r['with']['n']:>4}) {r['without']['recall']:>8.3f} ({r['without']['n']:>4}) {r['diff']:>+7.3f}")
    res["fp"] = fp_table(t, hit, rng)
    P("\nFALSE-POSITIVE rate on burst-free copies, by effect present in that background:")
    for r in res["fp"]:
        P(f"   {r['component']:<26} {r['recall']:>7.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]  n={r['n']}")

    # S/N50 by window-length group, PER MODEL GROUP (logistic fit); needs enough images
    from step2_eval_sweep import logistic_snr
    gnames = sorted({n.rsplit(" | ", 1)[0] for n in names}, key=lambda g: [n.rsplit(" | ", 1)[0] for n in names].index(g))
    P("\nS/N at 50% recall by NATIVE WINDOW LENGTH, per model group (logistic fit):")
    P(f"   {'window (samples)':<20}" + "".join(f"{g:>34}" for g in gnames))
    res["snr50_by_window"] = {}
    for lo, hi in zip([0, 64, 128, 256], [64, 128, 256, 1e9]):
        m = (scale > 0) & (t.num_time.values >= lo) & (t.num_time.values < hi)
        if m.sum() < 150:
            continue
        row, label = [], f"{lo}-{'inf' if hi > 1e8 else int(hi)} (n={int(m.sum())})"
        for g in gnames:
            idx = [i for i, n in enumerate(names) if n.rsplit(" | ", 1)[0] == g]
            xs = np.tile(snr[m], len(idx)); ys = H[idx][:, m].reshape(-1)      # both laid out model-major
            v = logistic_snr(xs, ys, 0.5); row.append(v)
            res["snr50_by_window"].setdefault(g, {})[label] = v
        P(f"   {label:<20}" + "".join(f"{v:>34.2f}" for v in row))

    json.dump(res, open(os.path.join(args.out, "diagnose_results.json"), "w"), indent=2, default=float)
    open(os.path.join(args.out, "diagnose_results.txt"), "w").write("\n".join(out))

    # ---------------- figure
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8))
    base_rec = hit[band].mean()
    def bars(ax, rows, title, xlabel):
        if not rows:
            ax.axis("off"); return
        x = np.arange(len(rows)); y = [r["recall"] for r in rows]
        err = [[r["recall"] - r["lo"] for r in rows], [r["hi"] - r["recall"] for r in rows]]
        ax.bar(x, y, color="#1b998b", yerr=err, capsize=4)
        ax.axhline(base_rec, color="#8a94a0", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([f"{r['label']}\n(n={r['n']})" for r in rows], fontsize=8.5)
        ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=10.5, color="#1f3a5f"); ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(f"recall at S/N {args.snr_lo:g}-{args.snr_hi:g}")
    bars(axs[0, 0], res["window"], "By native window length", "samples")
    bars(axs[0, 1], res["mask"], "By masked-channel fraction", "fraction of channels masked")
    bars(axs[1, 0], res.get("width", []), "By burst width (catalog)", "ms")
    ax = axs[1, 1]
    comp = res["comp"]
    if comp:
        y = np.arange(len(comp))
        ax.barh(y, [c["diff"] for c in comp], color=["#c0392b" if c["diff"] < 0 else "#1b998b" for c in comp])
        ax.set_yticks(y); ax.set_yticklabels([f"{c['component']} (n={c['with']['n']})" for c in comp], fontsize=8)
        ax.axvline(0, color="#8a94a0", lw=1)
        ax.set_title("Recall with minus without each effect", fontsize=10.5, color="#1f3a5f"); ax.set_xlabel("difference in recall")
    for a in axs.flat:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "diagnose_misses.png"), dpi=150)
    print(f"\nsaved {args.out}/diagnose_results.txt, diagnose_results.json, diagnose_misses.png, sweep_scores.npz")


if __name__ == "__main__":
    main()
