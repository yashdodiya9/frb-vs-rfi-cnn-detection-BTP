#!/usr/bin/env python3
"""
step3_convert_h5.py -- ONE-TIME conversion of the CHIME/FRB Catalog-2 waterfalls into a compact cache.

Reads each .h5 file (data, flag, good_freq, model + a few attributes), averages 16384 -> 256 channels, and stores
  X          N arrays of (256, T) float32  averaged Stokes-I waterfalls; T (window length) VARIES per file
  ch_ok      (N, 256) bool              channels kept
  model_prof N arrays of (T,) float32      frequency-summed best-fit model (or pulse_emission_region boxcar if missing)
  meta       one row per file (tns_name, event_id, dm, width_s, scat_s, beam, file) as a pickled object array
so the 53 GB are read only once. The cache is ~750 MB for all 4540 files.

The .h5 files and the cache stay OUTSIDE your git repository.

Usage:
    python step3_convert_h5.py --h5-dir ~/Desktop/using-h5-files/h5_files --out real_cache.npz --max-files 50   # test
    python step3_convert_h5.py --h5-dir ~/Desktop/using-h5-files/h5_files --out real_cache.npz --workers 4        # all
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from step3_real_lib import average_to_256


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def convert_one(path):
    import h5py
    try:
        with h5py.File(path, "r") as f:
            data, flag, good = f["data"][()], f["flag"][()], f["good_freq"][()]
            model = f["model"][()] if "model" in f else None          # a few files have no fitted model
            a = f.attrs
            has_params = "burst_parameters_json" in a          # ~2 % of files have no fit parameters
            if has_params:
                params = json.loads(_s(a["burst_parameters_json"]))
                width_s, scat_s = float(params["burst_width"][0]), float(params["scattering_timescale"][0])
            else:
                width_s, scat_s = float("nan"), float("nan")   # the window rule then falls back to 64 samples
            name = os.path.basename(path).split("_")[0]
            meta = {"file": os.path.basename(path), "tns_name": _s(a["tns_name"]) if "tns_name" in a else name,
                    "event_id": _s(a["event_id"]) if "event_id" in a else "",
                    "dm": float(a["dm_incoherent"]) if "dm_incoherent" in a else float("nan"),
                    "beam": int(a["beam_number"]) if "beam_number" in a else -1,
                    "width_s": width_s, "scat_s": scat_s, "has_params": has_params,
                    "num_time": int(a["num_time"]) if "num_time" in a else data.shape[1],
                    "res_time": float(a["res_time"]) if "res_time" in a else 0.00098304, "has_model": model is not None}
            region = np.asarray(a["pulse_emission_region"]).reshape(-1, 2) if "pulse_emission_region" in a else None
        if data.shape[0] != 16384 or flag.shape != data.shape:
            return path, None, f"unexpected shape {data.shape}"
        img, ok = average_to_256(data, flag.astype(bool), good.astype(bool))
        t = data.shape[1]
        if model is not None:
            prof = np.where(np.isfinite(model), model, 0.0).astype(np.float64).sum(axis=0).astype(np.float32)
        else:                                         # fall back to the pulse_emission_region(s) as a boxcar
            prof = np.zeros(t, np.float32)
            if region is not None and len(region):
                for lo, hi in region:
                    prof[max(int(lo), 0):min(int(hi) + 1, t)] = 1.0
            else:
                prof[t // 2 - 2:t // 2 + 3] = 1.0
        meta["T"] = t
        if not np.all(np.isfinite(img)):
            return path, None, "non-finite after averaging"
        return path, (img, ok, prof, meta), None
    except Exception as e:  # keep going; report at the end
        return path, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--out", default="real_cache.npz")
    ap.add_argument("--max-files", type=int, default=None, help="only convert the first N files (test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.h5_dir), "**", "*.h5"), recursive=True))
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        raise SystemExit("no .h5 files found")
    print(f"{len(files)} files; converting with {args.workers} worker(s) ...")
    X, OK, PROF, META, bad = [], [], [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (p, res, err) in enumerate(ex.map(convert_one, files, chunksize=4)):
            if res is None:
                bad.append((os.path.basename(p), err)); continue
            X.append(res[0]); OK.append(res[1]); PROF.append(res[2]); META.append(res[3])
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)")
    Xo, Po = np.empty(len(X), dtype=object), np.empty(len(X), dtype=object)    # variable window length per file
    for i in range(len(X)):
        Xo[i], Po[i] = X[i], PROF[i]
    np.savez(args.out, X=Xo, ch_ok=np.stack(OK), model_prof=Po, meta=np.array(META, dtype=object))
    ts = np.array([m["T"] for m in META])
    print("window length in the files (samples): min %d, median %d, max %d | %s" % (
        ts.min(), np.median(ts), ts.max(), ", ".join(f"{int(v)}: {int(n)}" for v, n in zip(*np.unique(ts, return_counts=True)))[:300]))
    print(f"{sum(not m['has_model'] for m in META)} files without a fitted model (burst zone taken from pulse_emission_region)")
    print(f"saved {len(X)} events to {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - t0:.0f}s; "
          f"{len(bad)} failed")
    if bad:
        from collections import Counter
        print("failure reasons:")
        for reason, n in Counter(e.split(":")[0] if "KeyError" not in e else "KeyError " + e.split("attribute:")[-1][:60] for _, e in bad).most_common():
            print(f"  {n:4d} x {reason}")
        for name, err in bad[:5]:
            print("  e.g.", name, err[:150])
    print(f"{sum(not m['has_params'] for m in META)} saved events have no fit parameters (window rule falls back to 64 samples)")


if __name__ == "__main__":
    main()
