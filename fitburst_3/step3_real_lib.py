"""
step3_real_lib.py -- turning real CHIME/FRB Catalog-2 waterfalls into the images our models were trained on.
(numpy only, so it can be tested without h5py / torch)

What a Catalog-2 .h5 file holds (from your inspect_h5.py output):
  data        (16384 freq, 162 time) Stokes-I, per-channel mean-subtracted and mean-normalised, already
              incoherently DEDISPERSED at the burst's DM, window centred on the burst (sample 81)
  flag        (16384, 162) True = valid sample          good_freq (16384,) True = usable channel
  model       (16384, 162) best-fit burst model (same units)
  freqs ascending 400.2 -> 800.2 MHz, 0.983 ms sampling (same as the simulation)

Conversion (done once, step3_convert_h5.py):
  * average 64 neighbouring channels (16384 -> 256, flag-aware); channels that are mostly flagged are set to 0,
    exactly like masked channels in the simulation
  * keep the full 162-sample window + the frequency-summed model profile (used to find burst-free samples)

Cropping (done at evaluation time):
  * POSITIVE: centred crop of the length the simulator's own window rule would give this burst (so the models
    see bursts at the same scale as in training); capped at the 162 samples that exist
  * NEGATIVE: windows taken from burst-free parts of the same file (real telescope background)
"""
import numpy as np

# Window length VARIES per file (e.g. 40, 62, 82, 162 samples); the burst always sits at sample num_time // 2.
NATIVE_TSAMP_S = 0.00098304
MIN_NUM_TIME, MAX_NUM_TIME = 32, 512  # same as sim_common.py
MIN_WINDOW_NATIVE_SAMPLES = 4
DISPLAY_PAD_FACTOR = 5.0
MAX_WINDOW_S = 0.4


def average_to_256(data, flag, good_freq, group=64, min_good_frac=0.5):
    """(16384, T) -> (256, T), flag-aware. Returns (img, channel_ok)."""
    n_f, t = data.shape
    assert n_f % group == 0
    n_out = n_f // group
    finite = np.isfinite(data)                       # inf/NaN samples count as missing (they caused overflow before)
    valid = flag & good_freq[:, None] & finite
    d = np.where(valid, np.where(finite, data, 0.0).astype(np.float64), 0.0)
    s = d.reshape(n_out, group, t).sum(axis=1)
    c = valid.reshape(n_out, group, t).sum(axis=1)
    img = s / np.maximum(c, 1)
    frac_good = good_freq.reshape(n_out, group).mean(axis=1)
    frac_valid_time = c.mean(axis=1) / group
    ok = (frac_good >= min_good_frac) & (frac_valid_time >= min_good_frac)
    img[~ok, :] = 0.0
    return img.astype(np.float32), ok


def rule_length(width_s, scat_s):
    """Window length (native samples) the SIMULATOR's rule gives this burst (sim_common.choose_grid, dedispersed).
    NaN/invalid width -> 64 samples."""
    if not (np.isfinite(width_s) and np.isfinite(scat_s)) or width_s <= 0:
        return 64
    extent = max(6 * width_s + 2.5 * scat_s, MIN_WINDOW_NATIVE_SAMPLES * NATIVE_TSAMP_S)
    window_s = min(DISPLAY_PAD_FACTOR * extent, MAX_WINDOW_S)
    return int(np.clip(round(window_s / NATIVE_TSAMP_S), MIN_NUM_TIME, MAX_NUM_TIME))


def window_length(width_s, scat_s, t_avail=None):
    """Rule length, but never longer than the window that exists in the file."""
    n = rule_length(width_s, scat_s)
    return min(n, int(t_avail)) if t_avail else n


def burst_center(model_profile, t):
    """Sample of the brightest burst component (peak of the frequency-summed fitted model; middle of the
    pulse_emission_region boxcar when there is no model). The FILE centre (t//2) is the MEAN of the sub-burst
    arrival times, which for widely separated components lies between them, on empty sky."""
    p = np.asarray(model_profile, float)
    p = np.where(np.isfinite(p), p, 0.0)
    if p.size != t or p.max() <= 0:
        return t // 2
    return int(round(np.flatnonzero(p >= 0.999 * p.max()).mean()))


def crop_positive(img, width_s, scat_s, center=None):
    """Crop of the simulator-rule length (or the whole window if shorter) around `center` (default: file centre)."""
    t = img.shape[1]
    w = window_length(width_s, scat_s, t)
    c = t // 2 if center is None else int(center)
    start = int(np.clip(c - w // 2, 0, t - w))
    return img[:, start:start + w].copy()


def mask_hot_channels(img, k=4.0):
    """EXPLORATORY: zero channels whose robust temporal scatter exceeds k x the median channel scatter
    (unflagged narrowband RFI lines). Not part of the training pipeline, so use only as a labelled experiment."""
    out = img.copy()
    med = np.median(out, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(out - med), axis=1)
    live = mad > 0
    if live.sum() < 8:
        return out
    hot = mad > k * np.median(mad[live])
    out[hot, :] = 0.0
    return out


def burst_zone(model_profile, margin=10, rel_threshold=0.01):
    """First/last sample (inclusive) of the burst (+ margin), from the frequency-summed fitted model
    (or the pulse_emission_region boxcar the converter stores when a file has no model)."""
    p = np.asarray(model_profile, float)
    p = np.where(np.isfinite(p), p, 0.0)              # a few fitted models contain inf: ignore those samples
    t = len(p)
    if p.max() <= 0:
        return max(t // 2 - max(3, t // 8), 0), min(t // 2 + max(3, t // 8), t - 1)
    idx = np.where(p > rel_threshold * p.max())[0]
    return max(int(idx.min()) - margin, 0), min(int(idx.max()) + margin, t - 1)


def crop_negatives(img, model_profile, width_s, scat_s, rng, min_len=32, margin=10):
    """Up to two burst-free windows (left / right of the burst zone) of length min(rule length, segment length).
    Short windows (e.g. 40 samples) leave no room: those files give no negatives."""
    zs, ze = burst_zone(model_profile, margin)
    w_rule = rule_length(width_s, scat_s)
    out = []
    for lo, hi in ((0, zs), (ze + 1, img.shape[1])):
        seg = hi - lo
        if seg >= min_len:
            w = min(w_rule, seg)
            start = int(rng.integers(lo, hi - w + 1))
            out.append(img[:, start:start + w].copy())
    return out


def build_crops(X, prof, widths, scats, idx, rng, center="peak", hot_k=None):
    """The ONE place where evaluation crops are built (used by step3_eval_real.py and step3_show_errors.py, so they
    always agree). Returns raws, labels, event_index, window_length, centre_shift (samples from the file centre)."""
    raws, labels, ev, win, shift = [], [], [], [], []
    for i in idx:
        t = X[i].shape[1]
        bc = burst_center(prof[i], t)                     # where the brightest component really is
        c = bc if center == "peak" else t // 2
        pos = crop_positive(X[i], widths[i], scats[i], c)
        negs = crop_negatives(X[i], prof[i], widths[i], scats[i], rng)
        for img, lab in [(pos, 1)] + [(n, 0) for n in negs]:
            raws.append(mask_hot_channels(img, hot_k) if hot_k else img)
            labels.append(lab); ev.append(i); win.append(img.shape[1]); shift.append(bc - t // 2 if lab == 1 else 0)
    return raws, np.array(labels), np.array(ev), np.array(win), np.array(shift)


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
