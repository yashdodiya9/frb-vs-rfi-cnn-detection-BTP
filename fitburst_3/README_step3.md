# Step 3 (real CHIME data): your trained models on real Catalog-2 bursts, no retraining

| file | what it does |
|---|---|
| step3_real_lib.py | channel averaging (16384 -> 256, flag-aware), window rule (identical to the simulator's), crops; handles variable window length per file |
| step3_convert_h5.py | ONE-TIME conversion of the .h5 files into one compact cache (~750 MB for all 4540) |
| step3_eval_real.py | scores every checkpoint on real bursts + real burst-free windows; recall vs catalog S/N |
| step3_real_vs_sim_figure.py | real vs simulated bursts, as the CNN sees them (a good slide) |

## Commands (working folder fitburst_3-step-0_all, venv active, h5py installed)
```
unzip -o ~/Downloads/fitburst_3_step3_real.zip

# 1) test on 50 files and note the time it takes
python step3_convert_h5.py --h5-dir ~/Desktop/using-h5-files/h5_files --out ~/Desktop/using-h5-files/real_cache_test.npz --max-files 50

# 2) all files (time = test time x ~90); keep the cache OUTSIDE the git repo
python step3_convert_h5.py --h5-dir ~/Desktop/using-h5-files/h5_files --out ~/Desktop/using-h5-files/real_cache.npz --workers 4

# 3) figure: real vs simulated
python step3_real_vs_sim_figure.py --cache ~/Desktop/using-h5-files/real_cache.npz --catalog chimefrbcat2.csv --sim-dir sim_output_v2

# 4) the evaluation
python step3_eval_real.py --cache ~/Desktop/using-h5-files/real_cache.npz --catalog chimefrbcat2.csv --groups "Step 2 models (interp)=runs/step2" "Area-averaging models=runs/step2d_area" --out runs/step3_real
```
If step 4 runs out of memory, add `--max-events 800`.

## Caveats to quote with the result
* Only DETECTED bursts exist in these files: biased to S/N >~ 9; no real undetected faint bursts, no labelled real-RFI false alarms.
* Negatives are burst-free windows of the same files: real telescope background, NOT real RFI candidates.
* bonsai_snr (real-time pipeline S/N) is not the simulated matched-filter S/N: do not overlay the two curves.
* The window length VARIES per file (e.g. 40, 62, 82, 162 samples). Each burst is cropped to the simulator's rule length if the file has that many samples, otherwise the whole window is used (the evaluation prints how many events that affects).
* Negatives (burst-free windows) are only possible from files with long windows; short-window files give none, so the negatives are a biased subset.
* A few files have no fitted 'model' and ~2 % have no 'burst_parameters_json' (no fit): their burst zone comes from pulse_emission_region and their window falls back to 64 samples.
* Infinite / NaN samples are treated as missing data (they overflowed the first version).

## Step 3b: look at the mistakes (run after step3_eval_real.py, same --max-events and --seed)
```
python step3_show_errors.py --cache ~/Desktop/using-h5-files/real_cache.npz --catalog chimefrbcat2.csv
```
Prints ensemble ROC AUC on real data, recall at P>0.5 and at thresholds set for 1 % / 5 % false alarms on real backgrounds,
whether false alarms come disproportionately from REPEATING sources, and lists the worst false alarms and misses;
saves runs/step3_real/real_errors.png.

## v5: crop on the brightest component (fix) + optional exploratory hot-channel masking
* The file's window is centred on the MEAN of the sub-burst arrival times; for widely separated components that centre is
  empty sky. `--center peak` (default now) crops around the brightest component of the fitted model; `--center middle`
  reproduces the first version. step3_eval_real.py prints how many bursts are affected.
* `--hot-k 4` (EXPLORATORY, off by default) zeroes channels whose scatter exceeds 4x the median (unflagged narrowband RFI lines).
  It is NOT part of the training pipeline, so report it only as a labelled experiment, never as the headline number.
* Run the evaluation and the gallery with the SAME --center/--hot-k/--max-events/--seed (the gallery checks alignment).

## v6: why are some real bursts missed? (run after the v5 commands)
* step3_eval_real.py now also prints recall by CHIME-fit status: ~2 % of files have no fitted parameters (CHIME's own fit
  failed); their window falls back to 64 samples, which is ALSO the rule length of many ordinary bursts.
* step3_inspect_event.py plots whole events (full window, channel-averaged series, model profile, crop position).
```
python step3_eval_real.py --cache ~/Desktop/using-h5-files/real_cache.npz --catalog chimefrbcat2.csv --groups "Step 2 models (interp)=runs/step2" "Area-averaging models=runs/step2d_area" --out runs/step3_real_peak
python step3_inspect_event.py --cache ~/Desktop/using-h5-files/real_cache.npz --catalog chimefrbcat2.csv --names FRB20190208B FRB20180912A FRB20190729B FRB20230706E --out runs/step3_real_peak/inspect.png
```

## v7: is the burst visible in its own waterfall? (step3_consistency.py)
FRB20190208B (catalog S/N 96) is missed by every model, and the matched filter of CHIME's own fitted model against the raw
data gives S/N -0.1 with or without flags: the supplied waterfall does not contain the signal. step3_consistency.py measures
this for every event and tabulates the CNN's recall by visibility (all bins shown; nothing is cut using the CNN's output).
```
python step3_consistency.py --cache ~/Desktop/using-h5-files/real_cache.npz --scores runs/step3_real_peak/real_scores.csv
```
