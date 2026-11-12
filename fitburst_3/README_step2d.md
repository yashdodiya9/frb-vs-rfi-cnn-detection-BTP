# Step 2d: block-averaging (area) vs interpolation when shrinking long windows

Hypothesis (from the Step 2c diagnostic): windows longer than 256 samples are shrunk to 256 bins by linear
interpolation, which reads ~2 samples per output bin and discards the rest; the same-S/N recall for 256-512-sample
windows was ~0.2 lower (S/N50 11.7 vs 8.6-10.2). Block-averaging keeps every sample.

Controlled test (white noise + Gaussian burst, matched-filter S/N in the 256-bin CNN input), area / interp:
  shrink 1.2x: 1.04-1.07 | 1.6x: 1.13-1.16 | 2x: 1.13-1.16 | 3x: 1.26-1.40 | 4x: 1.54-1.58
=> expected effect on the >=256-sample group: S/N50 about 1.15x lower (11.7 -> ~10.2); windows <=256 are
UNCHANGED (upsampling still uses interpolation, verified bit-identical). Do not expect a big overall change.

## SUCCESS CRITERION (written down before the run)
The change helps if, on the same sweep images:
  1. S/N50 of the >=256-sample group is at least 1.5 lower for the area models than for the Step 2 (interp) models,
  2. S/N50 of the shorter-window groups does not rise by more than 1.0,
  3. the false-positive rate on burst-free copies is not above the Step 2 models' (3.3 %).
Otherwise it is a null result (still useful: it settles the preprocessing for real data).

## What changed in the code
* step1_lib.py: area_resample_time(), resize_mode ('interp' default = old behaviour), RawScorer / score_checkpoint
  (each checkpoint is scored with the resize mode it was trained with; old checkpoints = interp).
* step1_multiseed.py: --resize-mode {interp,area}; the mode is stored in every checkpoint.
* step2_eval_sweep.py / step2_diagnose_sweep.py: score with each checkpoint's own mode; diagnose now prints
  S/N50 by window length PER MODEL GROUP.
* step1_eval_pairs.py, step2_eval_cross.py, step2_compare_same_split.py: refuse area checkpoints (they only know interp).

## Commands (working folder, venv active)
```
unzip -o ~/Downloads/fitburst_3_step2d.zip

# 0) smoke test (~minutes)
python step1_multiseed.py --frb-dir sim_output_v2 --rfi-dir sim_output_rfi_v2 --outdir runs/step2d_smoke --resize-mode area --seeds 42 --epochs 2

# 1) the real run: same data, same seeds/splits as runs/step2, only the preprocessing differs
caffeinate -i python step1_multiseed.py --frb-dir sim_output_v2 --rfi-dir sim_output_rfi_v2 --outdir runs/step2d_area --resize-mode area --seeds 42 1 2 3 4
python step1_report.py --rundir runs/step2d_area

# 2) a bigger sweep for statistical power (the >=256 group has only ~460 images in the 300-burst sweep)
python step2_make_sweep.py --catalog chimefrbcat2.csv --n 800 --seed 7 --outdir sim_sweep800

# 3) compare
python step2_eval_sweep.py     --sweep-dir sim_sweep800 --groups "Step 2 models (interp)=runs/step2" "Area-averaging models=runs/step2d_area" --out runs/step2d_sweep
python step2_diagnose_sweep.py --sweep-dir sim_sweep800 --catalog chimefrbcat2.csv --groups "Step 2 models (interp)=runs/step2" "Area-averaging models=runs/step2d_area" --out runs/step2d_sweep
```
