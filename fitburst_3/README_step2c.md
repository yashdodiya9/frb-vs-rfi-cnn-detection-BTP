# Step 2c: where do the misses concentrate?

`step2_diagnose_sweep.py` re-scores the sweep images (sim_sweep/, already generated) with all 10 checkpoints,
holds S/N fixed (band 8-20 by default) and splits the burst images by:
native window length, masked-channel fraction, burst width, scattering, and which RFI/instrument effects are present;
plus the false-positive rate on the burst-free copies split by effect. Every row shows n and a 95 % bootstrap CI.

## Command (working folder, venv active)
```
unzip -o ~/Downloads/fitburst_3_step2c.zip
python step2_diagnose_sweep.py --sweep-dir sim_sweep --catalog chimefrbcat2.csv \
    --groups "Step 1 models=runs/step1" "Step 2 models=runs/step2"
```
Outputs in runs/step2_sweep/: diagnose_results.txt / .json, diagnose_misses.png, sweep_scores.npz (per-image scores, reusable).

## Reading it
* Only trust a difference that is larger than the CIs. With ~300 bursts the CIs are wide (+-8 points per group).
* 'geo-mean S/N' per row must be similar across rows; otherwise S/N, not the property, explains the difference.
* If long windows (>256 samples) are clearly worse, the 256-bin interpolation is losing signal -> test block-averaging next.
* If more bursts would help the CIs, regenerate the sweep with --n 800 (about 10 minutes).
