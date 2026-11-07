# Step 2b: brightness sweep (recall vs S/N)

| file | what it does |
|---|---|
| step2_make_sweep.py | 300 bursts x 11 images: the SAME burst on the SAME background at 10 brightness levels (2 %-100 % of catalog fluence) + a burst-free copy. Measures each image's matched-filter S/N. Needs `fitburst`. |
| step2_eval_sweep.py | scores every checkpoint in runs/step1 and runs/step2: recall vs S/N, S/N at 50 % and 90 % recall (logistic fit), false-positive rate on the burst-free copies. Needs torch. |

## Commands (working folder, venv active)
```
unzip -o ~/Downloads/fitburst_3_step2b.zip
python step2_make_sweep.py --catalog chimefrbcat2.csv --n 300 --outdir sim_sweep
python step2_eval_sweep.py --sweep-dir sim_sweep --groups "Step 1 models=runs/step1" "Step 2 models=runs/step2"
```
Outputs: runs/step2_sweep/sweep_recall_vs_snr.png and sweep_results.json.

## Read the results with these caveats
* S/N is on the SIMULATOR's scale (noise sigma is a free constant there), not CHIME's. Use it for comparing models and for the shape of the curve, not as an absolute detection limit.
* It is the template-aware (best-case) S/N; the CNN needs more, so its 50 % point sits higher.
* Burst parameters come from catalog rows that were also in the training set (backgrounds differ).
* Scales below 0.15 are outside the training distribution (faint bursts were 15-60 %).
