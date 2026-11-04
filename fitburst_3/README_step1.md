# Step 1 add-ons (copy these files INTO your existing fitburst_3 folder)

| file | what it does |
|---|---|
| step1_lib.py | same model / preprocessing / augmentation / train loop as the notebook + RAM cache |
| step1_multiseed.py | trains N seeds (own split + init each), up to 60 epochs, early stop patience 10 |
| step1_report.py | mean ± std, convergence check, error breakdown by RFI type / burst brightness, figure |
| step1_make_pairs.py | paired images: SAME background, with / without burst |
| step1_eval_pairs.py | scores every seed model on the pairs (+ shuffled pairs) |

Seed 42 reproduces the notebook's split exactly, so it is directly comparable to your Step-0 run.
Data needed: sim_output/ and sim_output_rfi/ (already generated), chimefrbcat2.csv (for pairs).

## Commands (run inside fitburst_3, venv active)
```
unzip -o fitburst_3_step1.zip                      # puts the files next to your scripts

# 1) smoke test (a few minutes): proves everything runs + shows seconds per epoch
python3 step1_multiseed.py --seeds 42 --epochs 2 --outdir runs/step1_smoke

# 2) the real run (resumable; finished seeds are skipped if you stop and restart)
caffeinate -i python3 step1_multiseed.py --seeds 42 1 2 3 4

# 3) results
python3 step1_report.py --rundir runs/step1

# 4) paired "does it need the burst?" test
python3 step1_make_pairs.py --catalog chimefrbcat2.csv --n 600 --outdir sim_paired
python3 step1_eval_pairs.py --rundir runs/step1 --pairs-dir sim_paired
```
`caffeinate -i` stops the Mac sleeping mid-run. Add `--force-cpu` if the Apple GPU gives trouble.
