# Step 2a: physically correct impulsive RFI

Files in this zip (copy over your working folder; 3 files are REPLACED):
| file | change |
|---|---|
| sim_common.py | `add_impulsive_rfi` now renders zero-DM impulses as an inverted sweep after dedispersing at the candidate DM; 15 % are kept as the old vertical "mimic" streak (`IMPULSIVE_MIMIC_FRAC`) |
| simulate_frbs_fitburst.py | passes the burst's catalog DM to the renderer |
| simulate_rfi.py | passes the sampled catalog DM to the renderer |
| step2_eval_cross.py | scores the 5 Step-1 models on the NEW data without retraining |
| step2_impulsive_rfi_before_after.png | figure: old vs new impulsive RFI |

New metadata tags: `impulsive_rfi` = swept (new), `impulsive_mimic` = vertical streak (old style).
Old datasets (sim_output, sim_output_rfi) are untouched; new data go to *_v2 folders.

## Commands (working folder, venv active)
```
unzip -o ~/Downloads/fitburst_3_step2.zip

python simulate_frbs_fitburst.py --catalog chimefrbcat2.csv --outdir sim_output_v2
python simulate_rfi.py           --catalog chimefrbcat2.csv --outdir sim_output_rfi_v2

python step0_shortcut_audit.py --frb-dir sim_output_v2 --rfi-dir sim_output_rfi_v2 --max-n 1500   # must PASS (AUC < 0.58)

python step2_eval_cross.py --ckpt-root runs/step1 --frb-dir sim_output_v2 --rfi-dir sim_output_rfi_v2

caffeinate -i python step1_multiseed.py --frb-dir sim_output_v2 --rfi-dir sim_output_rfi_v2 --outdir runs/step2 --seeds 42 1 2 3 4
python step1_report.py --rundir runs/step2
```
Same seeds, same splits, same 60-epoch cap as Step 1: only the data changed, so the per-seed comparison is clean.
