# fitburst_3 — Step 0: remove the generator shortcut

## What was wrong (found by reading fitburst_2)
1. **Different noise pipelines per class.** The FRB generator built `burst*bandpass + noise + baseline` and stopped.
   The RFI generator additionally applied gain drift/jumps, time dropouts, scintillation, ADC quantization, narrowband
   RFI, impulsive/aircraft/periodic RFI, storms and pulsars to ~92 % of images. "Anything irregular" == RFI.
2. **`narrowband_bright` did nothing.** It edited `noise_scale_freq`, but noise was drawn from `NOISE_SIGMA * bandpass`.
3. **Different catalog subsets.** RFI used 4527 catalog rows for window sizing, FRB used 4081 (no `sub_num==0` filter in RFI).

## What changed
| file | change |
|---|---|
| `sim_common.py` | now holds ALL background code + `render_image()` (single path for both classes) + shared `load_catalog()` |
| `simulate_frbs_fitburst.py` | burst is injected via `render_image(..., burst_model=model)`; metadata gets `noise_components`, `label` |
| `simulate_rfi.py` | negatives = `render_image(..., burst_model=None)`; `--n` may exceed catalog size (sampled with replacement) |
| `step0_shortcut_audit.py` | burst-free classifier on edge columns; PASS needs AUC < 0.58 |
| `step0_cnn_shuffle_test.py` | trained-CNN test: shuffled / burst-erased images (untested, needs torch) |
| `train_frb_cnn_step0.ipynb` | same training code, outputs cleared, `OUTDIR=runs/step0_exp1` |

## Run order
```
python3 simulate_frbs_fitburst.py --catalog chimefrbcat2.csv --outdir sim_output
python3 simulate_rfi.py           --catalog chimefrbcat2.csv --outdir sim_output_rfi      # add --n 8000 for more negatives
python3 step0_shortcut_audit.py   --frb-dir sim_output --rfi-dir sim_output_rfi           # must PASS
# train: train_frb_cnn_step0.ipynb
python3 step0_cnn_shuffle_test.py --ckpt runs/step0_exp1/best_model.pt                    # shuffled acc ~0.5
# then re-run the Grad-CAM cells (alignment should clearly beat the 25 % chance level)
```
## Verification done in the sandbox (stand-in burst model, real fitburst not installed)
| generators | edge-only AUC |
|---|---|
| fitburst_2 (old) | 0.819 FAIL |
| fitburst_3 (new) | 0.502 PASS |
