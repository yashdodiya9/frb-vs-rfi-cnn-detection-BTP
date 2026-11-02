# Deep Learning Detection of Fast Radio Burst signals in radio telescope data

BTP project - Yash Dodiya (230021008), supervisor Prof. Mukul Bhattacharya,
Dept. of Astronomy, Astrophysics and Space Engineering.

A CNN that acts as a fast first-pass filter, separating genuine FRBs from RFI/noise in
dedispersed dynamic spectra (frequency x time images). Training data are simulated with the
CHIME/FRB `fitburst` model, with burst parameters bootstrapped from the CHIME/FRB Catalog 2.

## Project stages (one folder per stage, tagged in git)

| folder | git tag | what it is | test accuracy |
|---|---|---|---|
| `fitburst_1/` | `v1-midterm` | work up to the mid-term: four FRB simulator versions, RFI simulator, CNN runs `exp1`-`exp4` (`exp4` = mid-term report run) | 0.990 (exp4) |
| `fitburst_2/` | `v2-improvements` | extension, not in the report: shared `sim_common.py`, faint bursts, augmentation, Grad-CAM diagnostics | 0.982-0.988 |
| `fitburst_3/` | `v3-step0` | Step 0: removes a generator shortcut (one shared background renderer for both classes), plus audit and shuffle tests | 0.949 |
| `data/` | | `chimefrbcat2.csv`, CHIME/FRB Catalog 2 table used to sample burst parameters | |
| `docs/` | | mid-term evaluation report | |

**Reading the numbers:** the accuracy drop from stage 1/2 to stage 3 is expected. In `fitburst_1`/`fitburst_2`
the FRB and RFI images were rendered by different code paths (RFI images got drift, dropouts, quantization,
RFI etc.; FRB images did not), so the CNN could partly separate classes without seeing a burst. In `fitburst_3`
both classes share one renderer; a classifier on burst-free image edges scores AUC 0.48 (chance = 0.50) and
Grad-CAM attention sits on the burst. The 0.949 is the honest baseline.

Each `expN/` or `step0_exp1/` folder holds `history.json`, `test_metrics.json` and the result figures of one run.
Datasets (`sim_output*/`, `*.npz`) and model weights (`*.pt`) are not tracked; regenerate them with the
simulators, e.g. `python3 simulate_frbs_fitburst.py --catalog ../data/chimefrbcat2.csv`.

## Requirements
Python 3.12; `pip install -r fitburst_3/requirements.txt`; plus the CHIME/FRB
[`fitburst`](https://github.com/CHIMEFRB/fitburst) package (install from a clone with `pip install .`,
and keep the clone OUTSIDE this repo, otherwise Python imports the wrong `fitburst` folder).

## Status
Work in progress. Next: multi-seed training (mean +- std), augmentation / noise hardening, real CHIME data and
real RFI evaluation, pretrained CNNs (see the Next Steps section of the mid-term report).
