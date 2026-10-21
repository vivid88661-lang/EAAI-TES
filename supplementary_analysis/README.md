# Supplementary analysis for the 18 October 2026 revision

This folder contains the additional experiments reported in Section 3, Table 8
and Appendix Tables A4-A6 of the revised manuscript. Place it at the root of
the existing repository, beside `data/`, `src/` and `models/`.

The original repository files and frozen models remain unchanged. Table
numbers in the original root README refer to the earlier manuscript version;
the mapping below applies to these supplementary analyses.

## Manuscript mapping

| Manuscript item | Included evidence | Reproduction command |
|---|---|---|
| Table 8 | Seed-42 persistence, absolute-target LCA and state-anchored LCA on 25,599 common origins | `python supplementary_analysis/reproduce.py` |
| Appendix A4 | Direct forecasts at 30 s, 5 min and 15 min, with persistence and 30-point linear trend | Same command |
| Appendix A5 | State-anchored grouped OOF results for seeds 42, 123, 2024 and 3407 | Same command |
| Appendix A6 | E01/E02 source trajectories, processed data and four-seed frozen predictions | Same command |
| Section 3 | Shared windows, model training, epoch selection and frozen inference | Scripts under `src/` |

## Quick numerical reproduction

Run these commands from the repository root:

```bash
python -m pip install -r supplementary_analysis/requirements.txt
python supplementary_analysis/reproduce.py
```

This route requires NumPy and pandas only. It checks SHA-256 checksums,
reconstructs the historical persistence and linear-trend forecasts from the
case data, verifies alignment of all prediction origins, and recomputes every
reported supplementary table from the saved prediction rows. It also checks
the E01/E02 processed trajectories against the original third-discharge CSVs.
Generated tables and the verification result are written to
`supplementary_analysis/outputs/reproduced/` and compared with `reference/`.

The saved predictions and references are the frozen numerical records used
for the manuscript. Recalculating their metrics does not require retraining.

## Frozen-model inference

The training and inference environment is the original Python 3.10 environment
with TensorFlow 2.15.0 and the versions in the root `requirements.txt`. Use a
separate environment for this route:

```bash
python -m pip install -r supplementary_analysis/requirements-training.txt
python supplementary_analysis/src/inference.py --variant state_anchored
python supplementary_analysis/src/inference.py --variant direct
```

The first command applies the four frozen state-anchored models to E01/E02.
The second applies the three frozen direct-horizon models to the previously
examined reference condition `E_RI002`. All scalers, neural weights and linear
meta-model parameters are loaded without fitting. Outputs are written under
`outputs/inference/`. Small numerical differences can occur across hardware,
TensorFlow kernels and dependency versions.

## Optional full retraining

Full retraining is computationally expensive. Run each command in a separate
process and use an empty output directory. For the direct-horizon analysis:

```bash
python supplementary_analysis/src/train.py --variant direct --seed 42 --horizon 1 --output supplementary_analysis/outputs/direct_h01
python supplementary_analysis/src/train.py --variant direct --seed 42 --horizon 10 --output supplementary_analysis/outputs/direct_h10
python supplementary_analysis/src/train.py --variant direct --seed 42 --horizon 30 --output supplementary_analysis/outputs/direct_h30
```

For the state-anchored four-seed analysis:

```bash
python supplementary_analysis/src/train.py --variant state_anchored --seed 42 --output supplementary_analysis/outputs/state_seed42
python supplementary_analysis/src/train.py --variant state_anchored --seed 123 --output supplementary_analysis/outputs/state_seed123
python supplementary_analysis/src/train.py --variant state_anchored --seed 2024 --output supplementary_analysis/outputs/state_seed2024
python supplementary_analysis/src/train.py --variant state_anchored --seed 3407 --output supplementary_analysis/outputs/state_seed3407
```

The script imports the original LCA neural-network builders and fold-specific
preprocessor from `../src/preprocessing.py`. `src/common.py` preserves the
window construction used to generate the saved results. The training entry
point organizes that procedure into a portable interface. It writes new
results separately and never overwrites the frozen records. Full retraining
is not expected to be bitwise identical across environments.

To apply a newly retrained deployment model to its corresponding external
condition, use the output directory from `train.py`, for example:

```bash
python supplementary_analysis/src/inference.py --variant state_anchored --trained-run supplementary_analysis/outputs/state_seed42 --output supplementary_analysis/outputs/retrained_inference
```

## Evaluation and training protocol

- All models use 30 historical samples at a 30 s interval. Inputs include the
  simulated observed history of liquid fraction, inlet temperature, outlet
  temperature and cumulative discharged heat, plus geometry and the original
  derived historical heat feature. The setting is conditional rolling
  prediction with observed history.
- Common zero-based prediction origins are `t=29,...,n-31`. This leaves the
  same origins available at horizons 1, 10 and 30. There are 25,599 internal
  origins across D001-D032, 895 in E_RI002, 714 in E01 and 446 in E02.
- Internal evaluation uses five outer case-wise GroupKFold partitions and
  two inner folds within the corresponding outer training cases. The linear
  meta-learner fits only the inner OOF predictions; outer test cases do not
  enter preprocessing, epoch selection or meta-model fitting.
- The absolute-target models use ten epochs. For the state-anchored variant,
  each base learner selects the earliest epoch with the lowest case-averaged
  physical validation MSE over 30 epochs, using the first three-fold
  GroupKFold split of its fit cases. It is then refitted on all fit cases.
  Training runs through `max(10, selected_epoch)` and retains the checkpoint
  at the selected epoch. Recorded choices are in
  `configuration/selected_epochs.csv`.
- Temperature increments are multiplied by the training outlet-temperature
  MinMax scale without an additive offset. Heat increments in MJ are divided
  by their training-origin RMS. The linear meta-learner combines physical
  increments; the latest observed state is added once.
- A final deployment model fits all 32 development cases, using outer base
  OOF predictions to fit its meta-learner. E01/E02 do not enter fitting or
  parameter selection. E_RI002 is the previously examined external reference.
- Persistence uses the latest observed target. Linear trend fits an OLS line
  to the 30 historical target observations and extrapolates to the chosen
  horizon. Heat extrapolation is performed in physical J and reported in MJ.
- Table 8 uses seed 42. Tables A5/A6 calculate pooled RMSE separately for each
  seed and then report the mean and sample standard deviation (`ddof=1`).
  The pooled A6 row combines all 1,160 origins within each seed before RMSE
  is calculated; it is not an average of the two case RMSEs.

## E01/E02 data provenance

`data/source/` contains the original positive-time and zero-time sampled
third-discharge trajectories from the numerical solver. Each CSV has eight
columns, in order: time (min), liquid fraction, inlet temperature (degrees C),
outlet temperature (degrees C), interval discharged heat (J), cumulative
discharged heat (J), and two temperature-difference diagnostics (K). The
positive-time records are the samples used for preprocessing. Descriptors
are in `configuration/external_cases.json`.

The E01/E02 trajectories were generated using the numerical method described
in the manuscript. The numerical solver source, solver adapters and
material-property routines are not distributed in this repository. The
included sampled trajectories, case descriptors and processing code support
reproduction of the data-driven analysis from the supplied simulation data.
Regeneration of the underlying numerical simulations is outside the scope of
this public package.

To reconstruct a processed trajectory from the supplied sampled discharge
data:

```bash
python supplementary_analysis/src/prepare_external.py --case E01 --source supplementary_analysis/data/source/E01_cycle03_discharge.csv --output supplementary_analysis/outputs/E01.csv.gz
```

The 32 development trajectories and E_RI002 are reused from the original
repository `data/` folder, avoiding duplicate copies. All included files and
the required repository dependencies are identified in `checksums.json`.

## Folder contents

```text
configuration/  protocol, case descriptors, splits and selected epochs
data/           E01/E02 processed trajectories and source discharge CSVs
models/         frozen final neural weights, preprocessors and meta-models
predictions/    only the prediction columns needed for Tables 8/A4/A5/A6
reference/      numerical table references and per-seed values
src/            data processing, training, historical windows and inference
reproduce.py    deterministic source-data and table verification
```

The original repository's review-use notice also applies to these materials.
