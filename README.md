# TES LCA-Stacking reproducibility package

This repository contains the frozen implementation, processed case-wise data,
trained artifacts and evaluation scripts required to reproduce the main
**data-driven** modelling and evaluation results reported in the manuscript.

It is a publication package, not an archive of the complete development
history. The 2D MATLAB source solver, 1D reduced-order model, Lacroix validation
processing, internal audits and obsolete experiments are outside this
repository's scope. Consequently, Table 10 is not reproduced here.

## Paper-to-repository map

| Repository result | Manuscript result | Reproduction route |
|---|---|---|
| `results/main_metrics.csv` | Tables 6–7 | Recompute metrics from grouped OOF prediction rows |
| `results/external_metrics.csv` | Tables 8–9 | Apply frozen final artifacts to `E_RI002` |
| `results/multiseed_metrics.csv` | Appendix Table A2 | Frozen four-seed reference; retraining is optional and slow |
| `results/robustness_metrics.csv` | Tables 11–12 | Apply one restricted-input model at three noise levels |

Table 11 and Table 12 do not use two independent models. Table 11 is the
zero-noise restricted-input baseline; Table 12 uses the same frozen model and
same seven held-out cases at 0%, 0.5% and 1.0% inlet-temperature noise.

## Quick deterministic verification

The verified environment is Python 3.10.11. Create a clean environment and
install the minimal direct dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the default publication check:

```bash
python reproduce.py
```

This command:

1. verifies every processed case against `data/metadata.csv`;
2. reconstructs Tables 6–7 from case-wise grouped OOF predictions;
3. loads the frozen artifacts and reconstructs Tables 8–9 on `E_RI002`;
4. reconstructs Tables 11–12 from the restricted-input frozen model;
5. compares all generated metrics with the committed reference CSVs.

Generated files are written below `outputs/`; committed reference results are
never overwritten.

## Other modes

Generate the retained scientific figures from their source data:

```bash
python reproduce.py --mode figures
```

Run the quick verification and figure generation together:

```bash
python reproduce.py --mode all
```

Retrain the primary seed from the processed case data:

```bash
python reproduce.py --mode retrain
```

Repeat the complete train/evaluate pipeline for the four Appendix A2 seeds:

```bash
python reproduce.py --mode multiseed
```

The multiseed command is intentionally excluded from the default entry point:
it retrains all models for seeds `42`, `123`, `2024` and `3407` and can take a
long time. TensorFlow results are not promised to be bit-wise identical across
CPU/GPU, driver and BLAS combinations. The expected statistical result is the
same ranking and conclusion; `results/multiseed_metrics.csv` is the frozen
reference reported in the manuscript.

For a short software-path test, append `--smoke` to `--mode retrain` or
`--mode multiseed`. Smoke outputs are software checks and must not be reported
as manuscript results.

## Scientific contract

- Development set: complete cases `D001`–`D032`.
- Internal evaluation: five-fold `GroupKFold` by complete case ID.
- Windows: the preceding 30 samples, constructed only within one case.
- Sampling interval: 30 s; forecast horizon: one step.
- LCA meta-model: linear regression fitted on grouped base OOF predictions.
- External set: `E_RI002`, used only for transform and inference.
- Table 9 metrics: calculated consistently in the `log1p(Q_cum_J)` domain.
- `case_id` is a grouping label and is never a model input feature.

For Tables 11–12, multiplicative Gaussian noise is applied to the
inlet-temperature input `Tin_C` only. Metrics are first calculated separately
for the seven held-out cases. The reported aggregation is

```text
R2  = mean(case-wise R2)
MSE = mean(case-wise MSE)
MAE = mean(case-wise MAE)
RMSE = sqrt(mean(case-wise MSE))
```

The reported RMSE is therefore not the mean of the seven case-wise RMSE values.

## Repository layout

```text
README.md                 scope, commands and paper mapping
requirements.txt          verified minimal direct dependencies
reproduce.py              single reproduction entry point
src/                      preprocessing, models, training and evaluation code
data/                     processed complete-case data and metadata
models/                   frozen artifacts, scalers, configuration and splits
results/                  frozen reference metrics and scientific figure data
```

See `data/README.md`, `models/README.md` and `results/README.md` for detailed
file definitions.

## Availability and citation

The processed simulation data supporting the data-driven results are included
under `data/`. The source numerical solvers and the third-party experimental
benchmark are not redistributed. The latter should be accessed from the
published reference cited in the manuscript.

## Review-use notice

This repository is provided solely for reproducibility evaluation of the
associated manuscript during peer review. Reviewers and editors may inspect and
run the included code and data for that purpose. No open-source or open-data
licence is granted at this stage, and no broader permission to redistribute,
adapt for reuse, sublicense or commercially use the materials is granted.
Copyright remains with the manuscript authors. Any future licensing terms will
be stated explicitly in a later release.

When the manuscript is accepted, replace this paragraph with the final article
citation and archive a tagged repository release in a DOI-issuing repository.
