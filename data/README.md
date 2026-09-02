# Processed case-wise dataset

## Summary

This directory contains the machine-learning inputs used by the data-driven
reproduction chain. Each compressed CSV is one complete simulation
case; case boundaries are never inferred from randomly shuffled window rows.

- `development_cases/`: `D001`–`D032`, used for grouped OOF evaluation and
  full-development refitting.
- `external_cases/`: `E_RI002`, a locked transform-and-inference-only case.
- `metadata.csv`: stable case identity, relative file, operating descriptors,
  dataset role, row count and SHA-256 checksum.

`case_id` is used only as a grouping label. It is not supplied to the model as
an input feature.

## Variables and units

| Column | Definition | Unit |
|---|---|---|
| `case_id` | Stable complete-case identifier | categorical label |
| `time_min` | Elapsed time | min |
| `f_liquid` | Liquid-fraction state variable | dimensionless |
| `Tin_C` | Time-resolved inlet-fluid temperature | °C |
| `Tout_C` | Outlet-fluid temperature | °C |
| `Q_cum_J` | Cumulative heat | J |
| `delta_Q_J` | Incremental cumulative heat between stored rows | J |
| `log1p_Q_J` | `log(1 + Q_cum_J)` | transformed value |
| `Ri_m` | Manuscript `Ri` geometric parameter | m |
| `L_m` | Storage length | m |
| `Phi` | Manuscript `Phi` operating/geometric parameter | dimensionless |
| `Tin_ch_C` | Case-level charging inlet-temperature descriptor | °C |
| `split` | Original frozen dataset role | categorical label |

All cases use a 0.5 min (30 s) sampling interval. Missing values are not used
as special codes; the repository verification rejects missing required columns or
case-identity inconsistencies.

## Provenance

The CSV files are processed exports of the manuscript authors' numerical
simulation cases. Only the sheet and columns read by the machine-learning
pipeline were retained. The original workbooks, spatial fields, MATLAB solver
internals and intermediate development files are not required by this public
pipeline and are not redistributed.

## Access and review-use notice

These processed simulation data are supplied solely for reproducibility
evaluation of the associated manuscript during peer review. Reviewers and
editors may inspect and use the data for that purpose. No open-data licence or
broader permission for redistribution, adaptation, sublicensing or commercial
use is granted at this stage. No Lacroix experimental coordinates are included
here.
