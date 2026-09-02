# Frozen manuscript reference results

- `main_metrics.csv`: Tables 6–7, seed-42 grouped OOF metrics.
- `external_metrics.csv`: Tables 8–9, seed-42 locked external metrics.
- `multiseed_metrics.csv`: Appendix A2 four-seed mean and sample SD source.
- `robustness_metrics.csv`: Tables 11–12 restricted-input/noise source.
- `source_data/oof_predictions/`: prediction rows underlying Tables 6–7.
- `source_data/external_predictions.csv.gz`: source rows for the external
  prediction figure and an additional frozen reference.
- `source_data/figures/`: attention and loss-curve source data.
- `figures/`: compact reference PNGs retained for visual comparison.

Running `python reproduce.py` writes newly calculated results to
`outputs/reproduced/`; it does not overwrite this directory.

