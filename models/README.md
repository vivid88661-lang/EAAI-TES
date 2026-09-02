# Frozen models and experiment definitions

- `frozen_models/`: neural-network weights and trained meta/stage-two models
  used for exact frozen-version inference.
- `frozen_models/restricted_input/`: the temperature models used jointly by
  Tables 11–12.
- `scalers/`: preprocessing objects fitted on the corresponding development
  data only.
- `config.json`: window, sampling, feature, target, seed and metric contract.
- `splits/`: development, external and restricted-input held-out case lists.

The main frozen artifacts reproduce Tables 8–9. Tables 6–7 are grouped OOF
results and are therefore recomputed from the corresponding OOF prediction
rows, not from the full-development deployment refits.

The restricted-input artifacts intentionally omit `f_liquid` from the measured
input list. Their noise experiment perturbs `Tin_C` only.

The `.joblib` files should be loaded only from this trusted repository release.

