# Raw evaluation results

Every JSON is the direct output of `eval_protocol.py`. Each file records the full pinned
configuration under `protocol.audit`, so a run can be reproduced from the file itself.

- `friction/fclean_<policy>_<static>.json` — friction sweep, one process per (friction, policy) cell.
  `dynamic = 0.9 x static`; the override is recorded as `OVERRIDE static_friction_range` in `audit`.
- `budget_100k/rank_<algo>_<seed>_<eq|fin>_off.json` — `eq` = the 100k checkpoint (equal budget
  across all 8 algorithms), `fin` = the final checkpoint of that run.
- `sacd_training_curve/rank_sacdpk_<steps>_off.json` — SAC-D seed 46 evaluated every few thousand
  steps between 106k and 152k.

All runs used `--no_baselines` for the learned policies (one policy per process) except the
reference-point runs. Tables built from these files are in `../RESULTS_binary_rl.md`.
