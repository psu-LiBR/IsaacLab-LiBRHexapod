# Reproduce the DQN/DDQN measurements

The reported forward values in this package must be checked with the quantitative
evaluator, not with `play_discrete.py`.  The play program uses the separate
`Isaac-Goal-Flat-Hexapod-Binary-Play-v0` task and records a replay; it does not
produce the `x_disp_BL_per_cycle` metric.

Run this from the repository root in an Isaac Lab installation.  The command uses
the exact evaluator source committed on this branch and writes a JSON sidecar for
comparison.

```bash
CANDIDATE="switch-penalty-5x-to-100x/candidates/dqn-family/dqn/original-plus-added-switch/original-10x__added-10x__forward-1.0116-body-lengths"

./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/eval_protocol.py \
  --task Isaac-Goal-Flat-Hexapod-Binary-v0 \
  --num_envs 64 --steps 300 --seed 7 --warmup 1 \
  --goal_distance 2.0 --body_length_m 0.315 --gait_period_s 1.0 \
  --no_baselines \
  --policy net dqn_added_10x "$CANDIDATE/policy.pt" \
  --out dqn_added_10x_eval.json
```

For DDQN, change only the candidate path and policy label.  For example:

```bash
CANDIDATE="switch-penalty-5x-to-100x/candidates/dqn-family/ddqn/original-plus-added-switch/original-10x__added-10x__forward-0.9937-body-lengths"
```

The corresponding Windows launcher is `isaaclab.bat -p` in place of
`./isaaclab.sh -p`.

## What to compare

The JSON result must contain one object for the named policy and its
`x_disp_BL_per_cycle` field.  The package records 1.0116 for the DQN example and
0.9937 for the DDQN example.  The evaluator's printed configuration audit is part
of the reproduction record: retain it together with the JSON, because it records
the actual physics-material pinning, reset handling, task ID, and timestep.

Do not treat a `play_discrete.py` replay as a numerical reproduction.  If the
command above yields zero or a materially different value, retain the JSON and the
configuration-audit lines before changing reward weights or retraining.  That
distinguishes a checkpoint/load issue from a task/reset/physics mismatch.

## Known condition boundary

Training randomizes friction over `[0.18, 0.25]`.  The candidate metadata labels
the published evaluation condition as `0.21`; the current quantitative evaluator
pins that range's midpoint, `0.215`, and prints it in its audit.  Use the printed
audit as the authoritative value for a rerun, and do not silently substitute the
Play task's `0.21` setting.  The package will be reconciled against the original
measurement record before claiming bit-for-bit numerical agreement.
