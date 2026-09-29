# Reproduce the published DQN/DDQN results

Use the exact packaged checkpoint and its paired `config.json`.  The reported
forward values must be checked with the quantitative evaluator, not with
`play_discrete.py`.  The play program uses the separate
`Isaac-Goal-Flat-Hexapod-Binary-Play-v0` task and records a visual replay; it does
not produce the `x_disp_BL_per_cycle` metric.

| Policy | Packaged checkpoint | SHA-256 | Published `x_disp_BL_per_cycle` |
|---|---|---|---:|
| DQN, original + added switch at 10x | `candidates/dqn-family/dqn/original-plus-added-switch/original-10x__added-10x__forward-1.0116-body-lengths/policy.pt` | `96dc32c234b25939129066e0dd99c6adfc231d877423124ff6bc826c2fc262b8` | 1.0116 |
| DDQN, original + added switch at 10x | `candidates/dqn-family/ddqn/original-plus-added-switch/original-10x__added-10x__forward-0.9937-body-lengths/policy.pt` | `62fcc12a227454d1064de59733653d89a6501d31465e8a841c15024bf479d51c` | 0.9937 |

The full resolved reward-weight dictionary, training seed, training friction band,
and evaluation friction are in the paired `config.json`.  Confirm the checkpoint
hash against `validation.json` before evaluating it.

Run this from the repository root in an Isaac Lab installation.  The command uses
the exact evaluator source committed on this branch and writes a JSON sidecar for
comparison.

```bash
CANDIDATE="switch-penalty-5x-to-100x/candidates/dqn-family/dqn/original-plus-added-switch/original-10x__added-10x__forward-1.0116-body-lengths"

./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/eval_protocol.py \
  --task Isaac-Goal-Flat-Hexapod-Binary-v0 \
  --num_envs 64 --steps 300 --seed 7 --warmup 1 \
  --goal_distance 2.0 --friction 0.21 --body_length_m 0.315 --gait_period_s 1.0 \
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

## Visual replay using the same checkpoint

This is a visual sanity check only.  It uses the same packaged `policy.pt` at the
Play task's fixed 0.21 friction, but it does not replace the quantitative command
above and is not a claim of a byte-identical replacement for a replay that was
intentionally removed from Git history.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/play_discrete.py \
  --task Isaac-Goal-Flat-Hexapod-Binary-Play-v0 \
  --checkpoint "$CANDIDATE/policy.pt" \
  --num_envs 16 --steps 600 --video_length 600 --seed 42 \
  --out_dir reproduced_playback
```

## Known condition boundary

Training randomizes friction over `[0.18, 0.25]`.  The public candidate metadata,
metrics, HTML, and playback videos record the published condition as `0.21`; the
command above explicitly forces that value for both static and dynamic friction.
Without `--friction`, the legacy evaluator default is the training-range midpoint,
`0.215`, so omitting the flag is not a reproduction of the published package.
The JSON records the requested value and its audit records the applied material
values.  Keep both with any comparison result.
