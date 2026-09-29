# Reproduce any packaged candidate

This guide covers all delivered algorithms. Its filename preserves the existing DQN/DDQN documentation link.

## One candidate, one command

Use the same branch's code and an Isaac Sim 6 / Isaac Lab 3 installation compatible with this repository. Run from the repository root. Robot USD assets use Git LFS; policy files in this delivery do not.

```bash
git lfs pull --include="hexapod-assets/**"
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate dqn-moderate-0.6839 --out reproduced-dqn.json
```

On Windows, replace `./isaaclab.sh -p` with `isaaclab.bat -p`. Substitute any ID from [the table](README.md) for `dqn-moderate-0.6839`. Do not change reward weights or retrain to perform this check.

The evaluator looks up the candidate directory, verifies file hashes and applies its evaluation.json. Its loader requires the metadata alongside the packaged checkpoint, loads PPO's stored running normalization, and checks Masked PPO's 24 legal actions against the embedded mask. Missing/mismatched files cause an error rather than silently evaluating a different policy.

Test environment: Python 3.12, PyTorch 2.11.0+cu128, skrl 2.1.0; Kit 110.1.1+production.305458.6312fa25.gl. The exact export/CPU versions and fresh simulation results are recorded in validation.json. The repository's matching Isaac Sim installation is a prerequisite; installing a newer physics backend is not a controlled reproduction. PPO checkpoint evaluation requires skrl; ONNX deployment does not.

## Fixed measurement conditions

| Setting | Value |
|---|---|
| Task | Isaac-Goal-Flat-Hexapod-Binary-v0 |
| Environments / seed | 64 / 7 |
| Static and dynamic friction | 0.21, explicitly applied and audited |
| Goal | Fixed 2 m, no resampling or curriculum |
| Episode resets | Disabled during this finite measurement |
| Observation noise / joint reset noise | Disabled |
| Inference | Deterministic greedy, correct normalization and action mask |
| Warmup / measured control steps | 1 / 300 |
| Control timestep / spine period | 0.02 s / 1.0 s |
| Body length | 0.315 m |
| Forward displacement | Initial-heading projection, mean across environments, divided by 6 nominal cycles |
| Switching | Every command change, not only rising edges; maximum per individual leg over 5 complete phase-aligned cycles/environment |

Training friction was randomized over [0.18, 0.25]. The midpoint 0.215 is not the published evaluation value. Both the requested and loaded material values are recorded. The reward weights in config.json are the full resolved training weights; evaluation.json supplies the equivalent settings to the evaluator.

## Read the result

The output JSON contains `bl_per_cycle`, `checkpoint_sha256`, `measurement_valid`, `resets` and `screening`. Compare `screening.per_leg_max`, `dwell_min_s`, `dwell_p05_s`, `dwell_median_s`, `yaw_abs_max_deg` and `lateral_abs_max_bl` with metrics.json. The paired `.audit.json` records the actual scene, observation/action sizes, reward weights and physics settings. The `.npz` retains trajectories and actions for independent metric recomputation.

For the example above, the published forward value is approximately **0.6839187 BL/cycle**; the per-leg maximum vector is **[2, 2, 2, 3, 2, 2]**. The selected DDQN example `ddqn-moderate-0.7786` is approximately **0.7786 BL/cycle**, with maximum **2**. Full-precision values are in the manifest; validation.json reports the fresh reproduction difference rather than concealing it with rounding.

Use a new output path for each run. An existing result is preserved, not overwritten.

## If a result is zero or differs

Retain the command, output JSON, audit and log before adjusting anything. Check, in this order:

1. The checkpoint hash and candidate ID match the table, and the robot asset is actual USD rather than a Git LFS pointer.
2. The dedicated evaluator above was used. `play_discrete.py` uses a Play task and different reset/video settings. General `eval_protocol.py` is a separate baseline-comparison tool and does not produce this complete-cycle screening table.
3. The actual audit says friction 0.21, 64 environments, dt 0.02 and the expected 32 observation fields/joint order. Record the physics/backend versions.
4. Normalization and the masked action set were loaded. A policy that loads successfully with a different observation order or action restriction is not the same inference pipeline.
5. Compare the recorded action/trajectory and the candidate's metrics before considering retraining.

The earlier handoff did not make the exact quantitative protocol sufficiently clear. This delivery uses a dedicated entry point with the full measurement configuration and a sibling-metadata fix in the general evaluator. These are verified corrections; they do not establish which particular mismatch caused Jackson's earlier 0 BL result without his original command and log.

## ONNX and physical testing

Each candidate's README includes a CPU deployment dry-run command using its ONNX and this batch's deployment.example.yaml. Normalization, clipping, mask, argmax and six-bit decoding are inside ONNX. The hardware host supplies the observations and analytic spine wave. The five-second fake-IMU/in-memory-bus check does not move motors and cannot verify real localization, calibration, contact forces or locomotion.

The table's >0.8 group is retained for comparative assessment, not promoted as the preferred slow gait. Jackson chooses which candidates to test physically. Videos are shared separately through an agreed channel; none are committed in this update.
