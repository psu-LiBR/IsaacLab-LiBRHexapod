# Binary RL — Extended-Quadruped Reward Iteration (2026-09-11)

This directory records the Friday simulation round under the updated extended-quadruped dynamics. It is a reproducibility index, not a claim of hardware validation.

## Scope

The round compares four reward recipes under the same updated environment:

| Recipe | Meaning | Change relative to the repository version |
| --- | --- | --- |
| Repository Latest Version | control | Jackson's latest repository reward and environment |
| Candidate A | raw-magnitude shaping | stronger forward, stability, mechanical-cost, and failure terms |
| Candidate B | bounded shaping | bounded stability and mechanical-cost terms |
| Candidate C | quality-gated shaping | Candidate B plus a positive-quality gate on progress/success |

The maintained algorithm set is DQN, DDQN, PPO, masked PPO, SAC-D, and continuous SAC. The continuous SAC run is an action-space baseline and is not a binary-contact policy.

## Environment and protocol

- Updated extended-quadruped dynamics and analytic RL spine wave (Wave 1).
- Six binary leg-contact decisions encoded as one `Discrete(64)` action.
- Training static and dynamic friction randomized at startup in `[0.18, 0.25]`.
- The recorded primary evaluation used static/dynamic friction `0.21`, matching the `m021*` evaluation files and the Friday report. The source manifest still contains an earlier `0.215` launch metadata value; it is historical and must not be presented as the final evaluation condition.
- 64 parallel environments, 300 steps, `dt=0.02 s`, no resets during measurement, six-second rollout.
- Body-length basis recorded by the evaluator: `0.315 m`.
- Training budget: 192,000 vector steps per run; checkpoint interval: 2,000 steps.

## Source and run mapping

- Environment source commit: `929be07e10d1f8097b48f06a68b1af9e3eb554ac`.
- Friday reward-iteration source commit: `56033d64d2656cefa05a8222a058e264743e8c7f`.
- Full run mapping is preserved in the local experiment manifest `configs/manifest.json` and is summarized in `REPRODUCTION.md`.

## Reproduction

Read `scripts/reinforcement_learning/binary_rl/README_binary_rl.md` first. It defines the action encoding, observations, spine waves, training commands, evaluation flags, checkpoint handling, and ONNX export path. Use the exact source commits and manifest entries above; do not infer a checkpoint from a video filename.

The local experiment archive contains checkpoints, evaluation JSON/NPZ files, TensorBoard exports, report figures, and regular rendered videos. Because these artifacts are large, they are kept as release/download artifacts rather than committed into the source tree. Their local root is `A线_Jackson六足RL/04_环境与训练/新版训练_20260911/`. Files or metadata carrying `0.215` are legacy/supplemental and must be labeled separately.

## Interpretation boundary

The recorded simulation result is an in-simulation comparison. The 0.315 m body-length basis, approximately 1 s spine period, and displacement audit are documented locally in the measurement-audit directory. Hardware performance remains to be independently verified. The reward coefficients in this round are candidates for further coefficient tuning and one-factor-at-a-time ablations; they are not final settings.

## Historical reference

The earlier 2026-09-09 archive is a separate tripod-reference friction-sweep record. It must not be relabeled as the Friday extended-quadruped experiment. Its archive is linked from the repository's historical report index.
