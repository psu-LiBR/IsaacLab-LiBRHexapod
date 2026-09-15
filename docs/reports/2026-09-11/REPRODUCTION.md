# Reproduction and hardware handoff notes

## Software-side reproduction

1. Check out the Friday branch and the source commit recorded in `README.md`.
2. Use Isaac Lab 5.1 with the repository's normal launcher and the task `Isaac-Goal-Flat-Hexapod-Binary-v0`.
3. Recreate each run from `configs/manifest.json`: algorithm, recipe, seed, 4,096 training environments, 192,000 vector steps, friction ranges, and checkpoint interval are all recorded there.
4. Evaluate with `eval_protocol.py` using 64 environments, 300 steps, seed 7, warmup 1, all terminations disabled, and the explicit body-length and gait-period fields. Preserve the generated JSON sidecar and reward-term breakdown.
5. Use `export_binary_onnx.py` only after verifying the checkpoint's `run_meta.json`; the export must retain the six-leg bit ordering and the observation normalization metadata.

## Interface contract

- Action: one integer in `[0,63]`; bit `1` is stance target `0.460194 rad`, bit `0` is swing target `1.180398 rad`.
- Leg order and observation construction are defined in the source environment and wrapper files listed in `README_binary_rl.md`.
- The two spine joints are scripted by the environment and are not policy outputs. Learned policies use Wave 1; the tripod historical baseline uses a distinct anti-phase Wave 2.
- Do not send a simulation checkpoint to hardware without checking actuator mapping, joint order, units, observation normalization, action decoding, safety limits, and a low-level stop path.

## Evidence and limitations

The local evaluation archive and measurement audit are the source of truth for the numerical record. Videos are qualitative checks for forward motion, falls, torso contact, dragging, and constant-action behavior. A simulation score is not a hardware result. The measurement audit supports the recorded 0.315 m basis and approximately 1 s spine period for the disputed rollout, while leaving hardware transfer and broader codebase validation open.

## Artifact manifest

Large artifacts are intentionally outside the Git source tree. The local archive root, file names, hashes, and report assets are retained under `新版训练_20260911/`; when publishing, attach the generated archive and its SHA256 manifest to the GitHub release for this branch. Do not silently rename historical files or mix Wednesday tripod artifacts into this Friday package.
