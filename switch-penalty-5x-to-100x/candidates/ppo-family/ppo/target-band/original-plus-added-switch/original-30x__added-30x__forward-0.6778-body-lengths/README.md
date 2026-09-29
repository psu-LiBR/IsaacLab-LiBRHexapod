# ppo-moderate-0.6778

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

PPO, 30×, original + added command-switch cost. **0.6778 BL/cycle**, maximum **3** changes per leg per complete cycle. Yaw 32.7°; lateral drift 0.387 BL. Dwell min/P05/median: 0.12/0.14/0.48 seconds.

Representative displacement and command-switch behavior; compare the measured straightness and dwell rather than treating this as a rank.

Observed falls / torso-contact environment fractions: 0.0/0.0. Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate ppo-moderate-0.6778 --out reproduced-ppo-moderate-0.6778.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/ppo-family/ppo/target-band/original-plus-added-switch/original-30x__added-30x__forward-0.6778-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.
