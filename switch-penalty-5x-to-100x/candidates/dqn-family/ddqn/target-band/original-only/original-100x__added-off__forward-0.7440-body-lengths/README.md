# ddqn-moderate-0.7440

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DDQN, 100×, original only. **0.7440 BL/cycle**, maximum **3** changes per leg per complete cycle. Yaw 38.0°; lateral drift 0.382 BL. Dwell min/P05/median: 0.14/0.16/0.47 seconds.

Representative displacement and command-switch behavior; compare the measured straightness and dwell rather than treating this as a rank.

Observed falls / torso-contact environment fractions: 0.0/0.0. Yaw exceeds 35 degrees; Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate ddqn-moderate-0.7440 --out reproduced-ddqn-moderate-0.7440.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/dqn-family/ddqn/target-band/original-only/original-100x__added-off__forward-0.7440-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.
