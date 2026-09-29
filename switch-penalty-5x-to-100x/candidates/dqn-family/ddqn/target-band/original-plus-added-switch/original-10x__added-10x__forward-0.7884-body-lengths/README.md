# ddqn-moderate-0.7884

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DDQN, 10×, original + added command-switch cost. **0.7884 BL/cycle**, maximum **4** changes per leg per complete cycle. Yaw 37.7°; lateral drift 0.235 BL. Dwell min/P05/median: 0.06/0.12/0.24 seconds.

Representative displacement and command-switch behavior; compare the measured straightness and dwell rather than treating this as a rank.

Observed falls / torso-contact environment fractions: 0.0/0.0. Yaw exceeds 35 degrees; Includes dwell shorter than 0.1 s; Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate ddqn-moderate-0.7884 --out reproduced-ddqn-moderate-0.7884.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/dqn-family/ddqn/target-band/original-plus-added-switch/original-10x__added-10x__forward-0.7884-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.
