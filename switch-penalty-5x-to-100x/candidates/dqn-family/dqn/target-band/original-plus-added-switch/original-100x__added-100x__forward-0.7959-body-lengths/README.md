# dqn-moderate-0.7959

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DQN, 100×, original + added command-switch cost. **0.7959 BL/cycle**, maximum **2** changes per leg per complete cycle. Yaw 33.5°; lateral drift 0.559 BL. Dwell min/P05/median: 0.36/0.36/0.46 seconds.

Upper-band long-dwell alternative (0.36 s minimum); nearby 0.7986 is retained for lower lateral drift and three rather than two maximum switches.

Observed falls / torso-contact environment fractions: 0.0/0.0. Lateral drift exceeds 0.5 BL; Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate dqn-moderate-0.7959 --out reproduced-dqn-moderate-0.7959.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/dqn-family/dqn/target-band/original-plus-added-switch/original-100x__added-100x__forward-0.7959-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.
