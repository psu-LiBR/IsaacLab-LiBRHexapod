# dqn-higher-0.9092

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DQN, 10×, original-only. **0.9092 BL/cycle**, maximum **4** command changes per leg per complete cycle. Maximum yaw: 29.5 degrees; maximum lateral drift: 0.245 BL. Command dwell min/P05/median: 0.10/0.10/0.28 seconds.

## Selection rationale

Interior higher-displacement comparison with four maximum switches; fills the displacement interval between 0.8617 and 0.9486.

Recorded fall and torso-contact environment fractions are both zero. Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate dqn-higher-0.9092 --out reproduced-dqn-higher-0.9092.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/dqn-family/dqn/higher-displacement/original-only/original-10x__added-off__forward-0.9092-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).
