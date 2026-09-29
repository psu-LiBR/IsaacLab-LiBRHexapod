# dqn-higher-0.9761

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DQN, 10×, original-plus-added-switch. **0.9761 BL/cycle**, maximum **4** command changes per leg per complete cycle. Maximum yaw: 30.1 degrees; maximum lateral drift: 0.309 BL. Command dwell min/P05/median: 0.16/0.16/0.26 seconds.

## Selection rationale

Added-penalty four-switch comparison with 0.16 s minimum hold, compared with 0.06 s at 0.9983.

Recorded fall and torso-contact environment fractions are both zero. Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate dqn-higher-0.9761 --out reproduced-dqn-higher-0.9761.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/dqn-family/dqn/higher-displacement/original-plus-added-switch/original-10x__added-10x__forward-0.9761-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).
