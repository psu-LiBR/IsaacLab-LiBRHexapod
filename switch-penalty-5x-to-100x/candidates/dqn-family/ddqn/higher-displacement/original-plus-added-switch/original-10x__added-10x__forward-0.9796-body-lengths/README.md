# ddqn-higher-0.9796

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DDQN, 10×, original-plus-added-switch. **0.9796 BL/cycle**, maximum **2** command changes per leg per complete cycle. Maximum yaw: 26.9 degrees; maximum lateral drift: 0.210 BL. Command dwell min/P05/median: 0.12/0.12/0.42 seconds.

## Selection rationale

Two-switch higher-displacement alternative with 0.42 s median hold, distinct from the nearby four-switch 0.9937 candidate.

Recorded fall and torso-contact environment fractions are both zero. Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate ddqn-higher-0.9796 --out reproduced-ddqn-higher-0.9796.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/dqn-family/ddqn/higher-displacement/original-plus-added-switch/original-10x__added-10x__forward-0.9796-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).
