# ddqn-moderate-0.7998

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DDQN, 100×, original-plus-added-switch. **0.7998 BL/cycle**, maximum **2** command changes per leg per complete cycle. Maximum yaw: 29.0 degrees; maximum lateral drift: 0.336 BL. Command dwell min/P05/median: 0.34/0.34/0.44 seconds.

## Selection rationale

Upper target-band two-switch comparison with 29.0 degrees maximum yaw and 0.34 s minimum hold. Inclusion does not imply that near-0.8 displacement is preferred.

Recorded fall and torso-contact environment fractions are both zero. Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate ddqn-moderate-0.7998 --out reproduced-ddqn-moderate-0.7998.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/dqn-family/ddqn/target-band/original-plus-added-switch/original-100x__added-100x__forward-0.7998-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).
