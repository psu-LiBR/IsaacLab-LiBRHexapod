# dqn-moderate-0.7650

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DQN, 10×, original-only. **0.7650 BL/cycle**, maximum **4** command changes per leg per complete cycle. Maximum yaw: 31.9 degrees; maximum lateral drift: 0.288 BL. Command dwell min/P05/median: 0.12/0.14/0.42 seconds.

## Selection rationale

Four-switch alternative near 0.7656: smaller maximum drift (0.288 versus 0.544 BL), but shorter minimum command hold (0.12 versus 0.34 s).

Recorded fall and torso-contact environment fractions are both zero. Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate dqn-moderate-0.7650 --out reproduced-dqn-moderate-0.7650.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/dqn-family/dqn/target-band/original-only/original-10x__added-off__forward-0.7650-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).
