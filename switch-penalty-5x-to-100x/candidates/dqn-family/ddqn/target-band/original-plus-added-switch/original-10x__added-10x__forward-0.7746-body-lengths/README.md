# ddqn-moderate-0.7746

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DDQN, 10×, original-plus-added-switch. **0.7746 BL/cycle**, maximum **3** command changes per leg per complete cycle. Maximum yaw: 34.0 degrees; maximum lateral drift: 0.337 BL. Command dwell min/P05/median: 0.22/0.28/0.50 seconds.

## Selection rationale

Three-switch alternative with 0.22 s minimum hold; compared with the nearby two-switch 0.7786 candidate with 0.40 s minimum hold.

Recorded fall and torso-contact environment fractions are both zero. Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate ddqn-moderate-0.7746 --out reproduced-ddqn-moderate-0.7746.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/dqn-family/ddqn/target-band/original-plus-added-switch/original-10x__added-10x__forward-0.7746-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).
