# ppo-moderate-0.6234

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

PPO, 10×, original-plus-added-switch. **0.6234 BL/cycle**, maximum **4** command changes per leg per complete cycle. Maximum yaw: 34.0 degrees; maximum lateral drift: 0.243 BL. Command dwell min/P05/median: 0.02/0.04/0.29 seconds.

## Selection rationale

Lower-displacement added-penalty comparison: 0.243 BL maximum drift, but includes 0.02 s command pulses. Lower displacement alone does not establish slow limb transitions.

Recorded fall and torso-contact environment fractions are both zero. Includes dwell shorter than 0.1 s; Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate ppo-moderate-0.6234 --out reproduced-ppo-moderate-0.6234.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/ppo-family/ppo/target-band/original-plus-added-switch/original-10x__added-10x__forward-0.6234-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).
