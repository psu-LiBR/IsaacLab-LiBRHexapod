# dqn-moderate-0.7986

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DQN, 30×, original + added command-switch cost. **0.7986 BL/cycle**, maximum **3** changes per leg per complete cycle. Yaw 37.4°; lateral drift 0.320 BL. Dwell min/P05/median: 0.14/0.16/0.48 seconds.

Upper-band straightness tradeoff: lower drift than 0.7959, but shorter minimum dwell; retained for this distinct behavior despite similar displacement.

Observed falls / torso-contact environment fractions: 0.0/0.0. Yaw exceeds 35 degrees; Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate dqn-moderate-0.7986 --out reproduced-dqn-moderate-0.7986.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/dqn-family/dqn/target-band/original-plus-added-switch/original-30x__added-30x__forward-0.7986-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.
