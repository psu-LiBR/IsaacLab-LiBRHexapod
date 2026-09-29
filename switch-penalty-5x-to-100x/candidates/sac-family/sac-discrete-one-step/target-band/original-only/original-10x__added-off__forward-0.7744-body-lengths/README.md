# sac-discrete-one-step-moderate-0.7744

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

SAC-D 1-step, 10×, original only. **0.7744 BL/cycle**, maximum **2** changes per leg per complete cycle. Yaw 28.1°; lateral drift 0.823 BL. Dwell min/P05/median: 0.26/0.30/0.48 seconds.

Representative displacement and command-switch behavior; compare the measured straightness and dwell rather than treating this as a rank.

Observed falls / torso-contact environment fractions: 0.0/0.0. Lateral drift exceeds 0.5 BL; Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate sac-discrete-one-step-moderate-0.7744 --out reproduced-sac-discrete-one-step-moderate-0.7744.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/sac-family/sac-discrete-one-step/target-band/original-only/original-10x__added-off__forward-0.7744-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.
