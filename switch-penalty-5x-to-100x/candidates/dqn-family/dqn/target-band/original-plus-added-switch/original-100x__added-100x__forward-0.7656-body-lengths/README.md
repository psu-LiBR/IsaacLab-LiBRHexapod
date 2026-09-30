# dqn-moderate-0.7656

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DQN, 100×, original + added command-switch cost. **0.7656 BL/cycle**, maximum **2** changes per leg per complete cycle. Yaw 35.1°; lateral drift 0.544 BL. Dwell min/P05/median: 0.34/0.36/0.44 seconds.

Long-dwell representative with two maximum switches and 0.34 s minimum dwell. Compared with nearby 0.7580, it holds commands longer but drifts more.

Observed falls / torso-contact environment fractions: 0.0/0.0. Lateral drift exceeds 0.5 BL; Yaw exceeds 35 degrees; Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate dqn-moderate-0.7656 --out reproduced-dqn-moderate-0.7656.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/dqn-family/dqn/target-band/original-plus-added-switch/original-100x__added-100x__forward-0.7656-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.

## Simulation Video

[Play the matching OneDrive video](https://pennstateoffice365-my.sharepoint.com/personal/bqc5667_psu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbqc5667_psu_edu%2FDocuments%2FLIBR%2FExperiments%2FHexapod%2FVideos%2FRL_Robin%2FRL_Binary%2Fgithub-policy-videos%2Fdisplacement-0.6-to-0.8%2Fdqn-family%2Fdqn%2Foriginal-plus-added-switch%2Foriginal-100x__added-100x__forward-0.7656-body-lengths%2Fdqn-moderate-0.7656.mp4). Normal 1× playback; environment 0. The checkpoint SHA-256 matches this policy. [Full video catalog](../../../../../../VIDEO_CATALOG.md).
