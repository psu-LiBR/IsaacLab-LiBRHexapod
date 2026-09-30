# ddqn-moderate-0.7786

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DDQN, 100×, original + added command-switch cost. **0.7786 BL/cycle**, maximum **2** changes per leg per complete cycle. Yaw 33.5°; lateral drift 0.336 BL. Dwell min/P05/median: 0.40/0.40/0.46 seconds.

Representative displacement and command-switch behavior; compare the measured straightness and dwell rather than treating this as a rank.

Observed falls / torso-contact environment fractions: 0.0/0.0. Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate ddqn-moderate-0.7786 --out reproduced-ddqn-moderate-0.7786.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/dqn-family/ddqn/target-band/original-plus-added-switch/original-100x__added-100x__forward-0.7786-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.

## Simulation Video

[Play the matching OneDrive video](https://pennstateoffice365-my.sharepoint.com/personal/bqc5667_psu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbqc5667_psu_edu%2FDocuments%2FLIBR%2FExperiments%2FHexapod%2FVideos%2FRL_Robin%2FRL_Binary%2Fgithub-policy-videos%2Fdisplacement-0.6-to-0.8%2Fdqn-family%2Fddqn%2Foriginal-plus-added-switch%2Foriginal-100x__added-100x__forward-0.7786-body-lengths%2Fddqn-moderate-0.7786.mp4). Normal 1× playback; environment 0. The checkpoint SHA-256 matches this policy. [Full video catalog](../../../../../../VIDEO_CATALOG.md).
