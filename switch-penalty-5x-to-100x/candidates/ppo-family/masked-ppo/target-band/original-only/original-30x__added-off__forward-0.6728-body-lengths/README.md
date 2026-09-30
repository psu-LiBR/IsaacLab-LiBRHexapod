# masked-ppo-moderate-0.6728

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

Masked PPO, 30×, original only. **0.6728 BL/cycle**, maximum **3** changes per leg per complete cycle. Yaw 29.0°; lateral drift 0.273 BL. Dwell min/P05/median: 0.06/0.08/0.48 seconds.

Representative displacement and command-switch behavior; compare the measured straightness and dwell rather than treating this as a rank.

Observed falls / torso-contact environment fractions: 0.0/0.0. Includes dwell shorter than 0.1 s; Hardware performance not verified.

## Quantitative simulation (from repository root)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate masked-ppo-moderate-0.6728 --out reproduced-masked-ppo-moderate-0.6728.json
```

## CPU deployment dry run (no motor writes)

```bash
python scripts/sim2real_transfer/run_policy.py \
  --policy "switch-penalty-5x-to-100x/candidates/ppo-family/masked-ppo/target-band/original-only/original-30x__added-off__forward-0.6728-body-lengths/policy.onnx" \
  --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
```

Install the robot runtime's documented dependencies first. See the parent delivery guide for observation ordering and calibration. This dry run checks software execution only; it does not predict robot displacement.

## Simulation Video

[Play the matching OneDrive video](https://pennstateoffice365-my.sharepoint.com/personal/bqc5667_psu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbqc5667_psu_edu%2FDocuments%2FLIBR%2FExperiments%2FHexapod%2FVideos%2FRL_Robin%2FRL_Binary%2Fgithub-policy-videos%2Fdisplacement-0.6-to-0.8%2Fppo-family%2Fmasked-ppo%2Foriginal-only%2Foriginal-30x__added-off__forward-0.6728-body-lengths%2Fmasked-ppo-moderate-0.6728.mp4). Normal 1× playback; environment 0. The checkpoint SHA-256 matches this policy. [Full video catalog](../../../../../../VIDEO_CATALOG.md).
