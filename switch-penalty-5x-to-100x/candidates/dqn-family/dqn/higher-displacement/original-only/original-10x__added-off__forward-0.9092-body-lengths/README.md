# dqn-higher-0.9092

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DQN, 10×, original-only. **0.9092 BL/cycle**, maximum **4** command changes per leg per complete cycle. Maximum yaw: 29.5 degrees; maximum lateral drift: 0.245 BL. Command dwell min/P05/median: 0.10/0.10/0.28 seconds.

## Selection rationale

Interior higher-displacement comparison with four maximum switches; fills the displacement interval between 0.8617 and 0.9486.

Recorded fall and torso-contact environment fractions are both zero. Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate dqn-higher-0.9092 --out reproduced-dqn-higher-0.9092.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/dqn-family/dqn/higher-displacement/original-only/original-10x__added-off__forward-0.9092-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).

## Simulation Video

[Play the matching OneDrive video](https://pennstateoffice365-my.sharepoint.com/personal/bqc5667_psu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbqc5667_psu_edu%2FDocuments%2FLIBR%2FExperiments%2FHexapod%2FVideos%2FRL_Robin%2FRL_Binary%2Fgithub-policy-videos%2Fdisplacement-above-0.8%2Fdqn-family%2Fdqn%2Foriginal-only%2Foriginal-10x__added-off__forward-0.9092-body-lengths%2Fdqn-higher-0.9092.mp4). Normal 1× playback; environment 0. The checkpoint SHA-256 matches this policy. [Full video catalog](../../../../../../VIDEO_CATALOG.md).
