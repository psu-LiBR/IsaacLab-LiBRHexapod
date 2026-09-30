# dqn-moderate-0.6716

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

DQN, 30×, original-plus-added-switch. **0.6716 BL/cycle**, maximum **2** command changes per leg per complete cycle. Maximum yaw: 84.7 degrees; maximum lateral drift: 1.031 BL. Command dwell min/P05/median: 0.24/0.26/0.50 seconds.

## Selection rationale

Lowest available target-band DQN displacement. Retained as a poor-straightness comparison: 84.7 degrees maximum yaw and 1.031 BL maximum drift. It is not a preferred straight-gait candidate.

Recorded fall and torso-contact environment fractions are both zero. Lateral drift exceeds 0.5 BL; Yaw exceeds 35 degrees; Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate dqn-moderate-0.6716 --out reproduced-dqn-moderate-0.6716.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/dqn-family/dqn/target-band/original-plus-added-switch/original-30x__added-30x__forward-0.6716-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).

## Simulation Video

[Play the matching OneDrive video](https://pennstateoffice365-my.sharepoint.com/personal/bqc5667_psu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbqc5667_psu_edu%2FDocuments%2FLIBR%2FExperiments%2FHexapod%2FVideos%2FRL_Robin%2FRL_Binary%2Fgithub-policy-videos%2Fdisplacement-0.6-to-0.8%2Fdqn-family%2Fdqn%2Foriginal-plus-added-switch%2Foriginal-30x__added-30x__forward-0.6716-body-lengths%2Fdqn-moderate-0.6716.mp4). Normal 1× playback; environment 0. The checkpoint SHA-256 matches this policy. [Full video catalog](../../../../../../VIDEO_CATALOG.md).
