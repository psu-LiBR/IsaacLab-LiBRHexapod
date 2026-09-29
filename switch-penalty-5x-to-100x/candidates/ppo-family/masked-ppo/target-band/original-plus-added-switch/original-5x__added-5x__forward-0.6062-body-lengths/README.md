# masked-ppo-moderate-0.6062

[All candidates](../../../../../../README.md) · [Checkpoint](policy.pt) · [ONNX](policy.onnx) · [Reward configuration](config.json) · [Full metrics](metrics.json) · [Validation](validation.json)

Masked PPO, 5×, original-plus-added-switch. **0.6062 BL/cycle**, maximum **3** command changes per leg per complete cycle. Maximum yaw: 37.8 degrees; maximum lateral drift: 0.559 BL. Command dwell min/P05/median: 0.10/0.28/0.48 seconds.

## Selection rationale

Lower-end three-switch alternative with 0.10 s minimum hold; lateral drift reaches 0.559 BL.

Recorded fall and torso-contact environment fractions are both zero. Lateral drift exceeds 0.5 BL; Yaw exceeds 35 degrees; Hardware performance not verified.

## Simulation reproduction

Run from the repository root in the matching Isaac Lab environment. No retraining is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py --candidate masked-ppo-moderate-0.6062 --out reproduced-masked-ppo-moderate-0.6062.json
```

## CPU deployment dry run

```bash
python scripts/sim2real_transfer/run_policy.py --policy "switch-penalty-5x-to-100x/candidates/ppo-family/masked-ppo/target-band/original-plus-added-switch/original-5x__added-5x__forward-0.6062-body-lengths/policy.onnx" --profile binary --config switch-penalty-5x-to-100x/deployment.example.yaml --dry-run --fake-imu --duration 5
```

The dry run checks software execution without motor writes. Physical-robot performance remains unmeasured. [Prerequisites and troubleshooting](../../../../../../REPRODUCE_DQN_DDQN.md).

## Simulation Video

[Play the matching OneDrive video](https://pennstateoffice365-my.sharepoint.com/personal/bqc5667_psu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbqc5667_psu_edu%2FDocuments%2FLIBR%2FExperiments%2FHexapod%2FVideos%2FRL_Robin%2FRL_Binary%2Fgithub-policy-videos%2Fdisplacement-0.6-to-0.8%2Fppo-family%2Fmasked-ppo%2Foriginal-plus-added-switch%2Foriginal-5x__added-5x__forward-0.6062-body-lengths%2Fmasked-ppo-moderate-0.6062.mp4). Normal 1× playback; environment 0. The checkpoint SHA-256 matches this policy. [Full video catalog](../../../../../../VIDEO_CATALOG.md).
