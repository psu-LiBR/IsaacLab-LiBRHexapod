# Video Catalog

[Overview](README.md) · [Selection](SELECTION.md)

Video storage is separate from the Git repository. The destination is the shared OneDrive directory `RL_Robin/RL_Binary`. No video binaries or anonymous edit-sharing tokens are stored in this repository.

## Structure

```text
RL_Binary/
├── README.md
├── github-policy-videos/                  # 47 included policies: 33 target + 14 higher
│   ├── README.md
│   ├── video_index.csv                   # Candidate, checkpoint hash, video hash, metrics
│   ├── dqn-family/
│   ├── ppo-family/
│   └── sac-family/
│       └── <algorithm>/<group>/<condition>/<candidate>/<candidate-id>.mp4
└── additional-simulation-videos/          # Other 32 target-band numerical matches
    ├── README.md
    ├── video_index.csv
    └── current-study/<family>/<algorithm>/target-band/<status>/<condition>/*.mp4
```

The first collection mirrors the GitHub hierarchy beneath `candidates/`. Its candidate ID and checkpoint SHA-256 must match `candidate_manifest.json`; similar rounded BL/cycle values are not sufficient. Each index lists the exact video filename and SHA-256. Files retain their original frame rate and playback speed. The rendered environment is one instance; the published screening metrics aggregate 64 environments.

The second collection contains the 32 remaining target-band numerical matches, including the DDQN torso-contact case marked in its index. These are visual references, not published policy bundles or hardware-approved gaits. The target-band population is fully covered across the two collections; only fourteen representative higher-displacement videos are included. Historical and below-band videos are outside this delivery.

**Transfer status:** the destination README has been uploaded and upload access is verified. The 79-video transfer and playback verification are in progress; this catalog does not yet claim that all files are available remotely.
