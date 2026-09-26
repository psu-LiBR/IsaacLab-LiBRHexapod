# Preliminary · 6 algorithms × 4 rewards

[Open the English report](https://psu-libr.github.io/IsaacLab-LiBRHexapod/preliminary-6algorithms-4rewards/) · [Current switch-penalty policies](../switch-penalty-5x-to-100x/README.md)

Earlier simulation comparison of six algorithms under four reward definitions. The report preserves the mean-return curves, evaluation results and all 24 paired replays.

~~~text
index.html       English comparison and evaluation notes
report_assets/   Mean-return charts for the four reward definitions
videos/          Six algorithms × four reward definitions
~~~

Reward definitions in the report:
- Repository: existing repository reward.
- Candidate A: raw-magnitude constraints.
- Candidate B: bounded normalization.
- Candidate C: positive-reward quality gating added to the bounded formulation.

These are the historical report's defined reward labels, not new model versions. Different reward definitions produce incomparable training-return scales; compare common physical evaluation metrics instead.

This directory is a historical report and replay archive. It does not contain a complete standalone checkpoint deployment package. For the current checkpoint, ONNX, inference-metadata and configuration handoff, use the switch-penalty directory above. Historical videos use Git LFS; run git lfs pull after cloning. Video links in the online report use the GitHub media endpoint.

The older evaluation did not apply the current per-leg, per-complete-cycle switching screen. Its ranking must not be interpreted as hardware qualification or a controlled causal comparison with current results.
