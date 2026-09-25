# Reward Tuning V2 — 2026-09-24

This directory contains 18 simulation candidates: three per algorithm, selected from the two 09-24 waves under the strict per-leg, per-complete-cycle switch limit of **<=4**.

## Two waves

- **Wave 1 — original term only:** the original switch penalty is scaled by 5x, 10x, 30x, or 100x; Jackson's new term is zero.
- **Wave 2 — original term + Jackson term:** the same original scaled weight plus the new coefficient.

The exact values are printed on every card and in `candidate_manifest.json`: original switch weight, multiplier, new lambda, algorithm, run ID, checkpoint step, video, and model hashes.

## Use

Open `index.html`. Each candidate has its matching video, metric JSON, and checkpoint. `candidate_manifest.json` is the machine-readable index. The videos are simulation replays; passing the filter does not mean real-robot safety or deployment clearance. SAC-D 5-step candidates meet the switch-count filter but show near-stationary/poor-direction behavior and are included as explicit negative controls.

The evaluation uses `command_screen_v2_xyzw`, a 1.0 s cycle at 50 control steps. Switches are counted per leg, not summed across six legs.
