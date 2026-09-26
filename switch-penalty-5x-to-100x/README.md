# Switch penalty comparison: 5×–100×

This directory contains 18 simulation candidates: three per algorithm, selected from the two 09-24 waves under the strict per-leg, per-complete-cycle switch limit of **<=4**.

## Two waves

- **Wave 1 — original term only:** the original switch penalty is scaled by 5x, 10x, 30x, or 100x; Jackson's new term is zero.
- **Wave 2 — original term + Jackson term:** the same original scaled weight plus the new coefficient.

The exact values are printed on every card and in `candidate_manifest.json`: original switch weight, multiplier, new lambda, algorithm, configuration coefficients, candidate role, video, model, and hashes.

## Coefficient reference

The four tested multipliers are **5×, 10×, 30× and 100×**.

| Multiplier | Original switch weight (reference −0.005) | Wave 1 new λ | Wave 2 new λ (reference 0.0004) |
|---|---:|---:|---:|
| 5× | −0.025 | 0 | 0.002 |
| 10× | −0.05 | 0 | 0.004 |
| 30× | −0.15 | 0 | 0.012 |
| 100× | −0.5 | 0 | 0.04 |

These references define this sweep's multipliers, not every family's historical default. The source batch is identified by the two clearly named experiment waves above.

[Open the English report](https://psu-libr.github.io/IsaacLab-LiBRHexapod/switch-penalty-5x-to-100x/).

## Use

Open `index.html`. Each candidate has its matching video, metric JSON, and model file. `candidate_manifest.json` is the machine-readable index. The videos are simulation replays; passing the filter does not mean real-robot safety or deployment clearance. SAC-D 5-step candidates meet the switch-count filter but show near-stationary/poor-direction behavior and are included as explicit negative controls.

The evaluation uses `command_screen_xyzw`, a 1.0 s cycle at 50 control steps. Switches are counted per leg, not summed across six legs.
