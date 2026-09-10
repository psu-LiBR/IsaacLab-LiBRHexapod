# Binary-contact RL — friction sweep, equal-budget ranking, and an evaluation-reset finding

All numbers below were produced with `eval_protocol.py` on this branch.

> **⚠ The sweep tables in this file are stale — re-measure, do not rescale.** Four
> changes since these runs each moved the numbers. The four locomotion **anchors** have
> now been **re-measured on the current code** — see
> [Re-measured anchors (2026-09-10)](#re-measured-anchors-2026-09-10) below. The
> friction-sweep, equal-budget-ranking and SAC-D-along-training tables further down have
> **not** been re-collected and are still on the old basis:
>
> 1. **Body-length rebasing (2026-09).** The BL/cycle body-length constant changed from
>    0.265 m to **0.315 m** (updated HexapI USD front-to-rear leg spacing; ~0.311 m measured
>    from the leg-link bounding boxes). The window conversion `metres / 1.59` is now
>    `metres / 1.89`. Every value below is on the old 0.265 m basis and reads ~1.19× too
>    high. An end-to-end audit of the metric found **no arithmetic bug** (`step_dt` is
>    genuinely 0.02 s, the window is exactly 6.0 cycles, `x_displacement_m` is the per-env
>    mean net forward displacement, cross-checked against the `progress` reward term); the
>    old-vs-video gap is the rebasing plus the no-reset continuous eval window plus possible
>    video-fps slow-motion.
> 2. **Tripod baseline CSV.** Every `tripod` / `tripod contact replay` number was measured
>    against `tripod_B11BL0_sim.csv` (leg swing/stance switch at steps 7/31). The tripod
>    baseline moved to `tripod_extendedquad_sim.csv` (switch at 13/37) — same tripod
>    grouping and actions (25/38), different timing — so the tripod anchor and every
>    `vs tripod` ratio below need re-measuring.
> 3. **Spine wave split.** The scripted spine is now **two distinct waves**: the RL env
>    plays a fixed analytic traveling body wave (FrontLink sine, BackLink +90°), while the
>    tripod baseline alone uses an **anti-phase** analytic body wave (`BackLink = −FrontLink`,
>    `FrontLink_Joint(t) = −A_SPINE·sin(2π·t − π/4)`), regenerated from the MATLAB spec and
>    swapped in via `SpineSineAction.set_waveform`. Existing binary checkpoints predate this
>    and must be retrained. (See `README_binary_rl.md` §1.)
> 4. **DCMotor actuator swap (2026-09).** The hexapod actuators migrated from
>    `ImplicitActuatorCfg` (4.5 N·m cap, full torque at any joint speed) to `DCMotorCfg`
>    (XL430 torque-speed curve, 1.4 N·m cap, leg stiffness 20→50, added armature +
>    geartrain friction). Every fixed checkpoint in the friction-sweep / equal-budget /
>    SAC-D tables was scored under the old model. See
>    `source/isaaclab_assets/changelog.d/hexapod-dcmotor-actuator.rst`.
>
> Re-run the full sweep on the 0.315 m (`÷1.89`) basis and record fresh tables; do not
> apply a scale factor to the numbers here.

## Protocol

| item | value |
|---|---|
| task | `Isaac-Goal-Flat-Hexapod-Binary-v0`, `Discrete(64)` = 6 contact bits |
| window | 64 envs x 300 steps x 0.02 s = 6.00 s = 6 gait cycles (period 1.0 s) |
| unit | BL/cycle = forward displacement (m) / 1.89  (body **0.315 m** x 6 cycles); the sweep tables below are still on the old `/ 1.59` (0.265 m) basis — anchors re-measured, see [Re-measured anchors (2026-09-10)](#re-measured-anchors-2026-09-10) |
| resets | all three terminations neutralised; one continuous trajectory |
| randomisation | 14 items pinned (see the `audit` block in every result JSON) |
| seed | 7, warmup 1 step discarded |
| **isolation** | **one policy per process** — see the finding below |

Reference points — **historical** (bit-identical across 10 independent processes;
0.265 m / `÷1.59` basis, old `tripod_B11BL0_sim.csv` baseline, old fitted spine wave —
superseded by [Re-measured anchors (2026-09-10)](#re-measured-anchors-2026-09-10)):

| policy | BL/cycle (old basis) |
|---|---|
| all six feet down (action 63) | 0.0734 |
| all six feet up (action 0) | 0.0525 |
| uniform random | 0.0589 |
| **tripod contact replay** | **0.4014** |

## Re-measured anchors (2026-09-10)

> **Superseded by the Wave 1 sign fix (2026-09-10, later same day).** The non-tripod
> anchors below (all-stance / all-lift / uniform) and every learned-policy re-eval in this
> doc were measured with Wave 1 at `+A_SPINE`, which drives the scripted body wave the
> wrong way — it walked every policy backward from the goal. Wave 1 now carries `−A_SPINE`
> (the HexapI global spine-joint-sign flip). The tripod anchor is unaffected (it swaps in
> Wave 2). Re-run `eval_protocol.py` after retraining against the corrected env.

`eval_protocol.py` re-run on the **current** code: `A_SPINE = 0.9162978573` rad (analytic
Wave 1), body-length **0.315 m**, `tripod_extendedquad_sim.csv` leg timing + anti-phase
analytic Wave 2 spine (`BackLink = −FrontLink`, `FrontLink_Joint(t) = −A_SPINE·sin(2π·t − π/4)`)
for the tripod anchor, corrected (non-`--legacy_reset`) reset. RTX 4060, all defaults:
64 envs × 300 steps × 0.02 s = 6.00 s = 6.00 cycles, seed 7, warmup 1, goal 2.0 m;
friction pinned to 0.215 (midpoint of
the calibrated 0.18–0.25 band), `num_buckets` 1, 15 randomisation items pinned (full
`audit` in the JSON).

| anchor | x_disp (m) | BL/cycle (÷1.89) | fall_rate | survival_s | straightness |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-stance (action 63) | +0.0914 | +0.0484 | 0.0 | 6.00 | 0.123 |
| all-lift (action 0) | −0.0298 | −0.0158 | 0.0 | 6.00 | 0.059 |
| uniform random | −0.0453 | −0.0239 | 0.0 | 6.00 | 0.179 |
| **tripod-CSV bits** | **+0.834** | **+0.441** | **0.0** | **6.00** | **0.984** |

`tripod-CSV bits` `reward_terms` (per step, weight-applied): reward_per_step +0.004023;
`progress` +0.004121, `time_penalty` −6e-05, `dof_acc_l2` −2.1e-05, `action_rate_l2`
−1e-05, `ang_vel_z_l2` −6e-06, `dof_torques_l2` −1e-06, all others 0.0.

**Sign / phase convention — anti-phase Wave 2, sim-verified 2026-09-10.** Wave 2 is
regenerated analytically from the MATLAB gait generator as a genuine anti-phase pair:
`FrontLink_Joint(t) = −A_SPINE·sin(2π·t − π/4)` and `BackLink_Joint(t) = −FrontLink_Joint(t)`,
shared amplitude `A_SPINE = 0.9162978573` rad, −π/4 phase origin, `w = 2π/1.0 s`. The
per-joint global sign (which joint is + vs −) is fixed by a dedicated Isaac Sim forward-
displacement test; see the "CANONICAL SPINE-WAVE DEFINITION" block in
`hexapod_binary_env_cfg.py`. The committed `tripod_extendedquad_sim.csv` has byte-identical
spine columns because a MATLAB→Sim "body correction" negated the FrontLink column, collapsing
MATLAB's `yy = −xx` anti-phase pair into an in-phase one — that was the bug; Wave 2 no longer
reads the CSV spine columns.

**The anti-phase Wave 2 tripod anchor walks +0.834 m / +0.441 BL/cycle forward**
(straightness 0.98, `fall_rate` 0), verified in sim 2026-09-10. That essentially matches
the ~0.40 BL/cycle the extended-quad gait reaches on real hardware / the pre-DCMotor sim
(old `ImplicitActuatorCfg`, which delivered full torque at any joint speed). The earlier
~0.05–0.06 BL/cycle recorded here was the *in-phase* Wave 2 bug: the byte-identical CSV
spine columns barely bent the body, so almost no travel. Restoring the genuine anti-phase
pair brings back the full body-bend amplitude and the gait works — the spine wave does
most of the propulsion (`--spine_gain 0` shows the leg bits alone net ≈ 0). The
`--tripod_spine_phase_deg` sweep and its old numbers (peaking near +0.15 m / 0.08 BL/cycle
at φ ≈ 0) were on the in-phase wave and have **not** been re-run. The BL/cycle metric
itself is sound: `body_length_m` 0.315, `n_cycles` 6.0, `step_dt` 0.02 s, and
`x_displacement_m` is the per-env-mean net forward displacement — all confirmed in the
result JSON's `protocol` block. (The *other* open-loop baselines — all-stance nets only
+0.048 BL/cycle — genuinely are weak under `DCMotorCfg`: the ±0.92 rad spine wave at 1 Hz
under-tracks and the binary "lift" pose barely clears the ground.)

**Waist vs. legs (`--spine_gain 0`).** Freezing the waist collapses every open-loop
baseline to ~0 net forward motion:

| anchor | x_disp normal (m) | x_disp waist-frozen (m) |
| --- | ---: | ---: |
| all-stance | +0.0914 | −0.0000 |
| all-lift | −0.0298 | +0.0009 |
| uniform random | −0.0453 | +0.0135 |
| tripod-CSV bits | +0.834 | −0.0081 |

The tripod baseline's entire forward travel (+0.834 m) is the scripted body
wave; the tripod **leg** contact pattern alone nets −0.008 m (≈ 0) — waist-frozen figures
are spine-independent, so this holds under the anti-phase Wave 2. Legs slightly negative;
same story for all-stance (+0.09 m → 0). Replayed
open-loop, the 6-bit contact action space contributes essentially no propulsion on its
own — the waist wave does the walking.

**RL checkpoints — not refreshed, no current number.** The best existing checkpoints
(`runs_binary/pipeline_20260909_230028` `ddqn`/`sac_d` @100k, ranked +0.95 / +0.92
BL/cycle on the old 0.265 m + old-wave basis) were trained against the pre-split fitted
spine wave. Under the current analytic Wave 1 they **walk backward**: ddqn@100k x_disp
−0.9422 m (−0.4985 BL/cycle), sac_d@100k −0.8532 m (−0.4514), both `fall_rate` 0,
straightness 0.70 / 0.66, `progress` reward −0.0047 / −0.0043 per step. This matches the
standing note that binary checkpoints predate the spine split and must be retrained —
there is no valid "best RL gait" anchor until a policy is trained on the current env.

## Finding: `env.reset()` does not refresh the command manager

`ManagerBasedEnv.reset()` runs `_reset_idx()` -> `sim.forward()` -> `observation_manager.compute()`.
It never calls `command_manager.compute()`. `ManagerBasedRLEnv.step()` does call it, between
`_reset_idx()` and the observation. So the first observation returned by an explicit `env.reset()`
carries the base-frame goal from the *previous* rollout.

Measured directly (sum over 64 envs of each observation term, right after reset):

```
rollout 1   pose_command =   0.000000   projected_gravity = -64.000000   score 0.0734
rollout 2   pose_command =  93.018677   projected_gravity = -64.118843   score 0.0746
tripod      pose_command = 144.204041   projected_gravity = -64.232529   score 0.4014
next        pose_command = -60.634594   projected_gravity = -53.897243   score 0.2675   <- DQN
```
`base_ang_vel`, `joint_pos`, `joint_vel` and `actions` are exactly 0.000000 in every rollout,
so joints, velocities and the action buffer do reset correctly. With `--fix_reset_command` the
first frame reads `pose_command = 128.000000` = 64 envs x 2.0 m, matching `--goal_distance`.

Scope: **training is unaffected** (its resets go through `step()`). Only explicit-reset evaluation
rollouts are. Impact measured on one checkpoint: same weights, same settings, score 0.2679 when the
rollout follows the tripod replay in the same process, 0.5728 when it is the first rollout of its
own process (0.5728 reproduced bit-identically by two independent scripts).

`--fix_reset_command` is off by default so previously reported numbers reproduce bit-for-bit.


## Friction sweep

`static_friction_range` and `dynamic_friction_range` pinned to a single value per run,
`dynamic = 0.9 x static`, `num_buckets = 1`. One process per (friction, policy) cell.

Training used `static (0.2, 0.3)` / `dynamic (0.2, 0.25)` with 64 buckets.

| static friction | tripod | SAC-D (126k) | DQN (282k) | PPO (84k) |
|---:|---:|---:|---:|---:|
| 0.10 | 0.2888 | 0.0966 | 0.2119 | 0.0336 |
| 0.15 | 0.3994 | 0.2746 | 0.3242 | -0.0265 |
| 0.25 | 0.4014 | 0.5819 | 0.5728 | 0.2596 |
| 0.35 | 0.3917 | 0.5840 | 0.5693 | 0.2620 |
| 0.45 | 0.4405 | 0.1168 | 0.5452 | 0.2566 |
| 0.55 | 0.4501 | 0.0291 | 0.3575 | 0.2905 |
| 0.70 | 0.4468 | 0.0047 | 0.3932 | 0.3298 |
| 0.90 | 0.4168 | -0.0234 | 0.4520 | 0.3127 |
| 1.20 | 0.3796 | 0.0230 | 0.3905 | 0.2507 |
| 1.50 | -0.0068 | 0.2815 | 0.0757 | -0.2410 |

Fall rate (fraction of envs whose base contact exceeds 1.0 N at any point):

| static friction | tripod | SAC-D | DQN | PPO |
|---:|---:|---:|---:|---:|
| 0.10 | 0.000 | 0.000 | 0.297 | 0.000 |
| 0.15 | 0.000 | 0.000 | 0.297 | 0.000 |
| 0.25 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.35 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.45 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.55 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.70 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.90 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1.20 | 0.000 | 0.000 | 0.094 | 0.000 |
| 1.50 | 0.031 | 0.062 | 0.203 | 0.172 |

- tripod is flat from 0.15 to 1.20 (0.38-0.45) and fails at both ends: -0.2086 at 0.05 (net backward) and -0.0068 at 1.50.
- SAC-D peaks inside the training band (0.5819 at 0.25, 0.5840 at 0.35) and is at 0.0047 by 0.70.
- DQN holds 0.36-0.57 from 0.25 through 1.20 and is above tripod at 0.90 (0.4520 vs 0.4168).
- No policy falls between 0.25 and 0.90.

## Equal-budget ranking (100k steps, 8 algorithms x 4 seeds)

Same runs, each evaluated at its 100k checkpoint. No checkpoint selection.

| algorithm | s46 | s47 | s48 | s49 | mean | vs tripod |
|---|---:|---:|---:|---:|---:|---:|
| dqfd | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 100% |
| sac_d | 0.0425 | 0.1242 | 0.4287 | -0.0769 | 0.1296 | 32% |
| vmpo | 0.1069 | 0.0892 | 0.0654 | 0.1837 | 0.1113 | 28% |
| dqn | 0.2011 | 0.0565 | 0.1440 | -0.0965 | 0.0763 | 19% |
| pqn | -0.0676 | 0.0061 | 0.0185 | 0.1210 | 0.0195 | 5% |
| c51 | 0.0638 | -0.0327 | 0.1747 | -0.2076 | -0.0005 | -0% |
| qrdqn | -0.0023 | -0.0043 | -0.0134 | -0.1267 | -0.0367 | -9% |
| ppo | -0.0422 | 0.0940 | -0.0369 | -0.2322 | -0.0543 | -14% |

No run reaches the tripod replay. DQfD returns 0.4000 on all four seeds; it is trained on the
tripod demonstrations, so this is consistent with it reproducing the demonstration rather than
departing from it (its action histogram has not been checked yet).

DQN and SAC-D also have 300k checkpoints:

| algorithm | s46 | s47 | s48 | s49 |
|---|---:|---:|---:|---:|
| dqn @300k | 0.0542 | -0.0077 | 0.0730 | 0.1651 |
| sac_d @300k | 0.0689 | 0.0777 | 0.0552 | 0.2040 |

## Same run, along training: SAC-D seed 46

Every checkpoint evaluated in its own process.

| steps | BL/cycle | vs tripod | action entropy (nats) |
|---:|---:|---:|---:|
| 106000 | 0.0381 | 9% | 0.245 |
| 110000 | 0.0304 | 8% | 0.093 |
| 114000 | 0.3587 | 89% | 0.856 |
| 118000 | 0.2561 | 64% | 0.847 |
| 120000 | 0.5405 | 135% | 1.064 |
| 122000 | 0.2017 | 50% | 0.779 |
| 124000 | 0.2301 | 57% | 0.854 |
| 126000 | 0.5819 | 145% | 0.979 |
| 128000 | 0.2820 | 70% | 0.882 |
| 130000 | 0.0988 | 25% | 0.455 |
| 132000 | 0.3773 | 94% | 0.527 |
| 136000 | 0.5053 | 126% | 1.217 |
| 140000 | 0.4759 | 119% | 1.346 |
| 146000 | 0.5457 | 136% | 1.312 |
| 152000 | 0.3721 | 93% | 0.593 |

Best point 0.5819 (145% of tripod); mean over the 15 points 0.3263 (81%); 5 of 15 are above tripod.
Adjacent checkpoints 2000 steps apart differ by 2-5x (126k 0.5819 -> 128k 0.2820 -> 130k 0.0988).
Score tracks action entropy: the points above tripod have entropy 0.98-1.35, the points below 0.5 have entropy 0.09-0.46.

So the peak is not a single lucky checkpoint, but the policy does not hold it: reporting the maximum
(145%) and reporting the window mean (81%) give different answers to different questions.


## Not established

- Why the policy oscillates this strongly between adjacent checkpoints.
- Whether DQN's 0.5728 is a peak or a plateau: the other checkpoints of that run are not on this machine.
- Each friction cell is one checkpoint of one training seed; the friction curves are not algorithm-level claims.
- Static and dynamic friction were swept together (dynamic = 0.9 x static), so the two cannot be separated.
- The mechanism behind the residual `projected_gravity` offset (it is not explained by the command manager).
