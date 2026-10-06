# A2 reward record (binary-contact RL)

A record of the reward configuration Robin called **A2**, used for the DQN, DDQN, PPO and masked-PPO
contact-switch-penalty sweep (Sept 2026), plus how to reproduce the sweep's switch-penalty strengths with
the current training scripts. It is a reference for editing rewards by hand, not a preset: nothing in the
code reads this file.

- **Source:** branch `robin/binary-rl-extended-quadruped` (tip `f25cec23a1a`), `scripts/reinforcement_learning/binary_rl/overnight_config.py`
  (`COMMON_WEIGHTS`, `RAW_WEIGHTS`, `reward_metrics`) and each candidate's `config.json` (`resolved_reward_weights`).
  Those files are not part of this repo.
- **Task:** `Isaac-Goal-Flat-Hexapod-Binary-v0`, 4096 envs, training seed 46, training friction 0.18-0.25,
  evaluation friction 0.21, control step 0.02 s, spine Wave 1, DCMotor actuators.
- **Algorithms:** DQN, DDQN, PPO and masked PPO used A2. SAC-D used **J**, which is the env's own
  `_rebalance_binary_rewards` weights (see the last table). PPO and masked PPO also ran with
  `--no_value_preprocessor`.
- **Continuous SAC** was not part of the sweep.

## How a weight acts

`RewardManager` adds `weight x metric x dt` each step with `dt = 0.02 s`. Rate-style metrics are raw
per-second quantities. Event metrics (`reach`, `fall`) are divided by `dt` first, so they pay `weight` once
when the event fires, independent of `dt`.

## A2 reward terms

The "Current env" column is the term and weight in the binary env today (goal env plus
`_rebalance_binary_rewards`). **A2 replaced the whole term set** with its own metrics, so several terms are
not just reweighted versions of the current ones; the last column says how they differ.

| A2 term | A2 weight | A2 metric (per step) | Current env term | Current weight | A2 / current | Difference from the current term |
|---|---:|---|---|---:|---:|---|
| `progress` | 5.0 | Signed closing speed toward the goal in XY: `(prev_dist - dist) / dt` [m/s], zero on the first episode step | `progress` | 1.5 | 3.3x | Current term uses the 3D pose-command distance; first-step seeding is equivalent |
| `reach` | 2.0 | `reached x decay / dt`, `decay = 1 - 0.5 x clamp(t / 25 s, 0, 1)`. Reached = 3D distance < 0.2 m and no failure | `reach_bonus` | 2.0 | 1x | Same decay (min fraction 0.5); A2 gives a simultaneous failure priority over success |
| `fall` | -6.0 | `failed / dt`; failed = CenterLink contact force > 1 N | `fall_penalty` | -1.0 | 6x | Same event |
| `time` | -0.01 | Constant 1 | `time_penalty` | -0.003 | 3.3x | Same form |
| `contacts` | -1.0 | Count of torso links (CenterLink, BackLink, FrontLink) with force > 1 N | `undesired_contacts` | -0.5 | 2x | Same links and threshold |
| `limits` | -1.0 | Sum over joints of the excess beyond the soft position limits [rad] | `dof_pos_limits` | -1.0 | 1x | Same |
| `slide` | -0.01 | Sum over the 6 feet of foot XY speed x (foot force > 1 N) [m/s] | `feet_slide` | None (removed) | n/a | A2 keeps a small slide penalty; the current env removed it as unshapeable |
| `tilt` | -1.0 | `max(tilt - 10 deg, 0)^2` [rad^2], `tilt = acos(-g_b,z)` | none (`flat_orientation_l2` = 0) | 0.0 | n/a | A2-only term |
| `vertical` | -0.1 | World-frame z velocity squared [m^2/s^2] | `lin_vel_z_l2` | -1e-4 | 1000x | Current term uses the body-frame z velocity |
| `lateral` | -0.05 | Body-frame y velocity squared [m^2/s^2] | `lin_vel_y_l2` | -5e-4 | 100x | Same quantity |
| `roll_pitch` | -0.005 | Body roll and pitch rate squared, summed [rad^2/s^2] | `ang_vel_xy_l2` | -1e-6 | 5000x | Same quantity |
| `yaw` | -1e-4 | Body yaw rate squared [rad^2/s^2] | `ang_vel_z_l2` | -1e-4 | 1x | Same |
| `torque` | -3e-5 (DQN, DDQN); **-3e-4 (PPO, masked PPO)** | Sum over the 6 **leg** joints of applied torque squared [N^2 m^2] | `dof_torques_l2` | -3e-4 | 0.1x / 1x | A2 excludes the two spine joints; the current term sums the joints named in its config (all joints by default) |
| `acc` | -2.5e-9 | Sum over the 6 leg joints of joint acceleration squared | `dof_acc_l2` | -3e-8 | 0.08x | A2 excludes the spine joints |
| `switch` | -5e-4 default; swept (below) | `sum((a - a_prev)^2)` over the 6 leg bits (+-1), zero on the first episode step | `action_rate_l2` | -5e-4 | 1x | The current term has no first-step mask, so a reset step reads against the zero action |

`feet_air_time` is 0.0 in the current env and has no A2 counterpart.

To try A2 weights, edit the weights in `_rebalance_binary_rewards` or the goal env and use the table above to
match terms. Terms marked "A2-only" or with a different definition cannot be reproduced by weight changes alone.

## Contact-switch penalty

For +-1 bits `sum((a - a_prev)^2) = 4 x (number of flipped legs)`. So `action_rate_l2` already penalizes
every flip, and a flip costs `4 x |w| x dt` at weight `w`. This is the main knob.

The optional second term `action_switch_count` charges `weight` per flipped leg. It carries the same signal,
so it only adds a separately scaled knob. It is off by default.

### Command-line flags (every training script)

| Flag | Default | Effect |
|---|---|---|
| `--action_rate_multiplier M` | 1.0 | Multiplies the env's current `action_rate_l2` weight (-5e-4). Main knob |
| `--action_switch_penalty L` | 0.0 (off) | If `L > 0`, adds `action_switch_count` with weight `-L` (cost per flipped leg per step) |

Both apply to `train_discrete.py` (DQN, DDQN), `train_discrete_ppo.py` (PPO, masked PPO), `train_sac_d.py`
and `run_discrete_pipeline.py` (`--action-rate-multiplier`, `--action-switch-penalty`). `train_sac_continuous.py`
accepts only `--action_rate_multiplier` and rejects `--action_switch_penalty`. The resolved weights are
written to each run's `run_meta.json`.

### Robin's sweep strengths

Robin's multipliers were relative to `w0 = -0.005`, not to the env's current `-5e-4`. The "original" condition
set `w = m x w0` on the switch term; "original + added" also set `L = m x 0.0004`.

| Her multiplier | `switch` weight `w` | Cost per flipped leg, original (`4 x abs(w) x dt`) | Added `L` | Cost per flipped leg, original + added | Equivalent `--action_rate_multiplier` on the current env |
|---:|---:|---:|---:|---:|---:|
| 5x | -0.025 | 0.002 | 0.002 | 0.004 | 50 |
| 10x | -0.05 | 0.004 | 0.004 | 0.008 | 100 |
| 30x | -0.15 | 0.012 | 0.012 | 0.024 | 300 |
| 100x | -0.5 | 0.04 | 0.04 | 0.08 | 1000 |

The "added" condition simply doubles the per-flip cost at the same multiplier. Her 10x original-only run is
`--action_rate_multiplier 100`; the matching added term is `--action_switch_penalty 0.004`.

### Which strengths worked (from reviewing her results; not retrained here)

Evidence is weak: one training seed (46) per setting, one evaluation seed (7), 64 envs, a 6 s window, and the
same seed used to select and to report. Each family was also trained on a different reward base (A2 versus J), so
comparing across families is confounded.

| Family | Best-supported setting | Example result | Confidence |
|---|---|---|---|
| DQN | 10x original-only | 0.95 BL/cycle, max 2 switches per leg per cycle, all legs active | Moderate |
| DDQN | 10x original + added | 0.99 BL/cycle, max 4, all legs active | Moderate |
| PPO | 30x original-only | 0.69 BL/cycle, max 2, all legs active | Low |
| Masked PPO | 10x original-only | 0.65 BL/cycle; 2 legs pinned in stance | Very low |
| SAC-D 1-step | 5x original-only | 0.64 BL/cycle; 0.83 BL lateral drift | Very low |
| SAC-D 5-step | none | Every passing policy is near-stationary | n/a |

Caveats for any of these:

- Both the penalty and the switch metric act on commanded bits, not measured foot contact.
- Holding a leg's command constant costs nothing, so a penalty alone can be met by freezing legs, often
  permanently lifted. `switch_metrics.py` reports `frozen_legs` for this.
- A cycle under the per-leg limit can still be single-step (0.02 s) pulses; check `single_step_pulse_fraction`.

## J reward base (SAC-D, and the current env)

J is the env's rebalanced goal weights: `progress` 1.5, `reach_bonus` 2.0, `fall_penalty` -1.0,
`time_penalty` -0.003, `undesired_contacts` -0.5, `dof_pos_limits` -1.0, `feet_slide` None, `lin_vel_z_l2`
-1e-4, `lin_vel_y_l2` -5e-4, `ang_vel_xy_l2` -1e-6, `ang_vel_z_l2` -1e-4, `dof_torques_l2` -3e-4,
`dof_acc_l2` -3e-8, `action_rate_l2` -5e-4, `feet_air_time` 0.0, `flat_orientation_l2` 0.0. In the SAC-D sweep the
"original" term was this env's native `action_rate_l2` at `m x w0`.
