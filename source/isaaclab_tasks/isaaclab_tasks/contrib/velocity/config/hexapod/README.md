# Hexapod locomotion tasks

Baseline configs (`flat_env_cfg.py`, `rough_env_cfg.py`, `hexapod_goal_env_cfg.py`,
`hexapod_mimic_env_cfg.py`) are documented in their own modules.

`flat_env_cfg_rshape.py` and `hexapod_goal_tuned_env_cfg.py` hold reward-shaping
variants that trade the baseline's short, rapid steps for longer swing phases. The
reasoning behind each parameter is in the module docstring of the corresponding file.

## Goal-reaching baseline (`Isaac-Goal-Flat-Hexapod-v0`)

`HexapodGoalEnvCfg` (`hexapod_goal_env_cfg.py`) replaces velocity tracking with "reach a fixed point N
meters forward as fast as possible," progressing through `GOAL_DISTANCES = (1.0, 2.0, 3.5, 5.0)` m via a
success-rate-gated curriculum (`goal_distance_curriculum`, threshold 0.7 over non-overlapping windows of
4096 completed episodes). Reward = `progress` (weight 10.0, closing-speed toward goal, telescopes to net
displacement over an episode) + `reach_bonus` (weight 2500.0, fires once via the `reach_goal` termination,
radius 0.3 m) + `fall_penalty` (weight -1250.0, fires via `base_contact`) + `time_penalty` (-0.2/step) +
weakened anti-jump/anti-bounce shaping (`lin_vel_z_l2`, `ang_vel_xy_l2`, `dof_torques_l2`, `dof_acc_l2`,
`action_rate_l2`, `feet_air_time`, `undesired_contacts`, `dof_pos_limits`). See the **Goal-Reaching
System** section of the top-level `CLAUDE.md` for exact current weights, curriculum mechanics, and the
`num_steps_per_env=96` PPO setting (2x flat/rough/mimic) that is the main iteration-time driver for this
task. `episode_length_s=45.0` (vs. 20.0 for flat/rough).

The `BigStep`/`Slide06`/`Slide035`/`Minimal` variants below are layered on top of this goal-reaching
baseline, not on the flat-velocity baseline.

## Reward-shaping tasks

| Task id | Base | Differs from base by |
|---|---|---|
| `Isaac-Velocity-Flat-Hexapod-Rshape-v0` | flat | velocity std 0.15, `feet_air_time` 2.5 @ 0.16 s, `action_rate_l2` -0.03, `feet_slide` -0.6, friction (0.9, 1.0) / (0.7, 0.8) |
| `Isaac-Velocity-Flat-Hexapod-Rshape-Slide045-v0` | Rshape | `feet_slide` -0.45 |
| `Isaac-Velocity-Flat-Hexapod-Rshape-Slide035-v0` | Rshape | `feet_slide` -0.35 |
| `Isaac-Velocity-Flat-Hexapod-Rshape-Slide025-v0` | Rshape | `feet_slide` -0.25 |
| `Isaac-Velocity-Flat-Hexapod-Rshape-Play-v0` | Rshape | evaluation settings |
| `Isaac-Goal-Flat-Hexapod-BigStep-v0` | goal | `feet_air_time` 2.5 @ 0.16 s, `action_rate_l2` -0.03 |
| `Isaac-Goal-Flat-Hexapod-BigStep-Slide06-v0` | BigStep | `feet_slide` -0.6 |
| `Isaac-Goal-Flat-Hexapod-BigStep-Slide035-v0` | BigStep | `feet_slide` -0.35 |
| `Isaac-Goal-Flat-Hexapod-BigStep-Minimal-v0` | goal | `feet_air_time` 2.5 @ 0.16 s only |
| `Isaac-Goal-Flat-Hexapod-BigStep-Play-v0` | BigStep | evaluation settings |

**Dependency caveat:** the `BigStep`/`Slide06`/`Slide035`/`Minimal` goal variants were trained against the
goal-env revision on the `sihan-physical-goal-training` branch (a distance-proportional goal command);
they override reward terms only, so they also apply on top of the current `HexapodGoalEnvCfg`, but the
recorded training results correspond to that earlier revision, not necessarily current behavior.

## Running them

```bash
# train
./isaaclab.sh train --rl_library rsl_rl \
    --task Isaac-Velocity-Flat-Hexapod-Rshape-v0 --headless

# evaluate a checkpoint
./isaaclab.sh play --rl_library rsl_rl \
    --task Isaac-Velocity-Flat-Hexapod-Rshape-Play-v0 --num_envs 1 --checkpoint <path>

# replay an open-loop gait CSV (joint angles in radians) -- playReal.py has no
# --rl_library backend registration, so it is run directly by module path
./isaaclab.sh -p source/isaaclab_rl/isaaclab_rl/entrypoints/backends/playReal.py \
    --task Isaac-Velocity-Flat-Hexapod-Rshape-Play-v0 --num_envs 1 \
    --gait_csv <path> --gait_mode pos --gait_dt <seconds_per_row> --warmup_time 2.0
```

When evaluating, run each policy under the configuration it was trained with; the
baseline and reward-shaping configs use different ground friction. Keep `run_time`
below `episode_length_s`, otherwise the episode resets mid-run and the robot is
relocated.

Parameter values and task registrations are covered by
`source/isaaclab_tasks/test/test_hexapod_reward_shaping.py`.
