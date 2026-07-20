# Hexapod locomotion tasks

Baseline configs (`flat_env_cfg.py`, `rough_env_cfg.py`, `hexapod_goal_env_cfg.py`,
`hexapod_mimic_env_cfg.py`) are documented in their own modules.

`flat_env_cfg_rshape.py` and `hexapod_goal_tuned_env_cfg.py` hold reward-shaping
variants that trade the baseline's short, rapid steps for longer swing phases. The
reasoning behind each parameter is in the module docstring of the corresponding file.

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

## Running them

```bash
# train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Hexapod-Rshape-v0 --headless

# evaluate a checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Hexapod-Rshape-Play-v0 --num_envs 1 --checkpoint <path>

# replay an open-loop gait CSV (joint angles in radians)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/playReal.py \
    --task Isaac-Velocity-Flat-Hexapod-Rshape-Play-v0 --num_envs 1 \
    --gait_csv <path> --gait_mode pos --gait_dt <seconds_per_row> --warmup_time 2.0
```

When evaluating, run each policy under the configuration it was trained with; the
baseline and reward-shaping configs use different ground friction. Keep `run_time`
below `episode_length_s`, otherwise the episode resets mid-run and the robot is
relocated.

Parameter values and task registrations are covered by
`source/isaaclab_tasks/test/test_hexapod_reward_shaping.py`.
