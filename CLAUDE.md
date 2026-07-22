# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

The import above pulls in the repo-wide Isaac Lab guidelines (API design conventions, dependency policy,
`./isaaclab.sh -p` tooling usage, changelog-fragment workflow, commit/PR conventions). It is generic and
does not conflict with the hexapod-specific guidance below — treat both as in effect.

## Project Overview

Isaac Lab (v3.0) is a GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim (6.0+ — the
exact supported point-release range was not fully confirmed during the 2.3.2 -> 3.0 migration; verify
against `docs/source/setup/installation/` before relying on a specific patch version). This fork adds a
6-legged robot (LiBR Hexapod) with flat-terrain RL training configurations including an imitation-learning
warm-up system.

**Key runtime requirements:**

- Isaac Sim 6.0+ installed and on PATH (or at `_isaac_sim` symlink)
- Python 3.11, PyTorch 2.7.0 + CUDA 12.8
- Hexapod USD model at `hexapod-assets/USD/Hexapod_Flattened.usd` (repo-relative)
- Data output directory at `C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/`

## Common Commands

Isaac Lab 3.0 replaced the old per-library `scripts/reinforcement_learning/<library>/train.py` / `play.py`
scripts with a unified CLI subcommand dispatched through `isaaclab.bat` (confirmed by reading
`isaaclab.bat` and `source/isaaclab/isaaclab/cli/__init__.py`: `isaaclab.bat train ...` / `isaaclab.bat play ...`
run `scripts/reinforcement_learning/train.py` / `play.py`, which call
`isaaclab_rl.entrypoints.run_train_cli` / `run_play_cli`; those dispatch to a backend module in
`source/isaaclab_rl/isaaclab_rl/entrypoints/backends/` selected by `--rl_library`). All scripts must be run
via Isaac Sim's bundled Python (not system Python):

```bat
:: Train hexapod on flat terrain
isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-v0 --num_envs 4096

:: Resume training from checkpoint
isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-v0 --resume

:: Play/evaluate a checkpoint (logs joint positions to CSV)
isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1

:: Train hexapod with imitation warm-up then RL (two-phase, single run)
isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-Mimic-v0 --num_envs 4096

:: Play/evaluate a mimic checkpoint
isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-Mimic-Play-v0 --num_envs 1

:: Fine-tune a mimic checkpoint under the flat RL env (compatible observation space)
isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-v0 --checkpoint <path/to/mimic/model.pt>

:: Play open-loop gait from CSV (compare against RL policy rewards) -- playReal.py is a fork-only
:: script with no --rl_library backend registration, so it is run directly by module path, not through
:: the unified play subcommand
isaaclab.bat -p source/isaaclab_rl/isaaclab_rl/entrypoints/backends/playReal.py --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1 --gait_csv <path_to_csv> --gait_mode pos --gait_dt <seconds_per_row> --warmup_time 1.0

:: Train the goal-reaching curriculum (reach a fixed forward distance as fast as possible)
isaaclab.bat train --rl_library rsl_rl --task Isaac-Goal-Flat-Hexapod-v0 --num_envs 4096

:: Play/evaluate a goal-reaching checkpoint
isaaclab.bat play --rl_library rsl_rl --task Isaac-Goal-Flat-Hexapod-Play-v0 --num_envs 1

:: List all registered environments
isaaclab.bat -p scripts/environments/list_envs.py

:: Run with a specific checkpoint
isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-Play-v0 --checkpoint <path>
```

```bat
:: Code quality (run from repo root)
pre-commit run --all-files

:: Linting (ruff, line-length=120)
ruff check source/
ruff format source/
```

```bat
:: Run tests (requires Isaac Sim Python)
isaaclab.bat -p -m pytest source/ -m "not isaacsim_ci"

:: Run a single test file
isaaclab.bat -p -m pytest source/isaaclab_tasks/test/test_hexapod_goal_mdp.py -v
```

`test_hexapod_goal_mdp.py` loads `hexapod_goal_rewards.py`/`hexapod_goal_curriculum.py` directly via `importlib` (not through the `isaaclab_tasks` package) and only depends on `torch`, so — unlike most tests here — it can also run under plain system Python: `python -m pytest source/isaaclab_tasks/test/test_hexapod_goal_mdp.py`.

`test_hexapod_reward_shaping.py` asserts the reward-shaping/tuned config files (`flat_env_cfg_rshape.py`, `hexapod_goal_tuned_env_cfg.py`) and their gym registrations hold the documented parameter values; it only does source-text assertions (no `torch`/Isaac Sim import at all), so it also runs under plain system Python: `python -m pytest source/isaaclab_tasks/test/test_hexapod_reward_shaping.py`.

Training logs save to `logs/rsl_rl/<experiment_name>/<timestamp>/`.

`scripts/sim2real_transfer/` is a separate plain-Python package (no Isaac Sim dependency, own `requirements.txt`) for running exported policies on the real robot — see **Sim-to-Real Deployment** below for its commands.

## Repository Structure

```text
hexapod-assets/              # Hexapod USD model, OBJ meshes, and reference gait CSVs (repo-tracked)
  USD/Hexapod_Flattened.usd
  OBJ/                         # Per-link OBJ meshes for offline PyVista rendering
    CenterLink.obj  BackLink.obj  FrontLink.obj
    MiddleLeft.obj  MiddleRight.obj
    BackLeft.obj  BackRight.obj  FrontLeft.obj  FrontRight.obj
    local_fix.json               # Blender world-rotation matrices (generated; not needed for current pipeline)
    extract_blender_rotations.py # Blender scripting-tab helper to regenerate local_fix.json
  Sim Gaits/forward3_lleg30_amp65_sim.csv
source/
  isaaclab/          # Core framework: env managers, sensors, controllers, terrain
  isaaclab_assets/   # Robot and sensor config dataclasses (ArticulationCfg)
  isaaclab_tasks/    # Task definitions: reward/obs/termination/event MDP terms
  isaaclab_rl/       # RL-specific wrappers (RslRlVecEnvWrapper, export utilities)
    isaaclab_rl/entrypoints/backends/  # rsl_rl backend (train_rsl_rl.py, play_rsl_rl.py,
                                        # cli_args_rsl_rl.py) dispatched via `isaaclab.bat
                                        # train/play --rl_library rsl_rl`, plus fork-only
                                        # direct-invoke scripts with no --rl_library entry:
                                        # playReal.py, playTracking.py, play_physicsGait.py,
                                        # playpyvista.py, render_pyvista.py, train_mimic.bat
  isaaclab_mimic/    # Imitation learning support
scripts/
  environments/                   # Utility scripts: list_envs, random_agent, zero_agent
  sim2real_transfer/              # Standalone real-hardware deployment package; see below
```

## Architecture: How a Task is Defined

Tasks use a layered config inheritance pattern. Everything is a Python dataclass decorated with `@configclass`:

```text
LocomotionVelocityRoughEnvCfg          # source/isaaclab_tasks/.../velocity_env_cfg.py
  └── HexapodRoughEnvCfg               # config/hexapod/rough_env_cfg.py
        └── HexapodFlatEnvCfg          # config/hexapod/flat_env_cfg.py
              ├── HexapodFlatEnvCfg_PLAY
              ├── HexapodMimicEnvCfg   # config/hexapod/hexapod_mimic_env_cfg.py
              │     └── HexapodMimicEnvCfg_PLAY
              └── HexapodGoalEnvCfg    # config/hexapod/hexapod_goal_env_cfg.py
                    └── HexapodGoalEnvCfg_PLAY
```

The environment config holds nested sub-configs for:

- `scene` — robot, terrain, sensors (height scanner, contact sensor)
- `observations` — `policy` and `critic` groups with individual term configs
- `rewards` — each reward term has `.weight` and `.params`
- `terminations` — episode ending conditions
- `events` — resets and domain randomization (mass, friction, push)
- `curriculum` — terrain level progression
- `commands` — velocity command sampling ranges

The gym environment is instantiated by `ManagerBasedRLEnv` using these configs. Each sub-config (`rewards`, `observations`, etc.) is handled by a matching Manager class that dynamically calls the referenced MDP functions.

## Hexapod-Specific Details

**Robot asset:** `source/isaaclab_assets/isaaclab_assets/robots/hexapod.py`

- USD at `hexapod-assets/USD/Hexapod_Flattened.usd` (repo-relative path)
- 8 joints total: `FrontLink`, `BackLink` (spine), + 6 leg joints (`MiddleLeft/Right`, `BackLeft/Right`, `FrontLeft/Right`)
- **Two actuator groups** (split because spine and legs have different loading profiles):
  - `body_joints` (FrontLink, BackLink): stiffness=40, damping=0.4, velocity_limit=15.0 rad/s, effort_limit=4.5 N·m
    - Higher velocity limit required: body sin wave needs fast tracking that 5.5 rad/s cap would prevent
    - Lower stiffness than legs: reduces peak torque demand from sustained gravity loading on body sections
  - `leg_joints` (all 6 legs): stiffness=80, damping=0.9, velocity_limit=6.0 rad/s, effort_limit=4.5 N·m
  - Physical spec: Dynamixel XL430-W250-T, stall torque 1.4 N·m at 12V, no-load speed 5.97 rad/s
  - `effort_limit_sim` is NOT a 1:1 analog of physical torque; it caps the PD output and needs headroom for damping term (`damping × velocity` can exceed physical stall torque)
- Init pose: spine joints at 0.0 rad, all leg joints at -0.47 rad

**Gym registration:** `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/hexapod/__init__.py`

- `Isaac-Velocity-Flat-Hexapod-v0` / `Isaac-Velocity-Flat-Hexapod-Play-v0`
- `Isaac-Velocity-Rough-Hexapod-v0` / `Isaac-Velocity-Rough-Hexapod-Play-v0`
- `Isaac-Velocity-Flat-Hexapod-Mimic-v0` / `Isaac-Velocity-Flat-Hexapod-Mimic-Play-v0`
- `Isaac-Goal-Flat-Hexapod-v0` / `Isaac-Goal-Flat-Hexapod-Play-v0`
- Reward-shaping/tuned variants (`Isaac-Velocity-Flat-Hexapod-Rshape-*`, `Isaac-Goal-Flat-Hexapod-BigStep-*`) —
  see `flat_env_cfg_rshape.py` / `hexapod_goal_tuned_env_cfg.py` below and the task-specific
  `config/hexapod/README.md` for the full parameter table

**Flat env key settings** (`flat_env_cfg.py`):

- Velocity target: lin_vel_x=(0.2, 0.2) m/s training / (0.16, 0.16) play, y=0, yaw=0 (forward-only gait)
- No height scanner, no terrain curriculum, flat plane terrain
- Friction: static=(0.5, 0.6), dynamic=(0.35, 0.45) — tuned for PLA on wood
- Asymmetric actor-critic observations: actor sees proprioceptive-only (hardware-available), critic adds ground-truth base_lin_vel during training
- `obs_groups = {"policy": ["policy"], "critic": ["critic"]}` routes groups to actor/critic in PPO runner
- Action scale effectively 0.5: `q = q_default + 0.5 * action`
- `q_default` for legs: -0.47 rad (matches init_state); spine joints: 0.0 rad
- `track_ang_vel_z_exp` uses EMA (exponential moving average, alpha=0.98, ~33-step window) rather than instantaneous yaw rate — sinusoidal undulation produces zero net drift so the EMA reward stays near 1.0, while sustained turning shifts the mean and gets penalized

**Hexapod-specific files in config folder:**

- `hexapod_obs_cfg.py` — `HexapodFlatObservationsCfg` with separate `PolicyCfg` (no base_lin_vel) and `CriticCfg` (adds base_lin_vel) observation groups
- `hexapod_rewards.py` — custom reward functions:
  - `feet_air_time_per_leg`: per-leg duty-cycle aware air time (kept for reference; reverted to fixed-threshold `feet_air_time` in the active config because it produced too weak a signal)
  - `track_ang_vel_z_exp_ema`: EMA-smoothed yaw tracking reward (currently active)
- `hexapod_mimic_env_cfg.py` — `HexapodMimicEnvCfg` and `HexapodMimicEnvCfg_PLAY`; see **Mimic System** below
- `hexapod_mimic_rewards.py` — `joint_pos_imitation` and `spine_pos_imitation` reward functions; `MotionReference` is built lazily and cached in a module-level dict keyed by `(csv_path, gait_period, col_order)` so it is only constructed once across all envs
- `hexapod_mimic_motion.py` — `MotionReference` class: loads a reference gait from CSV (headerless or headered) or falls back to a built-in sinusoidal tripod gait; resamples to a 200-point uniform phase grid; transfers to GPU lazily on first `get_reference()` call
- `agents/rsl_rl_ppo_mimic_cfg.py` — `HexapodMimicPPORunnerCfg`: 3000 total iterations, `init_noise_std`/`entropy_coef` left at the inherited flat/rough defaults (1.0 / 0.01) to match the May 2026 working run configuration, logs to `logs/rsl_rl/hexapod_mimic/`
- `hexapod_goal_env_cfg.py` — `HexapodGoalEnvCfg` and `HexapodGoalEnvCfg_PLAY`; see **Goal-Reaching System** below
- `hexapod_goal_curriculum.py` — `goal_distance_curriculum`: success-rate-gated distance progression
- `hexapod_goal_rewards.py` — `progress_to_goal`, `termination_signal`, `constant_per_step`, `reached_goal_done`
- `hexapod_goal_obs_cfg.py` — `HexapodGoalObservationsCfg`: mirrors `HexapodFlatObservationsCfg` but swaps `velocity_commands` for `pose_command` (4-dim relative goal pose)
- `agents/rsl_rl_ppo_goal_cfg.py` — `HexapodGoalPPORunnerCfg`: 3000 iterations, `num_steps_per_env=96`, `gamma=0.999`, `entropy_coef=0.003`, logs to `logs/rsl_rl/hexapod_goal/`
- `flat_env_cfg_rshape.py` — `HexapodFlatRshapeEnvCfg` (+ `_PLAY` and `Slide045/035/025` foot-slide-weight variants): reward-shaping on top of the flat baseline that trades short/rapid steps for longer swing phases (looser velocity-tracking std, higher `feet_air_time` weight/threshold, added `feet_slide` penalty, higher ground friction)
- `hexapod_goal_tuned_env_cfg.py` — `HexapodGoalBigStepEnvCfg` family (+ `Slide06/035`, `Minimal`, `_PLAY`): carries the same flat-shaping deltas over to the goal-reaching task
- `README.md` — task-variant reference table (which reward params each `-Rshape-*`/`-BigStep-*` task ID changes) and CLI examples; kept current with the registrations in `__init__.py`

**Mimic System** (`hexapod_mimic_env_cfg.py`, `hexapod_mimic_rewards.py`, `hexapod_mimic_motion.py`):

Two-phase training in a single run controlled by curriculum terms (`modify_reward_weight`):

- **Phase 1 — Imitation** (iterations 0–799, `MIMIC_ITERATIONS=800`): `joint_pos_imitation` reward active at weight 3.0; RL rewards currently at full weight (`MIMIC_RL_SCALE=1.0`) so imitation and RL compete from the start. Safety/limit rewards (`dof_pos_limits`, `undesired_contacts`) stay at full weight throughout.
- **Phase 2 — RL polish** (iterations 800–3000): curriculum sets imitation weight to 0.0 and restores all RL reward weights at `MIMIC_DECAY_STEPS = 800 × 48 = 38 400` env steps. Checkpoint is compatible with `Isaac-Velocity-Flat-Hexapod-v0` (identical observation space).

`joint_pos_imitation` reward: per-joint Gaussian `exp(-(q-q_ref)² / σ²)` averaged across all 8 joints. Per-joint mean (not sum) is critical — summing would let the two high-amplitude spine joints drown the gradient for the 6 leg joints. Default `joint_sigma=0.4` rad (~±23°). Phase is computed as `(episode_elapsed_time % gait_period) / gait_period`.

On first call, `joint_pos_imitation` validates that `asset.data.joint_names` matches the expected Sim DOF order and auto-corrects with a reorder index if they differ (logs a warning). This makes the reward robust to Isaac Lab version changes that might alter joint sorting.

`MotionReference` joint ordering (Sim DOF order, matches `asset.data.joint_pos` and play_rsl_rl.py CSVs):

```text
0 BackLink  1 FrontLink  2 MiddleLeft  3 MiddleRight  4 BackLeft  5 BackRight  6 FrontLeft  7 FrontRight
```

Tripod A (swing first half-cycle): indices 3, 4, 6 (MiddleRight, BackLeft, FrontLeft).
Tripod B (swing second half-cycle): indices 2, 5, 7 (MiddleLeft, BackRight, FrontRight).

`MIMIC_CSV_PATH` defaults to `hexapod-assets/Sim Gaits/forward3_lleg30_amp65_sim.csv`. If the file is absent, the built-in sinusoidal tripod gait is used automatically — no CSV needed to start training.

PPO tuning for mimic: `init_noise_std`/`entropy_coef` are intentionally left at the inherited flat/rough defaults (1.0 / 0.01, see `rsl_rl_ppo_cfg.py`) rather than lowered, per the May 2026 working run configuration.

**Goal-Reaching System** (`hexapod_goal_env_cfg.py`, `hexapod_goal_curriculum.py`, `hexapod_goal_rewards.py`, `hexapod_goal_obs_cfg.py`):

`HexapodGoalEnvCfg` inherits `HexapodFlatEnvCfg` and replaces the velocity-tracking task with "reach a fixed point N meters forward as fast as possible":

- **Command**: `commands.base_velocity` is disabled; `commands.pose_command` (`UniformPose2dCommandCfg`) is pinned to a fixed `(distance, 0, heading=0)` per episode — resampling interval equals `episode_length_s` (45.0) so it never resamples mid-episode. Distance is set externally by the curriculum, not sampled randomly.
- **Observations**: `HexapodGoalObservationsCfg` swaps `velocity_commands` for `pose_command` (4-dim: x, y, z, heading in robot base frame); otherwise identical actor/critic asymmetry as the flat task.
- **Rewards**: velocity-tracking rewards (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`) are removed. Task reward = `progress_to_goal` (velocity component toward goal, weight 10.0 — the per-step integral telescopes to total displacement toward goal) + sparse `reach_bonus` (weight 2500.0, fires once via the `reach_goal` termination) + `fall_penalty` (weight −1250.0, fires via `base_contact`) + `time_penalty` (constant −0.2/step to select faster gaits). Anti-jump/anti-bounce shaping terms (`lin_vel_z_l2`, `ang_vel_xy_l2`, `flat_orientation_l2`, etc.) are kept but weakened relative to the flat template. `position_command_error_tanh`-style proximity rewards are deliberately **not** used — they reward lingering near the goal, incentivizing slow approaches.
- **Termination**: `reach_goal` fires when the robot is within `REACH_RADIUS` (0.3 m) of the goal.
- **Curriculum** (`goal_distance_curriculum`): tracks success/fall counts over non-overlapping windows of `CURRICULUM_WINDOW_SIZE` (4096) completed episodes; advances to the next stage in `GOAL_DISTANCES = (1.0, 2.0, 3.5, 5.0)` m once the windowed success rate reaches `CURRICULUM_SUCCESS_THRESHOLD` (0.7). State is stored as ad hoc attributes on the `env` object (`_goal_curriculum_stage`, etc.) rather than in a config field.
- `HexapodGoalEnvCfg_PLAY` fixes distance at the final curriculum stage (5.0 m), disables the curriculum and domain randomization events, and sets a wide fixed-world camera to view the whole 5 m path across 16 envs.
- PPO tuning (`rsl_rl_ppo_goal_cfg.py`): inherits `HexapodRoughPPORunnerCfg`, raises `gamma` to 0.999 (longer effective horizon needed for a 45 s sparse/shaped goal task vs. the flat task's short-horizon velocity tracking) and lowers `entropy_coef` to 0.003.

**playReal.py** (`source/isaaclab_rl/isaaclab_rl/entrypoints/backends/playReal.py`; run directly via
`isaaclab.bat -p <path> --task ...` — it has no `--rl_library` backend registration, so it is not
reachable through the unified `play` subcommand):

- Extends play_rsl_rl.py to support open-loop gait CSV playback for sim-to-real comparison
- Key args: `--gait_csv`, `--gait_mode pos`, `--gait_dt <sec/row>`, `--warmup_time`, `--run_time`
- In `pos` mode: CSV values are absolute joint positions (rad); converted to actions via `(pos - default) / scale` where default comes from `robot.data.default_joint_pos`
- `q_default_list` in script: `[0.0, 0.0, -0.47, -0.47, -0.47, -0.47, -0.47, -0.47]`
- Prints applied torques (N·m) and base velocity (body frame, vx/vy/vz + yaw_rate) every 20 steps
- Logs joint positions to CSV and displacement tracking per step
- Reward breakdown printed at end of episode

Distinct from `scripts/sim2real_transfer/` (see below): `playReal.py` replays a CSV gait open-loop *inside Isaac Sim* for reward comparison; `sim2real_transfer` runs a trained policy closed-loop on the *actual robot*.

**MATLAB gait conversion** (external, not in repo):

- Real DOF order: `[flink, blink, FR, FL, MR, ML, BR, BL]`
- Sim DOF order: `[BackLink, FrontLink, MiddleLeft, MiddleRight, BackLeft, BackRight, FrontLeft, FrontRight]`
- Reorder: `newOrder = [2 1 6 5 8 7 4 3]`
- Leg correction: `position_legs_rad = position_legs_rad * -1 + pi`
- Body corrections: BackLink unchanged, FrontLink negated (`*-1`)
- Ground contact encoder → rad: `val1 = (4096/2 + 300) → 2348 → 3.601 rad → -0.459 rad` (matches init_state -0.47)

**play_rsl_rl.py data logging:**

- Joint positions (rad) → `C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_Rad_*.csv`
- Displacement tracking → `sim_displacement_log_*.csv`
- Reward breakdown printed at end of episode

## Offline PyVista Rendering Pipeline

GPU driver 596.36 causes RTX scenedb crashes in Isaac Sim's normal rendering path on this machine. This driver **cannot be rolled back** due to enterprise security policy. The offline rendering pipeline bypasses Isaac Sim's RTX renderer entirely by logging per-frame body world poses during play and replaying them through VTK/PyVista on the CPU.

**Two-step workflow:**

**Step 1 — Log body poses** (runs under Isaac Sim Python, requires GPU for physics):

```bat
:: Runs the RL policy for N steps and saves body_poses_<timestamp>.npz alongside the checkpoint
:: playpyvista.py has no --rl_library backend registration, so run it directly by module path
:: (not through the unified play subcommand)
isaaclab.bat -p source/isaaclab_rl/isaaclab_rl/entrypoints/backends/playpyvista.py ^
    --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1 --num_steps 500
```

NPZ layout: `body_pos_w` (T, 9, 3), `body_quat_w` (T, 9, 4 xyzw), `body_names`, `dt`.
Saved to `<checkpoint_dir>/body_pose_log/body_poses_<timestamp>.npz`.

**Step 2 — Render to MP4** (plain Python, no Isaac Sim, no GPU required):

```bat
pip install pyvista numpy imageio imageio-ffmpeg
python source/isaaclab_rl/isaaclab_rl/entrypoints/backends/render_pyvista.py ^
    --npz <path/to/body_poses_*.npz> ^
    --obj-up-axis Y --camera-distance 1.8 --ground --out hexapod_render.mp4
```

Key `render_pyvista.py` arguments:

- `--npz` — required; path to the NPZ from playpyvista.py
- `--obj-up-axis Y` — always use `Y` for OBJ files exported from Blender 5.x (Y-up legacy convention)
- `--camera-distance SCALE` — multiplies the auto-fitted camera distance; >1 zooms out (default 1.0)
- `--ground` — adds a static ground plane at z=0 centered at the robot's frame-0 position
- `--ground-size M` — ground plane side length in meters (default 5.0)
- `--grid-spacing M` — ground grid line spacing in meters (default 0.25); requires `--ground`
- `--grid-color STR` — grid line color (default "gray")
- `--diagnose` — renders frame 0 from four camera angles as a PNG montage (use to debug orientations)
- `--local-fix JSON` — per-body Blender matrix_world correction; **not needed** for this robot (USD body frames already match OBJ local frames)
- `--frames` — write PNG sequence instead of MP4

**OBJ mesh coordinate system:**

- Blender 5.x OBJ export uses Y-up: OBJ +Y = Blender local +Z (up), OBJ +Z = Blender local -Y (depth)
- `render_pyvista.py` pre-applies `M_to_blender = [[1,0,0],[0,0,-1],[0,1,0]]` to all rest_points at load time
- The USD body frames happen to match the OBJ local frames for this robot; no additional per-body correction is needed
- Body ordering in NPZ (matches `asset.data.body_names` and Isaac DOF order):
  `CenterLink, BackLink, FrontLink, MiddleLeft, MiddleRight, BackLeft, BackRight, FrontLeft, FrontRight`

**playpyvista.py vs play_rsl_rl.py:**

`playpyvista.py` is a fork of `play_rsl_rl.py` kept as a separate file so `play_rsl_rl.py` can be merged from upstream cleanly. Differences: always logs body poses (no flag), adds `--num_steps` to stop after a fixed count. The hardcoded output paths in `play_rsl_rl.py`/`playpyvista.py` (CSV joint log, displacement log) still point to `C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/` and must be updated if the machine changes.

## Sim-to-Real Deployment (`scripts/sim2real_transfer/`)

Plain-Python package (**no Isaac Lab / Isaac Sim dependency**) that runs a trained, ONNX-exported hexapod policy on the real robot from its own host computer (e.g. a Raspberry Pi): real Dynamixel servos + real IMU (via ROS2 `/imu`), no simulator involved. Lives in its own environment — `pip install -r scripts/sim2real_transfer/requirements.txt` — separate from Isaac Lab's bundled Python; ROS2 (`rclpy`) must already be installed on that host for the `/imu` topic to exist. See `scripts/sim2real_transfer/README.md` for day-to-day usage.

**Bring-up order** (each stage removes one more layer of risk before the next; run from `scripts/sim2real_transfer/`):

```bash
# 1-2. Unit tests, then offline ONNX validation against a sim-recorded trace — no hardware at all.
python -m pytest tests/
python tools/validate_onnx.py --trace <sim_trace.csv> --policy <policy.onnx> --profile velocity

# 3. Full loop, no hardware attached at all.
python run_policy.py --policy <policy.onnx> --profile velocity --dry-run --fake-imu --duration 5

# 4. Live IMU + live encoders, servos never move (zero physical risk).
python run_policy.py --policy <policy.onnx> --profile velocity --config config/deployment.yaml --no-torque

# 5. First real motion — robot propped with legs off the ground, damped action scale.
python run_policy.py --policy <policy.onnx> --profile velocity --config config/deployment.yaml --action-scale-mult 0.2

# 6+. Full scale, then tethered ground contact, then free running.
python run_policy.py --policy <policy.onnx> --profile velocity --config config/deployment.yaml --log-csv run.csv
```

`config/deployment.yaml` is gitignored; copy it from `config/deployment.example.yaml` and calibrate every field marked `# CALIBRATE` (leg encoder `zero_tick` values, IMU `mount_offset_quat`) against the physical robot before trusting it — the example ships placeholder values only.

**Architecture** (`sim2real/` package):

- `profiles.py` — `ProfileSpec`/`ObsBuilder` for the `velocity` and `goal` profiles; builds the obs vector in the exact term order the RL policy was trained on — `gyro(3), gravity(3), command(3 or 4), joint_pos_rel(8), joint_vel(8), last_action(8)` — mirroring `HexapodFlatObservationsCfg.PolicyCfg` / `HexapodGoalObservationsCfg.PolicyCfg` from the main Isaac Lab config
- `joint_mapping.py` — **the single most safety-critical file in the package**: converts between sim DOF order (matches `asset.data.joint_names` / the policy's action order) and real DOF order (matches physical wiring/motor IDs), applying a per-joint `correction_group` (`unchanged` / `negate` / `leg_negate_plus_pi`) before converting radians to encoder ticks. Reorder → sign/offset correction → tick conversion are kept as separate, independently testable steps rather than one fused formula
- `deployment_config.py` — typed loader for `deployment.yaml`; every hardware fact (serial port, motor IDs, encoder zero ticks, soft joint limits, IMU mount offset, control rate) lives here and nowhere else — other modules never touch YAML directly
- `policy_runner.py` — onnxruntime wrapper around policies exported by `isaaclab_rl.rsl_rl.exporter.export_policy_as_onnx`; validates the loaded graph's obs/action dims against the requested `--profile` at construction so a mismatched policy/profile pairing fails immediately instead of producing garbage actions
- `control_loop.py` — the 50 Hz loop: reads IMU + encoder ticks, builds obs, runs inference, clips to soft limits, writes goal ticks. Runs `imu.RosImuReader`'s `rclpy` spin in a background thread while the servo/inference loop stays on the main thread (mirrors the real robot's own `hexapod_tripod_adaptive.py` + `combined_logger.py` threading split, which lives outside this repo). A `try`/`finally` guarantees `soft_stop_ramp` + `torque_enable(False)` run on normal exit, an unhandled exception, or Ctrl+C alike
- `safety.py` — `Watchdog` (trips on comms silence or a failed sanity check — non-finite obs, out-of-range target tick) and `ramp_to_target` (linear interpolation used for both soft-start and soft-stop, so the robot never snaps to a target pose instantly)
- `command_source.py` — swappable `CommandSource` interface; v1 only ships constant sources read from `deployment.yaml` (a future joystick/SSH-driven source can be added without touching `control_loop.py`)
- `localization.py` — `DeadReckoningLocalizer`, **goal profile only, and the weakest link in the pipeline**: integrates gyro-z for heading and assumes a constant forward speed for position (the real robot has no GPS/mocap/AprilTag localization). Bring up the `velocity` profile first since it has zero dependency on this class; validate it separately (known-distance walk test) before trusting the `goal` profile
- `tools/validate_onnx.py` — offline validation in two modes: `direct` (recorded obs → onnxruntime → diff vs. recorded action) and `pipeline` (additionally rebuilds obs from raw sensor fields via the real `ObsBuilder`, isolating obs-construction bugs from ONNX/export bugs); run on both the dev machine and the actual Pi since onnxruntime/opset behavior can differ by platform

Swapping policies only requires pointing `--policy` at a different exported `.onnx` file of the same `--profile` — no config or code changes.

**Tests**: `python -m pytest scripts/sim2real_transfer/tests/` (plain pytest, no Isaac Sim needed).

## Actuator Tuning Notes

The `ImplicitActuatorCfg` in Isaac Lab applies: `torque = clip(stiffness*(q_target - q) - damping*q_dot, -effort_limit, effort_limit)`

Key tuning insights from this project:

- `effort_limit_sim` must be large enough to accommodate both position error AND damping terms simultaneously: at max velocity, `damping × velocity_limit` alone can equal or exceed the effort limit
- The real Dynamixel XL430 runs internal PID at ~1 kHz with load awareness; the sim PD controller needs extra headroom to approximate this
- Body (spine) joints saturate much more easily than leg joints because they sustain gravity loading through the full sin wave cycle; splitting actuator groups allows independent tuning
- If effort_limit is raised but joint still saturates: check velocity_limit_sim — if the commanded trajectory requires higher joint velocity than the cap, position error accumulates and torque saturates regardless of effort headroom

## RSL-RL Compatibility

The repo uses rsl-rl < 4.0.0. `handle_deprecated_rsl_rl_cfg` in `source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py` strips parameters unsupported by the installed version:

- Removes `optimizer` field
- Removes `share_cnn_encoders` (added in rsl-rl >= 4.0.0)

`HexapodFlatPPORunnerCfg` in `agents/rsl_rl_ppo_cfg.py` sets `obs_groups = {"policy": ["policy"], "critic": ["critic"]}` to route asymmetric observation groups to actor and critic networks respectively.

## MDP Terms Location

All reward/observation/termination/event functions referenced by string in configs are defined in:

- `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/hexapod/hexapod_rewards.py` — hexapod-specific custom rewards
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/` — locomotion-specific terms
- `source/isaaclab/isaaclab/envs/mdp/` — general-purpose MDP terms (shared across tasks)

When a config references e.g. `mdp.feet_air_time`, look in the locomotion mdp directory first, then the core mdp directory.

## Code Style

- Line length: 120 characters (ruff enforced)
- Python 3.11 type annotations (pyright strict mode)
- Pre-commit hooks: ruff lint + ruff-format + trailing whitespace
- No mock databases in tests; integration tests use live Isaac Sim physics

## Claude Code Agent Delegation

- Main session model is Sonnet (`.claude/settings.local.json`).
- If you get stuck after a couple of failed attempts, hit a confusing bug, or want an
  independent second opinion before committing to a risky approach, delegate to the
  `fable-helper` subagent (runs on Fable 5) rather than continuing to guess.
- For an objective that decomposes into several independent workstreams (e.g. review N
  files, apply the same kind of change across N configs), use the `/fable-team` skill —
  it delegates to `fable-orchestrator` (Fable 5), which plans the breakdown and fans it
  out to up to 5 parallel Sonnet subagents, then integrates their results.
