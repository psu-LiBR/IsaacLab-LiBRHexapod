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
- Hexapod USD model at `hexapod-assets/USD/HexapiFlattened.usd` (repo-relative; a `Hexapod_Flattened.usd` path is commented out just above it in `hexapod.py`)
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

:: Train the binary contact-bit goal env (fork-only scripts, no --rl_library; e.g. categorical PPO).
:: Also: train_discrete.py (DQN/DDQN), train_sac_d.py, run_discrete_pipeline.py -- see binary_rl/README_binary_rl.md
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete_ppo.py --num_envs 4096 --timesteps 100000 --seed 42 --experiment_name ppo_s42

:: List all registered environments
isaaclab.bat -p scripts/environments/list_envs.py

:: Run with a specific checkpoint
isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Hexapod-Play-v0 --checkpoint <path>

:: Periodically record training videos (progress checks without a manual play run). --video_interval is in
:: env *steps*, not PPO iterations -- multiply the desired iteration interval by the task's num_steps_per_env
:: (48 for flat/rough/mimic tasks, 96 for the goal task) to get the --video_interval value. --video auto-enables
:: camera rendering. MP4s land in logs/rsl_rl/<experiment_name>/<timestamp>/videos/train/.
:: Example: every 250 iterations on the goal task (num_steps_per_env=96 -> 250*96=24000):
isaaclab.bat train --rl_library rsl_rl --task Isaac-Goal-Flat-Hexapod-v0 --num_envs 4096 --video --video_interval 24000 --video_length 200
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

- USD path in `HEXAPOD_CFG.spawn.usd_path`: **currently** `"hexapod-assets/USD/HexapiFlattened.usd"`
  (a `"hexapod-assets/USD/Hexapod_Flattened.usd"` line is present but commented out immediately above it —
  verify which USD is actually active before relying on the path).
- 8 joints total: `FrontLink`, `BackLink` (spine), + 6 leg joints (`MiddleLeft/Right`, `BackLeft/Right`, `FrontLeft/Right`)
- **Actuator model: `DCMotorCfg`** — migrated from `ImplicitActuatorCfg` (**breaking**: closed-loop
  joint dynamics change for every hexapod task, existing checkpoints must be retrained; see
  `source/isaaclab_assets/changelog.d/hexapod-dcmotor-actuator.rst`). Still **two groups** (spine vs.
  legs), now split only so `stiffness` can differ. Values below are **current on-disk** (still under
  active tuning); the file's comment header carries the full XL430-W250-T datasheet derivation.
  - Shared by both groups: `saturation_effort=1.4` N·m (== stall torque), `effort_limit=1.4` N·m,
    `velocity_limit=5.97` rad/s (== no-load speed, 11.1 V column, used directly — no voltage scaling),
    `damping=0.35`, `armature=1.3e-3` kg·m² (reflected rotor inertia ≈ J_rotor·258.5²; also needed for
    explicit-integration stability at `sim.dt=5e-3`), `friction=0.04` / `dynamic_friction=0.03` N·m
    (geartrain Coulomb / breakaway; modeled as a torque in Isaac Sim 5.0+, not a coefficient).
  - `body_joints` (FrontLink, BackLink): `stiffness=10.0`.
  - `leg_joints` (all 6 legs): `stiffness=50.0` (raised from 20 — a mild bump over the firmware
    P-Gain=640 default, motivated by open-loop tripod-gait tracking tests, trivially settable on the
    real servo).
  - DCMotor enforces the velocity-dependent torque-speed curve
    `tau_max(qd) = clip(saturation_effort * (1 - qd / velocity_limit), -inf, effort_limit)`, so
    deliverable torque collapses to ~0 as a joint approaches no-load speed — exactly like the real
    servo. This is the main sim2real gain over the old implicit model (which could deliver full torque
    at any speed). By design the sim will **not** perfectly track an aggressive open-loop reference gait.
  - `effort_limit_sim` / `velocity_limit_sim` are now left **unset**: DCMotor clips the physical
    envelope itself (no solver double-clip; the torque-speed curve governs joint speed).
  - Physical spec: Dynamixel XL430-W250-T — stall torque 1.4 N·m at 1.3 A, no-load speed 5.97 rad/s
    (57 rev/min) at 11.1 V, 258.5:1 gearing, 4096 pulse/rev (12 V column, for reference: 1.5 N·m stall,
    6.39 rad/s).
  - Rejected `ImplicitActuatorCfg` alternatives (kept for history, all **moot since the DCMotor swap**):
    `effort_limit_sim=4.5` N·m (3.2× physical stall — inflated headroom for the combined implicit
    PD+damping clamp); `body_joints` stiffness=80 ("too stiff -- small tracking lag generates huge
    torques; consistently saturates") and `velocity_limit_sim=5.5`; `leg_joints` stiffness=37 /
    damping=0.32 (max correctable error 1.4/37=0.038 rad before saturation) and `effort_limit_sim=1.4`.
- Init pose: spine joints (`FrontLink_Joint`, `BackLink_Joint`) at 0.0 rad. Leg joints are **currently
  +0.47 rad** under a `# For HEXAPI Implementation` block in `init_state.joint_pos`; an older
  `# For Trad Hexapod Implementation` block using **-0.47 rad** is present but commented out directly
  above it. Other parts of this doc (mimic gait tripod references, `playReal.py`'s `q_default_list`, the
  MATLAB conversion notes) still assume the -0.47 rad convention — reconcile the authoritative sign
  against `hexapod.py` directly before cross-referencing those sections.

**Gym registration:** `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/hexapod/__init__.py`

- `Isaac-Velocity-Flat-Hexapod-v0` / `Isaac-Velocity-Flat-Hexapod-Play-v0`
- `Isaac-Velocity-Rough-Hexapod-v0` / `Isaac-Velocity-Rough-Hexapod-Play-v0`
- `Isaac-Velocity-Flat-Hexapod-Mimic-v0` / `Isaac-Velocity-Flat-Hexapod-Mimic-Play-v0`
- `Isaac-Goal-Flat-Hexapod-v0` / `Isaac-Goal-Flat-Hexapod-Play-v0`
- `Isaac-Goal-Flat-Hexapod-Binary-v0` / `Isaac-Goal-Flat-Hexapod-Binary-Play-v0` (binary contact-bit action space; **no `rsl_rl_cfg_entry_point`** — trained by the fork-only scripts in `scripts/reinforcement_learning/binary_rl/`, see **Binary-Contact RL System** below)
- Reward-shaping/tuned variants (`Isaac-Velocity-Flat-Hexapod-Rshape-*`, `Isaac-Goal-Flat-Hexapod-BigStep-*`) —
  see `flat_env_cfg_rshape.py` / `hexapod_goal_tuned_env_cfg.py` below and the task-specific
  `config/hexapod/README.md` for the full parameter table

**Flat env key settings** (`flat_env_cfg.py`):

- Velocity target: lin_vel_x=(0.2, 0.2) m/s training / (0.16, 0.16) play, y=0, yaw=0 (forward-only gait)
- No height scanner, no terrain curriculum, flat plane terrain
- Friction: training randomizes static & dynamic friction over (0.18, 0.25); PLAY/eval pins both to
  (0.21, 0.21). This is a **2026-09 real-robot open-loop-gait friction-sweep calibration**
  (not flat-only): applied via the robot-side `events.physics_material`
  (`randomize_rigid_body_material`) ranges on the flat, rough, mimic, goal (incl. tuned/BigStep) and
  binary configs plus every `_PLAY` variant; terrain material and `friction_combine_mode` unchanged.
  The `Isaac-Velocity-Flat-Hexapod-Rshape-*` reward-shaping variants are the exception — they keep
  their own higher `static=(0.9, 1.0)/dynamic=(0.7, 0.8)`. Supersedes the earlier mixed ranges
  (flat-train `(0.2, 0.3)/(0.2, 0.25)`, the `(0.5, 0.6)/(0.35, 0.45)` "PLA on wood" guess in
  mimic-play, and the core `(0.8, 0.8)/(0.6, 0.6)` default the rough config fell through to). See
  `source/isaaclab_tasks/changelog.d/hexapod-friction-calibration.rst`.
- Asymmetric actor-critic observations: actor sees proprioceptive-only (hardware-available), critic adds ground-truth base_lin_vel during training
- `obs_groups = {"policy": ["policy"], "critic": ["critic"]}` routes groups to actor/critic in PPO runner
- Action scale effectively 0.5: `q = q_default + 0.5 * action`
- `q_default` for legs: currently +0.47 rad (matches `init_state`'s active "For HEXAPI Implementation" block; the -0.47 convention used elsewhere in this doc is unreconciled — see the Robot asset note above); spine joints: 0.0 rad
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
- `hexapod_binary_env_cfg.py` — `HexapodBinaryEnvCfg` / `HexapodBinaryEnvCfg_PLAY`: `HexapodGoalEnvCfg` with the 8-dim continuous joint action replaced by 6 leg contact bits + a scripted spine wave, and six inherited reward weights rebalanced by `_rebalance_binary_rewards()`; also holds the "CANONICAL SPINE-WAVE DEFINITION" block (Wave 1 / Wave 2 coefficients); see **Binary-Contact RL System** below
- `hexapod_binary_actions.py` — `SpineSineAction` / `SpineSineActionCfg`: zero-width (`action_dim == 0`) scripted action term that plays a truncated Fourier spine wave every step (no RL slot); `sin_coef`/`cos_coef` cfg fields, plus `set_waveform()` to swap Wave 1 → Wave 2 at runtime
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

**Goal-Reaching System** (`hexapod_goal_env_cfg.py`, `hexapod_goal_curriculum.py`, `hexapod_goal_rewards.py`, `hexapod_goal_obs_cfg.py`, `hexapod_goal_tuned_env_cfg.py`):

`HexapodGoalEnvCfg` inherits `HexapodFlatEnvCfg` and replaces the velocity-tracking task with "reach a fixed point N meters forward as fast as possible". Module-level constants (`hexapod_goal_env_cfg.py`, current on-disk — the earlier `(1.0, 2.0, 3.5, 5.0)` / `0.3` / `45.0` values are commented out directly above each): `GOAL_DISTANCES = (0.5, 1.0, 1.5, 2.0)` m, `REACH_RADIUS = 0.2` m, `EPISODE_LENGTH_S = 25.0` s, `CURRICULUM_SUCCESS_THRESHOLD = 0.7`, `CURRICULUM_DEMOTION_THRESHOLD = 0.4`, `CURRICULUM_WINDOW_FRACTIONS = (1.0, 1.0, 1.0, 1.0)` (per-stage fraction of `scene.num_envs` pooled into one curriculum window; all stages currently use a full window).

- **Command**: `commands.base_velocity = None`; `commands.pose_command` is a `UniformPose2dCommandCfg` with `resampling_time_range=(25.0, 25.0)` (== `episode_length_s`, so it never resamples mid-episode) and `ranges.pos_x` pinned to `(distance, distance)` for the active curriculum stage (`pos_y`/`heading` fixed at 0). The curriculum term mutates `command_term.cfg.ranges.pos_x` directly on every call — distance is set externally, never sampled.
- **Observations** (`HexapodGoalObservationsCfg`): `PolicyCfg` (6 terms, `enable_corruption=True`): `base_ang_vel` (`mdp.imu_ang_vel`, Unoise ±0.2), `projected_gravity` (Unoise ±0.05), `pose_command` (`mdp.generated_commands`, 4-dim: x, y, z, heading in robot base frame, no noise), `joint_pos` (`joint_pos_rel`, Unoise ±0.01), `joint_vel` (`joint_vel_rel`, Unoise ±1.5), `actions` (`last_action`). `CriticCfg` prepends `base_lin_vel` (ground-truth, Unoise ±0.1, `enable_corruption=False` for the whole critic group) ahead of the same 6 terms — asymmetric actor/critic as in the flat task.
- **Rewards** (current on-disk weights — `hexapod_goal_env_cfg.py` is under active tuning, verify against the file). The whole progress/terminal/time stack was **scaled down ~40–1250×** from the earlier `progress 10.0 / reach_bonus 2500 / fall_penalty -1250 / time_penalty -0.2` era (the "hexapod goal reward scale-down" work — to stop the policy "throwing itself" at the goal, since `progress_to_goal` is raw uncapped closing velocity and a leap/dive burst scored proportional to how fast it momentarily closed distance):
  - `progress = progress_to_goal(command_name="pose_command")`, weight **0.2**. Computes `(prev_dist_to_goal - curr_dist_to_goal) / env.step_dt` (closing-speed toward goal in m/s, using x/y/z of the 4-dim pose command), seeding `prev_dist = curr_dist` on the first post-reset step so no spurious cross-episode jump is scored. `RewardManager` multiplies by `weight * step_dt`, so `/step_dt` cancels and each step contributes `weight * (prev_dist - curr_dist)`; over an episode this telescopes to `weight * (initial_dist - final_dist)`.
  - `reach_bonus = time_decayed_termination_signal(termination_name="reach_goal", episode_length_s=25.0, min_fraction=0.5)`, weight **2.0** → fires between +2.0 (immediately after reset) and +1.0 (linearly decayed by `episode_length_s`) the one step `reach_goal` fires. Concentrates speed pressure into the single high-salience terminal transition.
  - `fall_penalty = termination_signal(termination_name="base_contact")`, weight **-1.0** → -1.0 the one step a fall is detected.
  - `time_penalty = constant_per_step()`, weight **-0.005**/step (the function returns a constant 1.0 tensor; the weight supplies sign/scale). `RewardManager` applies `value * weight * dt`, so it integrates to `weight * episode_duration_seconds` over an episode.
  - `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` are removed (`= None`).
  - Anti-jump / straight-line shaping, current value vs. the `HexapodFlatEnvCfg` baseline it overrides — all intentionally weaker than a full flat-walking template so a fast locomotion style can still be found: `lin_vel_z_l2` -1.0e-4 (flat: -1e-8), `ang_vel_xy_l2` -1.0e-6 (flat: 0.0), `dof_torques_l2` -3.0e-4 (flat: -5.0e-8), `dof_acc_l2` -2.5e-7 (flat: -8.5e-14), `action_rate_l2` -5.0e-4 (flat: -2.5e-6), `undesired_contacts` -1.0 (unchanged), `dof_pos_limits` -1.0 (unchanged), `flat_orientation_l2` left at the inherited flat value. `self.scene.height_scanner = None` (matches flat).
  - New goal-only terms (defined in `hexapod_goal_rewards.py`): `ang_vel_z_l2` -1.0e-4 (`goal_rewards.ang_vel_z_l2` — replaces the dropped `track_ang_vel_z_exp` so yaw rate is not left unpenalized; robot can still turn to face the goal but not spin wildly), `lin_vel_y_l2` -5.0e-4 (`goal_rewards.lin_vel_y_l2` — penalizes body-level sideways drift regardless of mechanism), `feet_slide` **-0.1** (`mdp.feet_slide` over the 6 leg bodies — a *small* continuous planted-foot-slide penalty, kept far below the Rshape/BigStep `-0.6/-0.35` because those were validated at much higher friction while this task runs the calibrated 0.18–0.25 band where forced slip is larger; raise from play videos only if "crazy" sliding persists). **This supersedes the old "deliberately not adopting `feet_slide`" stance.**
  - `feet_air_time`: weight **0.2** (flat: 0.25), `func` swapped to `goal_rewards.feet_air_time_ungated` and the `command_name` gate **removed entirely** (the flat default gates on `||command[:,:2]|| > 0.1` assuming a velocity command; retargeting to `pose_command` would turn that into a position-error gate that zeros the reward within 10 cm of the goal), `threshold` 0.12 s (flat: 0.1 s).
  - `position_command_error_tanh`-style proximity rewards are deliberately **not** used — lingering near the goal would accumulate reward, incentivizing slow approaches.
  - `reached_goal_bonus()` (a dead, unused alternative to `reach_bonus`) was deleted from `hexapod_goal_rewards.py`.
- **Termination**: `reach_goal` (`reached_goal_done`, radius `REACH_RADIUS` = 0.2 m) added on top of the inherited flat/rough terminations (`time_out`, `base_contact`, etc.).
- **Curriculum** (`goal_distance_curriculum`): state lives as ad hoc attributes on the shared `env` object (`_goal_curriculum_stage`, `_goal_curriculum_episodes`, `_goal_curriculum_successes`, `_goal_curriculum_falls`, `_goal_curriculum_last_rate`) — **global counters shared across all parallel envs, not per-env**. Each call filters `env_ids` to those with `episode_length_buf > 0` (skips the initial scene-setup call), and tallies `reach_goal` successes / `base_contact` falls from the termination manager; episodes that time out without either count toward the window denominator but neither the success nor fail numerator. Window size is **per-stage** (`window_sizes[stage]`, a tuple the same length as `distances`) rather than one hardcoded constant — `hexapod_goal_env_cfg.py` computes it as `max(1, round(scene.num_envs * CURRICULUM_WINDOW_FRACTIONS[stage]))` (fractions currently `(1.0, 1.0, 1.0, 1.0)` — freely tunable, so easier early stages *could* advance on less data) so window cost tracks `num_envs` instead of silently decoupling from it. Once the active stage's window closes, it computes `success_rate = successes/episodes`, advances one stage if `success_rate >= success_threshold` (0.7, inclusive) and not already at the last stage, **demotes one stage** if `demotion_threshold` is set (default `CURRICULUM_DEMOTION_THRESHOLD = 0.4`) and `success_rate` falls below it and the stage is not already the first, then unconditionally zeros the window counters either way — i.e. strictly non-overlapping windows. At the final stage (now 2.0 m) the stage index still clamps against advancing further (demotion can still bring it back down); the command distance updates every call, and success rate is still tracked/reported.
- `HexapodGoalEnvCfg_PLAY` fixes distance at the final curriculum stage (now 2.0 m), disables the curriculum, actor obs corruption and domain-randomization events (`base_external_force_torque`, `push_robot`), pins eval friction to `(0.21, 0.21)`, and sets a fixed-world camera (`viewer.eye=(-1.0,-3.0,1.5)`, `viewer.lookat=(2.5,0.0,0.2)`, `viewer.origin_type="world"`) across 16 envs (`scene.num_envs=16`, `env_spacing=8.0`).
- **`HexapodGoalBigStep*`/`Minimal` variants** (`hexapod_goal_tuned_env_cfg.py`, layered on top of `HexapodGoalEnvCfg`, not the flat baseline): `HexapodGoalBigStepEnvCfg` sets `action_rate_l2.weight = -0.03` and rebuilds `feet_air_time` at `weight=2.5`, `threshold=0.16 s` (both retargeted to `command_name="pose_command"`, mirroring `flat_env_cfg_rshape.py`'s shaping). `Slide06`/`Slide035` add a `feet_slide` penalty (`-0.6` / `-0.35`) on top of `BigStep`. `Minimal` inherits `HexapodGoalEnvCfg` directly (not `BigStep`) and only rebuilds `feet_air_time` (weight 2.5 @ 0.16 s). **Dependency caveats** (from the module docstring, both still apply): (1) these variants were trained against the goal-env revision on the `sihan-physical-goal-training` branch (a distance-proportional goal command) — recorded training results correspond to that earlier revision, not necessarily current behavior; (2) `feet_slide` was ported without the friction increase `flat_env_cfg_rshape.py`'s own docstring says is required for the penalty to be meaningful against physically-forced slip, so it risks penalizing forced slip rather than policy quality. (The base `HexapodGoalEnvCfg` now carries its *own* much smaller `feet_slide` at `-0.1`, deliberately kept low for exactly this reason — see the Rewards list above.)
- **PPO tuning** (`rsl_rl_ppo_goal_cfg.py`, `HexapodGoalPPORunnerCfg(HexapodRoughPPORunnerCfg)`): `max_iterations=3000`, **`num_steps_per_env=96`** (2x the `48` used by flat/rough/mimic — see iteration-time note below), `gamma=0.9995` (vs. parent 0.99; raised from an earlier 0.999), `lam=0.97` (vs. parent 0.95, explicit override added), `entropy_coef=0.003` (vs. parent 0.01 — kept *below* the parent baseline: raising it to 0.01 caused runaway action std >10, since the entropy term pulls std up every step and the only counterweight, the surrogate loss, is unusually noisy here given the sparse terminal rewards and the raised gamma/lam), `actor_hidden_dims=critic_hidden_dims=[128,128,128]` (smaller than the parent's `[512,256,128]`), `actor_obs_normalization=critic_obs_normalization=True` (parent leaves both `False`), `obs_groups={"policy": ["policy"], "critic": ["critic"]}`. Logs to `logs/rsl_rl/hexapod_goal/`. `gamma`/`lam` were raised together because the terminal `reach_bonus`/`fall_penalty` are larger than any per-step term and, at the prior `gamma=0.999`, were already discounted heavily by early-episode states over the ~1250-step, 25 s episode — pushing both further out extends the horizon over which that terminal credit propagates back. (The in-file comment still cites the old `0.999**2250` / 45 s episode figures — stale; the discount reasoning is unchanged.) `num_steps_per_env` was deliberately left at 96 (not lowered for iteration-time reasons, see below, and not yet raised toward ~150-192 as GAE-horizon coverage would suggest — an open question, not settled).
- **Iteration-time driver**: `num_steps_per_env=96` vs. `48` for every other hexapod task variant is the single clearest, directly-attributable ~2x multiplier on both rollout collection and the PPO update pass per iteration, holding `num_envs=4096` (default, unset by this task), `sim.dt=0.005` (200 Hz physics), `decimation=4` (50 Hz control), network size, and `num_learning_epochs=5`/`num_mini_batches=4` all constant vs. the flat/rough baseline. `episode_length_s=25.0` (vs. 20.0 for flat/rough) does **not** by itself add per-iteration cost — it only changes episode/curriculum cadence, not the fixed `num_steps_per_env` rollout length. Contact-sensor and IMU update at the full 200 Hz physics rate (`update_period=self.sim.dt`) and the robot has `enabled_self_collisions=True`, `solver_position_iteration_count=4` — both are baseline hexapod-sim costs shared identically by flat/rough/goal, not goal-specific. `height_scanner=None` for both flat and goal, so it is not a differentiator. No wall-clock/sec-per-iteration baseline is recorded anywhere in this repo for comparison.

**Binary-Contact RL System** (`hexapod_binary_env_cfg.py`, `hexapod_binary_actions.py`; training/eval scripts in `scripts/reinforcement_learning/binary_rl/`, see `README_binary_rl.md`):

`HexapodBinaryEnvCfg` / `HexapodBinaryEnvCfg_PLAY` inherit `HexapodGoalEnvCfg` / `_PLAY`. Observations, events, terminations, commands, and the distance curriculum are inherited unchanged — the **only structural change is the action space**:

- **6 leg contact bits** — one `BinaryJointPositionActionCfg` term per leg (policy action vector is 6-dim; the `binary_rl` scripts wrap it as `Discrete(64)` via `discrete_action_wrapper.py`). Convention: **1 / positive = stance (foot down), `STANCE_POS = 0.460` rad; 0 / negative = lift (foot up), `LIFT_POS = 1.180` rad** (leg limits `[-0.0873, +1.9199]` rad in the HexapI positive-leg convention). Bit / action-index order (matches hardware bit numbering; bits 2 & 4 inferred, pending hardware confirmation): `idx 0..5 = FrontRight, FrontLeft, MiddleRight, MiddleLeft, BackRight, BackLeft`.
- **Scripted spine wave** (`SpineSineAction`, **not** RL-controlled): a zero-width action term that drives the two spine joints every step from a truncated Fourier series `q(t) = offset + Σ_k [sin_coef[k]·sin((k+1)·w·t) + cos_coef[k]·cos((k+1)·w·t)]`, `w = 2π/period`, `period = GAIT_PERIOD_S = 1.0` s. There are **two distinct waves**, deliberately not unified (canonical definition + coefficients in `hexapod_binary_env_cfg.py`, "CANONICAL SPINE-WAVE DEFINITION" block):
  - **Wave 1 — the RL-env wave**, played during `Isaac-Goal-Flat-Hexapod-Binary-v0` training and `-Play-v0`: a fixed **analytic traveling body wave**, *not* fitted to any CSV. `FrontLink_Joint` is a pure sine `-0.9162978573·sin(2π·t/1.0)`; `BackLink_Joint` is the same sine shifted +90° (`-0.9162978573·cos(...)`) — magnitude `A_SPINE = 0.9162978573` rad (the **exact** open-loop gait-generator value: `deg2rad(70) · 12/16 = deg2rad(52.5)`) carried with the HexapI global spine-joint-sign flip so `sin_coef`/`cos_coef` hold **`-A_SPINE`** (corrected 2026-09-10 — a positive coefficient walked every re-evaluated policy backward; the old fitted wave had this flip as a negative amplitude and Wave 2's sim check independently confirmed the negative front-joint sign), shared offset `0.0`, only the phase differs by a quarter cycle, so the wave travels down the body. (Single-harmonic: `sin_coef`/`cos_coef` each hold one entry.) The BackLink-leads-FrontLink phase direction is still unverified (flag (a) in the config).
  - **Wave 2 — the tripod-baseline wave**: a single-harmonic **analytic** body wave regenerated straight from the MATLAB gait generator (**not** a CSV fit), **anti-phase** across the two spine joints (`BackLink_Joint = -FrontLink_Joint`, MATLAB `body_phase = π`): `FrontLink_Joint(t) = -A_SPINE·sin(w·t − π/4)`, `BackLink_Joint(t)` its negation (`_TRIPOD_FRONT_SIGN = -1.0`, sim-verified 2026-09-10 to walk the tripod baseline forward +0.83 m; the opposite sign walks it backward). Same `A_SPINE` amplitude (70°) and `0.0` offset as Wave 1. `tripod_extendedquad_sim.csv`'s byte-identical in-phase spine columns are a MATLAB→Sim export artifact (it folded the anti-phase pair onto one column) that Wave 2 deliberately does **not** reproduce. Used by `eval_protocol.py`'s `BASE_tripod_csv_bits` anchor **and** `play_discrete_closeup.py`'s `--gait_npz tripod` replay, swapped into the live `SpineSineAction` via `SpineSineAction.set_waveform`; never played by the RL env. The tripod baseline's *leg* contact-bit timing comes from `tripod_extendedquad_sim.csv` via `tripod_bit_demos.npz` (the npz never carried the spine).
  - `SpineSineActionCfg` fields are `sin_coef` / `cos_coef` (`dict[str, list[float]]`, one list entry per Fourier harmonic) plus unchanged `offset` / `period` — this **replaces** the earlier `amplitude` / `phase` (`dict[str, float]`) scalar fields.
  - **Breaking**: existing `Isaac-Goal-Flat-Hexapod-Binary-*` checkpoints must be retrained — the spine trajectory changed (was a per-joint sinusoid fitted to `tripod_B11BL0_sim.csv`'s spine columns, DOF-index-swapped and globally sign-flipped). Changelog fragment: `source/isaaclab_tasks/changelog.d/hexapod-binary-spine-wave.rst`.
- **Reward rebalance**: `__post_init__` calls `_rebalance_binary_rewards(self.rewards)` (identically for the train and play variants), which mutates six inherited goal reward weights (feet_slide, dof_acc_l2, feet_air_time, progress, undesired_contacts, time_penalty) for the discrete per-leg-bit action space. See `_rebalance_binary_rewards` in `hexapod_binary_env_cfg.py` for the current corrections and rationale (this file is under active tuning, like the actuator file — the docstring carries the `eval_protocol.py` evidence). The continuous `Isaac-Goal-Flat-Hexapod-v0` reward weights are untouched. Changelog fragment: `source/isaaclab_tasks/changelog.d/hexapod-binary-reward-rebalance.rst`.

The `binary_rl/` scripts are **fork-only, no `--rl_library` registration** — run directly, e.g. `isaaclab.bat -p scripts/reinforcement_learning/binary_rl/<script>.py ...`. Set: `train_discrete.py` (DQN / Double DQN), `train_discrete_ppo.py` (categorical PPO; `--mask` for masked-categorical PPO), `train_sac_d.py` (discrete SAC), `train_sac_continuous.py` (continuous SAC baseline on the 8-DOF `Isaac-Goal-Flat-Hexapod-v0`, *outside* the contact-bit space), `eval_protocol.py` (shared eval harness — all reported numbers come from it), `run_discrete_pipeline.py` (one-shot train→eval→rank→export pipeline), `export_binary_onnx.py` (ONNX export for sim2real). Runs land in `runs_binary/` (gitignored). A matching `binary` profile exists in `scripts/sim2real_transfer/` (its own `deployment.binary.example.yaml`).

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

**Note:** this pipeline was originally built to work around a GPU driver bug (driver 596.36 causing RTX
scenedb crashes in Isaac Sim's normal rendering path). That issue does not apply to the current setup —
Isaac Sim 6.0's normal RTX rendering path (`--video` / `--enable_cameras`) works fine on this machine. The
pipeline below is kept as an optional CPU-only alternative (e.g. for headless/no-GPU-for-rendering
scenarios), not as the required path for viewing training results.

The offline rendering pipeline bypasses Isaac Sim's RTX renderer entirely by logging per-frame body world poses during play and replaying them through VTK/PyVista on the CPU.

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

- `profiles.py` — `ProfileSpec`/`ObsBuilder` for the `velocity`, `goal`, and `binary` profiles; builds the obs vector in the exact term order the RL policy was trained on — `gyro(3), gravity(3), command(3 or 4), joint_pos_rel(8), joint_vel(8), last_action(8, or 6 for binary)` — mirroring `HexapodFlatObservationsCfg.PolicyCfg` / `HexapodGoalObservationsCfg.PolicyCfg` from the main Isaac Lab config. Total obs width is 33 (velocity), 34 (goal), **32 (binary)**. The `binary` policy emits only the six leg contact bits (its `last_action` obs term and action output are 6-dim, not 8); it drives the six leg joints from those bits (stance/lift snap), recreates the scripted spine sinusoid (Wave 1) host-side, and uses its own `deployment.binary.example.yaml`
- `joint_mapping.py` — **the single most safety-critical file in the package**: converts between sim DOF order (matches `asset.data.joint_names` / the policy's action order) and real DOF order (matches physical wiring/motor IDs), applying a per-joint `correction_group` (e.g. `unchanged` / `negate` / `leg_negate_plus_pi` — `joint_mapping.py` defines five) before converting radians to encoder ticks. Reorder → sign/offset correction → tick conversion are kept as separate, independently testable steps rather than one fused formula
- `deployment_config.py` — typed loader for `deployment.yaml`; every hardware fact (serial port, motor IDs, encoder zero ticks, soft joint limits, IMU mount offset, control rate) lives here and nowhere else — other modules never touch YAML directly
- `policy_runner.py` — onnxruntime wrapper around policies exported by `isaaclab_rl.rsl_rl.exporter.export_policy_as_onnx`; validates the loaded graph's obs/action dims against the requested `--profile` at construction so a mismatched policy/profile pairing fails immediately instead of producing garbage actions
- `control_loop.py` — the 50 Hz loop: reads IMU + encoder ticks, builds obs, runs inference, clips to soft limits, writes goal ticks. Runs `imu.RosImuReader`'s `rclpy` spin in a background thread while the servo/inference loop stays on the main thread (mirrors the real robot's own `hexapod_tripod_adaptive.py` + `combined_logger.py` threading split, which lives outside this repo). A `try`/`finally` guarantees `soft_stop_ramp` + `torque_enable(False)` run on normal exit, an unhandled exception, or Ctrl+C alike
- `safety.py` — `Watchdog` (trips on comms silence or a failed sanity check — non-finite obs, out-of-range target tick) and `ramp_to_target` (linear interpolation used for both soft-start and soft-stop, so the robot never snaps to a target pose instantly)
- `command_source.py` — swappable `CommandSource` interface; ships `ConstantVelocityCommand`, `ConstantGoalCommand` (`goal.mode: fixed`, a stationary world point) and `RecedingGoalCommand` (`goal.mode: receding`, a goal held `goal.lookahead_m` ahead of the robot's dead-reckoned position for continuous forward walking), all read from `deployment.yaml` — a future joystick/SSH-driven source can be added without touching `control_loop.py`
- `localization.py` — `DeadReckoningLocalizer`, **goal and binary profiles only, and the weakest link in the pipeline**: integrates gyro-z for heading and assumes a constant forward speed for position (the real robot has no GPS/mocap/AprilTag localization). Bring up the `velocity` profile first since it has zero dependency on this class; validate it separately (known-distance walk test) before trusting the `goal` or `binary` profile
- `tools/validate_onnx.py` — offline validation in two modes: `direct` (recorded obs → onnxruntime → diff vs. recorded action) and `pipeline` (additionally rebuilds obs from raw sensor fields via the real `ObsBuilder`, isolating obs-construction bugs from ONNX/export bugs); run on both the dev machine and the actual Pi since onnxruntime/opset behavior can differ by platform

Swapping policies only requires pointing `--policy` at a different exported `.onnx` file of the same `--profile` — no config or code changes.

**Tests**: `python -m pytest scripts/sim2real_transfer/tests/` (plain pytest, no Isaac Sim needed).

## Actuator Tuning Notes

The hexapod now uses `DCMotorCfg` (see **Robot asset** above). On top of the implicit PD law
`torque = stiffness*(q_target - q) - damping*q_dot`, `DCMotor` additionally clips the result every step
to the velocity-dependent envelope
`tau_max(qd) = clip(saturation_effort*(1 - qd/velocity_limit), -inf, effort_limit)` — peak deliverable
torque falls linearly from `saturation_effort` at zero speed to ~0 at `velocity_limit`.

Key tuning insights from this project:

- The torque-speed curve now handles peak-torque saturation directly. The spine keeps `stiffness=10`
  because its undulation is smooth and lightly loaded and the curve caps the peaks — under the old
  implicit model the low spine stiffness (and the inflated `effort_limit_sim=4.5`) was a *workaround*
  for that same saturation. The spine sinusoid still peaks near ~5.3 rad/s (≈ no-load speed), so it
  under-tracks its commanded amplitude — physically accurate for the real servo.
- Body (spine) joints saturate more easily than leg joints because they sustain gravity loading through
  the full sin-wave cycle; splitting actuator groups still lets `stiffness` be tuned per group
  (spine 10, legs 50).
- The real Dynamixel XL430 runs internal PID at ~1 kHz with load awareness; `armature` (reflected rotor
  inertia, ~20× the leg-link inertia at 258.5:1) plus geartrain `friction`/`dynamic_friction`
  approximate the parts of that the implicit model omitted.
- If a joint under-tracks a commanded trajectory it is now usually the torque-speed curve doing its job
  (commanded joint velocity approaching `velocity_limit`), not a tunable clamp — intended sim2real
  fidelity, not a bug to tune away.
- (Historical, moot since the DCMotor swap) Under `ImplicitActuatorCfg` the single `effort_limit_sim`
  clamp had to cover position error AND the damping term simultaneously — `damping × velocity_limit`
  alone could equal the effort limit — which is why the old sim value (4.5 N·m) ran well above physical
  stall (1.4 N·m).

## RSL-RL Compatibility

The repo pins **`rsl-rl-lib==5.4.1`** (`pyproject.toml`, core deps — the earlier "< 4.0.0" note
was stale). `handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)` in
`source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py` is a version-spanning shim the rsl_rl
train/play backends call with the actually-installed version
(`importlib.metadata.version("rsl-rl-lib")`); it mutates `agent_cfg` to bridge API changes:

- **< 4.0.0**: legacy `policy` is required; strips `optimizer` / `share_cnn_encoders`; clears
  the newer `actor` / `critic` / `student` / `teacher` model configs.
- **>= 4.0.0** (the pinned line): a legacy `policy = RslRlPpoActorCriticCfg(...)` block is
  deprecated — the shim infers `actor` + `critic` `RslRlMLPModelCfg`s from it, prints
  `[WARNING]` lines, then clears `policy`.
- **>= 5.0.0** (the pinned line): legacy stochastic params (`init_noise_std`,
  `noise_std_type`, …) are migrated into `distribution_cfg`.

Every hexapod PPO runner cfg (`HexapodRoughPPORunnerCfg` and its flat/goal/mimic subclasses in
`agents/rsl_rl_ppo_cfg.py`) still defines the network with the deprecated
`policy = RslRlPpoActorCriticCfg(...)` form, so each hexapod `train` / `play` run prints those
deprecation `[WARNING]`s on startup — harmless (the shim converts them); porting each cfg to
explicit `actor` / `critic` model configs would silence them.

`HexapodFlatPPORunnerCfg` also sets `obs_groups = {"policy": ["policy"], "critic": ["critic"]}` to
route asymmetric observation groups to actor and critic networks respectively.

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
