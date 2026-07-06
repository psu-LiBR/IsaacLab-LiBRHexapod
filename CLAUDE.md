# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab (v2.3.2) is a GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim (4.5/5.0/5.1). This fork adds a 6-legged robot (LiBR Hexapod) with flat-terrain RL training configurations including an imitation-learning warm-up system.

**Key runtime requirements:**

- Isaac Sim 4.5+ installed and on PATH (or at `_isaac_sim` symlink)
- Python 3.11, PyTorch 2.7.0 + CUDA 12.8
- Hexapod USD model at `hexapod-assets/USD/Hexapod_Flattened.usd` (repo-relative)
- Data output directory at `C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/`

## Common Commands

All scripts must be run via Isaac Sim's bundled Python (not system Python). Use `isaaclab.bat` on Windows:

```bat
:: Train hexapod on flat terrain
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Hexapod-v0 --num_envs 4096

:: Resume training from checkpoint
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Hexapod-v0 --resume

:: Play/evaluate a checkpoint (logs joint positions to CSV)
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1

:: Train hexapod with imitation warm-up then RL (two-phase, single run)
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Hexapod-Mimic-v0 --num_envs 4096

:: Play/evaluate a mimic checkpoint
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Hexapod-Mimic-Play-v0 --num_envs 1

:: Fine-tune a mimic checkpoint under the flat RL env (compatible observation space)
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Hexapod-v0 --checkpoint <path/to/mimic/model.pt>

:: Play open-loop gait from CSV (compare against RL policy rewards)
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/playReal.py --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1 --gait_csv <path_to_csv> --gait_mode pos --gait_dt <seconds_per_row> --warmup_time 1.0

:: List all registered environments
isaaclab.bat -p scripts/environments/list_envs.py

:: Run with a specific checkpoint
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Hexapod-Play-v0 --checkpoint <path>
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
```

Training logs save to `logs/rsl_rl/<experiment_name>/<timestamp>/`.

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
  isaaclab_mimic/    # Imitation learning support
scripts/
  reinforcement_learning/rsl_rl/  # train.py, play.py, playReal.py, playpyvista.py, render_pyvista.py
  environments/                   # Utility scripts: list_envs, random_agent, zero_agent
```

## Architecture: How a Task is Defined

Tasks use a layered config inheritance pattern. Everything is a Python dataclass decorated with `@configclass`:

```text
LocomotionVelocityRoughEnvCfg          # source/isaaclab_tasks/.../velocity_env_cfg.py
  └── HexapodRoughEnvCfg               # config/hexapod/rough_env_cfg.py
        └── HexapodFlatEnvCfg          # config/hexapod/flat_env_cfg.py
              ├── HexapodFlatEnvCfg_PLAY
              └── HexapodMimicEnvCfg   # config/hexapod/hexapod_mimic_env_cfg.py
                    └── HexapodMimicEnvCfg_PLAY
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

**Gym registration:** `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/__init__.py`

- `Isaac-Velocity-Flat-Hexapod-v0` / `Isaac-Velocity-Flat-Hexapod-Play-v0`
- `Isaac-Velocity-Rough-Hexapod-v0` / `Isaac-Velocity-Rough-Hexapod-Play-v0`
- `Isaac-Velocity-Flat-Hexapod-Mimic-v0` / `Isaac-Velocity-Flat-Hexapod-Mimic-Play-v0`

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
- `agents/rsl_rl_ppo_mimic_cfg.py` — `HexapodMimicPPORunnerCfg`: 3000 total iterations, `init_noise_std=0.25`, `entropy_coef=0.005`, logs to `logs/rsl_rl/hexapod_mimic/`

**Mimic System** (`hexapod_mimic_env_cfg.py`, `hexapod_mimic_rewards.py`, `hexapod_mimic_motion.py`):

Two-phase training in a single run controlled by curriculum terms (`modify_reward_weight`):

- **Phase 1 — Imitation** (iterations 0–799, `MIMIC_ITERATIONS=800`): `joint_pos_imitation` reward active at weight 3.0; RL rewards currently at full weight (`MIMIC_RL_SCALE=1.0`) so imitation and RL compete from the start. Safety/limit rewards (`dof_pos_limits`, `undesired_contacts`) stay at full weight throughout.
- **Phase 2 — RL polish** (iterations 800–3000): curriculum sets imitation weight to 0.0 and restores all RL reward weights at `MIMIC_DECAY_STEPS = 800 × 48 = 38 400` env steps. Checkpoint is compatible with `Isaac-Velocity-Flat-Hexapod-v0` (identical observation space).

`joint_pos_imitation` reward: per-joint Gaussian `exp(-(q-q_ref)² / σ²)` averaged across all 8 joints. Per-joint mean (not sum) is critical — summing would let the two high-amplitude spine joints drown the gradient for the 6 leg joints. Default `joint_sigma=0.4` rad (~±23°). Phase is computed as `(episode_elapsed_time % gait_period) / gait_period`.

On first call, `joint_pos_imitation` validates that `asset.data.joint_names` matches the expected Sim DOF order and auto-corrects with a reorder index if they differ (logs a warning). This makes the reward robust to Isaac Lab version changes that might alter joint sorting.

`MotionReference` joint ordering (Sim DOF order, matches `asset.data.joint_pos` and play.py CSVs):

```text
0 BackLink  1 FrontLink  2 MiddleLeft  3 MiddleRight  4 BackLeft  5 BackRight  6 FrontLeft  7 FrontRight
```

Tripod A (swing first half-cycle): indices 3, 4, 6 (MiddleRight, BackLeft, FrontLeft).
Tripod B (swing second half-cycle): indices 2, 5, 7 (MiddleLeft, BackRight, FrontRight).

`MIMIC_CSV_PATH` defaults to `hexapod-assets/Sim Gaits/forward3_lleg30_amp65_sim.csv`. If the file is absent, the built-in sinusoidal tripod gait is used automatically — no CSV needed to start training.

PPO tuning rationale for mimic: `init_noise_std=0.25` (down from default 1.0) prevents the policy from getting imitation reward "for free" by saturating joints at their limits, which causes value-function divergence. `entropy_coef=0.005` (down from 0.01) allows action std to decrease once the gradient supports deterministic tracking.

**playReal.py** (`scripts/reinforcement_learning/rsl_rl/playReal.py`):

- Extends play.py to support open-loop gait CSV playback for sim-to-real comparison
- Key args: `--gait_csv`, `--gait_mode pos`, `--gait_dt <sec/row>`, `--warmup_time`, `--run_time`
- In `pos` mode: CSV values are absolute joint positions (rad); converted to actions via `(pos - default) / scale` where default comes from `robot.data.default_joint_pos`
- `q_default_list` in script: `[0.0, 0.0, -0.47, -0.47, -0.47, -0.47, -0.47, -0.47]`
- Prints applied torques (N·m) and base velocity (body frame, vx/vy/vz + yaw_rate) every 20 steps
- Logs joint positions to CSV and displacement tracking per step
- Reward breakdown printed at end of episode

**MATLAB gait conversion** (external, not in repo):

- Real DOF order: `[flink, blink, FR, FL, MR, ML, BR, BL]`
- Sim DOF order: `[BackLink, FrontLink, MiddleLeft, MiddleRight, BackLeft, BackRight, FrontLeft, FrontRight]`
- Reorder: `newOrder = [2 1 6 5 8 7 4 3]`
- Leg correction: `position_legs_rad = position_legs_rad * -1 + pi`
- Body corrections: BackLink unchanged, FrontLink negated (`*-1`)
- Ground contact encoder → rad: `val1 = (4096/2 + 300) → 2348 → 3.601 rad → -0.459 rad` (matches init_state -0.47)

**play.py data logging:**

- Joint positions (rad) → `C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/HexapodRL_Rad_*.csv`
- Displacement tracking → `sim_displacement_log_*.csv`
- Reward breakdown printed at end of episode

## Offline PyVista Rendering Pipeline

GPU driver 596.36 causes RTX scenedb crashes in Isaac Sim's normal rendering path on this machine. This driver **cannot be rolled back** due to enterprise security policy. The offline rendering pipeline bypasses Isaac Sim's RTX renderer entirely by logging per-frame body world poses during play and replaying them through VTK/PyVista on the CPU.

**Two-step workflow:**

**Step 1 — Log body poses** (runs under Isaac Sim Python, requires GPU for physics):

```bat
:: Runs the RL policy for N steps and saves body_poses_<timestamp>.npz alongside the checkpoint
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/playpyvista.py ^
    --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1 --num_steps 500
```

NPZ layout: `body_pos_w` (T, 9, 3), `body_quat_w` (T, 9, 4 wxyz), `body_names`, `dt`.
Saved to `<checkpoint_dir>/body_pose_log/body_poses_<timestamp>.npz`.

**Step 2 — Render to MP4** (plain Python, no Isaac Sim, no GPU required):

```bat
pip install pyvista numpy imageio imageio-ffmpeg
python scripts/reinforcement_learning/rsl_rl/render_pyvista.py ^
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

**playpyvista.py vs play.py:**

`playpyvista.py` is a fork of `play.py` kept as a separate file so `play.py` can be merged from upstream cleanly. Differences: always logs body poses (no flag), adds `--num_steps` to stop after a fixed count. The hardcoded output paths in `play.py`/`playpyvista.py` (CSV joint log, displacement log) still point to `C:/Users/jrh6552/Hexapod/IsaacLab/Position Files/` and must be updated if the machine changes.

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

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_rewards.py` — hexapod-specific custom rewards
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/` — locomotion-specific terms
- `source/isaaclab/isaaclab/envs/mdp/` — general-purpose MDP terms (shared across tasks)

When a config references e.g. `mdp.feet_air_time`, look in the locomotion mdp directory first, then the core mdp directory.

## Code Style

- Line length: 120 characters (ruff enforced)
- Python 3.11 type annotations (pyright strict mode)
- Pre-commit hooks: ruff lint + ruff-format + trailing whitespace
- No mock databases in tests; integration tests use live Isaac Sim physics
