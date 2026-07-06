# Hexapod Goal Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scratch training reliably progress from one-meter goals to a five-meter speed-prioritized goal without rewarding deliberate falls.

**Architecture:** Keep goal reward functions pure except for reset-safe progress history stored per environment. Add one curriculum function that reads completed termination terms during reset, accumulates fixed evaluation windows on the environment, and mutates the pose-command range before command resampling. Configure phase-one reward and PPO settings in the existing goal-specific configuration classes.

**Tech Stack:** Python, PyTorch, Isaac Lab manager-based RL, RSL-RL PPO, pytest.

## Global Constraints

- Train from scratch with stages 1.0, 2.0, 3.5, and 5.0 m.
- Advance on at least 70% success in a non-overlapping window containing at least 4096 completed episodes; never regress.
- Use a 45 s episode, 0.3 m reach radius, +10 reward/m progress, -0.2 reward/s, +50 terminal success, and -25 terminal fall.
- Keep joint-limit protection; disable phase-one gait, posture, energy, contact, and smoothness shaping.
- Use PPO gamma 0.999, 96 rollout steps, actor/critic normalization, actor corruption, and an uncorrupted critic.

---

### Task 1: Reward and curriculum behavior

**Files:**
- Create: `source/isaaclab_tasks/test/manager_based/locomotion/velocity/hexapod/test_hexapod_goal_mdp.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_goal_rewards.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_goal_curriculum.py`

**Interfaces:**
- `progress_to_goal(env, command_name) -> torch.Tensor`
- `termination_signal(env, termination_name) -> torch.Tensor`
- `goal_distance_curriculum(env, env_ids, command_name, distances, success_threshold, window_size) -> dict[str, float]`

- [ ] **Step 1: Write failing reward tests** using lightweight fake command and termination managers. Assert forward distance changes `[5.0, 5.0] -> [4.8, 5.2]` at `step_dt=0.02` return `[10.0, -10.0]`, first post-reset progress is zero, and success/fall terminal terms mirror their named termination tensors.
- [ ] **Step 2: Run the focused test** with `python -m pytest source/isaaclab_tasks/test/manager_based/locomotion/velocity/hexapod/test_hexapod_goal_mdp.py -v`; expect failures for missing `termination_signal` and curriculum module.
- [ ] **Step 3: Implement minimal reward behavior.** Preserve the current reset-safe progress calculation and add:

```python
def termination_signal(env: ManagerBasedRLEnv, termination_name: str) -> torch.Tensor:
    return env.termination_manager.get_term(termination_name).float()
```

- [ ] **Step 4: Write failing curriculum tests.** Verify startup resets with zero episode length are ignored; a 69% window stays at 1.0 m; a 70% window advances exactly one stage and clears counters; repeated passing windows cap at 5.0 m; and the command range becomes `(distance, distance)`.
- [ ] **Step 5: Run tests and confirm the expected curriculum failures.**
- [ ] **Step 6: Implement `goal_distance_curriculum`.** Store `_goal_curriculum_stage`, `_goal_curriculum_episodes`, `_goal_curriculum_successes`, and `_goal_curriculum_falls` on `env`; count only `episode_length_buf[env_ids] > 0`; evaluate and clear counters whenever the window reaches `window_size`; update `command_term.cfg.ranges.pos_x`; return logging keys `distance`, `success_rate`, `episodes`, `successes`, and `falls`.
- [ ] **Step 7: Run the focused tests and confirm all reward/curriculum tests pass.**

### Task 2: Environment and PPO configuration

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_goal_env_cfg.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_goal_obs_cfg.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/agents/rsl_rl_ppo_goal_cfg.py`
- Extend test: `source/isaaclab_tasks/test/manager_based/locomotion/velocity/hexapod/test_hexapod_goal_mdp.py`

**Interfaces:**
- Environment imports and registers `goal_distance_curriculum` through `CurriculumTermCfg`.
- Environment uses `termination_signal` for terminal success and fall reward terms.

- [ ] **Step 1: Add failing source/config assertions** for 45 s duration, initial 1 m command, curriculum parameters, effective terminal weights (`2500.0` success and `-1250.0` fall at 0.02 s), progress weight 10, time weight -0.2, disabled shaping terms, critic corruption false, PPO gamma 0.999, rollout 96, and both normalizers true.
- [ ] **Step 2: Run the focused test and verify it fails against the old configuration.**
- [ ] **Step 3: Update environment configuration.** Set constants and command interval to 45 s; initialize at 1 m; register curriculum stages and thresholds; set progress/time terms; replace the custom success-only reward with named termination signals for success and fall; and set phase-one shaping terms to `None` while retaining `dof_pos_limits`.
- [ ] **Step 4: Update observations and PPO.** Set `critic.enable_corruption = False`; set `num_steps_per_env = 96`, `gamma = 0.999`, and actor/critic observation normalization to true.
- [ ] **Step 5: Run focused tests and confirm they pass.**

### Task 3: Integration verification and handoff

**Files:**
- Modify if needed: files from Tasks 1–2 only.

**Interfaces:** None beyond the registered Isaac Lab task.

- [ ] **Step 1: Run syntax compilation** for all new and modified Python modules with `python -m compileall` on the hexapod config directory and focused test.
- [ ] **Step 2: Run focused pytest** and record its pass/fail count.
- [ ] **Step 3: Run `git diff --check`** and inspect `git diff` to ensure only the approved goal-task behavior changed.
- [ ] **Step 4: Provide cluster commands** for a fresh training run and TensorBoard monitoring, explicitly warning not to resume the old `model_400.pt` checkpoint.
