# Binary-Contact RL for the Hexapod

Discrete-action RL on a binary contact interface, set up so that learned policies and
physics-informed gaits can be compared under identical conditions.

## 1. Action space

`Isaac-Goal-Flat-Hexapod-Binary-v0` — `Discrete(64)`.

Each of the six legs is either in stance or in swing, so an action is one integer in
`[0, 63]` whose bits give the contact pattern. The bit is mapped to a fixed joint angle:

| bit | meaning | leg joint target |
|---|---|---|
| 1 | stance | `0.460194 rad` |
| 0 | swing  | `1.180400 rad` |

The two body (spine) joints follow a prescribed sine/cosine wave with a 1.00 s period and
are **not** controlled by the policy. Both the leg angles and the spine wave are taken from
`tripod_B11BL0_sim.csv`, so the reference gait is representable inside this action space.

Everything else — observations, reward terms, network trunk, physics — is inherited
unchanged from `HexapodGoalEnvCfg`. The action interface is the only structural change.

For reference: the tripod gait occupies only two of the 64 actions, `25` and `38`
(complementary, `25 + 38 = 63`), alternating.

## 2. Files

**Maintained comparison set** (the algorithms this experiment reports on):

| File | Purpose |
|---|---|
| `hexapod_binary_env_cfg.py` | the binary-contact environment (under `isaaclab_tasks/contrib/velocity/config/hexapod/`) |
| `binary_action_mask.py` | bit <-> action helpers + the legal-action mask for masked PPO (torch-only, unit-tested) |
| `binary_common.py` | shared harness: CLI, env build, MLP, checkpoint sidecar, masked-categorical mixin |
| `discrete_action_wrapper.py` | decodes `Discrete(64)` into `[N, 6]` +/-1 contact bits; the joint targets are applied env-side |
| `train_discrete.py` | DQN and Double DQN (`--algo dqn\|ddqn`), via skrl |
| `train_discrete_ppo.py` | categorical PPO; `--mask` turns on the masked-categorical policy |
| `train_sac_d.py` | discrete SAC (categorical actor + twin critics), single-file torch |
| `train_sac_continuous.py` | continuous SAC baseline on the 8-DOF goal env (arXiv:2605.24975); comparison point *outside* the contact-bit action space |
| `binary_common.py` | shared harness: CLI, env build, MLP, checkpoint sidecar, masked mixin, n-step returns |
| `eval_protocol.py` | the shared evaluation protocol — all reported numbers come from this |
| `export_binary_onnx.py` | export a trained checkpoint to ONNX for sim-to-real (see §5) |
| `play_discrete.py` | replays a checkpoint in the viewer |
| `play_discrete_closeup.py` | renders a checkpoint to video |
| `extract_bit_demos.py` | builds `tripod_bit_demos.npz` from the tripod contact CSV |
| `tripod_bit_demos.npz` | tripod contact sequence, used as the baseline |

**Archived** (`archive/`, unmaintained — see `archive/README.md`): C51, QR-DQN, PQN, V-MPO,
DQfD, and `fold_qrdqn_meanQ.py`. Historical numbers for these are in `RESULTS_binary_rl.md`.

## 3. Training

Run from the repo root. On Windows use `isaaclab.bat -p` (the scripts also work under a
plain `python` if the `.venv` is active). `--timesteps` counts trainer iterations; each
advances all `--num_envs` envs one step, so `100000 x 4096 = 4.096e8` env-steps.

```bat
:: DQN  (Double DQN: --algo ddqn)
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete.py ^
  --algo dqn --num_envs 4096 --timesteps 100000 --seed 42 --experiment_name dqn_s42

:: categorical PPO
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete_ppo.py ^
  --num_envs 4096 --timesteps 100000 --seed 42 --experiment_name ppo_s42

:: masked categorical PPO (legal set: >= 4 stance legs, plus the two tripods 25/38 = 24 of 64)
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete_ppo.py ^
  --num_envs 4096 --timesteps 100000 --seed 42 --mask --experiment_name ppo_masked_s42

:: discrete SAC
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_sac_d.py ^
  --num_envs 4096 --timesteps 100000 --seed 42 --experiment_name sacd_s42

:: continuous SAC baseline (8-DOF goal env, NOT the contact-bit space) — arXiv:2605.24975
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_sac_continuous.py ^
  --num_envs 4096 --timesteps 60000 --seed 42 --experiment_name sacc_s42
```

Runs land in `runs_binary/<experiment_name>/` (gitignored) with a `run_meta.json` sidecar
(algorithm, obs/action dims, action mask) that the evaluator and ONNX exporter read.
Common flags (all scripts): `--num_envs`, `--timesteps`, `--seed`, `--experiment_name`,
`--directory`, `--checkpoint_interval`, `--resume <ckpt>`. DQN adds `--replay_size`
(total replay transitions, default 1e6); masked PPO adds `--mask_min_stance` /
`--mask_whitelist`; both SAC scripts add `--n_step` / `--replay_size` / `--batch_size`.

`train_sac_continuous.py` defaults to `--task Isaac-Goal-Flat-Hexapod-v0` (it is not a
binary-action script); the other five run on `Isaac-Goal-Flat-Hexapod-Binary-v0`.

Quick wiring check (a couple of minutes, no GPU-heavy load):

```bat
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/train_discrete.py ^
  --algo dqn --num_envs 512 --timesteps 1500 --replay_size 100000 --checkpoint_interval 500
```

## 4. Evaluation

All reported numbers come from `eval_protocol.py`, run identically for every method:

| Setting | Value | Why |
|---|---|---|
| parallel envs | 64 | results are the mean over 64 robots |
| horizon | 300 steps x 0.02 s = 6.00 s | the spine period is 1.00 s, so this is exactly 6 gait cycles |
| seed | 7 | fixed |
| terminations | all disabled | the robot never resets, so displacement is one continuous trajectory and cannot be inflated by reset teleports |
| domain randomisation | pinned term by term (14 items audited, printed into the output JSON) | repeated runs are bit-identical |
| first step | discarded (`--warmup 1`) | removes the start-up transient |
| primary metric | forward displacement, reported as BL/cycle | body length 0.265 m over 6 cycles, so BL/cycle = metres / 1.59 |

```bat
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/eval_protocol.py ^
  --num_envs 64 --steps 300 --seed 7 --warmup 1 ^
  --policy net "dqn_s42@100k"        runs_binary/dqn_s42/checkpoints/agent_100000.pt ^
  --policy net "ddqn_s42@100k"       runs_binary/ddqn_s42/checkpoints/agent_100000.pt ^
  --policy net "ppo_s42@100k"        runs_binary/ppo_s42/checkpoints/agent_100000.pt ^
  --policy net "ppo_masked_s42@100k" runs_binary/ppo_masked_s42/checkpoints/agent_100000.pt ^
  --policy net "sacd_s42@final"      runs_binary/sacd_s42/checkpoints/agent_final.pt ^
  --out eval_results.json
```

The tripod-CSV baseline uses `tripod_bit_demos.npz` next to the script by default (no
`--tripod_npz` needed). Every rollout — baselines included — now reports a `reward_terms`
breakdown, so the tripod gait and the learned policies are scored against the identical
reward terms. A masked-PPO checkpoint is auto-detected from its `run_meta.json` and its
greedy argmax is restricted to the recorded 24-action legal set. `--warmup >= 1` is
required (it flushes the IMU / contact-sensor buffers, which `env.reset()` leaves stale);
`--legacy_reset` reproduces pre-2026-09 result tables.

Four anchors are re-measured on every run and have been bit-identical across all
evaluations so far:

| Anchor | BL/cycle |
|---|---|
| all legs in stance | 0.0734 |
| all legs in swing | 0.0524 |
| uniform random action | 0.0589 |
| **tripod (baseline)** | **0.4014** |

> These values were measured with the **pre-2026-09** `eval_protocol.py` reset behaviour
> (`--legacy_reset`). The corrected default recomputes the goal command and flushes the
> sensor buffers after `env.reset()`, which shifts the anchors — re-run the full sweep and
> record the new anchors before quoting them in the paper.

A checkpoint is counted as beating the baseline only if all three hold: fall rate = 0,
survival = 6.00 s, and displacement > the (re-measured) tripod BL/cycle. Displacement
alone is not enough — a policy that falls at 0.16 s and slides can still register a high
number.

## 5. Sim-to-real export

`export_binary_onnx.py` turns a trained checkpoint into a single ONNX graph
(`obs [1, 32] float32 -> action [1, 6] float32` in `{-1, +1}`) that the
`scripts/sim2real_transfer` `binary` profile consumes directly. The graph folds in the
running obs-normalisation (PPO only), the greedy argmax over the 64 gait patterns, the
masked-run legal set, and the bit decode.

```bat
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/export_binary_onnx.py ^
  --checkpoint runs_binary/ppo_masked_s42/checkpoints/agent_100000.pt ^
  --out policies/ppo_masked_s42.onnx
```

On the deployment host (`scripts/sim2real_transfer/`, plain Python, no Isaac Sim):

```bash
# copy config/deployment.binary.example.yaml -> config/deployment.yaml and calibrate it
python tools/validate_onnx.py --mode direct --profile binary \
  --policy policies/ppo_masked_s42.onnx --trace <sim_trace.csv>
python run_policy.py --policy policies/ppo_masked_s42.onnx --profile binary \
  --config config/deployment.yaml --dry-run --fake-imu --duration 5
```

The `binary` profile drives the six leg joints from the policy bits (stance/lift snap) and
recreates the scripted spine sinusoid host-side. It uses the HexapI positive-leg joint
convention (stance 0.46 / lift 1.18 rad), so it needs its own deployment config —
`deployment.binary.example.yaml`, not `deployment.example.yaml`. See
`scripts/sim2real_transfer/README.md` and CLAUDE.md's "Sim-to-Real Deployment" section.

## 6. Notes

- `archive/train_dqfd.py` needs `--demo_npz`; the only demonstration available here is
  tripod, which is also the comparison baseline, so its result is not an independent
  answer to "can RL beat the physics-informed gait on its own".
- QR-DQN checkpoints must be passed through `archive/fold_qrdqn_meanQ.py` before the
  shared evaluator can load them.
- Checkpoints, evaluation JSON and videos are not committed here.
