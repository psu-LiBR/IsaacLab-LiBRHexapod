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
| 0 | swing  | `1.180398 rad` |

The two body (spine) joints follow a prescribed 1.00 s-period wave and are **not**
controlled by the policy. There are **two distinct spine waves**, deliberately not unified
(canonical coefficients in `hexapod_binary_env_cfg.py`, "CANONICAL SPINE-WAVE DEFINITION"):

- **Wave 1 — the RL-env wave** (`SpineSineAction` during `Isaac-Goal-Flat-Hexapod-Binary-v0`
  training and `-Play-v0`): a fixed **analytic traveling body wave**, *not* fitted to any
  CSV. `FrontLink_Joint` is a pure sine `−0.9162978573·sin(2π·t/1.0)`; `BackLink_Joint` is
  the same sine shifted +90° (`−0.9162978573·cos(...)`). Magnitude 0.9162978573 rad
  (exact open-loop gait-generator value: `deg2rad(70)·12/16 = deg2rad(52.5)`) carried with
  the HexapI global spine-joint-sign flip, so the `sin_coef`/`cos_coef` hold `−A_SPINE`
  (corrected 2026-09-10 — a positive coefficient walked every re-evaluated policy backward
  from the goal). Shared offset 0.0, only the phase differs by a quarter cycle — a wave
  travelling down the body. The BackLink-leads-FrontLink phase direction is still unverified.
- **Wave 2 — the tripod-baseline wave**: an **anti-phase** body wave, regenerated
  analytically from the user's MATLAB gait generator. `FrontLink_Joint` and `BackLink_Joint`
  are π out of phase, so `BackLink = −FrontLink` — a traveling / S-bend wave, not a rigid
  side-to-side wag. Same open-loop generator as Wave 1, so **same amplitude**
  (0.9162978573 rad), with a −π/4 phase origin; per joint,
  `FrontLink_Joint(t) = −A_SPINE·sin(2π·t − π/4)` with `BackLink_Joint(t)` its negation
  (`w = 2π/1.0 s`, single harmonic). The −π/4 origin matches the leg-contact schedule in
  `tripod_bit_demos.npz`, so spine and legs stay aligned at reset. The two spine columns in
  the committed `tripod_extendedquad_sim.csv` are byte-identical, but that was the bug: a
  MATLAB→Sim "body correction" negated the FrontLink column and collapsed MATLAB's
  `yy = −xx` anti-phase pair into an in-phase one. Wave 2 now bypasses the CSV entirely.
  Used **only** by `eval_protocol.py`'s `BASE_tripod_csv_bits` anchor and
  `play_discrete_closeup.py`'s `--gait_npz tripod` replay, swapped into the live
  `SpineSineAction` via `set_waveform` (see §4).

The leg *levels* (`STANCE_POS` / `LIFT_POS`) are identical in `tripod_B11BL0_sim.csv` and
`tripod_extendedquad_sim.csv`; the open-loop **baseline** replayed by `eval_protocol.py` /
`--gait_npz tripod` takes its leg contact timing from `tripod_extendedquad_sim.csv` (see §4).

Everything else — observations, reward terms, network trunk, physics — is inherited
unchanged from `HexapodGoalEnvCfg`. The action interface is the only structural change.

For reference: the tripod gait occupies only two of the 64 actions, `25` and `38`
(complementary, `25 + 38 = 63`), alternating.

## 2. Files

**Maintained comparison set** (the algorithms this experiment reports on):

| File | Purpose |
|---|---|
| `hexapod_binary_env_cfg.py` / `hexapod_binary_actions.py` | the binary-contact environment + `SpineSineAction` scripted-spine term (under `isaaclab_tasks/contrib/velocity/config/hexapod/`) |
| `binary_action_mask.py` | bit <-> action helpers + the legal-action mask for masked PPO (torch-only, unit-tested) |
| `binary_common.py` | shared harness: CLI + W&B bridge, env build, MLP, `run_meta.json` sidecar, masked-categorical mixin, n-step returns |
| `discrete_action_wrapper.py` | decodes `Discrete(64)` into `[N, 6]` +/-1 contact bits; the joint targets are applied env-side |
| `train_discrete.py` | DQN and Double DQN (`--algo dqn\|ddqn`), via skrl |
| `train_discrete_ppo.py` | categorical PPO; `--mask` turns on the masked-categorical policy |
| `train_sac_d.py` | discrete SAC (categorical actor + twin critics), single-file torch |
| `train_sac_continuous.py` | continuous SAC baseline on the 8-DOF goal env (arXiv:2605.24975); comparison point *outside* the contact-bit action space |
| `run_discrete_pipeline.py` | one-shot train -> eval -> rank -> export -> video pipeline over all five algorithms (see §7); runs the other scripts as subprocesses |
| `eval_protocol.py` | the shared evaluation protocol — all reported numbers come from this |
| `export_binary_onnx.py` | export a trained checkpoint to ONNX for sim-to-real (see §5) |
| `play_discrete.py` | runs a checkpoint's greedy (argmax) policy on the Play env headless and records a root-tracking MP4 |
| `play_discrete_closeup.py` | renders a checkpoint **or an open-loop gait** (`--gait_npz tripod` / `--gait_csv <csv>`) to a follow-cam video |
| `extract_bit_demos.py` | builds `tripod_bit_demos.npz` — **leg** contact bits only — from a tripod CSV (`CSV_DEFAULT` = `tripod_extendedquad_sim.csv`); the spine wave is configured separately in `hexapod_binary_env_cfg.py` and is not affected by re-running this |
| `tripod_bit_demos.npz` | tripod **leg** contact sequence (from `tripod_extendedquad_sim.csv`), used as the baseline; carries no spine data |

**Archived** (`archive/`, unmaintained — see `archive/README.md`): C51, QR-DQN, PQN, V-MPO,
DQfD, and `fold_qrdqn_meanQ.py`. Historical numbers for these are in `RESULTS_binary_rl.md`.

**Actuator-model diagnostics** (open-loop tripod replay; not part of the RL comparison):

| File | Purpose |
|---|---|
| `sweep_leg_actuator_gains.py` | replays the open-loop tripod gait while sweeping the six leg joints' PD stiffness × damping; writes a per-leg tracking-error / peak-torque table + CSV |

(A companion `compare_actuator_models.py` — tripod replay under ImplicitActuator / DCMotor /
DelayedPD, one Isaac Sim process each — is kept locally as an ad-hoc analysis script and is
not tracked in the repo; its 2026-09 output is cited in `hexapod_binary_env_cfg.py`'s
`_rebalance_binary_rewards` docstring.)

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

**Weights & Biases:** add `--wandb` to any training script to log it to W&B (this is the
binary-RL equivalent of the rsl_rl `--logger wandb --log_project_name` flags).
`--wandb_project` defaults to `discrete-RL`; `--wandb_group` / `--wandb_name` default to
the algorithm name and `--experiment_name`. No-ops with a warning if `wandb` is not
installed.

skrl 2.1 ships its own `SummaryWriter` (not `torch.utils.tensorboard` / `tensorboardX`),
so `wandb`'s `sync_tensorboard` does **not** see it — a plain `wandb.init` gives you a run
with config but no metric history. `binary_common.maybe_init_wandb` therefore wraps
skrl's `add_scalar` to forward every tracked scalar (losses, rewards, episode stats) to
`wandb.log`. SAC-D is single-file torch and logs its metrics explicitly. If an older run
has only config and no charts, that is the pre-bridge behaviour — re-run, or
`wandb sync runs_binary/<run>/` to import its tfevents.

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
| primary metric | forward displacement, reported as BL/cycle | body length 0.315 m over 6 cycles, so BL/cycle = metres / 1.89 |

> **Body-length rebasing (2026-09).** The body-length constant changed from 0.265 m to
> **0.315 m** (updated HexapI USD front-to-rear leg spacing; ~0.311 m measured from the
> leg-link bounding boxes). Every BL/cycle number recorded before this was on the
> 0.265 m / `÷1.59` basis and reads ~1.19× too high; do not silently rescale. The four
> locomotion **anchors have been re-measured** on the current code (see the anchor table
> below and `RESULTS_binary_rl.md` → "Re-measured anchors (2026-09-10)"); the
> friction-sweep / equal-budget tables in `RESULTS_binary_rl.md` **still must be
> re-measured**.
>
> An end-to-end audit of the BL/cycle **metric** (prompted by a 0.95 BL/cycle gait that
> looked ~0.5 on video, then by a tripod anchor that dropped to ~0.05) found **no
> arithmetic bug**: `step_dt` is genuinely 0.02 s (`sim.dt` 0.005 × `decimation` 4), the
> window is exactly 6.0 cycles, `BODY_LENGTH_M` is 0.315, and `x_displacement_m` is the
> per-env mean *net* forward displacement (cross-checked against the `progress` reward
> term). Two real (non-metric) bugs **were** found in the tripod baseline and fixed
> 2026-09-10: (1) `play_discrete_closeup.py --gait_npz tripod` was playing the RL
> traveling wave (Wave 1) on the tripod leg schedule — it now swaps in Wave 2 like
> `eval_protocol.py`; (2) Wave 2 is now an **anti-phase** analytic wave
> (`BackLink = −FrontLink`) at the shared `A_SPINE` amplitude, regenerated from the
> MATLAB spec rather than the byte-identical (bugged) CSV spine columns. Fixing the
> in-phase bug is what recovered the tripod anchor: on the anti-phase Wave 2 it now walks
> **+0.834 m / +0.441 BL/cycle** forward (Candidate-2 per-joint sign, verified in sim
> 2026-09-10), essentially matching the ~0.40 BL/cycle real-hardware / pre-DCMotor figure
> — the earlier ~0.05 was the in-phase wave barely bending the body. The opposite per-joint
> sign assignment (Candidate 1) walks the same leg schedule backward (−0.837 m). A `--tripod_spine_phase_deg`
> phase sweep on the anti-phase wave has not been re-measured. The
> remaining old-vs-video gap for *learned* policies
> is (i) the 0.265→0.315 rebasing (÷1.189), (ii) `eval_protocol.py` measuring a no-reset
> continuous window, and (iii) possible video slow-motion (RecordVideo tags 50 fps).

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
`--tripod_npz` needed). That file is the **leg** contact sequence from
`tripod_extendedquad_sim.csv` (swing/stance switch at steps 13/37); regenerate it with
`extract_bit_demos.py --out_dir scripts/reinforcement_learning/binary_rl` after changing
`CSV_DEFAULT`. The spine wave is **not** driven by this file. Learned policies run on the
analytic Wave 1 (see §1) baked into `hexapod_binary_env_cfg.py`; only the tripod baseline
swaps in Wave 2 (the anti-phase analytic body wave, `BackLink = −FrontLink`,
`FrontLink_Joint(t) = −A_SPINE·sin(2π·t − π/4)`, regenerated from the MATLAB spec) via
`SpineSineAction.set_waveform`. Every rollout — baselines included — reports a
`reward_terms` breakdown, so the tripod gait and the learned policies are scored against
the identical reward terms. A masked-PPO checkpoint is auto-detected from its
`run_meta.json` and its greedy argmax is restricted to the recorded 24-action legal set.
`--warmup >= 1` is required (it flushes the IMU / contact-sensor buffers, which
`env.reset()` leaves stale); `--legacy_reset` reproduces pre-2026-09 result tables.

Four anchors are re-measured on every run. Current values, **re-measured 2026-09-10** on
the analytic Wave 1 (`A_SPINE = 0.9162978573`), the 0.315 m body-length basis (`÷1.89`),
the corrected reset, and the `tripod_extendedquad_sim.csv` leg timing + anti-phase analytic
Wave 2 spine (`BackLink = −FrontLink`, `FrontLink_Joint(t) = −A_SPINE·sin(2π·t − π/4)`,
friction pinned to 0.215, midpoint of the calibrated 0.18–0.25 band):

| Anchor | x_disp (m) | BL/cycle (÷1.89) | fall_rate | straightness |
| --- | ---: | ---: | ---: | ---: |
| all legs in stance | +0.0914 | +0.0484 | 0.0 | 0.123 |
| all legs in swing | −0.0298 | −0.0158 | 0.0 | 0.059 |
| uniform random action | −0.0453 | −0.0239 | 0.0 | 0.179 |
| **tripod (baseline)** | **+0.834** | **+0.441** | **0.0** | **0.984** |

The tripod anchor's `fall_rate == 0` and straightness 0.98 at **+0.834 m** forward confirm
the Candidate-2 per-joint sign (`FrontLink = −A_SPINE·sin(2π·t − π/4)`, `BackLink` its
negation); the Candidate-1 assignment walks it backward (−0.837 m), verified in sim
2026-09-10 — see the CANONICAL block in `hexapod_binary_env_cfg.py`. The
`--tripod_spine_phase_deg` sweep has not been re-run on the anti-phase wave; −π/4 (the
phase the CSV leg schedule was designed with) is the default. A
`--spine_gain 0` run shows the tripod **leg** contact
pattern alone nets ≈ 0 (−0.008 m) — essentially all of the tripod baseline's forward
travel is the scripted body wave, not the contact bits. See
`RESULTS_binary_rl.md` → "Re-measured anchors (2026-09-10)" for the full tables,
reward-term breakdown, and the (backward-walking) result of re-evaluating the pre-split
DDQN/SAC-D checkpoints.

> **Historical (superseded).** The pre-2026-09 anchors were 0.0734 / 0.0524 / 0.0589 /
> **0.4014** BL/cycle, measured with `--legacy_reset`, the old `tripod_B11BL0_sim.csv`
> baseline, the old fitted spine wave, and the 0.265 m basis (`÷1.59`). The friction-sweep
> and equal-budget tables in `RESULTS_binary_rl.md` are still on that old basis — do not
> rescale, re-run. Now that the spine wave is a genuine anti-phase pair again, the current
> tripod anchor (+0.441 BL/cycle) essentially reproduces that **0.4014** legacy figure.

A checkpoint is counted as beating the baseline only if all three hold: fall rate = 0,
survival = 6.00 s, and displacement > the (re-measured) tripod BL/cycle. Displacement
alone is not enough — a policy that falls at 0.16 s and slides can still register a high
number.

### Visual check of the tripod baseline

`eval_protocol.py` only scores the tripod baseline headlessly. To *watch* it — and confirm
the binary env (leg bit → angle decode, the scripted spine wave, the spine DOF-swap fix)
is wired correctly — render it through `play_discrete_closeup.py`:

```bat
:: the committed tripod_bit_demos.npz (exactly what BASE_tripod_csv_bits replays)
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
  --gait_npz tripod --num_envs 4 --steps 600 --video_length 600

:: or straight from a gait CSV (leg columns thresholded to contact bits on the fly)
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/play_discrete_closeup.py ^
  --gait_csv "hexapod-assets/Sim Gaits/tripod_extendedquad_sim.csv" --num_envs 4
```

MP4 lands in `scripts/reinforcement_learning/binary_rl/videos_gait/` (override with
`--out_dir`). The script prints the gait period, the distinct action ids (tripod → `25`
and `38`), and the per-leg stance fraction so the replay is quantitatively checkable too.
Running `--gait_csv tripod_extendedquad_sim.csv` reproduces the `--gait_npz tripod` bit
table exactly (same two-level threshold rule as `extract_bit_demos.py`). Both tripod
replays now **swap in Wave 2** (the `−A_SPINE·sin(2π·t − π/4)` anti-phase body wave,
`BackLink = −FrontLink`), so the video matches what `eval_protocol.py`'s
`BASE_tripod_csv_bits` anchor scores — walking **forward** at ~`+0.441`
BL/cycle. (Before 2026-09-10 the closeup played the RL traveling wave
Wave 1 on the tripod leg schedule and walked backward; that was a script bug, not a gait
property.) A checkpoint replay (`--checkpoint`) still uses Wave 1, as trained.

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
recreates the analytic Wave 1 spine wave host-side. It uses the HexapI positive-leg joint
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

## 7. One-shot pipeline (`run_discrete_pipeline.py`)

Train every maintained discrete-optimisation policy in sequence, evaluate them, pick the
best, and export the whole candidate set to ONNX — in a single launch you can walk away
from. One training run at a time (single-GPU). Run from the repo root:

```bat
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/run_discrete_pipeline.py --iterations 2000
```

What it does, in order:

1. **Train** `dqn`, `ddqn`, `ppo`, `ppo_masked`, `sac_d` one after another (each starts
   only after the previous exits), each to `--iterations` trainer steps, checkpointing
   every `--checkpoint-interval` steps (250 → 1500 / 1750 / 2000 all land on disk).
   Each run is its own process and its own W&B run in project `discrete-RL` (group = algo
   name, run name `<tag>-<algo>`).
2. **Evaluate** the 1500 / 1750 / 2000 checkpoint of every algorithm through
   `eval_protocol.py` (64 envs, 300 steps, seed 7, all four baselines). One checkpoint
   per process by default — each is a "first rollout of its own process", the cleanest
   comparable number; `--eval-fast` puts everything in one process instead.
3. **Rank** by `x_disp_BL_per_cycle`. A checkpoint only counts as "beats tripod" if
   `fall_rate == 0`, it survives the full 6.00 s window, and its BL/cycle exceeds the
   re-measured tripod anchor. Winner = best of those, else best raw BL/cycle.
   *(Adjacent checkpoints swing 2–5× in BL/cycle — see `RESULTS_binary_rl.md` — so the
   bundle keeps all three per algo for you to try each on hardware, not just the winner.)*
4. **Export** every selected checkpoint to ONNX with `export_binary_onnx.py`, into
   `runs_binary/<tag>/onnx_bundle/`, the winner also copied as `BEST_<algo>_<iter>.onnx`.
5. **Record** an MP4 of each algorithm's best checkpoint with `play_discrete_closeup.py`
   (greedy argmax on the Play env, follow camera) → `onnx_bundle/<algo>_<iter>.mp4`, with
   the overall winner additionally copied to `BEST_<algo>_<iter>.mp4`.
   `--no-video` skips it; `--video-all` renders every ranked checkpoint;
   `--video-length` / `--video-num-envs` tune it.
6. **Log** the ranked table + per-policy reward-term breakdown + the rendered videos to a
   `discrete-RL` W&B run named `<tag>-eval`, with `eval_results.json` attached as an
   artifact.

### Output layout

```text
runs_binary/<tag>/                (<tag> defaults to pipeline_<timestamp>)
  pipeline_config.json
  dqn/  ddqn/  ppo/  ppo_masked/  sac_d/
    checkpoints/agent_*.pt
    run_meta.json
    train.log
  eval/                           per-checkpoint eval JSON + logs
  eval_results.json               merged
  onnx_bundle/
    <algo>_<iter>.onnx            one per selected checkpoint
    BEST_<algo>_<iter>.onnx       copy of the winner
    <algo>_<iter>.mp4             sim replay of each algorithm's best checkpoint (if --video)
    BEST_<algo>_<iter>.mp4        copy of the winner's replay
    videos/                       raw renders + per-render logs
    <algo>_run_meta.json
    manifest.json                 ranking + metrics + reward terms + best
    ranking.csv                   the same table, flat
    README.txt                    per-file on-robot test commands
    validate_all.sh               offline-validate every .onnx against a trace
```

`runs_binary*/` and `eval_results*.json` are gitignored — copy `onnx_bundle/` to the
robot host to test.

### Changing the iteration target

`--iterations N` is the skrl trainer-step count (one vectorised env step;
env-steps = `N * --num-envs`), applied **identically to every algorithm** — so the
comparison is at an equal env-step budget and PPO / masked PPO run fewer gradient updates
than DQN / SAC-D (rollout length 96). The equal-budget runs in `RESULTS_binary_rl.md`
used `N = 100000`; **2000 is a fast shakedown and will under-train PPO** — raise
`--iterations` for a publication-grade comparison. Checkpoints are taken every
`--checkpoint-interval` and the last three are evaluated/exported; override with
`--save-iters 1500,1750,2000`.

### Quick shakedown of every optimizer (`--smoke`)

```bat
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/run_discrete_pipeline.py --smoke
```

Runs all five algorithms end to end at a tiny scale (200 iters, 256 envs, one eval
process for all checkpoints, no video, no W&B) to confirm each one **trains, evals, and
exports without erroring** — ~6 Isaac Sim boots, roughly 15-25 min. It is *not* a real
comparison (the policies are untrained). Any tuned flag you pass explicitly still wins
(e.g. `--smoke --algos ppo ppo_masked --num-envs 512`). Output lands in
`runs_binary/smoke_<timestamp>/`.

### Other flags

`--algos dqn sac_d` (subset), `--num-envs`, `--seed`, `--goal-distance`, `--eval-fast`
(one eval process instead of one-per-checkpoint), `--force` (retrain existing),
`--skip-train` / `--skip-eval` (reuses an existing `eval_results.json`) / `--skip-export`,
`--no-video`, `--no-wandb`, `--no-keep-going` (stop on the first training failure),
`--export-no-check`. Re-running the same `--tag` resumes: any algorithm whose final
checkpoint exists is skipped.

### W&B

`wandb` must be installed in the Isaac Sim env and authenticated once (it already is on
this machine). Every W&B path no-ops with a warning if the package is missing. Per-run
training metrics (loss / reward / episode stats) show up because `maybe_init_wandb`
bridges skrl's custom TensorBoard writer to `wandb.log` — see the W&B note in §3.

### Testing a candidate on the robot

`onnx_bundle/README.txt` has the exact commands. Summary: copy the bundle to the
deployment host, `cp config/deployment.binary.example.yaml config/deployment.yaml` and
calibrate it, then per `.onnx`: `tools/validate_onnx.py --mode direct --profile binary`
→ `run_policy.py --profile binary --dry-run --fake-imu` → `--no-torque` → propped
`--action-scale-mult 0.2` → full. See `scripts/sim2real_transfer/README.md` and CLAUDE.md
"Sim-to-Real Deployment".
