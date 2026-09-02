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

| File | Purpose |
|---|---|
| `hexapod_binary_env_cfg.py` | the binary-contact environment (under `isaaclab_tasks/contrib/velocity/config/hexapod/`) |
| `discrete_action_wrapper.py` | maps `Discrete(64)` to joint targets |
| `train_discrete.py` | DQN |
| `train_sac_d.py` | discrete SAC (categorical actor + twin critics) |
| `train_discrete_c51.py` | C51 |
| `train_discrete_qrdqn.py` | QR-DQN |
| `train_discrete_pqn.py` | PQN (no replay buffer, no target network) |
| `train_discrete_ppo.py` | PPO |
| `train_vmpo.py` | V-MPO |
| `train_dqfd.py` | DQfD (needs a demonstration `.npz`) |
| `eval_protocol.py` | the shared evaluation protocol — all reported numbers come from this |
| `fold_qrdqn_meanQ.py` | folds QR-DQN quantile heads to mean-Q so the shared evaluator can load them |
| `play_discrete_closeup.py` | renders a checkpoint to video |
| `tripod_bit_demos.npz` | tripod contact sequence, used as the baseline and as DQfD demonstrations |

## 3. Training

```bash
# DQN / C51 / QR-DQN / PQN, 4096 envs
python scripts/reinforcement_learning/train_discrete.py \
  --num_envs 4096 --timesteps 100000 --seed 42 \
  --learning_starts 20 --random_timesteps 10 --batch_size 256 --memory_slots 200 \
  --experiment_name dqn_s42 --directory runs_binary --checkpoint_interval 2000

# discrete SAC
python scripts/reinforcement_learning/train_sac_d.py \
  --num_envs 4096 --timesteps 100000 --seed 42 \
  --batch_size 256 --memory_slots 200 --out_dir runs_binary/sacd_s42 --checkpoint_interval 2000

# PPO
python scripts/reinforcement_learning/train_discrete_ppo.py \
  --num_envs 4096 --timesteps 100000 --seed 42 \
  --rollouts 16 --learning_epochs 2 --mini_batches 2 \
  --experiment_name ppo_s42 --directory runs_binary --checkpoint_interval 2000
```

`--timesteps` counts trainer iterations; each iteration advances all `--num_envs`
environments by one step. So `100000 x 4096 = 4.096e8` env-steps.

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

```bash
python scripts/reinforcement_learning/eval_protocol.py \
  --num_envs 64 --steps 300 --seed 7 --warmup 1 \
  --tripod_npz scripts/reinforcement_learning/tripod_bit_demos.npz \
  --policy net "dqn_s42@100000" runs_binary/dqn_s42/checkpoints/agent_100000.pt \
  --out eval_results.json
```

Four anchors are re-measured on every run and have been bit-identical across all
evaluations so far:

| Anchor | BL/cycle |
|---|---|
| all legs in stance | 0.0734 |
| all legs in swing | 0.0524 |
| uniform random action | 0.0589 |
| **tripod (baseline)** | **0.4014** |

A checkpoint is counted as beating the baseline only if all three hold: fall rate = 0,
survival = 6.00 s, and displacement > 0.4014 BL/cycle. Displacement alone is not enough —
a policy that falls at 0.16 s and slides can still register 0.5012 BL/cycle.

## 5. Notes

- `train_dqfd.py` needs `--demo_npz`; the only demonstration available here is tripod,
  which is also the comparison baseline, so its result is not an independent answer to
  "can RL beat the physics-informed gait on its own".
- QR-DQN checkpoints must be passed through `fold_qrdqn_meanQ.py` before the shared
  evaluator can load them.
- Checkpoints, evaluation JSON and videos are not committed here.
