# Archived binary-contact RL experiments

These scripts were part of the collaborator's first-wave algorithm sweep on the
`Isaac-Goal-Flat-Hexapod-Binary-v0` env. They are **kept for reference only** and are
**not maintained or correctness-reviewed**. The maintained comparison set lives one
directory up: DQN / Double-DQN (`train_discrete.py`), PPO and masked-categorical PPO
(`train_discrete_ppo.py` / the rsl_rl path), discrete SAC (`train_sac_d.py`), and the
continuous SAC baseline.

| Script | Algorithm | Notes |
|---|---|---|
| `train_discrete_c51.py` | C51 (categorical distributional DQN) | fixed support `[-10, 10]`, 51 atoms |
| `train_discrete_qrdqn.py` | QR-DQN (quantile-regression DQN) | 51 quantiles; checkpoints must be folded first (see below) |
| `train_discrete_pqn.py` | PQN (parallelised Q-network, no replay / no target net) | TensorBoard logging only |
| `train_vmpo.py` | V-MPO (on-policy MPO) | hand-rolled, single file |
| `train_dqfd.py` | DQfD (deep Q from demonstrations) | needs `--demo_npz`; the only demo available is the tripod gait, which is also the comparison baseline, so its result is not an independent "can RL beat the gait" answer |
| `fold_qrdqn_meanQ.py` | utility | folds a QR-DQN quantile head to a mean-Q head so `../eval_protocol.py` can load it without changes |

Historical results for all of these are in `../RESULTS_binary_rl.md` (equal-budget
ranking table). Every one underperformed the tripod baseline in that sweep. Note that
those numbers predate the goal-task reward scale-down on this branch and were measured
with the pre-fix `eval_protocol.py` reset behaviour.

## Running an archived script

Each `train_*.py` here prepends the parent `binary_rl/` directory to `sys.path` so the
shared `discrete_action_wrapper` module still resolves. Run them by full path, e.g.:

```bat
isaaclab.bat -p scripts/reinforcement_learning/binary_rl/archive/train_discrete_c51.py --num_envs 512 --timesteps 1000
```

`fold_qrdqn_meanQ.py` is plain Python: `python fold_qrdqn_meanQ.py <ckpt_dir> <out_dir>`.
