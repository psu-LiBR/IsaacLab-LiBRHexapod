# Reward Definition

[Overview](README.md) · [Candidate configuration](candidate_manifest.json)

## Provenance and implementation

"Original" identifies the existing squared action-change term in this experiment, not an assurance that every reward component is unchanged from upstream. The DQN, DDQN, PPO and Masked PPO candidates use the experiment's A2 reward configuration; SAC-D uses the J configuration. Each candidate's `config.json` is authoritative for all resolved reward weights, not only the switch terms.

| Component | Source and behavior |
|---|---|
| Upstream squared action difference | [`action_rate_l2`](../source/isaaclab/isaaclab/envs/mdp/rewards.py) sums squared changes over action dimensions. It has no episode-first-step guard. |
| A2 experiment original term | [`reward_metrics`](../scripts/reinforcement_learning/binary_rl/overnight_config.py) exposes the same squared sum as `switch`, with episode-first-step masking. This reset-boundary modification must not be described as unchanged native upstream behavior. |
| J original term | Native `action_rate_l2`, including its initial zero-action transition. |
| Added command-switch term | [`action_switch_count`](../scripts/reinforcement_learning/binary_rl/switch_command_reward.py) counts changed binary commands, masks the first episode step, and divides by control dt before reward-manager integration. |
| Reward integration | [`RewardManager.compute`](../source/isaaclab/isaaclab/managers/reward_manager.py) multiplies the raw term by its configured weight and dt exactly once. |
| Applied configuration | [`evaluate_switch_candidate.py`](../scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py) applies `evaluation.json`, verifies the resolved weights against the training configuration and saves the runtime audit. |

The separate physical-contact transition function in the same source module is **not** the added term used by these delivered candidates. Both switch terms here act on high-level binary commands, not measured foot-ground transitions.

## Definitions

Let $a_{t,i}\in\{-1,+1\}$ be the binary command for leg $i$, $\Delta t=0.02\,\mathrm{s}$, and $g_t=0$ at the first episode step and $1$ otherwise. Define

$$
F_t=\sum_{i=1}^{6}\mathbf{1}[a_{t,i}\ne a_{t-1,i}],\qquad
S_t=\sum_{i=1}^{6}(a_{t,i}-a_{t-1,i})^2.
$$

For A2, the original contribution is

$$R_t^{\mathrm{original}}=w\,\Delta t\,g_t S_t.$$

For J, the original contribution is $w\,\Delta t\,S_t$ without $g_t$. The added raw term is $g_t F_t/\Delta t$ with reward-manager weight $-\lambda$, giving

$$R_t^{\mathrm{added}}=-\lambda\,\Delta t\,\frac{g_t F_t}{\Delta t}=-\lambda g_t F_t.$$

Away from initialization, binary commands give $S_t=4F_t$. Consequently each changed leg incurs original cost $4|w|\Delta t$ and added cost $\lambda$. There is no division by the number of legs and no second dt integration. A J reset starts from zero actions, so the initial squared sum is six rather than four times a binary flip count; the added term masks that boundary.

## Sweep coefficients

The declared references are $w_0=-0.005$ and $\lambda_0=0.0004$. Multipliers refer to these values, **not** the earlier A2 default switch weight of -0.0005. The original-only condition sets $w=mw_0,\lambda=0$; the original-plus-added condition sets $w=mw_0,\lambda=m\lambda_0$.

| Multiplier | Original weight w | Added cost λ (when enabled) | Original cost / changed leg | Combined cost / changed leg |
|---:|---:|---:|---:|---:|
| 5× | -0.025 | 0.002 | 0.002 | 0.004 |
| 10× | -0.050 | 0.004 | 0.004 | 0.008 |
| 30× | -0.150 | 0.012 | 0.012 | 0.024 |
| 100× | -0.500 | 0.040 | 0.040 | 0.080 |

At the same multiplier, the added condition doubles the normal-step binary-command switching cost. These are two implementations of a penalty on the same command changes, not independent command and contact penalties. Larger coefficients do not directly slow the analytic 1-second spine wave or limit physical joint transition speed. Reduced command switching need not produce monotonically smaller forward displacement.

## Configuration and validation

Each candidate contains the original-term name, reference, multiplier and actual weight; the added-term enabled flag, cost and signed reward weight; all resolved reward weights; the control timestep; spine coefficients; observation/action interface; training seed and friction range. `evaluation.json` specifies the separate fixed evaluation conditions. No training or reward adjustment is needed to evaluate the supplied checkpoint.

The published per-candidate validation receipts include fresh packaged-checkpoint simulation results. Reward configuration and matching reproduction evidence do not establish real-hardware safety or identify the cause of a differing result on another installation.
