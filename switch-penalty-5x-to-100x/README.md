# Switch penalty · 5×–100×

**25 simulation policies with paired checkpoints, ONNX exports, replays, reward configurations and measurements.**

[English comparison](https://psu-libr.github.io/IsaacLab-LiBRHexapod/switch-penalty-5x-to-100x/) · [Candidate index](candidate_manifest.json) · [Preliminary comparison](../preliminary-6algorithms-4rewards/)

## Two experiments

Both preserve each algorithm's selected reward configuration and reset handling.

- **Original only:** increase the existing switch penalty (DQN/PPO) or action_rate_l2 penalty (SAC-D).
- **Original + added switch:** apply the same original weight, plus a cost for each high-level binary leg-command flip. This does not count physical foot contacts.

| Multiplier | Original weight (reference −0.005) | Added cost: original only | Added cost: original + added (reference 0.0004) |
|---|---:|---:|---:|
| 5× | −0.025 | 0 | 0.002 |
| 10× | −0.05 | 0 | 0.004 |
| 30× | −0.15 | 0 | 0.012 |
| 100× | −0.5 | 0 | 0.04 |

These are this sweep's multiplier references, not every historical reward default. Each config.json includes the **full resolved reward-weight dictionary**, both coefficients and reset behavior. The added term uses command-flip count / dt, with the reward manager multiplying by dt once; the effective cost per flip is the number above.

## Directory structure

~~~text
switch-penalty-5x-to-100x/
├── README.md                      # Start here
├── index.html                     # English comparison, click-to-load videos
├── candidate_manifest.json        # Candidate paths, metrics and hashes
├── deployment.example.yaml        # This batch's observation joint order
└── candidates/
    ├── dqn-family/
    │   ├── dqn/                   # 5 candidates
    │   └── ddqn/                  # 5 candidates
    ├── ppo-family/
    │   ├── ppo/                   # 4 candidates
    │   └── masked-ppo/            # 4 candidates
    └── sac-family/
        ├── sac-discrete-one-step/ # 4 candidates
        └── sac-discrete-five-step/ # 3 low-displacement controls
~~~

Algorithms separate original-only/ and original-plus-added-switch/ where available. A folder such as original-10x__added-10x__forward-1.0116-body-lengths/ contains:

~~~text
policy.pt        # Original checkpoint
policy.onnx      # Deterministic binary inference
replay.mp4       # This checkpoint's simulation replay
config.json      # Exact reward weights and simulation configuration
run_meta.json    # Inference architecture and legal-action metadata
metrics.json     # Measurements, per-leg counts and command preview
validation.json  # Hashes and action-equivalence results
~~~

The two multipliers name separate penalty terms; added-off means no added term. Forward displacement distinguishes checkpoints without internal training IDs. Hashes bind the model, export and video.

## Selection and interpretation

**Hard filter: valid measurement, zero observed falls and torso contact, and at most four command changes for every individual leg in every complete cycle.** The six-leg sum is not used as the limit.

Within each algorithm: fewer maximum switches, then less lateral drift, then greater forward displacement. These are complementary tradeoffs, not a universal hardware ranking. Earlier candidates are retained, and additions broaden the comparison.

- **DQN/DDQN:** near-one-body-length displacement with 2–4 maximum switches; heading deviation remains.
- **PPO/Masked PPO:** useful forward motion but some brief dwell intervals despite passing switch counts.
- **SAC-D 1-step:** compare displacement against lateral drift; the lower-drift addition sacrifices displacement.
- **SAC-D 5-step:** three low-displacement controls, not preferred walking policies. Zero command changes does not imply physical stillness or stability.

Evaluation: 64 environments, seed 7, friction 0.21, body length 0.315 m, nominal cycle 1 second, control 50 Hz. A six-second replay spans six nominal cycles; switch statistics use five complete phase-aligned cycles per environment. Forward body lengths per cycle use the full six-second duration, **not division by five**. The video shows one environment; metrics aggregate the batch. No universal yaw, lateral-drift or dwell threshold has been agreed. Simulation screening is not hardware qualification.

## Load a policy

Use the existing binary deployment pipeline; no new robot controller is required.

~~~bash
git clone --branch robin/binary-rl-extended-quadruped https://github.com/psu-LiBR/IsaacLab-LiBRHexapod.git
cd IsaacLab-LiBRHexapod
git lfs pull
python -m pip install -r scripts/sim2real_transfer/requirements.txt
~~~

From the repository root, select a candidate:

~~~bash
CANDIDATE="switch-penalty-5x-to-100x/candidates/dqn-family/dqn/original-plus-added-switch/original-10x__added-10x__forward-1.0116-body-lengths"
python scripts/sim2real_transfer/run_policy.py \
  --policy "$CANDIDATE/policy.onnx" --profile binary \
  --config switch-penalty-5x-to-100x/deployment.example.yaml \
  --dry-run --fake-imu --duration 5
~~~

This uses simulated sensors and an in-memory motor bus; it does not move hardware. PowerShell users can substitute the quoted candidate path directly.

**Use this folder's deployment configuration.** All packaged training audits use the observation joint order FrontLink, BackLink, MiddleLeft, MiddleRight, FrontLeft, FrontRight, BackLeft, BackRight. The older generic binary example differs. Motor IDs, encoder offsets, sensor orientation and physical limits remain inherited hardware settings; confirm them against the current robot before following the [existing hardware bring-up guide](../scripts/sim2real_transfer/README.md).

ONNX interface: **float32[1,32] observations → float32[1,6] actions**. Observation order: gyro (3), projected gravity (3), goal command (4), relative joint position (8), joint velocity (8), previous binary actions (6). Output order: front right, front left, middle right, middle left, rear right, rear left. +1 is stance; −1 is swing. PPO normalization and Masked PPO's legal-action restriction are already folded into ONNX; do not apply them twice. Spine motion uses the existing host-side analytic wave, not the six policy outputs. The supplied configuration retains the audited 1-second wave and 50 Hz rate. Simulation uses a negative spine amplitude; the hardware configuration preserves the target branch's positive-amplitude correction for the physical motor mount. Do not copy the simulation sign directly onto the robot.

## Re-export and validation

All 25 original evaluation policies, export modules and ONNX files produced identical actions on **2,049 inputs per policy**: zero input, standard-normal inputs and broad clipping-stress inputs. These are synthetic software checks, not recorded hardware traces or a new simulation rollout. Exact hashes, versions and results are in each validation.json.

~~~bash
python -m pip install torch onnx onnxruntime
python scripts/reinforcement_learning/binary_rl/export_binary_onnx.py \
  --checkpoint "$CANDIDATE/policy.pt" \
  --run_meta "$CANDIDATE/run_meta.json" --out exported-policy.onnx
~~~

Embedded masks are preserved even when a checkpoint has been moved; conflicting metadata fails rather than silently changing the policy. Localization, physical calibration and actual locomotion still require robot-side testing.
