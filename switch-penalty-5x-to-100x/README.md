# Switch penalty · representative policies

**33 simulation candidates: 23 in 0.6–0.8 BL/cycle and 10 above 0.8.** Each candidate pairs its original checkpoint, ONNX, complete reward weights, inference metadata, measurements and validation. No retraining is required to evaluate these checkpoints. Hardware performance remains to be measured.

[Comparison page](https://psu-libr.github.io/IsaacLab-LiBRHexapod/switch-penalty-5x-to-100x/) · [Machine-readable index](candidate_manifest.json) · [Reproduction guide](REPRODUCE_DQN_DDQN.md) · [Robot runtime](../scripts/sim2real_transfer/README.md)

## Structure

```text
switch-penalty-5x-to-100x/
├── README.md                       # Overview, definitions and candidate tables
├── index.html                      # Interactive comparison page
├── candidate_pool_summary.json     # Evaluated, eligible and published counts
├── eligible_candidate_pool.csv     # All 177 numerical matches, including unselected
├── SELECTION.md                    # Coverage, tradeoffs and selection limitations
├── REWARD_DEFINITION.md             # Penalty provenance, formulas and coefficients
├── REPRODUCE_DQN_DDQN.md            # Reproduction guide for all six algorithms
├── VIDEO_CATALOG.md                # External video collections and identity mapping
├── candidate_manifest.json         # Published policy IDs, metrics and file hashes
├── validation_summary.json         # Recorded validation results
├── deployment.example.yaml         # CPU runtime interface and hardware configuration
└── candidates/
    ├── dqn-family/                 # dqn / ddqn
    ├── ppo-family/                 # ppo / masked-ppo
    └── sac-family/                 # sac-discrete-one-step / sac-discrete-five-step
        └── <algorithm>/<displacement-group>/<reward-condition>/<candidate>/
            ├── README.md          # Metrics, limitations and exact commands
            ├── policy.pt          # Original checkpoint
            ├── policy.onnx        # Exported inference graph
            ├── run_meta.json      # Normalization, architecture and action metadata
            ├── config.json        # Complete resolved reward/training configuration
            ├── evaluation.json    # Fixed simulation protocol
            ├── metrics.json       # Full-precision measurements
            ├── validation.json    # Simulation/export/CPU validation evidence
            └── index.html         # Candidate detail page
```

The candidate hierarchy is shared by all three families. Empty algorithm/group combinations are reported in the tables; no duplicate model is inserted to fill a quota.

**Reproduction chain:** table row → candidate ID → paired checkpoint/configuration and hashes → evaluation command → measured JSON and runtime audit. **Video correspondence:** candidate ID + checkpoint SHA-256 → the same hierarchy under `RL_Binary/github-policy-videos/` → matching MP4 and video SHA-256.

## Candidate population

The evaluated pool contains **690 unique checkpoints** from six algorithms, two penalty conditions and four multipliers. Repeated evaluations are deduplicated by checkpoint SHA-256; duplicate records have identical screening measurements. Historical reward studies and the separate native-action-rate control are outside this population.

Both groups require valid continuous measurements, zero resets, and a **maximum over individual legs and complete cycles of 2, 3 or 4 command changes**. The target group is **0.6 ≤ BL/cycle ≤ 0.8**; the comparison group is **BL/cycle > 0.8**. Exactly 0.8 belongs to the target group.

| Algorithm | 0.6–0.8: eligible | No recorded falls/contact | Published | >0.8: eligible | Published |
|---|---:|---:|---:|---:|---:|
| DQN | 9 | 9 | 5 | 66 | 5 |
| DDQN | 15 | 14 | 5 | 46 | 5 |
| PPO | 19 | 19 | 5 | 0 | 0 |
| Masked PPO | 19 | 19 | 5 | 0 | 0 |
| SAC-D 1-step | 3 | 3 | 3 | 0 | 0 |
| SAC-D 5-step | 0 | 0 | 0 | 0 | 0 |
| **Total** | **65** | **64** | **23** | **112** | **10** |

These are **65 + 112 numerical matches**, from which **23 + 10 representatives** are packaged. They are not the only available results. One target-band DDQN checkpoint (0.6194 BL/cycle) records torso contact in every environment and is excluded from the policy delivery; it remains explicitly marked in the numerical inventory and additional video collection. All 112 higher-displacement numerical matches have zero recorded falls and torso contact.

See [selection rationale](SELECTION.md), [all numerical matches](eligible_candidate_pool.csv), [reward definition](REWARD_DEFINITION.md), and [video catalog](VIDEO_CATALOG.md). Unselected candidates are not labeled failures solely because they were not packaged.

## Selection

Within each algorithm, start with the **0.6–0.8** group. Rows are in **ascending displacement**, not a performance ranking. The >0.8 group retains higher-displacement comparisons for separate assessment. Exactly 0.8 belongs to the first group; below 0.6 is outside this delivery.

All selected evaluations have valid continuous measurements, no observed falls or torso contact, and a **maximum of 2, 3 or 4 command changes over all individual legs and complete cycles**. This does not mean every leg switches at least twice. Selection spreads displacement, command-switch counts, reward conditions and dwell times, with straightness used as a preference. There is no hidden yaw/drift threshold. Some nearby displacement values are retained only for materially different dwell or straightness; see each candidate's rationale. Videos remain outside this repository.

| Algorithm | 0.6–0.8 | >0.8 | Availability and tradeoff |
|---|---:|---:|---|
| DQN | 5 | 5 | No suitable 0.60–0.66 point in evaluated pool; starts at 0.6839 |
| DDQN | 5 | 5 | 0.6194 omitted because every evaluated environment had torso contact |
| PPO | 5 | 0 | No >0.8 checkpoint satisfies maximum switching ≤4 |
| Masked PPO | 5 | 0 | Eligible values stop at 0.6816; no 0.7 band or >0.8 point to supply |
| SAC-D 1-step | 3 | 0 | All three numerical matches retained; substantial lateral drift is explicit |
| SAC-D 5-step | 0 | 0 | 22 evaluated points in 0.6–0.8 have 14–38 maximum switches; none meet ≤4 |

## Table definitions

- **BL/cycle:** mean forward displacement relative to each environment's initial heading, divided by body length 0.315 m and six nominal 1-second spine cycles. This is not a measured leg-gait period or a direct measure of limb speed. At this fixed period, 0.6–0.8 corresponds to mean forward speed 0.189–0.252 m/s.
- **Max switches:** maximum high-level binary-command changes for one leg in one complete, phase-aligned spine cycle, across 64 environments. Five complete cycles per environment enter this statistic. It is not the six-leg sum or physical foot-contact switching.
- **Yaw / drift:** maximum absolute heading change in degrees / lateral displacement in BL, relative to the initial heading, across the whole evaluation. Lower is straighter; these are not final-position-only values.
- **Dwell min / P05 / median:** seconds between consecutive changes of the same leg command; boundary-censored intervals excluded. This is how long a command is held, **not** how long the robot stays straight or a neutral joint pose. Small minima expose brief pulses even when max switches ≤4.
- **Original / added:** the original reward weight and extra cost per flipped leg. Original reference −0.005; added reference 0.0004. Multipliers are 5×, 10×, 30× or 100×. Both terms penalize commands; neither directly measures physical contacts. With binary ±1 and dt=0.02 s, original cost per flipped leg is 4×|weight|×dt; added cost is the listed value.

Each metrics.json includes the six individual leg maxima in order **front right, front left, middle right, middle left, rear right, rear left**, complete-cycle counts and a single-environment command/trajectory preview. Full resolved reward weights are in config.json.


## DQN family

### DQN

**0.6–0.8 BL/cycle**

| Candidate & files | BL/cycle | Max switches | Yaw ° | Drift BL | Dwell min / P05 / median (s) | Penalty |
|---|---:|---:|---:|---:|---|---|
| [dqn-moderate-0.6839](candidates/dqn-family/dqn/target-band/original-only/original-10x__added-off__forward-0.6839-body-lengths/README.md) | 0.6839 | 3 | 33.0 | 0.458 | 0.20 / 0.30 / 0.50 | 10× original only |
| [dqn-moderate-0.7580](candidates/dqn-family/dqn/target-band/original-only/original-30x__added-off__forward-0.7580-body-lengths/README.md) | 0.7580 | 4 | 29.8 | 0.427 | 0.10 / 0.26 / 0.48 | 30× original only |
| [dqn-moderate-0.7656](candidates/dqn-family/dqn/target-band/original-plus-added-switch/original-100x__added-100x__forward-0.7656-body-lengths/README.md) | 0.7656 | 2 | 35.1 | 0.544 | 0.34 / 0.36 / 0.44 | 100× original + added |
| [dqn-moderate-0.7959](candidates/dqn-family/dqn/target-band/original-plus-added-switch/original-100x__added-100x__forward-0.7959-body-lengths/README.md) | 0.7959 | 2 | 33.5 | 0.559 | 0.36 / 0.36 / 0.46 | 100× original + added |
| [dqn-moderate-0.7986](candidates/dqn-family/dqn/target-band/original-plus-added-switch/original-30x__added-30x__forward-0.7986-body-lengths/README.md) | 0.7986 | 3 | 37.4 | 0.320 | 0.14 / 0.16 / 0.48 | 30× original + added |

**Above 0.8 BL/cycle**

| Candidate & files | BL/cycle | Max switches | Yaw ° | Drift BL | Dwell min / P05 / median (s) | Penalty |
|---|---:|---:|---:|---:|---|---|
| [dqn-higher-0.8158](candidates/dqn-family/dqn/higher-displacement/original-plus-added-switch/original-100x__added-100x__forward-0.8158-body-lengths/README.md) | 0.8158 | 2 | 27.9 | 0.336 | 0.34 / 0.34 / 0.44 | 100× original + added |
| [dqn-higher-0.8617](candidates/dqn-family/dqn/higher-displacement/original-plus-added-switch/original-100x__added-100x__forward-0.8617-body-lengths/README.md) | 0.8617 | 2 | 25.0 | 0.265 | 0.34 / 0.34 / 0.40 | 100× original + added |
| [dqn-higher-0.9486](candidates/dqn-family/dqn/higher-displacement/original-only/original-10x__added-off__forward-0.9486-body-lengths/README.md) | 0.9486 | 2 | 27.7 | 0.142 | 0.04 / 0.12 / 0.28 | 10× original only |
| [dqn-higher-0.9983](candidates/dqn-family/dqn/higher-displacement/original-plus-added-switch/original-5x__added-5x__forward-0.9983-body-lengths/README.md) | 0.9983 | 4 | 29.9 | 0.137 | 0.06 / 0.06 / 0.26 | 5× original + added |
| [dqn-higher-1.0116](candidates/dqn-family/dqn/higher-displacement/original-plus-added-switch/original-10x__added-10x__forward-1.0116-body-lengths/README.md) | 1.0116 | 3 | 29.8 | 0.337 | 0.14 / 0.14 / 0.26 | 10× original + added |

### DDQN

**0.6–0.8 BL/cycle**

| Candidate & files | BL/cycle | Max switches | Yaw ° | Drift BL | Dwell min / P05 / median (s) | Penalty |
|---|---:|---:|---:|---:|---|---|
| [ddqn-moderate-0.6582](candidates/dqn-family/ddqn/target-band/original-plus-added-switch/original-100x__added-100x__forward-0.6582-body-lengths/README.md) | 0.6582 | 3 | 43.9 | 0.620 | 0.16 / 0.16 / 0.50 | 100× original + added |
| [ddqn-moderate-0.7440](candidates/dqn-family/ddqn/target-band/original-only/original-100x__added-off__forward-0.7440-body-lengths/README.md) | 0.7440 | 3 | 38.0 | 0.382 | 0.14 / 0.16 / 0.47 | 100× original only |
| [ddqn-moderate-0.7520](candidates/dqn-family/ddqn/target-band/original-plus-added-switch/original-30x__added-30x__forward-0.7520-body-lengths/README.md) | 0.7520 | 2 | 44.7 | 0.408 | 0.16 / 0.20 / 0.47 | 30× original + added |
| [ddqn-moderate-0.7786](candidates/dqn-family/ddqn/target-band/original-plus-added-switch/original-100x__added-100x__forward-0.7786-body-lengths/README.md) | 0.7786 | 2 | 33.5 | 0.336 | 0.40 / 0.40 / 0.46 | 100× original + added |
| [ddqn-moderate-0.7884](candidates/dqn-family/ddqn/target-band/original-plus-added-switch/original-10x__added-10x__forward-0.7884-body-lengths/README.md) | 0.7884 | 4 | 37.7 | 0.235 | 0.06 / 0.12 / 0.24 | 10× original + added |

**Above 0.8 BL/cycle**

| Candidate & files | BL/cycle | Max switches | Yaw ° | Drift BL | Dwell min / P05 / median (s) | Penalty |
|---|---:|---:|---:|---:|---|---|
| [ddqn-higher-0.8310](candidates/dqn-family/ddqn/higher-displacement/original-plus-added-switch/original-100x__added-100x__forward-0.8310-body-lengths/README.md) | 0.8310 | 3 | 27.5 | 0.350 | 0.30 / 0.32 / 0.44 | 100× original + added |
| [ddqn-higher-0.8861](candidates/dqn-family/ddqn/higher-displacement/original-only/original-100x__added-off__forward-0.8861-body-lengths/README.md) | 0.8861 | 2 | 26.5 | 0.252 | 0.18 / 0.28 / 0.38 | 100× original only |
| [ddqn-higher-0.9253](candidates/dqn-family/ddqn/higher-displacement/original-plus-added-switch/original-5x__added-5x__forward-0.9253-body-lengths/README.md) | 0.9253 | 4 | 28.9 | 0.260 | 0.02 / 0.10 / 0.28 | 5× original + added |
| [ddqn-higher-0.9937](candidates/dqn-family/ddqn/higher-displacement/original-plus-added-switch/original-10x__added-10x__forward-0.9937-body-lengths/README.md) | 0.9937 | 4 | 28.8 | 0.181 | 0.08 / 0.14 / 0.32 | 10× original + added |
| [ddqn-higher-1.0025](candidates/dqn-family/ddqn/higher-displacement/original-plus-added-switch/original-30x__added-30x__forward-1.0025-body-lengths/README.md) | 1.0025 | 2 | 30.4 | 0.296 | 0.14 / 0.16 / 0.26 | 30× original + added |

## PPO family

### PPO

**0.6–0.8 BL/cycle**

| Candidate & files | BL/cycle | Max switches | Yaw ° | Drift BL | Dwell min / P05 / median (s) | Penalty |
|---|---:|---:|---:|---:|---|---|
| [ppo-moderate-0.6173](candidates/ppo-family/ppo/target-band/original-plus-added-switch/original-10x__added-10x__forward-0.6173-body-lengths/README.md) | 0.6173 | 2 | 28.5 | 0.442 | 0.04 / 0.06 / 0.50 | 10× original + added |
| [ppo-moderate-0.6550](candidates/ppo-family/ppo/target-band/original-plus-added-switch/original-5x__added-5x__forward-0.6550-body-lengths/README.md) | 0.6550 | 2 | 27.5 | 0.453 | 0.16 / 0.18 / 0.52 | 5× original + added |
| [ppo-moderate-0.6778](candidates/ppo-family/ppo/target-band/original-plus-added-switch/original-30x__added-30x__forward-0.6778-body-lengths/README.md) | 0.6778 | 3 | 32.7 | 0.387 | 0.12 / 0.14 / 0.48 | 30× original + added |
| [ppo-moderate-0.7115](candidates/ppo-family/ppo/target-band/original-only/original-10x__added-off__forward-0.7115-body-lengths/README.md) | 0.7115 | 4 | 31.0 | 0.244 | 0.02 / 0.02 / 0.46 | 10× original only |
| [ppo-moderate-0.7519](candidates/ppo-family/ppo/target-band/original-plus-added-switch/original-10x__added-10x__forward-0.7519-body-lengths/README.md) | 0.7519 | 4 | 29.7 | 0.258 | 0.08 / 0.12 / 0.44 | 10× original + added |

**Above 0.8 BL/cycle**

No checkpoint meets the selection conditions.

### Masked PPO

**0.6–0.8 BL/cycle**

| Candidate & files | BL/cycle | Max switches | Yaw ° | Drift BL | Dwell min / P05 / median (s) | Penalty |
|---|---:|---:|---:|---:|---|---|
| [masked-ppo-moderate-0.6018](candidates/ppo-family/masked-ppo/target-band/original-plus-added-switch/original-5x__added-5x__forward-0.6018-body-lengths/README.md) | 0.6018 | 4 | 31.6 | 0.274 | 0.02 / 0.06 / 0.42 | 5× original + added |
| [masked-ppo-moderate-0.6327](candidates/ppo-family/masked-ppo/target-band/original-only/original-10x__added-off__forward-0.6327-body-lengths/README.md) | 0.6327 | 2 | 27.7 | 0.330 | 0.04 / 0.04 / 0.50 | 10× original only |
| [masked-ppo-moderate-0.6497](candidates/ppo-family/masked-ppo/target-band/original-only/original-10x__added-off__forward-0.6497-body-lengths/README.md) | 0.6497 | 2 | 32.8 | 0.382 | 0.28 / 0.30 / 0.54 | 10× original only |
| [masked-ppo-moderate-0.6728](candidates/ppo-family/masked-ppo/target-band/original-only/original-30x__added-off__forward-0.6728-body-lengths/README.md) | 0.6728 | 3 | 29.0 | 0.273 | 0.06 / 0.08 / 0.48 | 30× original only |
| [masked-ppo-moderate-0.6816](candidates/ppo-family/masked-ppo/target-band/original-only/original-10x__added-off__forward-0.6816-body-lengths/README.md) | 0.6816 | 4 | 26.9 | 0.243 | 0.02 / 0.02 / 0.41 | 10× original only |

**Above 0.8 BL/cycle**

No checkpoint meets the selection conditions.

## SAC-D family

### SAC-D 1-step

**0.6–0.8 BL/cycle**

| Candidate & files | BL/cycle | Max switches | Yaw ° | Drift BL | Dwell min / P05 / median (s) | Penalty |
|---|---:|---:|---:|---:|---|---|
| [sac-discrete-one-step-moderate-0.6101](candidates/sac-family/sac-discrete-one-step/target-band/original-plus-added-switch/original-10x__added-10x__forward-0.6101-body-lengths/README.md) | 0.6101 | 2 | 34.5 | 0.638 | 0.42 / 0.42 / 0.50 | 10× original + added |
| [sac-discrete-one-step-moderate-0.6356](candidates/sac-family/sac-discrete-one-step/target-band/original-only/original-5x__added-off__forward-0.6356-body-lengths/README.md) | 0.6356 | 2 | 34.4 | 0.831 | 0.26 / 0.34 / 0.52 | 5× original only |
| [sac-discrete-one-step-moderate-0.7744](candidates/sac-family/sac-discrete-one-step/target-band/original-only/original-10x__added-off__forward-0.7744-body-lengths/README.md) | 0.7744 | 2 | 28.1 | 0.823 | 0.26 / 0.30 / 0.48 | 10× original only |

**Above 0.8 BL/cycle**

No checkpoint meets the selection conditions.

### SAC-D 5-step

**0.6–0.8 BL/cycle**

No checkpoint meets the selection conditions.

**Above 0.8 BL/cycle**

No checkpoint meets the selection conditions.

## Reproduce a candidate

Use this branch with its matching Isaac Sim / Isaac Lab installation. Fetch the robot assets with Git LFS (the small policy files themselves are ordinary Git files):

```bash
git lfs pull --include="hexapod-assets/**"
./isaaclab.sh -p scripts/reinforcement_learning/binary_rl/evaluate_switch_candidate.py \
  --candidate dqn-moderate-0.6839 --out reproduced-dqn.json
```

On Windows use `isaaclab.bat -p`. Replace only the candidate ID to select another table row. The evaluator automatically loads and verifies its paired files and configuration. It writes measurement JSON, an actual runtime audit, and a compressed trajectory. Use a new output filename for each evaluation; it refuses to overwrite a result.

The exact published protocol is 64 environments, seed 7, fixed 2 m goal, 0.21 friction, no observation noise or resets, deterministic greedy inference, one warmup step followed by 300 steps at 50 Hz. PPO normalization and Masked PPO's 24-action restriction are loaded explicitly. The script applies the original reward configuration and checks the resolved weights, observation shape, timestep and loaded material friction. General `play_discrete.py` and `eval_protocol.py` serve other comparison protocols; their numbers must not be substituted for this table.

See [reproduction and troubleshooting](REPRODUCE_DQN_DDQN.md) for expected output, dependencies and what to retain if a result differs. Simulation reproduction is an acceptance test of this delivery, not a promise that a different physics version or real robot yields identical numbers.

## Deployment files

Every candidate directory contains:

| File | Use |
|---|---|
| policy.pt | Original checkpoint, unmodified |
| policy.onnx | CPU inference graph, input float32 [1,32], output float32 [1,6] in −1/+1 |
| run_meta.json | Architecture, observation normalization and legal-action metadata |
| config.json | All resolved reward weights, multiplier coefficients and training/inference configuration |
| evaluation.json | Exact quantitative-evaluation settings |
| metrics.json | Historical measurements and command/trajectory preview |
| validation.json | Export, CPU pipeline and fresh simulation verification receipts |
| README.md | Candidate-specific metrics, tradeoff and copyable commands |

The graph folds in observation normalization, clipping, action masking, argmax and bit decoding; do not apply those again. The host supplies gyro (3), projected gravity (3), goal command (4), relative joint positions (8), joint velocities (8) and previous six-bit action (6). Use this batch's [deployment.example.yaml](deployment.example.yaml), which preserves its audited joint observation order. The six actions use front-right, front-left, middle-right, middle-left, rear-right, rear-left order; +1 denotes stance. The host runs the spine wave separately at 1 second and control at 50 Hz.

The hardware spine amplitude retains the target branch's positive motor-mount correction; simulation uses the negative amplitude. Existing calibration is preserved. Hardware localization and real sensor signals are not validated by a fake-IMU dry run. The deployment example's fixed goal is 5 m; the quantitative simulation deliberately uses 2 m, so these are not the same navigation task.

## Validation scope

The per-candidate validation receipt is authoritative; the aggregate [validation summary](validation_summary.json) records the completed checks. Original evaluation inference, packaged inference, export module and ONNX are compared on 2,049 synthetic inputs per candidate. Each ONNX also runs through the actual CPU deployment pipeline for five seconds with fake IMU and an in-memory motor bus. Fresh simulator runs use these packaged checkpoints and the code in this branch.

The previous delivery emphasized large forward displacement. This selection replaces that default with a moderate-displacement comparison first, while explicitly preserving representative >0.8 cases. Earlier commits preserve the previous set; unselected candidates are not silently relabeled as failures. No videos or workflow changes are part of this update.
