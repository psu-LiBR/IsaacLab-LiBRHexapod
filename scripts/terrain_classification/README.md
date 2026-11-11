# Terrain classification data preparation

This is a feature preparation baseline, not a trained classifier. Run independently of Isaac Sim with Python and NumPy.

## Recording contract

Use CSV files produced by `scripts/sim2real_transfer/sim2real/logging_utils.py`. Each file must contain one terrain only; trim starts, stops, and terrain transitions before preparation. Labels are supplied by the operator, not inferred from robot commands. Actual labeled recordings are not yet included.

Create a manifest CSV with these columns:

```csv
path,label,session_id,split
recordings/run01.csv,gravel,day01,train
recordings/run02.csv,gravel,day02,validation
recordings/run03.csv,gravel,day03,test
```

Paths are relative to the manifest. Choose actual terrain names after inspecting available data. All recordings from a collection session must have the same split. Keep related repeats in the same session. Every terrain needs independent sessions in train, validation, and test; the preparation script enforces session isolation but does not enforce class coverage.

```sh
python scripts/terrain_classification/prepare.py data/manifest.csv data/terrain_features.npz
```

## Initial features

Default windows span 1 second with a 0.5 second stride. Each channel contributes mean, standard deviation, root mean square, and range, yielding 120 features:

- Three body gyro channels: rotation and vibration response.
- Three projected gravity channels: body tilt response. These are gravity direction, not linear acceleration.
- Eight measured joint positions and eight measured velocities: gait response.
- Eight commanded target minus measured position channels: joint tracking error. Targets and measurements in the existing logger may be from different points in a control cycle, so this is a cycle-level proxy, not synchronized servo error.

Uses physical measurement columns instead of profile-dependent joint observation offsets. Commands and actions are excluded initially to reduce shortcuts based on collection behavior. Balance gait, speed, payload, policy, battery condition, and slope across terrains; record them in collection notes. Log schema and units should match across runs. Do not mix simulated and real recordings without explicitly tracking the domain.

Timestamps must be finite and strictly increasing. Invalid numeric data rejects a recording. Windows with fewer than three samples, gaps over 0.1 seconds, or insufficient boundary coverage are skipped. A run producing no windows fails. The maximum gap is configurable and must be checked against actual loop timing. Statistics are sample-weighted; heavily irregular data needs resampling before frequency-domain features are added.

## Training handoff

The NPZ contains `X`, `y`, `feature_names`, `split`, `session_id`, `source`, `start_s`, and window configuration. Load with `numpy.load(..., allow_pickle=False)`. Select rows by the predefined split. Fit imputation, feature selection, and scaling on training rows only. Keep validation for tuning and test untouched until choosing a model. Do not randomly split windows: adjacent windows overlap.

Start by comparing a majority-class baseline, standardized logistic regression, and a tree ensemble. Report macro F1, balanced accuracy, per-class confusion, and results by session. Treat session count as the independent sample count when assessing confidence. Compare gyro/gravity alone with the full feature set to check whether joint features generalize across gaits. No accuracy claims are possible until labeled recordings exist.

Future collection improvements: raw linear acceleration, synchronized sensor timestamps, motor current/load when available, and explicit policy/gait/speed metadata. The current run logger does not record raw linear acceleration or motor current. Frequency-band energy can be evaluated once sensor sample rates and timestamp jitter are measured.
