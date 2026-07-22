# sim2real_transfer

Runs a trained hexapod RL policy (exported to ONNX by `play.py`) on the real
robot: real IMU (via ROS2 `/imu`) + real Dynamixel servos + a Raspberry Pi,
completely wirelessly. See `CLAUDE.md`'s sim2real plan for the full design
rationale, key facts (obs layout, DOF ordering, motor IDs, control rate), and
the staged bring-up checklist -- this file only covers day-to-day usage.

This package is plain Python with **no Isaac Lab / Isaac Sim dependency**. It
runs in its own environment on the robot's host computer (see
`requirements.txt`); ROS2 (`rclpy`) must already be installed there for the
`/imu` topic to exist.

## Setup

```bash
pip install -r requirements.txt
cp config/deployment.example.yaml config/deployment.yaml
# edit config/deployment.yaml: serial port, motor IDs, and especially the
# fields marked "# CALIBRATE" -- do not trust the example's placeholder values.
```

**TODO before stage 4 (live encoders): tune the USB-serial latency timer.**
The U2D2's FTDI chip defaults to a 16ms USB latency timer on Linux. At three
bus transactions per control cycle (position read, velocity read, goal
write), that's up to 48ms of pure driver overhead per loop -- already over
budget for a 50Hz (20ms) loop, regardless of the Dynamixel baud rate (raising
baud rate barely helps -- transmission time itself is ~250us at 1Mbps, not
the bottleneck). Set it to 1ms:

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# make it persistent across reboots with a udev rule, e.g.:
# ACTION=="add", SUBSYSTEM=="usb-serial", ATTR{latency_timer}="1"
```

Verify actual loop timing on the Pi (not just inference time) before trusting
the 50Hz assumption once this is set.

## Bring-up order (see CLAUDE.md for why)

```bash
# 1-2. Unit tests, then offline ONNX validation against a sim-recorded trace --
#      no hardware involved at all.
python -m pytest tests/
python tools/validate_onnx.py --trace <sim_trace.csv> --policy <policy.onnx> --profile velocity

# 3. Full loop, no hardware attached at all.
python run_policy.py --policy <policy.onnx> --profile velocity --dry-run --fake-imu --duration 5

# 4. Live IMU + live encoders, servos never move (zero physical risk).
python run_policy.py --policy <policy.onnx> --profile velocity --config config/deployment.yaml --no-torque

# 5. First real motion -- robot propped with legs off the ground, damped action scale.
python run_policy.py --policy <policy.onnx> --profile velocity --config config/deployment.yaml --action-scale-mult 0.2

# 6+. Full scale, then tethered ground contact, then free running.
python run_policy.py --policy <policy.onnx> --profile velocity --config config/deployment.yaml --log-csv run.csv
```

## Swapping policies

`--policy` just points at a different exported `.onnx` file (matching
`--profile`); `PolicyRunner` shape-checks it against the profile at load, so a
mismatched pairing fails immediately instead of producing garbage actions. No
config or code changes needed to switch between checkpoints of the same
profile (`velocity` or `goal`).

## Layout

- `sim2real/` -- the package: `joint_mapping.py` (Sim<->Real DOF conversion,
  most safety-critical file), `profiles.py` (obs schemas), `deployment_config.py`
  (typed `deployment.yaml` loader), `policy_runner.py` (onnxruntime wrapper),
  `dynamixel_bus.py` / `imu.py` (hardware interfaces, each with a
  hardware-free dry-run/fake counterpart), `command_source.py` /
  `localization.py` (velocity/goal command handling), `safety.py` (watchdog,
  startup/shutdown ramps), `logging_utils.py`, `control_loop.py` (ties it all
  together).
- `run_policy.py` -- CLI entry point.
- `tools/validate_onnx.py` -- offline sim-vs-onnx trace validation; run before
  ever touching hardware. Direct mode needs only a CSV with `obs_*`/`action_*`
  columns; pipeline mode additionally needs the raw fields that
  `isaaclab.bat play --rl_library rsl_rl ... --dump_obs_action_csv <path>`
  produces.
- `tools/calibrate_encoders.py`, `tools/check_body_alignment.py`,
  `tools/log_imu_rotation_test.py`, `tools/log_imu_axis_alignment_test.py` --
  standalone hardware calibration/diagnostic helpers, not part of the bring-up
  sequence above.
- `config/deployment.example.yaml` -- hardware facts template; copy to
  `deployment.yaml` (gitignored) and calibrate before real bring-up.
- `tests/` -- `python -m pytest tests/` from this directory, or
  `python -m pytest scripts/sim2real_transfer/tests/` from the repo root.
