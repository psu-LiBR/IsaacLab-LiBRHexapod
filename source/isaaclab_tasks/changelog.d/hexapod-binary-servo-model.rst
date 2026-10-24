Added
^^^^^

* Added the opt-in ``Isaac-Goal-Flat-Hexapod-Binary-Servo-v0`` and
  ``Isaac-Goal-Flat-Hexapod-Binary-Servo-Play-v0`` environments
  (:class:`HexapodBinaryServoEnvCfg` / :class:`HexapodBinaryServoEnvCfg_PLAY`). They replace the leg actuator
  model of the binary-contact goal task with one fitted to real XL430 servo logs: firmware-default leg stiffness
  (5.16 N m/rad, randomised 0.8-1.75x), larger effective armature and a 3-5 physics-step command delay. Each
  environment also draws a battery-voltage scale (0.85-1.0) applied to the servo velocity limit, stall torque
  and effort limit. The play variant is deterministic. Policies must be trained on the ``-Servo-`` task ID to
  benefit.
* Added ``ServoDCMotor`` / ``ServoDCMotorCfg`` and ``with_servo_model`` to ``isaaclab_assets.robots.hexapod_actuators``.

Changed
^^^^^^^

* Hexapod ``_PLAY`` environment configs now pin the servo command delay to a fixed value so evaluations are
  reproducible.
