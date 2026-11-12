Added
^^^^^

* Added the ``Isaac-Goal-Flat-Hexapod-Mimic-Servo-v0`` and ``Isaac-Goal-Flat-Hexapod-Mimic-Servo-Play-v0``
  environments (:class:`HexapodGoalMimicServoEnvCfg` / :class:`HexapodGoalMimicServoEnvCfg_PLAY`). They train the
  continuous goal + imitation task on the same fitted XL430 servo model as ``Isaac-Goal-Flat-Hexapod-Binary-Servo-v0``
  (firmware-default leg stiffness, larger effective armature, 3-5 physics-step leg delay, per-environment gain and
  battery-voltage randomisation). Runs log to ``logs/rsl_rl/hexapod_goal_mimic_servo/``.
