Changed
^^^^^^^

* Recalibrated the ground friction of the hexapod locomotion, goal and binary-contact
  environments against the real-robot open-loop-gait friction sweep (2026-09).
  Evaluation/PLAY configs now pin the effective contact friction to ``mu = 0.21`` (the
  single best sweep row) and training configs randomize it over ``0.18-0.25``, applied
  through the robot-side ``events.physics_material`` ``randomize_rigid_body_material``
  ranges only (the terrain material and ``friction_combine_mode`` are unchanged). Affects
  :class:`~isaaclab_tasks.contrib.velocity.config.hexapod.flat_env_cfg.HexapodFlatEnvCfg`,
  :class:`~isaaclab_tasks.contrib.velocity.config.hexapod.rough_env_cfg.HexapodRoughEnvCfg`,
  the mimic, goal (including goal-tuned/BigStep) and binary-contact families, and their
  ``_PLAY`` variants. The ``-Rshape-*`` reward-shaping variants are unchanged -- they keep
  their own higher ``(0.9, 1.0) / (0.7, 0.8)`` validation friction.
* Migration: previous friction ranges were flat-train ``(0.2, 0.3) / (0.2, 0.25)``,
  flat-play ``(0.17, 0.21) / (0.13, 0.17)``, mimic-play ``(0.5, 0.6) / (0.35, 0.45)``;
  the goal, mimic and binary configs inherited flat-train's ``(0.2, 0.3) / (0.2, 0.25)``
  through ``super().__post_init__()``, and only the rough config (and its ``_PLAY``) fell
  through to the core default ``(0.8, 0.8) / (0.6, 0.6)``. All are now ``(0.18, 0.25)``
  for training and ``(0.21, 0.21)`` for evaluation.
