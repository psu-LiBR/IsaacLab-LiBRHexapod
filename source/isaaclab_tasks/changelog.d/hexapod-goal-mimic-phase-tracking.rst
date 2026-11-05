Added
^^^^^

* Added ``joint_pos_imitation_phase``, a hexapod imitation reward that tracks the reference gait phase from the
  robot's pose so the policy can choose its own gait frequency, and enabled it in
  ``Isaac-Goal-Flat-Hexapod-Mimic-v0`` (set ``GOAL_MIMIC_PHASE_TRACKING = False`` to restore the time-locked reward).
