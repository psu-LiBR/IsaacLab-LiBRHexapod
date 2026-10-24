Added
^^^^^

* Added ``DelayedDCMotor`` / ``DelayedDCMotorCfg`` and ``with_fixed_actuator_delay`` in
  ``isaaclab_assets.robots.hexapod_actuators``: a DC motor whose commands are applied after a per-environment
  random delay, and a helper that pins the delay to the middle of its range for evaluation.

Changed
^^^^^^^

* **Breaking:** The LiBR Hexapod (:data:`~isaaclab_assets.robots.hexapod.HEXAPOD_CFG`) actuators now use
  ``DelayedDCMotorCfg`` with a command delay fitted to real robot logs (legs 8-12 and spine 11-15 physics steps,
  redrawn per environment at every reset), so every hexapod task sees the servo lag. Existing hexapod policies
  must be retrained. To restore the previous behavior set ``HEXAPOD_ACTUATOR_DELAY = False`` in ``hexapod.py``.
