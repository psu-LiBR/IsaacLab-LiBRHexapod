Changed
^^^^^^^

* **Breaking:** Migrated the LiBR Hexapod (:data:`~isaaclab_assets.robots.hexapod.HEXAPOD_CFG`) leg and
  spine actuators from :class:`~isaaclab.actuators.ImplicitActuatorCfg` to
  :class:`~isaaclab.actuators.DCMotorCfg` matched to the Dynamixel XL430-W250-T
  (stall 1.4 N·m, no-load 5.97 rad/s at 11.1 V, 258.5:1 gearing), adding the
  velocity-dependent torque-speed limit, reflected-rotor ``armature`` and geartrain
  ``friction``. This changes the closed-loop joint dynamics for every hexapod task, so
  existing hexapod policy checkpoints must be retrained. To restore the previous
  behavior, revert the ``actuators`` block to the ``ImplicitActuatorCfg`` groups
  (``leg_joints`` stiffness=20 / damping=0.9, ``body_joints`` stiffness=10 / damping=0.3).
