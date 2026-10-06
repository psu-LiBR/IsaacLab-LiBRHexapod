Changed
^^^^^^^

* Updated the hexapod velocity, goal, mimic and binary-contact environments for the upstream ``develop``
  base configuration. The ``contact_forces`` sensor and ``base_com`` event are plain configs now, so the
  hexapod configs no longer address their ``.default`` / ``.physx`` sub-fields.
* Added :class:`~isaaclab_tasks.contrib.velocity.config.hexapod.rough_env_cfg.HexapodPhysicsCfg`, which keeps
  PhysX (``isaacsim_physx``) as the default physics backend of every hexapod task. The upstream locomotion
  default is Newton MJWarp, but the hexapod actuator model and friction calibration were tuned on PhysX.
  Select another backend with ``physics=<name>`` only after re-validating the gait.
* Ported the hexapod PPO runner configs to the ``actor`` / ``critic`` model form of ``rsl-rl-lib`` 5.5.
  ``actor_obs_normalization`` / ``critic_obs_normalization`` became ``obs_normalization`` on each model and
  the ``obs_groups`` keys became ``actor`` / ``critic``. Update any script that read ``cfg.policy.*`` to
  read ``cfg.actor.*`` / ``cfg.critic.*`` instead.
* Set ``use_newton_actuators=False`` on the hexapod simulation config so the ``DCMotorCfg`` actuators keep
  using the Isaac Lab execution path the actuator tuning was validated on. Upstream now defaults this flag to
  ``True`` (native Newton actuator adapter, also used under PhysX) and deprecates the old path; remove the
  override once the two paths have been compared.
