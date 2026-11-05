Changed
^^^^^^^

* **Breaking:** Added a ``min_stance_legs`` penalty
  (:func:`~isaaclab_tasks.contrib.velocity.config.hexapod.hexapod_binary_rewards.min_stance_legs_penalty`,
  weight ``-0.2``, ``min_legs=4``, which may be set to 3 or 4) to the
  ``Isaac-Goal-Flat-Hexapod-Binary-*`` tasks (including the ``-Servo-`` variants) that penalizes
  steps with fewer than ``min_legs`` legs commanded to stance. Retrain binary checkpoints trained
  before this change under the new reward.
