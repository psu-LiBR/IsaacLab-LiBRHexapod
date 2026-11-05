Changed
^^^^^^^

* Changed the ``Isaac-Goal-Flat-Hexapod-Mimic-v0`` task to hold the ``action_rate_l2`` weight at -5e-3 for the
  whole run instead of ramping it to -0.03 after the imitation blend, so the RL phase is not pushed back toward the
  reference gait. To restore the old behavior, re-add the ``restore_action_rate_l2`` curriculum ramp.
