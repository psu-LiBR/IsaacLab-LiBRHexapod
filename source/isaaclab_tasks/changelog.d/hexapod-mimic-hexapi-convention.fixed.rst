* Fixed the hexapod mimic imitation reward (``Isaac-Velocity-Flat-Hexapod-Mimic-*``) tracking a mis-signed
  reference: the committed ``hexapod-assets/Sim Gaits`` CSVs and the built-in gait were written for the old
  ``Hexapod_Flattened.usd`` (legs stance -0.46 / lift -1.18 rad), while ``HexapiFlattened.usd`` uses the mirrored
  convention (legs +0.46 / +1.18 rad, limits -5..110 deg, spine sign-flipped), so the reward pulled the legs
  outside their joint limits. :class:`~isaaclab_tasks.contrib.velocity.config.hexapod.hexapod_mimic_motion.MotionReference`
  now takes ``csv_convention`` (``"legacy"``, the default, negates all eight CSV columns on load; ``"hexapi"`` uses
  a CSV as is), exposed as ``csv_convention`` on ``joint_pos_imitation`` / ``spine_pos_imitation`` and as
  ``MIMIC_CSV_CONVENTION`` in the mimic env config; the built-in gait now stands at ``+0.47`` rad. The spine
  sign flip follows the documented HexapI convention and has not been checked in simulation. **Breaking:**
  existing mimic checkpoints were trained against the old reference and should be retrained.
