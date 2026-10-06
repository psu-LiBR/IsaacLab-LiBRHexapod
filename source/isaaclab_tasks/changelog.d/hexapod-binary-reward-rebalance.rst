Changed
^^^^^^^

* Rebalanced six inherited reward weights on the ``Isaac-Goal-Flat-Hexapod-Binary-v0``
  / ``-Play-v0`` environments for the contact-bit action space: removed ``feet_slide``
  (it scored forced slip from the scripted spine wave, not policy behaviour), cut
  ``dof_acc_l2`` from ``-2.5e-7`` to ``-3.0e-8`` and disabled ``feet_air_time``
  (``weight = 0.0``) since neither is shapeable by a discrete per-leg bit, raised
  ``progress`` from ``0.2`` to ``1.5`` so a forward walk to the goal is clearly
  net-positive under the weaker DCMotor actuator, halved ``undesired_contacts`` from
  ``-1.0`` to ``-0.5`` to cut the asymmetric spine-contact tax the 6-bit policy cannot
  counter, and eased the ``time_penalty`` floor from ``-0.005`` to ``-0.003``. The
  continuous ``Isaac-Goal-Flat-Hexapod-v0`` reward weights are unchanged; retrain
  binary-action checkpoints under the new weights.
