Added
^^^^^

* Added :meth:`~isaaclab_tasks.contrib.velocity.config.hexapod.hexapod_binary_actions.SpineSineAction.set_waveform`
  to replace the per-joint spine Fourier coefficients at runtime without rebuilding the
  env (used by ``eval_protocol.py`` to score the reference tripod gait against its own
  spine wave).

Changed
^^^^^^^

* **Breaking:** Changed the scripted spine (body-bending) wave of the
  ``Isaac-Goal-Flat-Hexapod-Binary-v0`` / ``-Play-v0`` environments to a fixed analytic
  traveling wave: ``FrontLink_Joint`` follows a pure sine and ``BackLink_Joint`` the same
  sine shifted +90 deg (a cosine), with a shared magnitude and offset -- a quarter-cycle
  wave travelling down the body, no longer fitted to a gait CSV. The per-joint Fourier
  coefficients carry ``-A_SPINE`` (the HexapI global spine-joint-sign flip): a positive
  coefficient drives the body wave the wrong way and walks the policy backward from the
  goal, matching both the negative amplitude of the earlier CSV-fitted wave and the
  in-sim-verified sign of the reference-tripod (Wave 2) wave. This changes the
  open-loop spine trajectory and therefore the closed-loop dynamics every binary-contact
  policy was trained against: retrain existing ``Isaac-Goal-Flat-Hexapod-Binary-*``
  checkpoints. The separate reference-tripod (Wave 2) wave, redefined below, is exported
  only for ``eval_protocol.py``'s reference-tripod baseline.
* Redefined the reference-tripod (Wave 2) spine wave as a single-harmonic analytic wave
  that is *anti-phase* across the two spine joints: ``FrontLink_Joint`` follows
  ``-A_SPINE * sin(2*pi*t - pi/4)`` and ``BackLink_Joint`` its negation
  ``+A_SPINE * sin(2*pi*t - pi/4)`` (MATLAB ``body_phase = pi``; the per-joint global
  sign was verified in Isaac Sim 2026-09-10 to walk the tripod baseline forward),
  regenerated analytically from the MATLAB gait generator rather than fitted to the
  committed ``tripod_extendedquad_sim.csv`` (whose byte-identical spine columns were a
  MATLAB-to-Sim export artifact). Same amplitude as the RL traveling wave (``A_SPINE``,
  from the open-loop gait generator at amp 70 deg). This replaces the earlier
  in-phase two-harmonic least-squares fit of the amp-65 CSV. ``play_discrete_closeup.py``'s
  ``--gait_npz tripod`` / ``--gait_csv tripod_extendedquad_sim.csv`` replay swaps this wave
  in too.
* **Breaking:** Replaced :class:`~isaaclab_tasks.contrib.velocity.config.hexapod.hexapod_binary_actions.SpineSineActionCfg`
  fields ``amplitude`` and ``phase`` (``dict[str, float]``) with ``sin_coef`` and
  ``cos_coef`` (``dict[str, list[float]]``, one entry per Fourier harmonic); ``offset``
  and ``period`` are unchanged. Migration: convert ``A`` / ``phi`` to the fundamental
  pair ``a1 = A * cos(phi)``, ``b1 = A * sin(phi)`` and pass ``sin_coef={j: [a1, ...]}``,
  ``cos_coef={j: [b1, ...]}``.
