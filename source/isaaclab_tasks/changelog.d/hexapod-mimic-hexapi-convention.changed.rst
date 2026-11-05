* **Breaking:** Changed the built-in sinusoidal tripod gait of the hexapod mimic reference
  (``MotionReference`` in ``hexapod_mimic_motion.py``) to stand at ``+0.47`` rad instead of ``-0.47`` rad, which
  lay outside the HexapI leg joint limits (``-5`` to ``110`` deg). It now matches the positive-leg convention
  of ``HexapiFlattened.usd``. Only mimic tasks run with ``MIMIC_CSV_PATH = None`` are affected; retrain their
  checkpoints.
