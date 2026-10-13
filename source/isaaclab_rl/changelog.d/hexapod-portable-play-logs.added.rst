* Added ``--save_play_logs`` and ``--play_log_dir`` to the hexapod play scripts (``play_rsl_rl``, ``playReal``,
  ``playpyvista``, ``play_physicsGait``, ``playTracking``). The per-step CSV logs (joint positions, base
  displacement) are now opt-in: ``--save_play_logs`` writes them to ``<checkpoint_dir>/play_logs/`` and
  ``--play_log_dir <dir>`` writes them to ``<dir>`` (and also enables them).
