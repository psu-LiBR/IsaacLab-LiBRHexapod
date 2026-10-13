* Fixed the hexapod play scripts (``play_rsl_rl``, ``playReal``, ``playpyvista``, ``play_physicsGait``,
  ``playTracking``) crashing with ``PermissionError`` on any machine other than the original author's: the
  joint-position and displacement CSV logs were written to a hardcoded ``C:/Users/<user>/...`` folder. The logs
  are now off by default; when enabled they go to ``<checkpoint_dir>/play_logs/``
  (``logs/rsl_rl/<experiment>/gait_playback/play_logs/`` for ``playReal`` gait-CSV replay), file names are
  timestamped per run instead of overwritten, and a log that cannot be written only prints a warning instead
  of stopping playback.
