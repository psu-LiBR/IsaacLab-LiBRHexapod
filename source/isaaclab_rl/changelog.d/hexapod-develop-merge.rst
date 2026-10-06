Changed
^^^^^^^

* Rebuilt the ``play_rsl_rl`` backend on the upstream ``run(argv)`` structure and re-added the hexapod
  ``--dump_obs_action_csv`` option, the joint-position and displacement CSV logs and the per-term reward
  summary.
* Ported the fork-only ``playReal``, ``playpyvista``, ``playTracking`` and ``play_physicsGait`` scripts to the
  same structure. Run them as modules, for example
  ``python -m isaaclab_rl.entrypoints.backends.playReal --task <task> ...``, since they now use package-relative
  imports. Replace ``--use_pretrained_checkpoint`` with ``--checkpoint pretrained`` and ``--video`` clip control
  with ``--video_length`` / ``--video_interval``.
* Updated ``train_mimic.bat`` to call the ``isaaclab`` command (or ``uv run``) instead of ``isaaclab.bat``.

Removed
^^^^^^^

* Removed ``rsl_rl/utils.py`` (``handle_deprecated_rsl_rl_cfg`` and ``handle_deprecated_rsl_rl_checkpoint``) and
  its deprecation test, following upstream. Define runner configs with ``actor`` / ``critic`` models instead of
  the legacy ``policy`` block.
