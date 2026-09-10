# Binary-contact RL meeting archive: September 9, 2026

The [meeting release](https://github.com/psu-LiBR/IsaacLab-LiBRHexapod/releases/tag/robin-report-20260909)
contains the complete historical report bundle. Download and unzip
`hexapod-binary-rl-20260909.zip`, then open `START_HERE.html` locally.
The bundle keeps the original relative paths so its videos play alongside the reports.

Read the release's `AUDIT_NOTES.md` first. The original reports are historical
snapshots and remain unchanged; later code fixes do not retroactively alter their
experimental conditions. The archive contains 91 source files: six standalone
report HTML pages, speech notes, report-generation scripts and fragments, figures
and 11 JSON inputs, plus all 28 historical videos. Two videos whose names contain
`INVALID` are preserved only as invalidated historical material.

| Release asset | Purpose |
| --- | --- |
| `hexapod-binary-rl-20260909.zip` | Complete local-viewing bundle with original reports, data, scripts, and videos |
| `AUDIT_NOTES.md` | Qualifications to historical scientific claims |
| `README.md` | Entry points, provenance boundaries, and exclusions |
| `MANIFEST.json` | Per-file byte counts and SHA-256 hashes |
| `SHA256SUMS.txt` | Checksums for the downloadable archive and companion files |

The publication verifies archive integrity, JSON parsing, and local HTML
dependencies. It does not rerun the simulator or independently remeasure the
historical results. Training checkpoints and raw TensorBoard event files were
not present in the meeting-report directory and are outside this archive.
Result data and media are release attachments rather than source-tree files.

中文：下载发布附件 ZIP，解压后打开 `START_HERE.html`；先读勘误。
历史报告、数据及视频原样保留，后续奖励与 extquad 改动不回填到旧实验。
本次核验文件完整性及依赖，不代表重新完成历史仿真实验。
