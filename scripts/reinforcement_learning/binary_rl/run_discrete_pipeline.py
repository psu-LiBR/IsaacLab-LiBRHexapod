# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-shot train -> evaluate -> rank -> export pipeline for the binary-contact discrete-RL set.

Sequential (single-GPU) driver that, in one launch:

  1. trains each discrete-optimisation policy in turn -- DQN, Double-DQN, categorical PPO,
     masked PPO, SAC-D -- to ``--iterations`` trainer steps (equal env-step budget for
     every algorithm), checkpointing every ``--checkpoint-interval`` steps so the
     1500 / 1750 / 2000 checkpoints all land on disk with the defaults;
  2. evaluates the three selected checkpoints of every algorithm through the shared
     ``eval_protocol.py`` harness (64 envs, 300 steps, seed 7), one checkpoint per
     process by default (see ``--eval-fast``) so the rollouts cannot contaminate each
     other -- the isolation ``RESULTS_binary_rl.md`` documents as necessary;
  3. ranks every checkpoint by the harness forward-displacement metric (BL/cycle), only
     counting a checkpoint as "beats tripod" when it also has ``fall_rate == 0`` and
     survives the full window;
  4. exports every selected checkpoint to ONNX (``export_binary_onnx.py``) into one
     bundle folder -- the winner flagged ``BEST_*`` -- ready for on-robot testing;
  5. records an MP4 of the best checkpoint of every algorithm so they can be compared
     side by side (``play_discrete_closeup.py``; the overall winner is additionally
     flagged ``BEST_*``; ``--no-video`` to skip, ``--video-all`` for every ranked
     checkpoint);
  6. logs the training curves (each trainer's ``--wandb``), the evaluation table +
     per-policy reward-term breakdown, and the winner's video to W&B project ``discrete-RL``.

This script does NOT import Isaac Sim: it runs each stage script as its own
``sys.executable <script>`` subprocess (Isaac Sim can only launch once per process). The
pipeline is itself launched via ``isaaclab.bat -p`` so it already has the right
interpreter + environment; re-invoking ``isaaclab.bat`` for the stages does NOT nest.

Run from the repo root::

    isaaclab.bat -p scripts/reinforcement_learning/binary_rl/run_discrete_pipeline.py --iterations 2000

Resume after an interruption: run it again with the same ``--tag`` -- any algorithm whose
final checkpoint already exists is skipped unless ``--force`` is given. ``--skip-train`` /
``--skip-eval`` / ``--skip-export`` run individual stages.

Note on ``--iterations``: this is the skrl trainer-step count (one vectorised env step;
env-steps = iterations * num_envs). It is applied identically to every algorithm, so the
comparison is at an equal env-step budget -- PPO / masked PPO then run fewer gradient
updates than DQN / SAC-D (rollout length 96), which is the intended trade-off. The
equal-budget runs in ``RESULTS_binary_rl.md`` used 100000; 2000 is a fast shakedown, so
raise ``--iterations`` for a publication-grade comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # scripts/reinforcement_learning/binary_rl -> repo root

TASK = "Isaac-Goal-Flat-Hexapod-Binary-v0"
VIDEO_TASK = "Isaac-Goal-Flat-Hexapod-Binary-Play-v0"

# algo name -> (training script, extra CLI args). The name is also the run-folder name,
# the W&B group, and the eval/export label prefix. Edit this dict to change the set.
ALGOS: dict[str, tuple[str, list[str]]] = {
    "dqn": ("train_discrete.py", ["--algo", "dqn"]),
    "ddqn": ("train_discrete.py", ["--algo", "ddqn"]),
    "ppo": ("train_discrete_ppo.py", []),
    "ppo_masked": ("train_discrete_ppo.py", ["--mask"]),
    "sac_d": ("train_sac_d.py", []),
}


def launcher_prefix() -> list[str]:
    """How to run a stage script.

    The pipeline is launched with ``isaaclab.bat -p run_discrete_pipeline.py``, so this
    process already runs under the Isaac Sim venv interpreter (``sys.executable``) with
    the environment ``isaaclab`` set up. Each stage script does its own
    ``from isaaclab.app import AppLauncher`` and only needs that interpreter + environment
    -- so call it directly. Re-invoking ``isaaclab.bat`` here does NOT nest: its CLI
    wrapper re-runs this pipeline's argv instead of the stage script.
    """
    return [sys.executable]


def run_streamed(cmd: list[str], log_path: Path) -> int:
    """Run ``cmd`` from the repo root, teeing combined stdout/stderr to the console and ``log_path``.

    A background reader thread pumps the pipe so a stray Isaac Sim / Omniverse helper that
    inherits the stdout handle and lingers past exit cannot wedge the pipeline: once the
    child process itself exits, this returns within a few seconds regardless.
    """
    import threading

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(cmd)}\n  (log: {log_path})", flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None

        def _pump() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)

        reader = threading.Thread(target=_pump, daemon=True)
        reader.start()
        rc = proc.wait()
        reader.join(timeout=10)
        return rc


def selected_iters(iterations: int, interval: int, override: str) -> list[int]:
    """The checkpoint iterations to evaluate / export (default: the last three intervals)."""
    if override:
        return sorted({int(x) for x in override.replace(",", " ").split() if x.strip()})
    return [i for i in (iterations - 2 * interval, iterations - interval, iterations) if i > 0]


def resolve_ckpt(run_dir: Path, it: int, tol: int) -> Path | None:
    """The checkpoint for iteration ``it`` -- exact ``agent_<it>.pt``, else the nearest within ``tol``."""
    cdir = run_dir / "checkpoints"
    exact = cdir / f"agent_{it}.pt"
    if exact.exists():
        return exact
    cands: list[tuple[int, Path]] = []
    for f in cdir.glob("agent_*.pt"):
        stem = f.stem.split("_", 1)[1]
        if stem.isdigit():
            cands.append((abs(int(stem) - it), f))
    if cands:
        cands.sort()
        if cands[0][0] <= tol:
            return cands[0][1]
    return None


def latest_ckpt(run_dir: Path) -> Path | None:
    """The newest checkpoint in ``run_dir/checkpoints`` -- the highest numeric ``agent_<n>.pt``.

    Fallback for an algorithm whose training terminated early (e.g. the machine was shut
    down) and so has no checkpoint near a requested iteration. Non-numeric stems
    (``agent_final.pt``) do not contribute to the max but are returned if they are the only
    checkpoints present.
    """
    cdir = run_dir / "checkpoints"
    if not cdir.is_dir():
        return None
    numbered: list[tuple[int, Path]] = []
    others: list[Path] = []
    for f in cdir.glob("agent_*.pt"):
        stem = f.stem.split("_", 1)[1]
        (numbered.append((int(stem), f)) if stem.isdigit() else others.append(f))
    if numbered:
        return max(numbered)[1]
    return sorted(others)[-1] if others else None


# --------------------------------------------------------------------------------- train
def train_all(args: argparse.Namespace, pipe_dir: Path) -> dict[str, bool]:
    ok: dict[str, bool] = {}
    tol = max(1, args.checkpoint_interval // 2)
    for algo in args.algos:
        script, extra = ALGOS[algo]
        run_dir = pipe_dir / algo
        if not args.force and resolve_ckpt(run_dir, args.iterations, tol) is not None:
            print(f"[pipeline] {algo}: final checkpoint present -- skipping training (use --force to retrain)")
            ok[algo] = True
            continue
        cmd = [
            *launcher_prefix(),
            str(HERE / script),
            "--task",
            TASK,
            "--num_envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--timesteps",
            str(args.iterations),
            "--checkpoint_interval",
            str(args.checkpoint_interval),
            "--directory",
            str(pipe_dir),
            "--experiment_name",
            algo,
            *extra,
        ]
        if args.wandb:
            cmd += [
                "--wandb",
                "--wandb_project",
                args.wandb_project,
                "--wandb_group",
                algo,
                "--wandb_name",
                f"{args.tag}-{algo}",
            ]
        t0 = time.time()
        rc = run_streamed(cmd, run_dir / "train.log")
        final_ok = resolve_ckpt(run_dir, args.iterations, tol) is not None
        ok[algo] = rc == 0 and final_ok
        print(
            f"[pipeline] {algo}: exit={rc} final_ckpt={'ok' if final_ok else 'MISSING'} "
            f"({(time.time() - t0) / 60:.1f} min)"
        )
        if not ok[algo] and not args.keep_going:
            raise SystemExit(f"[pipeline] {algo} training failed and --no-keep-going is set; stopping")
    return ok


# ---------------------------------------------------------------------------------- eval
def _eval_cmd(args: argparse.Namespace, out_json: Path, policy_args: list[str], baselines: bool) -> list[str]:
    cmd = [
        *launcher_prefix(),
        str(HERE / "eval_protocol.py"),
        "--task",
        TASK,
        "--num_envs",
        str(args.eval_num_envs),
        "--steps",
        str(args.eval_steps),
        "--seed",
        "7",
        "--warmup",
        "1",
        "--goal_distance",
        str(args.goal_distance),
        "--out",
        str(out_json),
        *policy_args,
    ]
    if not baselines:
        cmd.append("--no_baselines")
    return cmd


def evaluate(args: argparse.Namespace, pipe_dir: Path, iters: list[int]) -> Path | None:
    """Evaluate every selected checkpoint; return the merged results JSON path."""
    eval_dir = pipe_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    merged = pipe_dir / "eval_results.json"
    tol = max(1, args.checkpoint_interval // 2)

    found: list[tuple[str, int, Path]] = []
    for algo in args.algos:
        seen: set[Path] = set()
        for it in iters:
            p = resolve_ckpt(pipe_dir / algo, it, tol)
            if p is None:
                # training may have stopped early -- fall back to this algo's newest checkpoint
                p = latest_ckpt(pipe_dir / algo)
                if p is None:
                    print(f"[pipeline] eval: no checkpoint near iter {it} for {algo} -- skipping")
                    continue
                print(f"[pipeline] eval: {algo} has no ckpt near {it}; using latest {p.name} instead")
            if p in seen:
                continue  # several requested iters can resolve to the same fallback checkpoint
            seen.add(p)
            stem = p.stem.split("_", 1)[1]
            found.append((algo, int(stem) if stem.isdigit() else it, p))
    if not found:
        print("[pipeline] eval: no checkpoints found; skipping evaluation")
        return None

    all_results: list[dict] = []
    protocol: dict | None = None

    if args.eval_fast:
        # one process: baselines + every checkpoint (fast, less isolated)
        policy_args = [x for algo, it, p in found for x in ("--policy", "net", f"{algo}@{it}", str(p))]
        rc = run_streamed(_eval_cmd(args, merged, policy_args, baselines=True), pipe_dir / "eval.log")
        if not merged.exists():
            print(f"[pipeline] eval: exit={rc}, {merged} MISSING")
            return None
        return merged

    # default: baselines once, then one process per checkpoint (each a "first rollout of
    # its own process", the cleanest / most comparable number -- see RESULTS_binary_rl.md)
    base_json = eval_dir / "_baselines.json"
    run_streamed(_eval_cmd(args, base_json, [], baselines=True), eval_dir / "_baselines.log")
    if base_json.exists():
        d = json.loads(base_json.read_text())
        protocol = d.get("protocol")
        all_results += d.get("results", [])

    for algo, it, p in found:
        name = f"{algo}@{it}"
        oj = eval_dir / f"{algo}_{it}.json"
        policy_args = ["--policy", "net", name, str(p)]
        run_streamed(_eval_cmd(args, oj, policy_args, baselines=False), eval_dir / f"{algo}_{it}.log")
        if oj.exists():
            d = json.loads(oj.read_text())
            protocol = protocol or d.get("protocol")
            all_results += [r for r in d.get("results", []) if r.get("policy") == name]
        else:
            all_results.append({"policy": name, "error": "eval produced no output"})

    merged.write_text(json.dumps({"protocol": protocol, "results": all_results}, indent=2))
    return merged


def rank(eval_json: Path) -> tuple[list[dict], dict | None, dict]:
    """Rank the policy checkpoints: beats-tripod first, then BL/cycle.

    Returns ``(rows, winner, anchors)`` where ``anchors`` maps each ``BASE_*`` baseline
    to its BL/cycle (the tripod entry is the comparison reference).
    """
    data = json.loads(eval_json.read_text())
    results = data.get("results", [])
    anchors = {
        r["policy"]: r.get("x_disp_BL_per_cycle") for r in results if str(r.get("policy", "")).startswith("BASE_")
    }
    tripod_bl = anchors.get("BASE_tripod_csv_bits")
    rows: list[dict] = []
    for r in results:
        name = r.get("policy", "")
        if "@" not in name or "error" in r:
            continue
        algo, _, it = name.partition("@")
        bl = r.get("x_disp_BL_per_cycle")
        window_s, surv, fall = r.get("window_s"), r.get("survival_s"), r.get("fall_rate")
        legit = (
            bl is not None
            and fall == 0
            and window_s is not None
            and surv is not None
            and surv >= window_s - 1e-6
            and (tripod_bl is None or bl > tripod_bl)
        )
        rows.append(
            {
                "policy": name,
                "algo": algo,
                "iter": int(it),
                "bl_per_cycle": bl,
                "vs_tripod_pct": (
                    round(100.0 * bl / tripod_bl, 1) if (bl is not None and tripod_bl not in (None, 0)) else None
                ),
                "x_displacement_m": r.get("x_displacement_m"),
                "reward_per_step": r.get("reward_per_step"),
                "fall_rate": fall,
                "survival_s": surv,
                "action_entropy_nats": r.get("action_entropy_nats"),
                "mean_stance_legs": r.get("mean_stance_legs"),
                "frac_5plus_stance": r.get("frac_5plus_stance"),
                "beats_tripod": bool(legit),
                "reward_terms": r.get("reward_terms", {}),
                "checkpoint": r.get("checkpoint"),
            }
        )
    rows.sort(
        key=lambda d: (d["beats_tripod"], d["bl_per_cycle"] if d["bl_per_cycle"] is not None else -1e9),
        reverse=True,
    )
    return rows, (rows[0] if rows else None), anchors


# -------------------------------------------------------------------------------- export
def export_all(args: argparse.Namespace, pipe_dir: Path, rows: list[dict], best: dict | None, anchors: dict) -> Path:
    bundle = pipe_dir / "onnx_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for algo in args.algos:
        rm = pipe_dir / algo / "run_meta.json"
        if rm.exists():
            shutil.copy2(rm, bundle / f"{algo}_run_meta.json")

    exported: list[dict] = []
    for r in rows:  # ranked order
        ckpt = Path(r["checkpoint"]) if r.get("checkpoint") else None
        if ckpt is None or not ckpt.exists():
            print(f"[pipeline] export: missing checkpoint for {r['policy']}; skipping")
            continue
        onnx_name = f"{r['algo']}_{r['iter']}.onnx"
        cmd = [
            *launcher_prefix(),
            str(HERE / "export_binary_onnx.py"),
            "--checkpoint",
            str(ckpt),
            "--out",
            str(bundle / onnx_name),
        ]
        if args.export_no_check:
            cmd.append("--no_check")
        rc = run_streamed(cmd, bundle / f"{r['algo']}_{r['iter']}.export.log")
        entry = {
            **{k: r[k] for k in ("policy", "algo", "iter", "bl_per_cycle", "fall_rate", "beats_tripod")},
            "onnx": onnx_name,
            "export_ok": rc == 0 and (bundle / onnx_name).exists(),
        }
        exported.append(entry)
        if best is not None and r["policy"] == best["policy"] and entry["export_ok"]:
            shutil.copy2(bundle / onnx_name, bundle / f"BEST_{onnx_name}")

    manifest = {
        "task": TASK,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "iterations": args.iterations,
        "best_policy": best["policy"] if best else None,
        "tripod_bl_per_cycle": anchors.get("BASE_tripod_csv_bits"),
        "baseline_anchors_bl_per_cycle": anchors,
        "ranking": [{k: v for k, v in r.items() if k != "reward_terms"} for r in rows],
        "reward_terms": {r["policy"]: r["reward_terms"] for r in rows},
        "onnx_files": exported,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_ranking_csv(bundle / "ranking.csv", rows, anchors)
    _write_bundle_readme(bundle / "README.txt", rows, best, anchors)
    _write_validate_script(bundle / "validate_all.sh", rows)
    return bundle


# --------------------------------------------------------------------------------- video
def record_videos(args: argparse.Namespace, rows: list[dict], best: dict | None, bundle: Path) -> list[Path]:
    """Render one MP4 per algorithm's best checkpoint (or every ranked one with ``--video-all``).

    Default: the highest-ranked evaluated checkpoint of each algorithm -- ``rows`` is
    already sorted best-first, so the first row seen per ``algo`` is that algorithm's best.
    Each render is named ``<algo>_<iter>.mp4`` for easy side-by-side comparison; the overall
    winner is additionally copied to ``BEST_<algo>_<iter>.mp4``. ``--video-all`` renders
    every ranked checkpoint instead.

    Uses ``play_discrete_closeup.py`` -- greedy argmax on the Play env with a follow
    camera. Non-fatal: a missing MP4 (e.g. no ffmpeg) is warned and skipped. The
    masked-PPO legal set is not re-applied here (a trained masked policy rarely argmaxes
    to an illegal action; the ONNX export in the bundle still enforces it).
    """
    vids = bundle / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    if args.video_all:
        targets = rows
    else:
        # one per algorithm: first row per algo == that algo's highest-ranked checkpoint
        per_algo: dict[str, dict] = {}
        for r in rows:
            per_algo.setdefault(r["algo"], r)
        targets = list(per_algo.values())
    made: list[Path] = []
    for r in targets:
        ckpt = Path(r["checkpoint"]) if r.get("checkpoint") else None
        if ckpt is None or not ckpt.exists():
            print(f"[pipeline] video: missing checkpoint for {r['policy']}; skipping")
            continue
        tag = f"{r['algo']}_{r['iter']}"
        stage_dir = vids / tag
        cmd = [
            *launcher_prefix(),
            str(HERE / "play_discrete_closeup.py"),
            "--task",
            VIDEO_TASK,
            "--checkpoint",
            str(ckpt),
            "--num_envs",
            str(args.video_num_envs),
            "--steps",
            str(args.video_length),
            "--video_length",
            str(args.video_length),
            "--seed",
            "7",
            "--out_dir",
            str(stage_dir),
        ]
        rc = run_streamed(cmd, vids / f"{tag}.log")
        mp4s = sorted(stage_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime) if stage_dir.exists() else []
        if not mp4s:
            print(f"[pipeline] video: no MP4 for {r['policy']} (rc={rc}); see {vids / (tag + '.log')}")
            continue
        prefix = "BEST_" if (best and r["policy"] == best["policy"]) else ""
        dest = bundle / f"{prefix}{tag}.mp4"
        shutil.copy2(mp4s[-1], dest)
        made.append(dest)
        print(f"[pipeline] video: {dest}")
    return made


def _write_ranking_csv(path: Path, rows: list[dict], anchors: dict) -> None:
    cols = [
        "rank",
        "policy",
        "algo",
        "iter",
        "bl_per_cycle",
        "vs_tripod_pct",
        "x_displacement_m",
        "reward_per_step",
        "fall_rate",
        "survival_s",
        "action_entropy_nats",
        "mean_stance_legs",
        "frac_5plus_stance",
        "beats_tripod",
    ]
    lines = [",".join(cols)]
    # baseline anchors first, as rank-0 reference rows (tripod is the comparison point)
    tripod_bl = anchors.get("BASE_tripod_csv_bits")
    for name, bl in anchors.items():
        pct = round(100.0 * bl / tripod_bl, 1) if (bl is not None and tripod_bl not in (None, 0)) else None
        vals = {"rank": 0, "policy": name, "algo": "baseline", "bl_per_cycle": bl, "vs_tripod_pct": pct}
        lines.append(",".join(str(vals.get(c, "")) for c in cols))
    for i, r in enumerate(rows, 1):
        lines.append(",".join(str(v) for v in [i, *[r.get(c) for c in cols[1:]]]))
    path.write_text("\n".join(lines) + "\n")


def _write_bundle_readme(path: Path, rows: list[dict], best: dict | None, anchors: dict) -> None:
    tripod_bl = anchors.get("BASE_tripod_csv_bits")
    lines = [
        "Binary-contact discrete-RL ONNX bundle",
        "=" * 38,
        "",
        f"Best policy: {best['policy'] if best else 'n/a'}"
        + (
            f"   ({best['bl_per_cycle']} BL/cycle = {best.get('vs_tripod_pct')}% of tripod, "
            f"beats_tripod={best['beats_tripod']})"
            if best
            else ""
        ),
        f"Tripod reference this run: {tripod_bl} BL/cycle",
        "All anchors this run: "
        + ", ".join(f"{k[5:] if k.startswith('BASE_') else k}={v}" for k, v in anchors.items()),
        "",
        "Ranking (eval_protocol.py; 64 envs x 300 steps = 6 gait cycles; seed 7; one",
        "checkpoint per process). BL/cycle = forward displacement (m) / 0.315 m / 6 cycles.",
        "A checkpoint only 'beats tripod' if fall_rate == 0, it survives the full 6.00 s,",
        "and BL/cycle exceeds the re-measured tripod anchor (see ranking.csv / manifest.json).",
        "Low action_entropy_nats + high frac_5plus_stance == the policy is just standing.",
        "",
        f"  {'rank':>4}  {'policy':<16} {'BL/cycle':>9} {'%trip':>6} {'fall':>6} {'entropy':>8} {'beats':>6}  onnx",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"  {i:>4}  {r['policy']:<16} {r['bl_per_cycle']!s:>9} {r.get('vs_tripod_pct')!s:>6} "
            f"{r['fall_rate']!s:>6} {r['action_entropy_nats']!s:>8} {str(r['beats_tripod']):>6}  "
            f"{r['algo']}_{r['iter']}.onnx"
        )
    lines += [
        "",
        "Test one on the robot (from scripts/sim2real_transfer/, plain Python, no Isaac Sim):",
        "",
        "  1. cp config/deployment.binary.example.yaml config/deployment.yaml  and calibrate it",
        "     (every '# CALIBRATE' field: leg encoder zero_tick, IMU mount_offset_quat).",
        "  2. python tools/validate_onnx.py --mode direct --profile binary \\",
        "         --policy <this_folder>/<file>.onnx --trace <sim_trace.csv>",
        "  3. python run_policy.py --policy <this_folder>/<file>.onnx --profile binary \\",
        "         --config config/deployment.yaml --dry-run --fake-imu --duration 5",
        "  4. then the --no-torque / propped / --action-scale-mult 0.2 bring-up ladder in",
        "     scripts/sim2real_transfer/README.md and CLAUDE.md 'Sim-to-Real Deployment'.",
        "",
        "validate_all.sh runs step 2 for every .onnx here (edit TRACE at the top first).",
        "<algo>_run_meta.json records obs/action dims; for ppo_masked it also carries the",
        "legal action set folded into the ONNX graph.",
        "",
        "<algo>_<iter>.mp4 (in this folder, if --video was on) is a sim replay of each",
        "algorithm's best checkpoint for side-by-side comparison; the overall winner is also",
        "copied to BEST_<algo>_<iter>.mp4. videos/ holds the raw renders and per-render logs.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_validate_script(path: Path, rows: list[dict]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "# Offline-validate every policy in this bundle against a sim-recorded trace.",
        "# Run from scripts/sim2real_transfer/ .  Set TRACE to your recorded obs/action CSV.",
        "set -euo pipefail",
        'TRACE="${1:-sim_trace.csv}"',
        'BUNDLE="$(cd "$(dirname "$0")" && pwd)"',
        "",
    ]
    for r in rows:
        f = f"{r['algo']}_{r['iter']}.onnx"
        lines.append(f'echo "=== {f}  (rank: BL/cycle {r["bl_per_cycle"]}, beats_tripod {r["beats_tripod"]}) ==="')
        lines.append(
            f'python tools/validate_onnx.py --mode direct --profile binary --policy "$BUNDLE/{f}" --trace "$TRACE"'
        )
        lines.append("")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------------- wandb
def log_eval_to_wandb(
    args: argparse.Namespace,
    rows: list[dict],
    best: dict | None,
    eval_json: Path,
    videos: list[Path] | None = None,
) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        print("[pipeline] wandb not installed; skipping eval upload")
        return
    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.tag}-eval",
        group="eval",
        job_type="eval",
        config={"iterations": args.iterations, "algos": args.algos, "tag": args.tag},
    )
    cols = [
        "rank",
        "policy",
        "algo",
        "iter",
        "bl_per_cycle",
        "x_displacement_m",
        "reward_per_step",
        "fall_rate",
        "survival_s",
        "action_entropy_nats",
        "mean_stance_legs",
        "frac_5plus_stance",
        "beats_tripod",
    ]
    table = wandb.Table(columns=cols)
    for i, r in enumerate(rows, 1):
        table.add_data(i, *[r.get(c) for c in cols[1:]])
    run.log({"eval/ranking": table})
    for r in rows:
        for term, val in (r.get("reward_terms") or {}).items():
            run.log({f"eval_reward_terms/{r['policy']}/{term}": val})
    if best:
        run.summary["best_policy"] = best["policy"]
        run.summary["best_bl_per_cycle"] = best["bl_per_cycle"]
        run.summary["best_beats_tripod"] = best["beats_tripod"]
    for v in videos or []:
        try:
            key = "eval/best_video" if v.name.startswith("BEST_") else f"eval/video_{v.stem}"
            run.log({key: wandb.Video(str(v), format="mp4")})
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] wandb: could not attach {v.name}: {exc}")
    art = wandb.Artifact(f"{args.tag}-eval", type="evaluation")
    art.add_file(str(eval_json))
    run.log_artifact(art)
    run.finish()


# ---------------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=2000, help="trainer steps per algorithm (skrl --timesteps)")
    p.add_argument("--checkpoint-interval", type=int, default=250, help="checkpoint every N steps")
    p.add_argument(
        "--save-iters", default="", help="iterations to eval/export (default: last 3 intervals -> 1500,1750,2000)"
    )
    p.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--goal-distance", type=float, default=2.0)
    p.add_argument("--eval-num-envs", type=int, default=64)
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument(
        "--eval-fast", action="store_true", help="evaluate every checkpoint in one process (faster, less isolated)"
    )
    p.add_argument("--tag", default="", help="pipeline run tag / output subfolder (default: pipeline_<timestamp>)")
    p.add_argument(
        "--out-root", default=str(REPO_ROOT / "runs_binary"), help="parent dir for the pipeline output folder"
    )
    p.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True, help="log to W&B (project --wandb-project)"
    )
    p.add_argument("--wandb-project", default="discrete-RL")
    p.add_argument("--force", action="store_true", help="retrain even if the final checkpoint exists")
    p.add_argument(
        "--keep-going", action=argparse.BooleanOptionalAction, default=True, help="continue if one algorithm fails"
    )
    p.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render an MP4 of the best checkpoint of every algorithm (play_discrete_closeup.py)",
    )
    p.add_argument(
        "--video-all", action="store_true", help="render every ranked checkpoint, not just the best-per-algorithm"
    )
    p.add_argument("--video-length", type=int, default=600, help="video length in env steps (600 = 12 s at 50 Hz)")
    p.add_argument("--video-num-envs", type=int, default=16)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--export-no-check", action="store_true", help="pass --no_check to export_binary_onnx.py")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="fast end-to-end shakedown of EVERY algo (tiny run, one eval process, no video/W&B) -- "
        "checks each optimizer trains/evals/exports without error; not a real comparison. "
        "Any of the tuned flags you pass explicitly still win.",
    )
    args = p.parse_args()

    if args.smoke:

        def _explicit(*names: str) -> bool:
            return any(a == n or a.startswith(n + "=") for a in sys.argv[1:] for n in names)

        if not _explicit("--iterations"):
            args.iterations = 200
        if not _explicit("--checkpoint-interval"):
            args.checkpoint_interval = 100
        if not _explicit("--save-iters"):
            args.save_iters = str(args.iterations)
        if not _explicit("--num-envs"):
            args.num_envs = 256
        if not _explicit("--eval-fast", "--no-eval-fast"):
            args.eval_fast = True
        if not _explicit("--eval-num-envs"):
            args.eval_num_envs = 32
        if not _explicit("--eval-steps"):
            args.eval_steps = 60
        if not _explicit("--video", "--no-video"):
            args.video = False
        if not _explicit("--wandb", "--no-wandb"):
            args.wandb = False
        if not _explicit("--tag"):
            args.tag = "smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        print("[pipeline] SMOKE MODE: fast shakedown of every algo -- NOT a real comparison")

    if not args.tag:
        args.tag = "pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    pipe_dir = Path(args.out_root) / args.tag
    pipe_dir.mkdir(parents=True, exist_ok=True)
    iters = selected_iters(args.iterations, args.checkpoint_interval, args.save_iters)

    print(f"[pipeline] tag          : {args.tag}")
    print(f"[pipeline] output dir   : {pipe_dir}")
    print(f"[pipeline] algorithms   : {', '.join(args.algos)}")
    print(f"[pipeline] iterations   : {args.iterations}  (env-steps ~= {args.iterations * args.num_envs:,})")
    print(f"[pipeline] checkpoints  : every {args.checkpoint_interval}; eval/export at {iters}")
    print(f"[pipeline] W&B          : {'project ' + args.wandb_project if args.wandb else 'disabled'}")
    if args.iterations < 20000:
        print(
            "[pipeline] NOTE: --iterations is well below the 100000 used for the equal-budget\n"
            "[pipeline]       comparison in RESULTS_binary_rl.md; expect under-trained policies\n"
            "[pipeline]       (PPO especially). Raise --iterations for a real comparison."
        )
    (pipe_dir / "pipeline_config.json").write_text(json.dumps({**vars(args), "selected_iters": iters}, indent=2))

    train_ok = {a: True for a in args.algos}
    if not args.skip_train:
        train_ok = train_all(args, pipe_dir)

    eval_json = pipe_dir / "eval_results.json"
    if args.skip_eval:
        if not eval_json.exists():
            print("[pipeline] --skip-eval but no eval_results.json to reuse; stopping after training")
            return
        print(f"[pipeline] --skip-eval: reusing {eval_json} for ranking / export / video")
    else:
        eval_json = evaluate(args, pipe_dir, iters)
        if eval_json is None:
            raise SystemExit("[pipeline] evaluation produced no results; cannot rank / export")
    rows, best, anchors = rank(eval_json)
    tripod_bl = anchors.get("BASE_tripod_csv_bits")

    print(f"\n[pipeline] ===== ranking (tripod reference: {tripod_bl} BL/cycle) =====")
    for i, r in enumerate(rows, 1):
        print(
            f"  {i:>2}. {r['policy']:<16} {r['bl_per_cycle']!s:>9} BL/cyc  ({r.get('vs_tripod_pct')}% of tripod)  "
            f"fall={r['fall_rate']}  H={r['action_entropy_nats']}  beats_tripod={r['beats_tripod']}"
        )
    if best:
        print(
            f"[pipeline] winner: {best['policy']}  "
            f"({best['bl_per_cycle']} BL/cycle = {best.get('vs_tripod_pct')}% of tripod, "
            f"beats_tripod={best['beats_tripod']})"
        )

    bundle = pipe_dir / "onnx_bundle"
    if args.skip_export:
        bundle.mkdir(parents=True, exist_ok=True)
        _write_ranking_csv(bundle / "ranking.csv", rows, anchors)
        _write_bundle_readme(bundle / "README.txt", rows, best, anchors)
        print("[pipeline] --skip-export: ONNX export skipped; ranking.csv / README.txt still written")
    else:
        bundle = export_all(args, pipe_dir, rows, best, anchors)

    videos: list[Path] = []
    if args.video:
        videos = record_videos(args, rows, best, bundle)

    log_eval_to_wandb(args, rows, best, eval_json, videos)

    print(f"\n[pipeline] DONE. output: {bundle}")
    if not args.skip_export:
        print(f"[pipeline]   manifest : {bundle / 'manifest.json'}")
        print(f"[pipeline]   ranking  : {bundle / 'ranking.csv'}")
        print(f"[pipeline]   how-to   : {bundle / 'README.txt'}")
    for v in videos:
        print(f"[pipeline]   video    : {v}")
    failed = [a for a, v in train_ok.items() if not v]
    if failed:
        print(f"[pipeline] NOTE: training did not complete cleanly for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
