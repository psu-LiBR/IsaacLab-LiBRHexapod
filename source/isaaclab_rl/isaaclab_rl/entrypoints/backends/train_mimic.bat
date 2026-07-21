@echo off
REM ============================================================================
REM  train_mimic.bat — Two-phase hexapod training: Imitation → RL fine-tune
REM ============================================================================
REM
REM  PHASE 1:  Train with Isaac-Velocity-Flat-Hexapod-Mimic-v0.
REM            The joint_pos_imitation reward guides the policy toward the
REM            reference gait.  After MIMIC_ITERATIONS the curriculum
REM            automatically zeroes the imitation reward, so training
REM            continues as pure RL within the same run.
REM
REM  Usage — single integrated run (recommended):
REM    train_mimic.bat
REM
REM  Usage — explicit two-phase with separate RL fine-tune:
REM    train_mimic.bat --two-phase
REM    This runs Phase 1 to completion, then automatically resumes with the
REM    standard flat env for an additional RL fine-tuning session.
REM
REM  Common overrides:
REM    set NUM_ENVS=2048          (default: 4096)
REM    set MIMIC_ITERS=600        (default: 3000, covers both phases)
REM    set RL_ITERS=2000          (only used in --two-phase mode)
REM    set CSV_PATH=C:\path\to\ref.csv   (override reference gait CSV)
REM
REM  Requirements:
REM    - isaaclab.bat must be on PATH or this script must be run from the
REM      root of the Isaac Lab repository (where isaaclab.bat lives).
REM    - Hexapod USD and Isaac Sim must be configured (see CLAUDE.md).
REM ============================================================================

setlocal EnableDelayedExpansion

REM --- Defaults (override via environment variable or edit here) --------------
if not defined NUM_ENVS    set NUM_ENVS=4096
if not defined MIMIC_ITERS set MIMIC_ITERS=3000
if not defined RL_ITERS    set RL_ITERS=2000
if not defined LOG_ROOT    set LOG_ROOT=logs\rsl_rl\hexapod_mimic

REM Locate isaaclab.bat (try repo root, then PATH).
set ISAACLAB=isaaclab.bat
if not exist "%ISAACLAB%" (
    echo [train_mimic] ERROR: isaaclab.bat not found. Run this script from the
    echo               Isaac Lab repository root, or add it to PATH.
    exit /b 1
)

REM ============================================================================
REM  Single integrated run  (default, no --two-phase flag)
REM ============================================================================
if /I not "%~1"=="--two-phase" (
    echo.
    echo ========================================================
    echo  Hexapod Mimic + RL  ^(integrated^)
    echo    Env:        Isaac-Velocity-Flat-Hexapod-Mimic-v0
    echo    Num envs:   %NUM_ENVS%
    echo    Max iters:  %MIMIC_ITERS%
    echo    Phase 1 imitation weight decays automatically at
    echo    iteration ~800  ^(see hexapod_mimic_env_cfg.py^).
    echo ========================================================
    echo.

    call %ISAACLAB% train --rl_library rsl_rl ^
        --task Isaac-Velocity-Flat-Hexapod-Mimic-v0 ^
        --num_envs %NUM_ENVS% ^
        --max_iterations %MIMIC_ITERS%

    if errorlevel 1 (
        echo [train_mimic] Phase 1 training failed.
        exit /b 1
    )
    echo.
    echo [train_mimic] Training complete. Checkpoint saved under %LOG_ROOT%\.
    echo To evaluate:
    echo   isaaclab.bat play --rl_library rsl_rl ^
        --task Isaac-Velocity-Flat-Hexapod-Mimic-Play-v0 --num_envs 1
    exit /b 0
)

REM ============================================================================
REM  Explicit two-phase run  (--two-phase flag)
REM ============================================================================
echo.
echo ========================================================
echo  Hexapod Mimic + RL  ^(two-phase^)
echo    Phase 1: Isaac-Velocity-Flat-Hexapod-Mimic-v0
echo             %MIMIC_ITERS% iterations ^(mimic phase ends
echo             automatically at ~800 iters via curriculum^)
echo    Phase 2: Isaac-Velocity-Flat-Hexapod-v0
echo             %RL_ITERS% additional iterations
echo ========================================================
echo.

REM ---- Phase 1: Mimic --------------------------------------------------------
echo [train_mimic] Starting Phase 1 — imitation warm-up ...
call %ISAACLAB% train --rl_library rsl_rl ^
    --task Isaac-Velocity-Flat-Hexapod-Mimic-v0 ^
    --num_envs %NUM_ENVS% ^
    --max_iterations %MIMIC_ITERS%

if errorlevel 1 (
    echo [train_mimic] Phase 1 failed.
    exit /b 1
)
echo [train_mimic] Phase 1 complete.

REM ---- Locate latest checkpoint from Phase 1 ---------------------------------
REM The runner saves to logs\rsl_rl\hexapod_mimic\<timestamp>\model_<iter>.pt
REM We sort by date and pick the most recent model_*.pt file.
set LATEST_CKPT=
for /f "delims=" %%F in (
    'dir /b /s /o-d "%LOG_ROOT%\model_*.pt" 2^>nul'
) do (
    if not defined LATEST_CKPT set LATEST_CKPT=%%F
)

if not defined LATEST_CKPT (
    echo [train_mimic] WARNING: Could not auto-locate checkpoint under %LOG_ROOT%.
    echo               Start Phase 2 manually:
    echo   isaaclab.bat train --rl_library rsl_rl ^
        --task Isaac-Velocity-Flat-Hexapod-v0 --checkpoint ^<path^>
    exit /b 1
)
echo [train_mimic] Using checkpoint: %LATEST_CKPT%

REM ---- Phase 2: RL fine-tune -------------------------------------------------
echo.
echo [train_mimic] Starting Phase 2 — RL fine-tuning from mimic checkpoint ...
call %ISAACLAB% train --rl_library rsl_rl ^
    --task Isaac-Velocity-Flat-Hexapod-v0 ^
    --num_envs %NUM_ENVS% ^
    --max_iterations %RL_ITERS% ^
    --checkpoint "%LATEST_CKPT%"

if errorlevel 1 (
    echo [train_mimic] Phase 2 failed.
    exit /b 1
)

echo.
echo [train_mimic] Two-phase training complete.
echo   Phase 1 log: %LOG_ROOT%
echo   Phase 2 log: logs\rsl_rl\hexapod_flat\
endlocal
