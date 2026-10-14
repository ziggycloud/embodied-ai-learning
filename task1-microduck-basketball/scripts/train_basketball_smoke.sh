#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_DIR/microduck-playground"
export WANDB_MODE=disabled
uv run --locked python "$TASK_DIR/scripts/run_warp_compat.py"   scripts/finetune_basketball.py   "${BASKETBALL_CHECKPOINT:-artifacts/basketball/checkpoint.pt}"   --run-name basketball-no-mempool-smoke   --num-envs 64 --iterations 5 --learning-rate 2e-5   --push-interval-s 1.5 3 --save-interval 5
