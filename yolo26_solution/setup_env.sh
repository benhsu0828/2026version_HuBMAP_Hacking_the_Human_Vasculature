#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 用既有 root uv 環境(不另建 conda)
# -----------------------------------------------------------------------------
# root .venv 已含 ultralytics 8.4.51(內建 yolo26-seg)。本檔僅做 uv sync + 自檢,
# 與 mmdet_v3_solution 的 setup_env.sh 風格對齊,讓「一鍵驗證環境」有入口。
# 用法:bash yolo26_solution/setup_env.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] uv sync(root 環境)"
uv sync

echo "[setup] 自檢 ultralytics / yolo26-seg.yaml / CUDA"
uv run python - <<'PY'
import os, glob
import ultralytics
import torch
print("ultralytics :", ultralytics.__version__)
print("torch       :", torch.__version__, "| cuda:", torch.cuda.is_available())
base = os.path.dirname(ultralytics.__file__)
hit = glob.glob(base + "/cfg/models/**/yolo26-seg.yaml", recursive=True)
assert hit, "找不到 yolo26-seg.yaml — 請升級 ultralytics"
print("yolo26-seg  :", hit[0])
print("OK")
PY

echo "[setup] 完成。接著可執行:bash yolo26_solution/run_all.sh"
