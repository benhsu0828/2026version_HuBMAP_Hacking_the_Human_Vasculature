#!/usr/bin/env bash
# =============================================================================
<<'COMMENT'
run_all.sh — YOLO26-seg 兩階段一鍵訓練(走 root uv 環境,不另建 conda)
  Stage1(全部 ds2 粗練, 預設 15ep, lr 2e-3, 輕度 aug)
    └─ 自動接力 best.pt ─┐
  Stage2(ds1 精練, 預設 25ep, lr 5e-4→~5e-8, 重度 aug)× N folds

全程 `uv run`(root .venv 內含 ultralytics 8.4.51 / yolo26-seg)。

用法:
  bash yolo26_solution/run_all.sh                      # prepare + Stage1 + Stage2 ×4
  bash yolo26_solution/run_all.sh --folds 1            # 單模型(快速驗證兩階段)
  bash yolo26_solution/run_all.sh --skip-prepare       # 跳過資料前處理
  bash yolo26_solution/run_all.sh --skip-stage1 --stage1-ckpt <best.pt>   # 只跑 Stage2
  bash yolo26_solution/run_all.sh --folds 1 --epochs1 1 --epochs2 1       # smoke test
COMMENT
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- 預設參數 --------------------------------------------------------------
FOLDS=4
SKIP_PREPARE=0
SKIP_STAGE1=0
STAGE1_CKPT=""
EPOCHS1=""
EPOCHS2=""
DEVICE="0"
DATA_ROOT="data"
DATA_OUT="yolo26_solution/yolo26_data"
PRETRAINED="${PRETRAINED:-yolo26x-seg.pt}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folds)        FOLDS="$2"; shift 2 ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    --skip-stage1)  SKIP_STAGE1=1; shift ;;
    --stage1-ckpt)  STAGE1_CKPT="$2"; shift 2 ;;
    --epochs1)      EPOCHS1="$2"; shift 2 ;;
    --epochs2)      EPOCHS2="$2"; shift 2 ;;
    --device)       DEVICE="$2"; shift 2 ;;
    --data-root)    DATA_ROOT="$2"; shift 2 ;;
    *) echo "未知參數: $1"; exit 1 ;;
  esac
done

# 找某次 run 的最佳(優先)或最後權重
find_ckpt() {
  local run_dir="$1"
  if [ -f "$run_dir/weights/best.pt" ]; then echo "$run_dir/weights/best.pt"; return; fi
  if [ -f "$run_dir/weights/last.pt" ]; then echo "$run_dir/weights/last.pt"; return; fi
  echo ""
}

# 解析 Stage1 權重:先精確路徑,找不到再 glob 挑最新的 pretrain*(救回 ultralytics 自動加序號的目錄)
resolve_stage1_ckpt() {
  local exact; exact="$(find_ckpt "$S1_DIR")"
  if [ -n "$exact" ]; then echo "$exact"; return; fi
  local newest
  newest="$(ls -t yolo26_solution/runs/stage1/pretrain*/weights/best.pt 2>/dev/null | head -n1 || true)"
  [ -z "$newest" ] && newest="$(ls -t yolo26_solution/runs/stage1/pretrain*/weights/last.pt 2>/dev/null | head -n1 || true)"
  echo "$newest"
}

# ---- 0) 資料前處理 ----------------------------------------------------------
if [ "$SKIP_PREPARE" -eq 0 ]; then
  echo "==> [0] 產生兩階段 YOLO 資料(folds=$FOLDS)"
  uv run python yolo26_solution/prepare_data.py \
    --data-root "$DATA_ROOT" --out "$DATA_OUT" --folds "$FOLDS"
fi

# ---- 1) Stage1 粗練(全 ds2)------------------------------------------------
S1_DIR="yolo26_solution/runs/stage1/pretrain"
if [ "$SKIP_STAGE1" -eq 0 ]; then
  echo "==> [1] Stage1 粗練(全部 ds2)"
  cmd=(uv run python yolo26_solution/train_yolo.py --stage 1
       --data "$DATA_OUT/stage1.yaml" --weights "$PRETRAINED"
       --name pretrain --device "$DEVICE")
  [ -n "$EPOCHS1" ] && cmd+=(--epochs "$EPOCHS1")
  echo "    ${cmd[*]}"
  "${cmd[@]}"
  STAGE1_CKPT="$(resolve_stage1_ckpt)"
fi

# ---- 解析 Stage1 接力權重 ---------------------------------------------------
if [ -z "$STAGE1_CKPT" ]; then STAGE1_CKPT="$(resolve_stage1_ckpt)"; fi
if [ -z "$STAGE1_CKPT" ] || [ ! -f "$STAGE1_CKPT" ]; then
  echo "找不到 Stage1 權重,請用 --stage1-ckpt 指定。"; exit 1
fi
echo "==> Stage2 將載入 Stage1 權重:$STAGE1_CKPT"

# ---- 2) Stage2 精練(ds1)× folds -------------------------------------------
last_fold=$((FOLDS - 1))
for K in $(seq 0 "$last_fold"); do
  echo "==> [2] Stage2 精練 fold$K(ds1)"
  cmd=(uv run python yolo26_solution/train_yolo.py --stage 2 --fold "$K"
       --data "$DATA_OUT/stage2_fold${K}.yaml" --weights "$STAGE1_CKPT"
       --device "$DEVICE")
  [ -n "$EPOCHS2" ] && cmd+=(--epochs "$EPOCHS2")
  echo "    ${cmd[*]}"
  "${cmd[@]}"
done

echo ""
echo "==> 全部完成。最終 fold 權重:"
for K in $(seq 0 "$last_fold"); do
  echo "    yolo26_solution/runs/stage2/fold${K}/weights/best.pt"
done
