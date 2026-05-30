#!/usr/bin/env bash
# =============================================================================
# package_for_kaggle.sh — 打包 YOLO26 fold 權重 + 推理程式 + 離線 wheels 成 Kaggle dataset
# -----------------------------------------------------------------------------
# 產出 yolo26_solution/weights_to_upload_yolo26/:
#   yolo/fold{K}.pt      (來自 runs/stage2/fold{K}/weights/best.pt)
#   code/*.py            (train.py / dataprocess.py / predict_ensemble.py,推理重用)
#   wheels/*.whl         (Internet OFF 用;ultralytics + opencv + pycocotools)
#   dataset-metadata.json
#
# 用法:
#   FOLDS=4 DATASET_SLUG=paohuah/hubmap-yolo26-2stage bash yolo26_solution/package_for_kaggle.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STAGE="$SCRIPT_DIR/weights_to_upload_yolo26"
DATASET_SLUG="${DATASET_SLUG:-paohuah/hubmap-yolo26-2stage}"
FOLDS="${FOLDS:-4}"

rm -rf "$STAGE"
mkdir -p "$STAGE/yolo" "$STAGE/code" "$STAGE/wheels"

# 1) fold 權重 --------------------------------------------------------------
last=$((FOLDS - 1))
for K in $(seq 0 "$last"); do
  src="yolo26_solution/runs/stage2/fold${K}/weights/best.pt"
  if [ ! -f "$src" ]; then
    echo "缺 $src — 請先完成訓練(bash yolo26_solution/run_all.sh)"; exit 1
  fi
  cp "$src" "$STAGE/yolo/fold${K}.pt"
  echo "  + yolo/fold${K}.pt"
done

# 2) 推理程式(重用 root 既有檔)--------------------------------------------
cp train.py dataprocess.py predict_ensemble.py "$STAGE/code/"
echo "  + code/{train,dataprocess,predict_ensemble}.py"

# 3) 離線 wheels(Kaggle Internet OFF)--------------------------------------
if compgen -G "ultra_wheels/*.whl" > /dev/null 2>&1; then
  cp ultra_wheels/*.whl "$STAGE/wheels/" 2>/dev/null || true
fi
# uv 沒有 download 子指令;走 venv 內真正的 pip。notebook 以 --no-deps 安裝(Kaggle 已內建
# torch/numpy 等),故這裡也 --no-deps 只抓 3 個套件本體。指定 py3.11 + manylinux 確保 wheel 相容。
uv run python -m pip download -d "$STAGE/wheels" --no-deps \
  --only-binary=:all: --python-version 3.11 --platform manylinux2014_x86_64 \
  ultralytics opencv-python-headless pycocotools \
  || echo "(提醒)pip download 失敗,請手動把 wheel 放進 $STAGE/wheels/"

# 4) metadata ---------------------------------------------------------------
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "hubmap-yolo26-2stage",
  "id": "$DATASET_SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

echo ""
echo "完成 staging → $STAGE"
echo "編輯 dataset-metadata.json 的 id 後上傳:"
echo "  kaggle datasets create  -p $STAGE --dir-mode zip          # 第一次"
echo "  kaggle datasets version -p $STAGE --dir-mode zip -m v2    # 後續更新"
