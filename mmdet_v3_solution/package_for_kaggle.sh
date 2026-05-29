#!/usr/bin/env bash
# =============================================================================
# package_for_kaggle.sh — 打包 mmdet_v3_solution（InternImage-T Cascade Mask R-CNN）
# 上 Kaggle。staging 結構與 root 那份 package_for_kaggle.sh 對齊：
#   weights_to_upload_v3/
#     ├── code/{kaggle_infer.py, models/}     # 推理腳本 + 自訂 backbone
#     ├── configs/{_base_/, *.py}              # Config.fromfile 三層 _base_ 鏈
#     ├── weights/best_coco_segm_mAP_*.pth    # stage 2 最新 best ckpt
#     ├── wheels/*.whl                          # 離線安裝用（HuBMAP submit 強制關網）
#     ├── dataset-metadata.json                  # kaggle datasets create 用
#     └── README.md
#
# 用法：
#   bash package_for_kaggle.sh                                    # 全自動
#   bash package_for_kaggle.sh --ckpt <path>                       # 指定 ckpt
#   bash package_for_kaggle.sh --skip-wheels                       # 已抓過 wheel 跳過下載
#   bash package_for_kaggle.sh --torch-version 2.1 --cu-version cu121
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- 預設參數 ----------------------------------------------------------------
OUT="${ROOT}/weights_to_upload_v3"
CKPT=""
SLUG="${KAGGLE_USERNAME:-paohuah}/hubmap-internimage-cascade"
TITLE="HuBMAP InternImage-T Cascade Weights"
TORCH_VER="2.1"
CU_VER="cu121"
PY_VER="311"
SKIP_WHEELS=0
SKIP_CKPT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)            CKPT="$2"; shift 2 ;;
        --out)             OUT="$2"; shift 2 ;;
        --dataset-slug)    SLUG="$2"; shift 2 ;;
        --torch-version)   TORCH_VER="$2"; shift 2 ;;
        --cu-version)      CU_VER="$2"; shift 2 ;;
        --python-version)  PY_VER="$2"; shift 2 ;;
        --skip-wheels)     SKIP_WHEELS=1; shift ;;
        --skip-ckpt)       SKIP_CKPT=1; shift ;;
        *) echo "未知參數: $1"; exit 1 ;;
    esac
done

# ---- 1. auto-find ckpt（若沒指定）-------------------------------------------
if [[ -z "$CKPT" && "$SKIP_CKPT" -eq 0 ]]; then
    CKPT=$(ls -t "$ROOT"/work_dirs/stage2/best_coco_segm_mAP_epoch_*.pth 2>/dev/null | head -n1 || true)
    if [[ -z "$CKPT" ]]; then
        echo "❌ 找不到 work_dirs/stage2/best_coco_segm_mAP_epoch_*.pth，請用 --ckpt 指定"
        exit 1
    fi
    echo "==> auto-found ckpt: $CKPT"
fi

echo "==> 輸出 staging: $OUT"
echo "==> Kaggle dataset slug: $SLUG"
echo "==> 鎖 wheel 目標: torch $TORCH_VER + $CU_VER + Python 3.${PY_VER#3}（cp${PY_VER}）"

# ---- 2. 建立 staging 樹（保留 wheels/ 若 --skip-wheels）----------------------
if [[ "$SKIP_WHEELS" -eq 0 ]]; then
    rm -rf "$OUT"
else
    # 保留 wheels/ 但其餘清掉
    find "$OUT" -mindepth 1 -maxdepth 1 ! -name 'wheels' -exec rm -rf {} + 2>/dev/null || true
fi
mkdir -p "$OUT"/{code,configs/_base_,weights,wheels}

# ---- 3. 複製 code（kaggle_infer.py + models/）-------------------------------
echo "==> 複製推理程式碼"
cp "$ROOT/kaggle_infer.py" "$OUT/code/kaggle_infer.py"
cp -r "$ROOT/models" "$OUT/code/models"
# 清掉 __pycache__ 避免 Kaggle 載入時碰到二進位 .pyc 版本差
find "$OUT/code" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "  + code/kaggle_infer.py"
echo "  + code/models/  ($(du -sh "$OUT/code/models" | cut -f1))"

# ---- 4. 複製 configs（含三層 _base_ 鏈）------------------------------------
echo "==> 複製 configs（_base_ 鏈三層）"
cp "$ROOT/configs/_base_/default_runtime.py"             "$OUT/configs/_base_/"
cp "$ROOT/configs/cascade_mask_rcnn_internimage_t_fpn.py" "$OUT/configs/"
cp "$ROOT/configs/stage2_dataset1_finetune.py"           "$OUT/configs/"
echo "  + configs/_base_/default_runtime.py"
echo "  + configs/cascade_mask_rcnn_internimage_t_fpn.py"
echo "  + configs/stage2_dataset1_finetune.py"

# ---- 5. 複製 ckpt ----------------------------------------------------------
if [[ "$SKIP_CKPT" -eq 0 ]]; then
    echo "==> 複製 ckpt"
    CKPT_NAME=$(basename "$CKPT")
    cp "$CKPT" "$OUT/weights/$CKPT_NAME"
    echo "  + weights/$CKPT_NAME ($(du -h "$CKPT" | cut -f1))"
fi

# ---- 6. 下載 wheels（離線安裝用）-------------------------------------------
if [[ "$SKIP_WHEELS" -eq 0 ]]; then
    echo "==> 下載 wheels 到 $OUT/wheels/"
    PIP_DOWNLOAD_FLAGS=(
        --platform manylinux2014_x86_64
        --python-version "$PY_VER"
        --only-binary=:all:
        --dest "$OUT/wheels"
    )

    # (a) 一般 PyPI 套件（mmdet 全家桶 + 配套）
    #     mmengine / mmdet / mmpretrain 是純 Python（無編譯），跨 torch 版本可用。
    #     albumentations 鎖 1.3.x 避開 1.4+ 嚴格 key 驗證。
    echo "  [a] PyPI 套件（純 Python wheel）"
    python -m pip download "${PIP_DOWNLOAD_FLAGS[@]}" \
        "mmengine>=0.10,<1.0" \
        "mmdet==3.3.0" \
        "mmpretrain==1.2.0" \
        "albumentations>=1.3,<1.4" \
        "pycocotools" \
        "opencv-python-headless" \
        "addict" \
        "terminaltables" \
        "mat4py" \
        "einops" \
        "ordered_set" \
        "model_index" \
        "shapely" \
        "yapf" \
        2>&1 | tail -5

    # (b) mmcv 2.1.0 從 openmmlab wheel index（綁 torch + cu 版本）
    #     index 不支援 --platform 過濾，先下回所有再人工挑 cp / linux 那顆。
    echo "  [b] mmcv 2.1.0 從 openmmlab(torch${TORCH_VER}+${CU_VER})"
    MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/${CU_VER}/torch${TORCH_VER}/index.html"
    python -m pip download \
        --no-deps \
        --dest "$OUT/wheels" \
        --index-url "$MMCV_INDEX" \
        "mmcv==2.1.0" 2>&1 | tail -5

    # 清掉 manylinux2014 之外的 mmcv wheel（avx 等變體 + py 不符版本）
    # Kaggle 目標：cp311 + linux x86_64
    find "$OUT/wheels" -name 'mmcv-*.whl' ! -name "*cp${PY_VER}*linux_x86_64*" -delete 2>/dev/null || true

    echo "  下載完成: $(ls "$OUT/wheels" | wc -l) 個 wheel ($(du -sh "$OUT/wheels" | cut -f1))"
fi

# ---- 7. dataset-metadata.json ----------------------------------------------
cat > "$OUT/dataset-metadata.json" <<EOF
{
  "title": "$TITLE",
  "id": "$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
echo "==> 寫入 dataset-metadata.json (slug=$SLUG)"

# ---- 8. README -------------------------------------------------------------
CKPT_BASENAME=$(basename "${CKPT:-best_coco_segm_mAP_epoch_X.pth}")
cat > "$OUT/README.md" <<EOF
# HuBMAP InternImage-T Cascade Weights

mmdet v3 + InternImage-T(DCNv3_pytorch) + Cascade Mask R-CNN 單模推理。

## 內容
- \`code/kaggle_infer.py\` — 推理入口（CLI: --config --checkpoint --img-dir --out --no-compile）
- \`code/models/\` — 自訂 InternImage backbone（觸發 mmdet registry 註冊）
- \`configs/\` — 訓練用 config 完整鏈（Config.fromfile 三層 \`_base_\` 展開）
- \`weights/$CKPT_BASENAME\` — stage 2 訓練選出的 best ckpt
- \`wheels/\` — Kaggle 離線安裝用 wheel（mmdet 全家桶 + mmcv 2.1.0 預編）

## wheel 版本鎖定
- target: **torch ${TORCH_VER}+${CU_VER}** / **Python 3.${PY_VER#3}** / **linux x86_64**
- 若 Kaggle 升 torch / Python 版本，需用同 script 加 \`--torch-version / --cu-version / --python-version\` 重抓

## 在 Kaggle Notebook 用法
見 \`hubmap_internimage_infer.ipynb\`（沿用 root 的 dual_stream notebook 風格）。

## 重新打包
\`\`\`bash
bash mmdet_v3_solution/package_for_kaggle.sh                # 全自動
bash mmdet_v3_solution/package_for_kaggle.sh --skip-wheels  # 只更新 code/configs/ckpt
\`\`\`
EOF

# ---- 9. summary ------------------------------------------------------------
echo
echo "✅ 打包完成: $OUT"
du -sh "$OUT"/* 2>/dev/null
echo
echo "下一步："
echo "  1. 編輯 $OUT/dataset-metadata.json，把 slug 改成你的 kaggle username"
echo "  2. kaggle datasets create  -p $OUT --dir-mode zip       # 第一次"
echo "  3. kaggle datasets version -p $OUT --dir-mode zip -m v2 # 後續更新"
