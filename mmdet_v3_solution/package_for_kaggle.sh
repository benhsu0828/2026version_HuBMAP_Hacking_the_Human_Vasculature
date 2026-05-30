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
MMCV_VER="2.1.0"
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
        --mmcv-version)    MMCV_VER="$2"; shift 2 ;;
        --skip-wheels)     SKIP_WHEELS=1; shift ;;
        --skip-ckpt)       SKIP_CKPT=1; shift ;;
        *) echo "未知參數: $1"; exit 1 ;;
    esac
done

# ---- 1.5 Python 版本與 mmcv/torch 兼容性自動修正 ----------------------------
# openmmlab wheel index 對 cp312 只有 mmcv 2.2.0 + torch2.3+cu121 這組（已實測），
# 其他組合都沒 cp312 wheel。預設 mmcv 2.1.0 + torch 2.1 在 cp310/cp311 沒問題，
# 但碰到 cp312 必須自動切。
if [[ "$PY_VER" == "312" ]]; then
    if [[ "$MMCV_VER" == "2.1.0" || "$TORCH_VER" == "2.1" ]]; then
        echo "→ Python 3.12 偵測：自動切換 mmcv 2.1.0/torch 2.1 → mmcv 2.2.0/torch 2.3"
        echo "  (openmmlab 對 cp312 只有這一組預編 wheel)"
        MMCV_VER="2.2.0"
        TORCH_VER="2.3"
    fi
fi

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

# ---- 2. 建立 staging 樹（保留 --skip-* 對應的目錄，不要互相波及）------------
# code/ configs/ 永遠重建（便宜），wheels/ 與 weights/ 依 --skip-* 決定保不保留。
PRESERVE_PAT='!'   # find -not-name 的開頭
PRESERVE_ARGS=()
[[ "$SKIP_WHEELS" -eq 1 ]] && PRESERVE_ARGS+=(! -name 'wheels')
[[ "$SKIP_CKPT"   -eq 1 ]] && PRESERVE_ARGS+=(! -name 'weights')
if [[ -d "$OUT" ]]; then
    find "$OUT" -mindepth 1 -maxdepth 1 "${PRESERVE_ARGS[@]}" -exec rm -rf {} + 2>/dev/null || true
fi
mkdir -p "$OUT"/{code,configs/_base_,weights,wheels,mmcv_src}

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

    # (b) mmcv 從 openmmlab wheel index（綁 torch + cu 版本）
    #     注意：openmmlab 的 index 是純 HTML 列表（不是 PyPI simple index 結構），
    #     必須用 --find-links（-f）；用 --index-url 會找不到（'No matching distribution'）。
    #     --platform + --python-version 鎖 Kaggle target（cpXYZ + linux x86_64）。
    echo "  [b] mmcv ${MMCV_VER} 從 openmmlab(torch${TORCH_VER}+${CU_VER})"
    MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/${CU_VER}/torch${TORCH_VER}/index.html"
    python -m pip download \
        --no-deps \
        --platform manylinux1_x86_64 \
        --python-version "$PY_VER" \
        --only-binary=:all: \
        --dest "$OUT/wheels" \
        -f "$MMCV_INDEX" \
        "mmcv==${MMCV_VER}" 2>&1 | tail -10

    # 確認真的抓到 mmcv wheel
    if ! ls "$OUT/wheels"/mmcv-${MMCV_VER}-*.whl >/dev/null 2>&1; then
        echo "❌ mmcv ${MMCV_VER} wheel 抓失敗。可能原因："
        echo "   - 該 (torch, cu) 組合在 openmmlab 沒這版 mmcv → 改 --torch-version / --cu-version / --mmcv-version"
        echo "   - 網路問題 → 重跑或手動 curl '$MMCV_INDEX' 確認頁面有 cp${PY_VER} 的 linux wheel"
        exit 1
    fi

    echo "  下載完成: $(ls "$OUT/wheels" | wc -l) 個 wheel ($(du -sh "$OUT/wheels" | cut -f1))"
fi

# ---- 6.5 下載 mmcv source tarball（離線 source build 用）-------------------
# Kaggle submit 時關網→不能 git clone，所以把 mmcv 原始碼也打包進 dataset。
# notebook cell 3 用 `pip install --no-build-isolation /path/to/tarball` 即可離線 build。
# 體積極小（~5MB），每次都重抓不痛不癢，所以不加 skip flag。
MMCV_TARBALL="$OUT/mmcv_src/mmcv-${MMCV_VER}.tar.gz"
echo "==> 下載 mmcv ${MMCV_VER} source tarball（離線 source build 用）"
curl -fsSL -o "$MMCV_TARBALL" \
    "https://github.com/open-mmlab/mmcv/archive/refs/tags/v${MMCV_VER}.tar.gz"
echo "  + mmcv_src/mmcv-${MMCV_VER}.tar.gz ($(du -h "$MMCV_TARBALL" | cut -f1))"

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
# HuBMAP InternImage-T Cascade（mmdet v3 單模）

## 內容
| 路徑 | 用途 |
|---|---|
| \`code/kaggle_infer.py\` | 推理入口（CLI: \`--config --checkpoint --img-dir --out --no-compile\`） |
| \`code/models/\` | InternImage backbone（觸發 mmdet registry） |
| \`configs/\` | 三層 \`_base_\` 鏈，Config.fromfile 用 |
| \`weights/$CKPT_BASENAME\` | stage 2 best ckpt |
| \`wheels/\` | mmdet 全家桶 + 配套（不含 mmcv）|
| \`mmcv_src/mmcv-${MMCV_VER}.tar.gz\` | mmcv 原始碼，Kaggle 上 source build 對齊新 ABI（離線可跑） |

## Kaggle 環境鎖定（必須）
target: **Python 3.${PY_VER#3}** / **torch ${TORCH_VER}+${CU_VER}**

Kaggle latest 環境用 py3.12 + torch 2.10，預編 wheel 全部 ABI 不合。
解法是 fork 一個 2024 年的舊 notebook 繼承凍結環境（py3.10 + torch 2.1）。

## 用法
見 \`hubmap_internimage_infer.ipynb\`。cell 3 同時做兩件事：
1. 白名單裝 wheel（科學 stack 用 Kaggle 預裝版，不蓋）
2. 從 \`mmcv_src/\` 本地 tarball source build mmcv（\`--no-build-isolation --no-deps\` → 完全離線可跑）

## 重新打包
\`\`\`bash
bash mmdet_v3_solution/package_for_kaggle.sh                   # 全自動
bash mmdet_v3_solution/package_for_kaggle.sh --skip-wheels     # 只更新 code/configs/ckpt
bash mmdet_v3_solution/package_for_kaggle.sh --python-version 311  # Kaggle Python 漂移時調整
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
