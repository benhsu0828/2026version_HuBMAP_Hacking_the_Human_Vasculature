#!/usr/bin/env bash
# 把 YOLO + MedSAM-2 LoRA + vendored MedSAM2 + 推理程式碼打包到 weights_to_upload/
# 之後用 `kaggle datasets create -p weights_to_upload` 上傳。
#
# 用法：
#   ./package_for_kaggle.sh                 # 預設打包 fold 0-3
#   ./package_for_kaggle.sh --folds 0 1     # 只打包 fold 0, 1
#   ./package_for_kaggle.sh --dataset-slug benten/hubmap-dual-stream-weights

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT}/weights_to_upload"
FOLDS=(0 1 2 3)
SLUG="${KAGGLE_USERNAME:-paohuah}/hubmap-dual-stream-weights"
TITLE="HuBMAP Dual-Stream Weights (YOLO + MedSAM-2 LoRA)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folds)
            shift; FOLDS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do FOLDS+=("$1"); shift; done
            ;;
        --dataset-slug)
            SLUG="$2"; shift 2 ;;
        --out)
            OUT="$2"; shift 2 ;;
        *)
            echo "未知參數: $1"; exit 1 ;;
    esac
done

echo "==> 打包到 $OUT"
echo "==> Folds: ${FOLDS[*]}"
echo "==> Dataset slug: $SLUG"

rm -rf "$OUT"
mkdir -p "$OUT"/{yolo,medsam2/base,code}

# --- 1. YOLO weights ---
echo "==> 複製 YOLO fold weights"
for f in "${FOLDS[@]}"; do
    src="$ROOT/kaggle_weights/fold${f}.pt"
    if [[ ! -f "$src" ]]; then
        echo "  ⚠ $src 不存在，跳過"
        continue
    fi
    cp "$src" "$OUT/yolo/fold${f}.pt"
    echo "  + yolo/fold${f}.pt ($(du -h "$src" | cut -f1))"
done

# --- 2. MedSAM-2 base + LoRA adapter + decoder ---
echo "==> 複製 MedSAM-2 base checkpoint"
base="$ROOT/pretrained/MedSAM2_latest.pt"
if [[ -f "$base" ]]; then
    cp "$base" "$OUT/medsam2/base/MedSAM2_latest.pt"
    echo "  + medsam2/base/MedSAM2_latest.pt ($(du -h "$base" | cut -f1))"
else
    echo "  ⚠ $base 不存在，請先下載 (uv run download)"
fi

echo "==> 複製 LoRA adapter + decoder per fold"
for f in "${FOLDS[@]}"; do
    src_dir="$ROOT/runs/medsam2/fold${f}"
    dst_dir="$OUT/medsam2/fold${f}"
    if [[ ! -d "$src_dir" ]]; then
        echo "  ⚠ $src_dir 不存在，未訓練？跳過 fold $f"
        continue
    fi
    mkdir -p "$dst_dir"
    if [[ -d "$src_dir/lora_best" ]]; then
        cp -r "$src_dir/lora_best" "$dst_dir/lora_best"
    fi
    if [[ -f "$src_dir/decoder_best.pt" ]]; then
        cp "$src_dir/decoder_best.pt" "$dst_dir/decoder_best.pt"
    fi
    echo "  + medsam2/fold${f}/  ($(du -sh "$dst_dir" | cut -f1))"
done

# --- 3. Vendored MedSAM2 (tarball，省 inode 數，加速 Kaggle 上傳/下載) ---
echo "==> 打包 third_party/MedSAM2 為 tarball"
if [[ -d "$ROOT/third_party/MedSAM2" ]]; then
    tar -C "$ROOT/third_party" \
        --exclude='MedSAM2/checkpoints' \
        --exclude='MedSAM2/examples' \
        --exclude='MedSAM2/notebooks' \
        --exclude='__pycache__' \
        -czf "$OUT/third_party_MedSAM2.tar.gz" MedSAM2
    echo "  + third_party_MedSAM2.tar.gz ($(du -h "$OUT/third_party_MedSAM2.tar.gz" | cut -f1))"
else
    echo "  ⚠ third_party/MedSAM2 不存在"
fi

# --- 3b. 下載 wheels（Kaggle 無 Internet 用）---
echo "==> 下載推理用 wheels (Python 3.11 linux_x86_64)"
WHEEL_DIR="$OUT/wheels"
mkdir -p "$WHEEL_DIR"
# Kaggle 環境：Python 3.11，平台 linux_x86_64
# peft / safetensors / hydra-core / iopath + 它們的傳遞依賴
# Kaggle 已預裝 torch / transformers / accelerate / huggingface_hub。
# 用 --no-deps 只抓真正缺的 4 個套件 + 它們不在 Kaggle 上的直接依賴
uv run pip download \
    --platform manylinux2014_x86_64 \
    --python-version 311 \
    --only-binary=:all: \
    --no-deps \
    --dest "$WHEEL_DIR" \
    "peft>=0.12" "safetensors>=0.4" "hydra-core>=1.3" "iopath>=0.1.9" \
    "omegaconf>=2.3" "antlr4-python3-runtime" "portalocker>=2.0" \
    "ultralytics>=8.3" \
    2>&1 | tail -5
echo "  wheels 總數: $(ls "$WHEEL_DIR" | wc -l)"
du -sh "$WHEEL_DIR"

# --- 4. 推理程式碼 ---
echo "==> 複製推理程式碼"
for p in medsam2_stream.py predict_dual_stream.py predict_ensemble.py train.py dataprocess.py; do
    cp "$ROOT/$p" "$OUT/code/$p"
    echo "  + code/$p"
done

# --- 5. Kaggle dataset metadata ---
cat > "$OUT/dataset-metadata.json" <<EOF
{
  "title": "$TITLE",
  "id": "$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
echo "==> 寫入 dataset-metadata.json (slug=$SLUG)"

# --- 6. 簡易 README ---
cat > "$OUT/README.md" <<'EOF'
# HuBMAP Dual-Stream Weights

YOLO Fast Stream + MedSAM-2 (LoRA + decoder FT) Heavy Stream.

## 內容
- `yolo/foldK.pt` — YOLOv8x-seg 4-fold
- `medsam2/base/MedSAM2_latest.pt` — MedSAM-2 base checkpoint (Ma Lab)
- `medsam2/foldK/lora_best/` — PEFT LoRA adapter (image encoder, rank 16)
- `medsam2/foldK/decoder_best.pt` — full-FT mask decoder
- `third_party_MedSAM2.tar.gz` — vendored MedSAM2 source (commit 332f30d)
- `code/*.py` — 推理腳本

## 在 Kaggle Notebook 用法
見 `hubmap_dual_stream_infer.ipynb`。
EOF

echo
echo "✅ 打包完成: $OUT"
du -sh "$OUT"/* 2>/dev/null
echo
echo "下一步（必須加 --dir-mode zip 才會上傳子資料夾）："
echo "  1. 編輯 $OUT/dataset-metadata.json 把 slug 改成你的 kaggle username"
echo "  2. uv run kaggle datasets create  -p $OUT --dir-mode zip          # 第一次上傳"
echo "  3. uv run kaggle datasets version -p $OUT --dir-mode zip -m v2    # 後續更新"
