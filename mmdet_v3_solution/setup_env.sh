#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 為本方案建立「獨立」的 Conda 環境
# -----------------------------------------------------------------------------
# 優勢：Conda 會在環境內安裝獨立的 cuda-nvcc，解決本機 CUDA 11.5 太舊的問題。
# 安裝鏈：conda (python, cuda-nvcc, ninja) -> pytorch -> mmcv (原始碼編譯) -> 雜項
#
# 用法：
#   bash setup_env.sh                 # 標準安裝（使用 Conda）
#   UV_VENV_CLEAR=1 bash setup_env.sh # 重建既有環境（相容舊環境清理參數）
#   bash setup_env.sh --cuda-build    # 額外編譯 DCNv3 CUDA ops（進階）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 定義 Conda 環境名稱與版本設定（改用黃金組合：PyTorch 2.4+ / CUDA 12.1）
# 註：為確保 mmcv 與 DCNv3 編譯成功，這裡使用生態系最穩定的 CUDA 12.1 組合
ENV_NAME="mmdet-v3-env"
PY_VERSION="3.11"
CUDA_VERSION="12.8"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"

CUDA_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --cuda-build) CUDA_BUILD=1 ;;
    *) echo "未知參數: $arg" ; exit 1 ;;
  esac
done

# 1) 前置檢查 ----------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] 找不到 conda，請先安裝 Miniconda 或 Anaconda！"
  exit 1
fi

# 檢查是否需要清除舊環境
if [ "${UV_VENV_CLEAR:-0}" = "1" ]; then
  echo "[setup] 偵測到清除參數，正在移除既有 Conda 環境: $ENV_NAME"
  conda env remove -n "$ENV_NAME" -y || true
fi

# 2) 建立 Conda 環境並安裝環境內專用編譯器 (nvcc) -----------------------------
if ! conda info --envs | grep -q "$ENV_NAME"; then
  echo "[setup] 正在建立 Conda 環境: $ENV_NAME (Python $PY_VERSION)"
  # 同時安裝編譯所需的 nvcc, gxx, ninja
  conda create -n "$ENV_NAME" python="$PY_VERSION" -c conda-forge -y
fi

# 取得 Conda 環境中的 Python 與 Pip 路徑 (不需手動 conda activate)
CONDA_PREFIX="$(conda info --base)/envs/$ENV_NAME"
PY="$CONDA_PREFIX/bin/python"
PIP="$CONDA_PREFIX/bin/pip"

echo "[setup] 正在環境中安裝專用 nvcc 編譯器與建置工具 (CUDA $CUDA_VERSION)..."
conda install -n "$ENV_NAME" -c "nvidia/label/cuda-${CUDA_VERSION}.0" cuda-nvcc -y
conda install -n "$ENV_NAME" -c conda-forge gxx_linux-64=11 ninja -y

# 3) 安裝特定 CUDA 版本的 PyTorch --------------------------------------------
echo "[setup] 正在安裝 PyTorch $TORCH_VERSION + CUDA $CUDA_VERSION"
$PIP install torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu128

# 4) 安裝建置依賴與雜項套件 --------------------------------------------------
echo "[setup] 安裝建置工具與雜項依賴"
$PIP install "setuptools<81" wheel psutil opencv-python-headless pycocotools ensemble-boxes shapely "albumentations>=1.3,<1.4"

# 5) 從原始碼編譯 OpenMMLab 套件（MMCV） -------------------------------------
echo "[setup] 安裝 mmengine"
$PIP install mmengine

echo "[setup] 開始編譯 mmcv==2.1.0（這會花費幾分鐘，請稍候...）"
# 強制指定使用環境內剛裝好的 nvcc 進行編譯
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
# Blackwell sm_120 + 常見 arch；+PTX 讓未來卡也能 JIT。注意要與 conda 內 nvcc 12.8 對齊。
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0+PTX"
export FORCE_CUDA=1

# 砍掉前一次裝失敗的舊 wheel + 清 pip 快取裡的 mmcv wheel
$PIP uninstall -y mmcv mmcv-full 2>/dev/null || true
$PIP cache remove "mmcv*" 2>/dev/null || true

# 關鍵：OpenMMLab 在 PyPI 只發 wheel 沒發 sdist，所以 --no-binary mmcv 會 silently 退回 wheel。
# 必須直接從 git tag 裝 → pip 拿到 source tree 別無選擇只能編。
MMCV_WITH_OPS=1 $PIP install \
    "mmcv @ git+https://github.com/open-mmlab/mmcv.git@v2.1.0" \
    --no-build-isolation -v
$PIP install "mmdet>=3.1.0,<3.4.0" --no-build-isolation
$PIP install "mmpretrain>=1.0.0" --no-build-isolation

# 6)（可選）編譯 DCNv3 CUDA ops，僅在 --cuda-build 時執行 --------------------
if [ "$CUDA_BUILD" -eq 1 ]; then
  echo "[setup] (進階) clone 官方 InternImage 並編譯 ops_dcnv3"
  THIRD="$SCRIPT_DIR/third_party"
  mkdir -p "$THIRD"
  if [ ! -d "$THIRD/InternImage" ]; then
    git clone --depth 1 https://github.com/OpenGVLab/InternImage.git \
      "$THIRD/InternImage"
  fi
  ( cd "$THIRD/InternImage/detection/ops_dcnv3" && "$PY" setup.py build install )
  echo "[setup] DCNv3 CUDA ops 編譯完成；config 可改 backbone.core_op='DCNv3'"
fi

# 7) 自檢 --------------------------------------------------------------------
echo "[setup] 自檢 ..."
PYTHONPATH="$SCRIPT_DIR" "$PY" - <<'PYEOF'
import torch, mmcv, mmdet
try:
    import models.intern_image
    from mmdet.registry import MODELS
    print('InternImage registered ->', MODELS.get('InternImage'))
except ImportError:
    pass

print('torch     :', torch.__version__, '| cuda:', torch.cuda.is_available())
print('mmcv      :', mmcv.__version__)
print('mmdet     :', mmdet.__version__)
PYEOF

echo "[setup] 完成。之後訓練請用：conda activate $ENV_NAME 之後執行訓練指令。"