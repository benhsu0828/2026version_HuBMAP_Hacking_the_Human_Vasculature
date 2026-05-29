#!/usr/bin/env bash
# Install mmcv / mmdet / mmpretrain into the active uv venv.
#
# Two-part hack required for torch 2.12 + cu130 + python 3.13:
#   1. mmcv has no pre-built wheel for this triple on download.openmmlab.com,
#      so it falls back to source build.
#   2. Source build's setup.py imports `pkg_resources`, which setuptools 81+
#      removed. We pin setuptools<81 into the venv and use --no-build-isolation
#      so the build picks up the venv's old setuptools.
#
# Run AFTER `uv sync --extra training`. Requires nvcc/CUDA toolkit on PATH for
# source compilation (only needed for mmcv).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activating .venv..." >&2
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

uv run python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"

# Step 1 — pin setuptools<81 inside the venv (so pkg_resources stays importable).
uv pip install "setuptools<81" wheel

# Step 2 — install a CUDA toolkit matching torch (system nvcc may be too old).
#          nvidia-cuda-nvcc ships the cu13 nvcc inside the venv, no sudo needed.
uv pip install "nvidia-cuda-nvcc"

# Locate the venv nvcc and prepend its dir to PATH for the mmcv build.
NVCC_PATH="$(find "$VIRTUAL_ENV/lib" -name nvcc -executable -path '*/nvidia/cu*/bin/*' | head -1)"
if [[ -z "$NVCC_PATH" ]]; then
    echo "ERROR: couldn't locate cu13 nvcc under $VIRTUAL_ENV" >&2
    exit 1
fi
CUDA_HOME_LOCAL="$(dirname "$(dirname "$NVCC_PATH")")"
export PATH="$CUDA_HOME_LOCAL/bin:$PATH"
export CUDA_HOME="$CUDA_HOME_LOCAL"
echo "Using nvcc: $(which nvcc)  CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -2

# Step 3 — build mmcv against the matched nvcc. --no-build-isolation lets the
#          build pick up the venv's torch and setuptools<81.
uv pip install --no-build-isolation "mmcv>=2.1.0,<2.3.0"

# Step 3 — mmdet and mmpretrain are pure python; isolation is fine.
uv run mim install "mmdet>=3.3.0"
uv run mim install "mmpretrain>=1.2.0"

echo
echo "=== installed mm* versions ==="
uv run python -c "import mmcv, mmdet, mmengine, mmpretrain; \
    print(f'mmcv={mmcv.__version__}'); \
    print(f'mmdet={mmdet.__version__}'); \
    print(f'mmengine={mmengine.__version__}'); \
    print(f'mmpretrain={mmpretrain.__version__}')"
