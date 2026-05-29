#!/usr/bin/env bash
# =============================================================================
<<'COMMENT'
run_all.sh — 一鍵跑完兩階段訓練（Conda 環境版）
  Stage1（Dataset2 粗練, 10ep, lr 2e-4, 輕度 aug）
    └─ 自動接力權重 ─┐
  Stage2（Dataset1 精練, 20ep, lr 1e-4→1e-7, 重度 aug）
全程使用 Conda mmdet-v3-env 環境並開啟 --amp（AMP 混合精度）。

用法：
  bash run_all.sh                              # 建環境 + 跑完整兩階段
  # 跳過建環境
  bash run_all.sh --skip-install
  bash run_all.sh --skip-install --epochs1 1 --epochs2 1   # smoke test
  bash run_all.sh --skip-install --stage 2 --stage1-ckpt /home/ben/nycu_hw/2026version_HuBMAP_Hacking_the_Human_Vasculature/mmdet_v3_solution/work_dirs/stage1/best_coco_segm_mAP_epoch_8.pth          # 只跑 Stage2
  bash run_all.sh --compile                    # 訓練時開 torch.compile
COMMENT
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# ---- 1) 動態動定位 Conda 環境中的 Python 執行檔 -----------------------------
ENV_NAME="mmdet-v3-env"
if ! command -v conda >/dev/null 2>&1; then
  echo "找不到 conda 命令，請確保已經安裝 Miniconda 或 Anaconda！"; exit 1
fi

CONDA_BASE="$(conda info --base)"
CONDA_ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
PY="$CONDA_ENV_DIR/bin/python"

# ---- 2) 預設參數 --------------------------------------------------------------
SKIP_INSTALL=0
STAGE="all"
STAGE1_CKPT=""
EPOCHS1=""
EPOCHS2=""
COMPILE=0
WORK_DIR="work_dirs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install) SKIP_INSTALL=1; shift ;;
    --stage)        STAGE="$2"; shift 2 ;;
    --stage1-ckpt)  STAGE1_CKPT="$2"; shift 2 ;;
    --epochs1)      EPOCHS1="$2"; shift 2 ;;
    --epochs2)      EPOCHS2="$2"; shift 2 ;;
    --compile)      COMPILE=1; shift ;;
    --work-dir)     WORK_DIR="$2"; shift 2 ;;
    *) echo "未知參數: $1"; exit 1 ;;
  esac
done

# ---- 3) 建環境 (呼叫對應的 Conda setup_env) ----------------------------------
if [ "$SKIP_INSTALL" -eq 0 ]; then
  echo "==> [0/2] 建立 Conda 環境 (setup_env.sh)"
  bash "$SCRIPT_DIR/setup_env.sh"
fi

# 再次驗證環境中的 Python 是否有效存在
if [ ! -x "$PY" ]; then
  echo "找不到 Conda 環境下的 Python: $PY"
  echo "請先確保執行過 setup_env.sh 且順利完成環境建置（或移除 --skip-install）"
  exit 1
fi

# 組裝共用 cfg-options
extra_cfg_stage1=()
extra_cfg_stage2=()
[ -n "$EPOCHS1" ] && extra_cfg_stage1+=("train_cfg.max_epochs=$EPOCHS1")
[ -n "$EPOCHS2" ] && extra_cfg_stage2+=("train_cfg.max_epochs=$EPOCHS2")
if [ "$COMPILE" -eq 1 ]; then
  extra_cfg_stage1+=("compile=True")
  extra_cfg_stage2+=("compile=True")
fi

# 找出某 work_dir 內最佳（優先）或最後的 checkpoint
find_ckpt() {
  local wd="$1"
  local best
  best=$(ls -t "$wd"/best_coco_segm_mAP_epoch_*.pth 2>/dev/null | head -n1 || true)
  if [ -n "$best" ]; then echo "$best"; return; fi
  if [ -f "$wd/last_checkpoint" ]; then cat "$wd/last_checkpoint"; return; fi
  ls -t "$wd"/epoch_*.pth 2>/dev/null | head -n1 || true
}

# ---- 4) Stage 1 訓練 --------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "1" ]; then
  echo "==> [1/2] Stage1 粗練 (Dataset2)"
  cmd=("$PY" train.py configs/stage1_dataset2_pretrain.py
     --work-dir "$WORK_DIR/stage1" --amp
     --cfg-options
       train_dataloader.batch_size=8
       train_dataloader.num_workers=8
       auto_scale_lr.enable=True
       auto_scale_lr.base_batch_size=2)

  if [ "${#extra_cfg_stage1[@]}" -gt 0 ]; then
    cmd+=(--cfg-options "${extra_cfg_stage1[@]}")
  fi
  echo "    ${cmd[*]}"
  "${cmd[@]}"
fi

# ---- 5) 解析 Stage1 接力權重 --------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "2" ]; then
  if [ -z "$STAGE1_CKPT" ]; then
    STAGE1_CKPT="$(find_ckpt "$WORK_DIR/stage1")"
  fi
  if [ -z "$STAGE1_CKPT" ] || [ ! -f "$STAGE1_CKPT" ]; then
    echo "找不到 Stage1 權重，請用 --stage1-ckpt 指定。"; exit 1
  fi
  echo "==> Stage2 將載入 Stage1 權重：$STAGE1_CKPT"
fi

# ---- 6) Stage 2 訓練 --------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "2" ]; then
  echo "==> [2/2] Stage2 精練 (Dataset1)"
  cmd=("$PY" train.py configs/stage2_dataset1_finetune.py
     --work-dir "$WORK_DIR/stage2" --amp
     --cfg-options
       "load_from=$STAGE1_CKPT"
       train_dataloader.batch_size=8
       train_dataloader.num_workers=8
       auto_scale_lr.enable=True
       auto_scale_lr.base_batch_size=2
       default_hooks.logger.interval=10
       "${extra_cfg_stage2[@]}")

  echo "    ${cmd[*]}"
  "${cmd[@]}"
fi

echo "==> 全部完成。最終權重在 $WORK_DIR/stage2/"