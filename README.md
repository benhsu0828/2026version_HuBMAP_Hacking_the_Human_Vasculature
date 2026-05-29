# 2026version_HuBMAP_Hacking_the_Human_Vasculature
NYCU DLCV Final project for HuBMAP - Hacking the Human Vasculature

## Create uv env
If you clone this repo (with `uv.lock` and `pyproject.toml`), just run:
```bash
uv sync
```

Otherwise, to set up from scratch:
```bash
uv init --bare --python 3.11
uv add numpy pandas kaggle
uv add torch torchvision pillow einops tqdm scikit-image wandb
uv add ultralytics opencv-python-headless pycocotools shapely
```

## Download Dataset
```bash
uv run kaggle auth login

uv run kaggle competitions download -c hubmap-hacking-the-human-vasculature
```

## data process
```bash
uv run python dataprocess.py --out yolo_data --pad 128 # 一次性產出 4-fold YOLO 資料
```

## Train

```bash
uv run python train.py train --fold 0 --epochs 5       # smoke test

# 完整 baseline：單 fold 80 epochs（RTX 4090 約 2-4 小時）
uv run python train.py train --fold 0 --epochs 80

# 全部 4-fold（最終提交用，4-fold ensemble）
set -e  # 一失敗就停
for f in 0 1 2 3; do
  uv run python train.py train --fold $f --weights yolo26x-seg.pt --wandb \
    2>&1 | tee runs/seg/fold${f}_train.log
done


# 推理 test 集 → 存 .npz 預測檔
uv run python train.py predict \
  --weights runs/seg/fold0/weights/best.pt \
  --src dataset/test \
  --out preds/fold0 \
  --tta

# 產出 submission.csv
uv run python train.py submit \
  --preds preds/fold0 \
  --out submission.csv

```

走完整 Kaggle 提交（推薦，能看 Private LB 真分數）：

訓練 4 fold → 把 runs/seg/fold*/weights/best.pt 打包成 Kaggle Dataset 上傳
在 Kaggle 開新 Notebook，貼 inference 程式（可以複用你的 train.py predict + submit 邏輯）
用你貼的 kaggle competitions submit 指令推上去

### 上傳到 kaggle
```bash
Step 1：把 4 個 best.pt 上傳成 Kaggle Dataset

mkdir -p kaggle_weights
cp runs/seg/fold0-2/weights/best.pt kaggle_weights/fold0.pt   # 或改名後 fold0/weights/best.pt
cp runs/seg/fold1/weights/best.pt kaggle_weights/fold1.pt
cp runs/seg/fold2/weights/best.pt kaggle_weights/fold2.pt
cp runs/seg/fold3/weights/best.pt kaggle_weights/fold3.pt

cd kaggle_weights
uv run kaggle datasets init -p .
# 編輯生成的 dataset-metadata.json，填上 title 和 id (例如 "paohuah/hubmap-yolo26x-4fold")
uv run kaggle datasets create -p .
cd ..
Step 2：在 Kaggle 開新 Notebook
進 https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/code → New Notebook
Add Data → 加 hubmap-hacking-the-human-vasculature 和剛上傳的 weights dataset
在 notebook 裡複製 train.py 的 pad_test_tile + cmd_predict + cmd_submit + encode_rle 邏輯
對 /kaggle/input/hubmap-hacking-the-human-vasculature/test/*.tif 推理 → 寫出 /kaggle/working/submission.csv
Save Version → 等它跑完
Step 3：用 kaggle CLI 提交那個 notebook

uv run kaggle competitions submit \
    -c hubmap-hacking-the-human-vasculature \
    -k paohuah/<NOTEBOOK_SLUG> \
    -v <VERSION_NUMBER> \
    -m "yolo26x-seg 4fold ensemble"
（也可以直接在 notebook 頁面點 Submit to Competition 按鈕，更簡單
```