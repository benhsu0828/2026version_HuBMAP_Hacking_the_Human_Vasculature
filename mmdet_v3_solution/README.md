# MMDetection v3.x — HuBMAP 細胞/血管實例分割解決方案

獨立、自包含的一套方案：**Cascade Mask R-CNN + InternImage-T + FPN**，兩階段訓練
（Dataset2 粗練 → Dataset1 精練），Kaggle T4 上以 FP16 + torch.compile + TTA + 形態學
後處理做推理。

> 設計取捨：InternImage 的核心算子 DCNv3 預設使用**純 PyTorch 版**（`DCNv3_pytorch`），
> **零 CUDA 編譯**，因此在 PRO 6000（Blackwell, torch 2.12）與 Kaggle 2×T4 都能直接跑。
> 若要極致速度，可 `bash setup_env.sh --cuda-build` 編譯官方 ops 並把 config 的
> `backbone.core_op` 改為 `'DCNv3'`。另附 ConvNeXt-V2-Tiny 切換區塊（見骨架 config 註解）。

## 目錄

```
mmdet_v3_solution/
├── setup_env.sh        # 建立獨立 uv 環境 .venv-mmdet + 裝 mmcv/mmdet/mmpretrain
├── run_all.sh          # 一鍵：Stage1 → 自動接力 → Stage2
├── train.py            # 標準 MMDet v3 訓練 runner
├── kaggle_infer.py     # Part 3：推理 + TTA + 形態學 + RLE 提交
├── models/intern_image.py     # InternImage backbone（含純 PyTorch DCNv3）
├── configs/
│   ├── _base_/default_runtime.py
│   ├── cascade_mask_rcnn_internimage_t_fpn.py  # Part 1：模型骨架
│   ├── stage1_dataset2_pretrain.py             # Stage1：輕度 aug, lr 2e-4
│   └── stage2_dataset1_finetune.py             # Stage2：重度 aug, lr 1e-4→1e-7
└── tools/prepare_coco.py       # 產生 COCO json
```

## 1. 環境（uv 專案式）

依賴都宣告在 `pyproject.toml`，安裝走 `uv sync`（不用 pip）。

```bash
cd mmdet_v3_solution
bash setup_env.sh                 # 內部執行 uv sync，環境建在 .venv-mmdet
UV_VENV_CLEAR=1 bash setup_env.sh # 砍掉重建
```
結尾自檢會印出 `InternImage registered -> <class 'models.intern_image.InternImage'>`。

- 換 CUDA runtime：改 `pyproject.toml` 的 `[[tool.uv.index]]` url（`cu128`→`cu126`…）
  與 `torch`/`torchvision` 版本。
- **重要**：torch wheel 內建 CUDA runtime（執行 torch 不需本機 toolkit），但 `mmcv`
  full 版會**從原始碼編譯** CUDA ops，需要本機 `nvcc` 且版本與 torch 的 CUDA runtime
  相容（例如 `+cu128` 需 CUDA 12.8 toolkit）。若本機 nvcc 版本不符（如 11.5），請先裝
  對應 CUDA toolkit，或改用 OpenMMLab devel 容器編譯。

## 2. 準備資料（COCO 格式）

```bash
.venv-mmdet/bin/python tools/prepare_coco.py \
    --polygons ../polygons.jsonl --tile-meta ../tile_meta.csv --out-dir ../data
```
產生 `dtrain0i.json`（Dataset1 train）、`dval0i.json`（val）、`dtrain_dataset2.json`
（Dataset2）。影像目錄由 config 的 `img_prefix='../data/train'` 指定，請依實際調整。

## 3. 兩階段訓練（一鍵）

```bash
# 完整：建環境 + Stage1(10ep) + Stage2(20ep)，全程 AMP
bash run_all.sh

# 已建好環境後重跑
bash run_all.sh --skip-install

# smoke test：各 1 epoch，驗證接力/AMP/loss
bash run_all.sh --skip-install --epochs1 1 --epochs2 1

# 只跑 Stage2 並指定 Stage1 權重
bash run_all.sh --skip-install --stage 2 \
    --stage1-ckpt work_dirs/stage1/best_coco_segm_mAP_epoch_10.pth

# 訓練時開 torch.compile
bash run_all.sh --skip-install --compile
```
最終權重在 `work_dirs/stage2/`。

| 階段 | 資料 | epoch | lr (AdamW) | scheduler | 增強 |
|------|------|-------|-----------|-----------|------|
| Stage1 粗練 | Dataset2（弱標註） | 10 | 2e-4 | Cosine→2e-5 | 輕度（翻轉 + 小幅仿射）|
| Stage2 精練 | Dataset1（乾淨） | 20 | 1e-4 | Cosine→1e-7 | 重度（AutoAugment + Albu Elastic/Rotate90）|

## 4. 推理（Kaggle）

```bash
.venv-mmdet/bin/python kaggle_infer.py \
    --config configs/stage2_dataset1_finetune.py \
    --checkpoint work_dirs/stage2/best_coco_segm_mAP_epoch_20.pth \
    --img-dir /kaggle/input/<comp>/test \
    --out submission.csv
```
內含：FP16 autocast、torch.compile（失敗自動 fallback）、6 視角 TTA
（orig/hflip/vflip/rot90/180/270）、形態學開運算（腐蝕→膨脹）+ 移除小碎片、
COCO RLE→zlib→base64 提交（只輸出 blood_vessel）。

## 5. 打包到 Kaggle（離線）

Kaggle Notebook 無外網，需把相依與權重做成 Dataset 後離線安裝：
1. 在與 Kaggle 同款 CUDA/torch 的環境用 `pip download` 或 `mim download` 收集
   mmengine / mmcv / mmdet / mmpretrain / pycocotools wheels。
2. 把 wheels + `models/` + `configs/` + 訓練好的 `*.pth` 上傳為 Kaggle Dataset。
3. Notebook 第一格 `pip install --no-index --find-links=<dataset_wheels> ...`，
   再把本資料夾加入 `sys.path`，即可呼叫 `kaggle_infer.py` 的函式。
   （可參考專案根目錄既有的 `package_for_kaggle.sh` 與 `ultra_wheels/` 作法。）
> 用 `core_op='DCNv3_pytorch'` 時**完全不需**上傳任何編譯好的 CUDA 擴充，最省事。

## 注意事項

- 須從 `mmdet_v3_solution/` 目錄執行，讓 `custom_imports=['models']` 能找到 backbone。
- 切 ConvNeXt-V2-Tiny：見 `configs/cascade_mask_rcnn_internimage_t_fpn.py` 內註解，
  記得把 neck `in_channels` 改成 `[96,192,384,768]`。
- batch_size 預設 2（1024 解析度 + Cascade）。顯存夠可調大；T4 推理建議單張處理。
- Stage2 的 Albu `ElasticTransform` 後 bbox 由角點近似；alpha 已設溫和值，主要保留
  mask 邊界細節，對 Cascade 影響可忽略。
