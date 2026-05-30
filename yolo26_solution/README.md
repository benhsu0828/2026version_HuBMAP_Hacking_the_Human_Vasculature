# YOLO26-seg — HuBMAP 第三名兩階段方案

把第三名的**兩階段策略**移植到 **YOLO26x-seg**,跑在你既有的 root `uv` 環境
(`ultralytics 8.4.51`,內建 `yolo26-seg`),不需 mmcv/conda,環境零負擔。

- **Stage1(粗練)**:全部 **dataset2(1211 張雜訊標註)**,載入 COCO 預訓練 `yolo26x-seg.pt`,
  epoch 少、lr 高、輕度增強。
- **Stage2(精練)**:**dataset1(422 張乾淨標註)** 做 **4-fold**,載入 Stage1 權重,
  epoch 多、lr 低退火到極小、重度增強。

推理**重用** repo 既有 [predict_ensemble.py](../predict_ensemble.py)(4-fold WBF + 4-way TTA)+
[train.py](../train.py) 的 `submit`(COCO-RLE)。之後 YOLO26 的 `submission.csv` / `preds/*.npz`
可在 Kaggle 端與第一名 InternImage 再 ensemble(本資料夾不含 ensemble 程式)。

## 目錄

```
yolo26_solution/
├── setup_env.sh                 # uv sync + 驗證 ultralytics / yolo26-seg / CUDA
├── prepare_data.py              # 產 stage1(ds2) + stage2(ds1 4-fold)YOLO 資料(重用 dataprocess.py)
├── train_yolo.py                # 兩階段訓練入口(--stage 1/2,stage 超參預設)
├── run_all.sh                   # 一鍵 Stage1 → 自動接力 → Stage2 ×folds
├── hubmap_yolo26_infer.ipynb    # Kaggle 推理 notebook(YOLO-only,離線 wheels)
└── package_for_kaggle.sh        # 打包 fold*.pt + code + wheels → Kaggle dataset
```

## 快速開始

```bash
# 0)(可選)驗證環境;root .venv 已備妥
bash yolo26_solution/setup_env.sh

# 1) 一鍵兩階段(預設 4-fold)
bash yolo26_solution/run_all.sh

# 單模型快速驗證 / smoke test
bash yolo26_solution/run_all.sh --folds 1
bash yolo26_solution/run_all.sh --folds 1 --epochs1 1 --epochs2 1
```

最終權重:`yolo26_solution/runs/stage2/fold{0..3}/weights/best.pt`。

`run_all.sh` 旗標:`--folds N`、`--skip-prepare`、`--skip-stage1 --stage1-ckpt <pt>`、
`--epochs1/--epochs2`、`--device`。

## 兩階段超參(`train_yolo.py` 預設,可 `--epochs` 覆寫)

| 階段 | 資料 | epoch | lr0 | lrf(cosine) | 增強 |
|---|---|---|---|---|---|
| Stage1 | ds2(1211) | 15 | 2e-3 | 0.1 → 2e-4 | 輕(degrees 30、無 mixup/copy_paste、close_mosaic 5) |
| Stage2 | ds1(422)/fold | 25 | 5e-4 | 1e-4 → ~5e-8 | 重(degrees 180、mixup 0.1、copy_paste 0.3、close_mosaic 15) |

共同:`imgsz=768`(512 + 2×128 鄰居拼接 padding)、`AdamW`、`cos_lr`、`amp`、EMA(內建)。

## 為什麼 4-fold?

ds1 乾淨資料只有 ~422 張,單一 split 驗證很抖;[dataprocess.py](../dataprocess.py) 的
`make_splits` 按 **WSI(1,2) × 左右半邊** 切 4 塊(防鄰居拼接 leak),每張 ds1 輪流當 val,
CV 更可信;4 個模型再用 WBF 融合通常 +2~5% mAP。不想花成本就 `--folds 1` 先驗證兩階段是否有效。

## 推理 + 上 Kaggle

```bash
# 1) 打包(fold 權重 + 推理程式 + 離線 wheels)
FOLDS=4 DATASET_SLUG=yourname/hubmap-yolo26-2stage bash yolo26_solution/package_for_kaggle.sh

# 2) 上傳(編輯 dataset-metadata.json 的 id 後)
kaggle datasets create -p yolo26_solution/weights_to_upload_yolo26 --dir-mode zip

# 3) Kaggle 開 Notebook → 貼 hubmap_yolo26_infer.ipynb 各 cell
#    Add Data 加「你的 weights dataset」+「比賽 dataset」→ Run → /kaggle/working/submission.csv
```

`hubmap_yolo26_infer.ipynb` 自動支援 1~4 個 fold 權重(`yolo/fold*.pt`),離線裝 ultralytics,
跑 `predict_ensemble.py`(WBF+TTA)→ `train.py submit`,Internet OFF 可提交。

## 重用對照(不重造輪子)

- 資料:[dataprocess.py](../dataprocess.py) 的 `build_padded_tile / polygons_to_yolo_txt / make_splits / TILE / NAMES`
- 推理:[predict_ensemble.py](../predict_ensemble.py)(WBF+TTA)、[train.py](../train.py) `submit`(RLE)
- notebook:仿 [hubmap_dual_stream_infer.ipynb](../hubmap_dual_stream_infer.ipynb)(去掉 MedSAM heavy stream)
- bash:仿 [mmdet_v3_solution/run_all.sh](../mmdet_v3_solution/run_all.sh)(`find_ckpt` + stage 接力)

## 注意

- 不修改 repo 既有檔(`train.py / dataprocess.py / predict_ensemble.py` 仍是 baseline),新增全在本資料夾。
- 資料根預設 `data/`(含 `tile_meta.csv` + `polygons.jsonl` + `train/*.tif`)。
