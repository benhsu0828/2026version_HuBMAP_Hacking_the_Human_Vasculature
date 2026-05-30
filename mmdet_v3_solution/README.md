# MMDetection v3 — HuBMAP InternImage-T Cascade 方案

**Cascade Mask R-CNN + InternImage-T + FPN**,兩階段訓練(Dataset2 粗練 → Dataset1 精練),
Kaggle T4 上 FP16 + 6 視角 TTA + 形態學後處理推理。

InternImage 核心算子用**純 PyTorch 版** `DCNv3_pytorch`(零 CUDA 編譯) → 本地 Blackwell 與
Kaggle T4 都能直接跑。極致速度可 `bash setup_env.sh --cuda-build` 編 official ops 並把
`backbone.core_op` 改 `'DCNv3'`。

## 目錄

```
mmdet_v3_solution/
├── setup_env.sh                  # 建 conda env mmdet-v3-env(py3.11 + torch2.8+cu128 + 編 mmcv 2.1)
├── run_all.sh                    # 一鍵 Stage1 → Stage2(自動接力)
├── train.py                      # 訓練入口(含 Albu + torch.load monkey-patch)
├── kaggle_infer.py               # Kaggle 推理:TTA + 形態學 + RLE
├── package_for_kaggle.sh         # 打包成 Kaggle dataset(wheels + mmcv source tarball + ckpt)
├── hubmap_internimage_infer.ipynb # Kaggle 推理 notebook(7 cells,離線 submit 友善)
├── models/intern_image.py        # InternImage backbone(純 PyTorch DCNv3)
├── configs/
│   ├── _base_/default_runtime.py
│   ├── cascade_mask_rcnn_internimage_t_fpn.py    # 骨架
│   ├── stage1_dataset2_pretrain.py               # 輕度 aug,lr 2e-4
│   └── stage2_dataset1_finetune.py               # 重度 aug,lr 1e-4→1e-7
└── tools/prepare_coco.py         # 產 COCO json
```

## 快速開始

```bash
# 1. 建環境(conda env mmdet-v3-env;source build mmcv 對齊 Blackwell + torch 2.8 ABI)
bash setup_env.sh

# 2. 產資料(COCO json + 圖檔符號連結)
/home/ben/miniconda3/envs/mmdet-v3-env/bin/python tools/prepare_coco.py \
    --polygons ../polygons.jsonl --tile-meta ../tile_meta.csv --out-dir ../data

# 3. 訓練(Stage1 10ep + Stage2 20ep,全程 AMP)
bash run_all.sh --skip-install
```

最終權重在 `work_dirs/stage2/best_coco_segm_mAP_epoch_X.pth`。

| 階段 | 資料 | epoch | lr | scheduler | aug |
|---|---|---|---|---|---|
| Stage1 | Dataset2(髒) | 10 | 2e-4 | Cosine→2e-5 | ShiftScaleRotate |
| Stage2 | Dataset1(乾淨) | 20 | 1e-4 | Cosine→1e-7 | AutoAugment + Albu Elastic + FilterAnnotations |

`bash run_all.sh --help` 看 `--epochs1/2 --stage --compile --stage1-ckpt` 等旗標。

## 推理 + 上 Kaggle

### 本地測試
```bash
.../mmdet-v3-env/bin/python kaggle_infer.py \
    --config configs/stage2_dataset1_finetune.py \
    --checkpoint work_dirs/stage2/best_coco_segm_mAP_epoch_X.pth \
    --img-dir <test_dir> --out submission.csv --no-compile
```

### 打包並上傳 Kaggle dataset
```bash
bash package_for_kaggle.sh                              # 全自動 staging → weights_to_upload_v3/
kaggle datasets create  -p weights_to_upload_v3 --dir-mode zip       # 第一次
kaggle datasets version -p weights_to_upload_v3 --dir-mode zip -m v2 # 後續更新
```

### Kaggle Notebook(離線 submit 友善)

⚠️ **必須 fork 2024 年的舊 notebook 繼承凍結環境**(py3.10 + torch 2.1.2),不能用 Kaggle latest
(py3.12 + torch 2.10 的 mmcv wheel 全部 ABI 不合)。

把本地 `hubmap_internimage_infer.ipynb` 7 個 cell 貼到 fork 出的 notebook,Add Data 加你的
`hubmap-internimage-cascade` + 比賽 dataset → cell 3 會白名單裝 wheel + 從本地 tarball
source build mmcv(對齊 Kaggle 的新 GCC ABI),整段過程**無需網路**,Submit Internet OFF 可直接跑。

## 已知坑(都已寫進程式碼,提醒用)

| 位置 | 修法 | 為什麼 |
|---|---|---|
| [train.py:18-72](train.py#L18-L72) | `Albu._postprocess_results` monkey-patch | mmdet 3.3.0 順序錯,空 idx_mapper 噴 IndexError |
| [train.py:75-91](train.py#L75-L91) | `torch.load(weights_only=False)` monkey-patch | torch 2.6+ 預設改 True,擋下 mmengine 的 HistoryBuffer |
| [stage2_dataset1_finetune.py:55-62](configs/stage2_dataset1_finetune.py#L55-L62) | `FilterAnnotations(min_gt_bbox_wh=(2,2), keep_empty=True)` 插在 Albu 前 | albu 1.3+ 嚴格 check_bbox,AutoAugment 把 bbox 推出畫面會炸 |
| [setup_env.sh:80-87](setup_env.sh#L80-L87) | mmcv 從 git tag 裝(`mmcv @ git+...@v2.1.0` + `--no-binary mmcv`) | OpenMMLab PyPI 只發 wheel 沒 sdist,`--no-binary` 沒效;必須走 git 才會 source build |
| [Kaggle notebook cell 3](hubmap_internimage_infer.ipynb) | 白名單裝 wheel + 從 mmcv_src/ tarball source build | 不蓋 Kaggle 預裝科學 stack(numpy/scipy);對齊 Kaggle 新 ABI;離線可跑 |

## 注意事項

- **必須從 `mmdet_v3_solution/` 執行**,讓 `custom_imports=['models']` 找得到 backbone
- `batch_size=2` 預設(1024² + Cascade),顯存夠可調大;run_all.sh 支援 `--cfg-options train_dataloader.batch_size=8`
- 切 ConvNeXt-V2-Tiny:見 `configs/cascade_mask_rcnn_internimage_t_fpn.py` 註解,記得改 neck `in_channels`
- 純 `DCNv3_pytorch` → 不需上傳任何編好的 CUDA ops 到 Kaggle
