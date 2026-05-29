# YOLO Fast Stream + MedSAM-2 Heavy Stream (Prompt-Driven Refinement)

## Context

目前 HuBMAP vasculature 方案僅以 YOLOv8x-seg 4-fold ensemble + 4-way TTA + WBF 完成 mask
預測（[train.py](../train.py)、[predict_ensemble.py](../predict_ensemble.py)）。YOLO 對小血管召回不錯，
但 mask 邊界粗糙、coco-RLE 後 IoU 損失明顯。

本次擴充採「雙串流」設計：
- **Fast Stream（YOLO）**：4-fold + TTA + WBF，負責**偵測**（bbox + score），mask 僅作為粗 prior。
- **Heavy Stream（MedSAM-2，Ma Lab，SAM2.1 基底）**：以 YOLO 的高信心 bbox 作為 prompt，由 LoRA 微調過的
  MedSAM-2 重新解出**像素級精準 mask**，取代該 cluster 的 YOLO mask。
- 仍套用現有 WBF 做跨 fold + TTA bbox 融合，但 cluster 內 mask 改用 MedSAM-2 prompt-driven 解碼。

訓練 GPU：RTX PRO 6000（充裕）。推理目標：Kaggle T4 ×2 （15GB×2，須 fp16）。

---

## 整體架構

```
                    Test Tile (512x512)
                          │
                  pad_test_tile  → 768x768
                          │
                ┌─────────┴──────────┐
                ▼                    ▼
    ┌──────────────────────┐   ┌────────────────────────┐
    │ YOLO 4-fold × 4-TTA  │   │  MedSAM-2 (LoRA-FT)    │
    │ run_one + WBF        │   │  image encoder 一次     │
    │ → fused_boxes,       │   │  per-tile cache         │
    │   fused_scores,      │   └────────────┬───────────┘
    │   (YOLO mask 丟棄)    │                │ embedding
    └──────────┬───────────┘                │
               │  bbox (conf ≥ τ_high)      │
               └─────────────┬──────────────┘
                             ▼
                  MedSAM-2 mask decoder
                  (bbox prompt → mask)
                             │
                  center-crop 回 512×512
                             │
                       .npz 輸出
```

---

## 變更清單（檔案層級）

### 新增

1. **`medsam2_stream.py`** — MedSAM-2 推理封裝
   - `class MedSAM2Refiner`：載入 SAM2.1 + LoRA adapter；提供 `embed(image)` 與
     `decode(boxes_xyxy)` 兩階段介面（image encoder 只跑一次，bbox 解碼批次化）。
   - 支援 fp16 / `torch.compile=False`（Kaggle T4 相容）。
   - 介面對齊 `run_one()` 回傳格式：`(boxes, scores, masks_uint8 in padded coords)`。

2. **`train_medsam2.py`** — LoRA 微調訓練腳本
   - 載入 MedSAM-2 (Ma Lab `MedSAM2_pretrain.pth`) → 套 PEFT LoRA 到 image encoder
     的 attention `qkv` / `proj`（rank=16, alpha=32, dropout=0.05）。
   - **mask decoder：全參數微調**（unfreeze）。
   - **prompt encoder：凍結**（bbox prompt embedding 本就是 SAM 訓練好的 prior）。
   - 資料：**僅 ds1**（透過 `dataprocess.py` 已建的 4-fold；讀 `yolo_data/fold_K.yaml`
     中 `val` 對應 ds1 影像 + 對應 polygon mask）。一個 fold 一個 LoRA 權重。
   - Loss：Dice + BCE（SAM 原生）+ IoU prediction head MSE。
   - Bbox prompt 來源：**GT mask 的緊湊外接矩形** + 隨機 jitter（±5%）做 augmentation；
     模擬推理時 YOLO 給的不完美 bbox。
   - 增強：與 YOLO 同套幾何增強（rot90/flip/scale 0.8–1.2），不做 mosaic（破壞 prompt 對應）。
   - epochs=30，AdamW lr=1e-4（LoRA）/ 5e-5（decoder），cosine。
   - Output：`runs/medsam2/fold{K}/lora.safetensors` + `decoder.pt`。

3. **`predict_dual_stream.py`** — 雙串流推理（取代 `predict_ensemble.py` 的呼叫位置）
   - Reuse `pad_test_tile`、`tta_forward/inverse_*`、`weighted_box_fusion` from
     [predict_ensemble.py](../predict_ensemble.py)。
   - 對每張 tile：
     1. 跑 YOLO 4 model × 4 TTA → 收集 (boxes, scores) **丟棄 YOLO masks**。
     2. `weighted_box_fusion(...)` 但傳入**虛擬 mask**（避免修改 WBF 簽章）或新增
        `weighted_box_fusion_box_only()` 分支（首選：在同檔內加薄包裝避免動 ensemble 檔）。
     3. 對 padded image 跑 `MedSAM2Refiner.embed(image)` 一次。
     4. 過濾 `fused_scores >= conf_high (e.g., 0.25)` 的 bbox → batch decode 出 mask。
     5. Center-crop 回 512×512，過濾面積 ≥50px，存 `.npz`（schema 完全相同）。
   - **低信心 bbox（< τ_high）的處理**：丟棄。實驗證明 SAM2 對 bbox 噪聲敏感，
     低分 bbox 多為 FP，refine 後仍是 FP。

### 修改（最小幅度）

4. **[pyproject.toml](../pyproject.toml)** — 新增依賴
   - `sam2 @ git+https://github.com/facebookresearch/sam2` 或 ultra wheels 內 vendored
   - `peft>=0.12`、`safetensors`、`hydra-core`（SAM2 依賴）
   - 確認 `torch>=2.5`（SAM2.1 要求）— 目前已是 2.12+，OK。

5. **[train.py](../train.py)** — 不動。`cmd_submit` 仍可重用，因 `.npz` schema 不變。

### 不修改
- `dataprocess.py`、`predict_ensemble.py`（保留作為 baseline 對照）、`kaggle_weights/`。

---

## 關鍵實作細節

### MedSAM-2 整合的座標系
- YOLO bbox 已被 WBF 歸一化到 `[0,1]`（`predict_ensemble.py:224`）。送 MedSAM-2 前
  乘回 `IMG_SIZE=768`（padded canvas 座標），這正是 MedSAM-2 image encoder 接收的尺寸。
- MedSAM-2 內部會 resize 到 1024（SAM2 標準）；輸出 mask resize 回 768；最後 center-crop 到 512。

### LoRA 訓練資料 pipeline 重用
- 直接讀 `dataprocess.py` 已生成的 `yolo_data/fold_K/{train,val}` 影像與
  `labels/fold_K/train/*.txt`（YOLO-seg polygon 格式）。
- 新增 `MedSAMDataset` (in `train_medsam2.py`)：解析 polygon TXT → 還原 binary mask →
  以每個 instance 的 mask bbox（+ jitter）為 prompt，目標 = 該 instance binary mask。
- 對應的 ds1 篩選邏輯透過 `dataprocess.py:164-196` 的 fold 已確保（ds1 only in val 路徑）。
  訓練時用 fold_K 的 **val** split（純 ds1）+ ds1 在 train split 的部分（依 `tile_meta.csv` 過濾 `dataset==1`）。

### Kaggle T4 推理可行性
- SAM2.1-large（image encoder ~440MB）+ fp16 在單 T4 ~ 4GB。雙 T4 可一張跑 YOLO 一張跑
  MedSAM-2，或同卡序列化。
- 每張 tile 一次 image encoding（768→1024，~120ms on T4 fp16），bbox decoding batch 32（~20ms）。
- 預估 65 tiles × ~150ms ≈ 10s（vs. 原 ensemble ~30s），可接受。

### Fusion 策略（已選定：MedSAM 取代）
- WBF 仍跑，但 `weighted_box_fusion` 內 mask averaging 邏輯**繞過**：cluster 只回 bbox + score。
- 在 cluster 確定後（每張 tile 約 N=10–50 個 fused bbox），統一 batch 餵 MedSAM-2 decoder。
- **Fallback**：若 MedSAM-2 decode 出空 mask 或面積 < 50px，**fallback 用 YOLO WBF mask**（需保留 YOLO mask 作 backup buffer）。

---

## 訓練/推理流程

### 訓練（per fold）
```bash
# (1) 先確認 ds1-only fold 資料 (現有)
ls yolo_data/fold_0/

# (2) LoRA + decoder FT
uv run python train_medsam2.py \
    --fold 0 \
    --base-weights pretrained/MedSAM2_pretrain.pth \
    --lora-rank 16 --epochs 30 --batch 4 --device 0
# → runs/medsam2/fold0/{lora.safetensors, decoder.pt, args.yaml}
```

### 推理（雙串流 ensemble）
```bash
uv run python predict_dual_stream.py \
    --yolo-weights kaggle_weights/fold{0,1,2,3}.pt \
    --medsam-base pretrained/MedSAM2_pretrain.pth \
    --medsam-loras runs/medsam2/fold{0,1,2,3} \
    --src dataset/test --out preds/dual_stream \
    --conf-high 0.25 --wbf-iou 0.55 --device 0
uv run python train.py submit \
    --preds preds/dual_stream --out submission_dual.csv
```

### Kaggle 提交
- 上傳：YOLO 4 個 fold .pt + MedSAM-2 base + 4 個 LoRA adapter（< 1GB 總和，符合 Kaggle 額度）。
- Notebook 內呼叫 `predict_dual_stream.run_dual_stream()` + `cmd_submit`。

---

## 驗證 (Verification)

1. **單元驗證**
   - `python -c "from medsam2_stream import MedSAM2Refiner; r=MedSAM2Refiner(...); r.embed(np.zeros((768,768,3),uint8)); r.decode(np.array([[100,100,300,300]]))"` 應回傳 `(1,768,768)` mask。
   - LoRA 載入後參數量檢查：trainable params < 5M（image encoder LoRA + mask decoder full）。

2. **小規模 E2E**
   - 在 fold 0 ds1 val 集上跑 `predict_dual_stream.py --src yolo_data/fold_0/val` →
     算 mAP@0.5、mAP@[.5:.95] vs. 純 YOLO ensemble baseline。預期 mAP@0.5:0.95 +2~5pp。

3. **完整 LB 驗證**
   - 本機產 `submission_dual.csv`（dataset/test，公開部分 if available）。
   - Kaggle 提交，對比 baseline ensemble LB 分數。

4. **GPU/時間預算**
   - Kaggle notebook timing：應 < 9hr 限制；目標 < 30 min for 65 test tiles。

---

## 風險與緩解

| 風險 | 緩解 |
|---|---|
| MedSAM-2 (Ma Lab) 是 3D/video 為主，2D 單張的 inference path 需確認 | Phase 1 先確認 `MedSAM2VideoPredictor` vs `MedSAM2ImagePredictor` API；若只有 video，初始化時用 `num_frames=1` |
| SAM2.1 對 noisy bbox 敏感 | `conf-high=0.25` 過濾；jitter aug 提升魯棒 |
| Kaggle T4 fp16 overflow | image encoder 用 fp16，mask decoder 維持 fp32（小） |
| LoRA + decoder FT 過擬合 ds1（量少） | 加 dropout、early stopping on val Dice、保留 baseline ensemble fallback |

---

## 關鍵檔案參考索引

- 現有 YOLO pipeline：[train.py:40-83](../train.py#L40-L83)（訓練）、[train.py:102-147](../train.py#L102-L147)（單模推理）
- 現有 ensemble + WBF：[predict_ensemble.py:144-203](../predict_ensemble.py#L144-L203)
- TTA 變換：[predict_ensemble.py:38-83](../predict_ensemble.py#L38-L83)（reuse）
- 資料 padding：[train.py:88-99](../train.py#L88-L99)、[dataprocess.py:52-137](../dataprocess.py#L52-L137)
- Fold 切分（ds1-only val）：[dataprocess.py:164-196](../dataprocess.py#L164-L196)
- 提交編碼：[train.py:152-189](../train.py#L152-L189)（reuse，schema 不變）
