"""HuBMAP Vasculature — 雙串流推理 (YOLO Fast + MedSAM-2 Heavy)

Pipeline:
    對每張 test tile（padded 768x768）：
      1. YOLO 4-fold × 4-TTA → 16 份 (boxes, scores, masks)
      2. WBF 跨模型分群 → fused (boxes, scores, yolo_masks)
      3. 過濾 fused_scores >= conf_high
      4. MedSAM-2 對 padded tile embed 一次
      5. 對每個 fused bbox 跑 MedSAM-2 mask decoder → 取代 yolo mask
      6. 若 MedSAM mask 為空（< min_area），fallback 用 yolo WBF mask
      7. center-crop 回 512×512，存 .npz（schema 與 train.py 相容）

用法：
    uv run python predict_dual_stream.py \\
        --yolo-weights kaggle_weights/fold0.pt kaggle_weights/fold1.pt \\
                       kaggle_weights/fold2.pt kaggle_weights/fold3.pt \\
        --medsam-ckpt pretrained/MedSAM2_latest.pt \\
        --medsam-cfg configs/sam2.1_hiera_t512.yaml \\
        --medsam-loras runs/medsam2/fold0 runs/medsam2/fold1 \\
                       runs/medsam2/fold2 runs/medsam2/fold3 \\
        --src dataset/test --out preds/dual_stream
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from train import IMG_SIZE, PAD, pad_test_tile
from dataprocess import TILE
from predict_ensemble import TTA_MODES, run_one, weighted_box_fusion


# ---------- 單張 tile 雙串流 ----------

def run_dual_stream_tile(
    yolo_models: list,
    refiners: list,           # MedSAM2Refiner 列表（每 fold 一個 LoRA）
    padded_img: np.ndarray,   # (768, 768, 3) BGR uint8
    modes: list[str],
    conf: float,
    iou: float,
    wbf_iou: float,
    conf_high: float,
    min_area: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (masks_512 uint8 (K, TILE, TILE), scores (K,))。"""

    # --- Stream 1: YOLO 4-fold × TTA ---
    all_boxes, all_scores, all_masks = [], [], []
    for m in yolo_models:
        for mode in modes:
            b, s, ms = run_one(m, padded_img, mode, conf, iou, device)
            if len(b) == 0:
                continue
            all_boxes.append(b / IMG_SIZE)
            all_scores.append(s)
            all_masks.append(ms)

    if not all_boxes:
        return np.zeros((0, TILE, TILE), np.uint8), np.zeros((0,), np.float32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    yolo_masks_pool = np.concatenate(all_masks, axis=0)
    T = len(yolo_models) * len(modes)

    fused_boxes_n, fused_scores, fused_yolo_masks = weighted_box_fusion(
        boxes, scores, yolo_masks_pool, iou_thr=wbf_iou, T=T,
    )
    if len(fused_boxes_n) == 0:
        return np.zeros((0, TILE, TILE), np.uint8), np.zeros((0,), np.float32)

    # --- Stream 2: MedSAM-2 refinement on high-conf boxes ---
    high_keep = fused_scores >= conf_high
    high_boxes_n = fused_boxes_n[high_keep]
    high_scores = fused_scores[high_keep]
    high_yolo_masks = fused_yolo_masks[high_keep]

    if len(high_boxes_n) == 0:
        return np.zeros((0, TILE, TILE), np.uint8), np.zeros((0,), np.float32)

    high_boxes_px = (high_boxes_n * IMG_SIZE).astype(np.float32)
    # RGB 餵 SAM；YOLO 餵的是 BGR(cv2 讀)
    img_rgb = padded_img[:, :, ::-1].copy()

    # 多 fold MedSAM ensemble：每個 refiner 對同一張圖 embed + decode，mask 投票（>=半數）
    accum = np.zeros((len(high_boxes_px), IMG_SIZE, IMG_SIZE), np.float32)
    iou_accum = np.zeros((len(high_boxes_px),), np.float32)
    for ref in refiners:
        ref.embed(img_rgb)
        m, ious = ref.decode(high_boxes_px)
        accum += m.astype(np.float32)
        iou_accum += ious
    n_ref = len(refiners)
    # 多數投票二值化（>=半數的 refiner 認為是 mask）
    medsam_masks = (accum >= (n_ref / 2.0)).astype(np.uint8)
    medsam_ious = iou_accum / max(n_ref, 1)

    # --- Fusion: MedSAM 主、YOLO fallback ---
    final_masks = np.zeros_like(medsam_masks)
    for i in range(len(high_boxes_px)):
        mm = medsam_masks[i]
        if mm.sum() >= min_area:
            final_masks[i] = mm
        else:
            final_masks[i] = high_yolo_masks[i]

    # center crop → 512
    final_masks = final_masks[:, PAD:PAD + TILE, PAD:PAD + TILE]
    keep = final_masks.reshape(len(final_masks), -1).sum(1) >= min_area
    final_masks = final_masks[keep]
    final_scores = high_scores[keep]
    # 用 MedSAM iou 微調 score（可選）— 這裡保守只用 YOLO WBF score
    return final_masks, final_scores


# ---------- 主流程 ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-weights", nargs="+", required=True)
    ap.add_argument("--medsam-ckpt", default="pretrained/MedSAM2_latest.pt")
    ap.add_argument("--medsam-cfg", default="configs/sam2.1_hiera_t512.yaml")
    ap.add_argument("--medsam-loras", nargs="*", default=None,
                    help="LoRA adapter 資料夾（每 fold 一個）；None 代表用 base checkpoint")
    ap.add_argument("--medsam-decoders", nargs="*", default=None,
                    help="可選：每 fold 一個 decoder.pt（與 --medsam-loras 對應）")
    ap.add_argument("--src", default="dataset/test")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO NMS 前 conf")
    ap.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU")
    ap.add_argument("--wbf-iou", type=float, default=0.55)
    ap.add_argument("--conf-high", type=float, default=0.25,
                    help="進入 MedSAM refine 的 fused score 下限")
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--tta-modes", nargs="+", default=None, choices=TTA_MODES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    from medsam2_stream import MedSAM2Refiner

    yolo_models = [YOLO(w) for w in args.yolo_weights]

    lora_dirs = args.medsam_loras or [None]
    decoders = args.medsam_decoders or [None] * len(lora_dirs)
    refiners = [
        MedSAM2Refiner(
            ckpt=args.medsam_ckpt, cfg=args.medsam_cfg,
            lora_dir=ld, decoder_ckpt=dc,
            device=args.device, fp16=args.fp16,
        )
        for ld, dc in zip(lora_dirs, decoders)
    ]

    modes = args.tta_modes or TTA_MODES
    print(f"YOLO={len(yolo_models)}, MedSAM refiners={len(refiners)}, TTA={modes}")

    test_dir = Path(args.src)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta) if args.meta else test_dir.parent / "tile_meta.csv"
    test_ids = [p.stem for p in test_dir.glob("*.tif")] + [p.stem for p in test_dir.glob("*.png")]
    if meta_path.exists():
        test_meta = pd.read_csv(meta_path)
        test_meta = test_meta[test_meta["id"].isin(test_ids)].reset_index(drop=True)
    else:
        test_meta = pd.DataFrame(columns=["id", "source_wsi", "i", "j"])

    for tid in tqdm(test_ids, desc="dual-stream"):
        padded = pad_test_tile(tid, test_meta, test_dir)
        masks, scores = run_dual_stream_tile(
            yolo_models, refiners, padded, modes,
            conf=args.conf, iou=args.iou, wbf_iou=args.wbf_iou,
            conf_high=args.conf_high, min_area=args.min_area, device=args.device,
        )
        np.savez_compressed(
            out_dir / f"{tid}.npz",
            masks=masks, scores=scores,
            classes=np.zeros((len(masks),), np.int32),
        )


if __name__ == "__main__":
    main()
