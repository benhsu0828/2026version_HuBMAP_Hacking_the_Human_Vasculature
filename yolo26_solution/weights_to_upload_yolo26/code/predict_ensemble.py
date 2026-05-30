"""HuBMAP Vasculature — 4-fold WBF + Mask 平均 + 4-way TTA ensemble 推理

對每張 test tile：
    4 個模型 × 4 種 TTA（orig / hflip / vflip / rot90）= 16 份預測
    → 用 WBF (Weighted Box Fusion) 合併 bbox
    → 對應 mask 依 confidence 加權平均後 0.5 二值化
    → 寫 .npz（masks/scores/classes），格式與 train.py submit 相容

本機用法：
    uv run python predict_ensemble.py \
        --weights kaggle_weights/fold0.pt kaggle_weights/fold1.pt \
                  kaggle_weights/fold2.pt kaggle_weights/fold3.pt \
        --src dataset/test --out preds/ensemble
    uv run python train.py submit --preds preds/ensemble --out submission_ens.csv

Kaggle 用法：整檔貼進 notebook（或從 train.py 一併複製 pad_test_tile / encode_rle），
對 /kaggle/input/.../test/*.tif 呼叫 run_ensemble。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from train import IMG_SIZE, PAD, pad_test_tile
from dataprocess import TILE

TTA_MODES = ["orig", "hflip", "vflip", "rot90"]


# ---------- TTA 變換 ----------

def tta_forward(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "orig":
        return img
    if mode == "hflip":
        return np.ascontiguousarray(img[:, ::-1])
    if mode == "vflip":
        return np.ascontiguousarray(img[::-1, :])
    if mode == "rot90":
        return np.ascontiguousarray(np.rot90(img, k=1))
    raise ValueError(mode)


def tta_inverse_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    """mask shape (N, H, W) 或 (H, W)。將預測還原回 orig 座標。"""
    if mode == "orig":
        return mask
    if mode == "hflip":
        return np.ascontiguousarray(mask[..., :, ::-1])
    if mode == "vflip":
        return np.ascontiguousarray(mask[..., ::-1, :])
    if mode == "rot90":
        # forward = np.rot90(k=1)，逆向 = np.rot90(k=-1)
        return np.ascontiguousarray(np.rot90(mask, k=-1, axes=(-2, -1)))
    raise ValueError(mode)


def tta_inverse_box(boxes_xyxy: np.ndarray, mode: str, size: int) -> np.ndarray:
    """boxes_xyxy: (N, 4) 在「變換後」的 padded canvas 座標；還原到 orig 座標。
    假設正方形 canvas，邊長 size。"""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    if mode == "orig":
        return boxes_xyxy.copy()
    if mode == "hflip":
        return np.stack([size - x2, y1, size - x1, y2], axis=1)
    if mode == "vflip":
        return np.stack([x1, size - y2, x2, size - y1], axis=1)
    if mode == "rot90":
        # forward: (x, y) -> (y, size-x)；inverse: (x', y') -> (size-y', x')
        nx1 = size - y2
        nx2 = size - y1
        ny1 = x1
        ny2 = x2
        return np.stack([nx1, ny1, nx2, ny2], axis=1)
    raise ValueError(mode)


# ---------- 單一 (model, TTA) 預測 ----------

def run_one(
    model,
    padded_img: np.ndarray,
    mode: str,
    conf: float,
    iou: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳 (boxes_xyxy_in_orig_padded, scores, masks_in_orig_padded uint8)，已過濾 class==0。"""
    img_t = tta_forward(padded_img, mode)
    res = model.predict(
        img_t, imgsz=IMG_SIZE, conf=conf, iou=iou,
        retina_masks=True, augment=False, verbose=False, device=device,
    )[0]
    if res.masks is None or len(res.masks.data) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0, IMG_SIZE, IMG_SIZE), np.uint8))
    masks = res.masks.data.cpu().numpy().astype(np.uint8)   # (N, H, W)
    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)  # 變換後座標
    scores = res.boxes.conf.cpu().numpy().astype(np.float32)
    classes = res.boxes.cls.cpu().numpy().astype(np.int32)

    keep = classes == 0
    masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
    if len(masks) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0, IMG_SIZE, IMG_SIZE), np.uint8))

    # 反變換回 orig padded 座標
    masks = tta_inverse_mask(masks, mode)
    boxes = tta_inverse_box(boxes, mode, IMG_SIZE)
    # 確保 mask shape == (N, IMG_SIZE, IMG_SIZE)
    if masks.shape[1:] != (IMG_SIZE, IMG_SIZE):
        # ultralytics 偶爾會輸出 imgsz - 1 的 mask，這邊安全 resize
        resized = np.zeros((len(masks), IMG_SIZE, IMG_SIZE), np.uint8)
        for i, m in enumerate(masks):
            resized[i] = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        masks = resized
    return boxes, scores, masks


# ---------- WBF (手刻) + Mask 加權平均 ----------

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """單對 xyxy 的 IoU，a/b 形狀 (4,)。"""
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(
    boxes: np.ndarray,      # (M, 4) 已歸一化到 [0,1]
    scores: np.ndarray,     # (M,)
    masks: np.ndarray,      # (M, H, W) uint8 — orig padded 座標
    iou_thr: float = 0.55,
    T: int = 16,
    skip_box_thr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳 (fused_boxes_xyxy_norm, fused_scores, fused_masks uint8)。"""
    if len(boxes) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0, masks.shape[1], masks.shape[2]) if masks.ndim == 3
                         else (0, IMG_SIZE, IMG_SIZE), np.uint8))

    mask_keep = scores >= skip_box_thr
    boxes = boxes[mask_keep]; scores = scores[mask_keep]; masks = masks[mask_keep]
    if len(boxes) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0, masks.shape[1], masks.shape[2]), np.uint8))

    order = np.argsort(-scores)
    boxes = boxes[order]; scores = scores[order]; masks = masks[order]

    clusters: list[list[int]] = []
    fused_boxes: list[np.ndarray] = []  # 暫存 cluster 的當前融合 box

    for i in range(len(boxes)):
        best_iou = 0.0
        best_c = -1
        for c_idx, fb in enumerate(fused_boxes):
            iou = _iou_xyxy(boxes[i], fb)
            if iou > best_iou:
                best_iou = iou; best_c = c_idx
        if best_c >= 0 and best_iou >= iou_thr:
            clusters[best_c].append(i)
            idxs = clusters[best_c]
            w = scores[idxs]
            fused_boxes[best_c] = (boxes[idxs] * w[:, None]).sum(0) / w.sum()
        else:
            clusters.append([i])
            fused_boxes.append(boxes[i].copy())

    H, W = masks.shape[1], masks.shape[2]
    out_boxes = np.zeros((len(clusters), 4), np.float32)
    out_scores = np.zeros((len(clusters),), np.float32)
    out_masks = np.zeros((len(clusters), H, W), np.uint8)
    for c_idx, idxs in enumerate(clusters):
        w = scores[idxs]
        out_boxes[c_idx] = fused_boxes[c_idx]
        # 標準 WBF 對 cluster size 的折扣
        out_scores[c_idx] = w.mean() * min(len(idxs), T) / T
        # mask 加權平均 → 二值化
        acc = np.zeros((H, W), np.float32)
        for j, s in zip(idxs, w):
            acc += s * masks[j].astype(np.float32)
        avg = acc / w.sum()
        out_masks[c_idx] = (avg >= 0.5).astype(np.uint8)
    return out_boxes, out_scores, out_masks


# ---------- 對單張 tile 跑完整 ensemble ----------

def run_ensemble(
    models: list,
    padded_img: np.ndarray,
    modes: list[str],
    conf: float,
    iou: float,
    wbf_iou: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (masks_center_cropped uint8 (K, TILE, TILE), scores (K,))。"""
    all_boxes, all_scores, all_masks = [], [], []
    for m in models:
        for mode in modes:
            b, s, ms = run_one(m, padded_img, mode, conf, iou, device)
            if len(b) == 0:
                continue
            all_boxes.append(b / IMG_SIZE)  # 歸一化
            all_scores.append(s)
            all_masks.append(ms)
    if not all_boxes:
        return (np.zeros((0, TILE, TILE), np.uint8), np.zeros((0,), np.float32))
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    T = len(models) * len(modes)
    _, fs, fm = weighted_box_fusion(boxes, scores, masks, iou_thr=wbf_iou, T=T)
    # center crop 回 512
    fm = fm[:, PAD:PAD + TILE, PAD:PAD + TILE]
    keep = fm.reshape(len(fm), -1).sum(1) >= 50
    return fm[keep], fs[keep]


# ---------- 主流程 ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="多個 .pt 權重")
    ap.add_argument("--src", default="dataset/test")
    ap.add_argument("--meta", default=None, help="test tile_meta.csv；無則 reflect padding")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05, help="WBF 前的候選 conf（建議低）")
    ap.add_argument("--iou", type=float, default=0.7, help="模型內 NMS IoU（建議鬆）")
    ap.add_argument("--wbf-iou", type=float, default=0.55, help="WBF 分群閾值")
    ap.add_argument("--tta-modes", nargs="+", default=None,
                    choices=TTA_MODES, help=f"預設 {TTA_MODES}")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    models = [YOLO(w) for w in args.weights]
    modes = args.tta_modes or TTA_MODES
    print(f"模型數={len(models)}, TTA 模式={modes}, 每張 tile {len(models)*len(modes)} 份預測")

    test_dir = Path(args.src)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.meta) if args.meta else test_dir.parent / "tile_meta.csv"
    test_ids = [p.stem for p in test_dir.glob("*.tif")] + [p.stem for p in test_dir.glob("*.png")]
    if meta_path.exists():
        test_meta = pd.read_csv(meta_path)
        test_meta = test_meta[test_meta["id"].isin(test_ids)].reset_index(drop=True)
    else:
        test_meta = pd.DataFrame(columns=["id", "source_wsi", "i", "j"])

    for tid in tqdm(test_ids, desc="ensemble"):
        padded = pad_test_tile(tid, test_meta, test_dir)
        masks, scores = run_ensemble(
            models, padded, modes,
            conf=args.conf, iou=args.iou, wbf_iou=args.wbf_iou, device=args.device,
        )
        np.savez_compressed(
            out_dir / f"{tid}.npz",
            masks=masks, scores=scores,
            classes=np.zeros((len(masks),), np.int32),
        )


if __name__ == "__main__":
    main()
