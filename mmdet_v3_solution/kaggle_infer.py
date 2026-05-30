# -*- coding: utf-8 -*-
"""Part 3 — Kaggle 線上推理 + 後處理（單模型）。

可直接整檔貼進 Kaggle Notebook，或以 CLI 執行：
    python kaggle_infer.py --config configs/stage2_dataset1_finetune.py \
        --checkpoint work_dirs/stage2/best_coco_segm_mAP_epoch_20.pth \
        --img-dir /kaggle/input/.../test --out submission.csv

效能與準確度優化：
  1. FP16 推理：torch.autocast('cuda', float16)，省顯存、提速（T4 友善）。
  2. torch.compile：算子融合加速；Cascade RoI head 有動態流程可能 graph-break，
     故包 try/except，失敗自動退回 eager。
  3. TTA：orig + hflip + vflip + rot90/180/270 共 6 視角，逆變換回原座標後集成。
  4. 形態學後處理：開運算（先腐蝕再膨脹）去毛刺/孤立噪點 + 移除過小連通域。
  5. 提交：只輸出 blood_vessel(label 0)，COCO RLE → zlib → base64（HuBMAP 格式）。
"""
import argparse
import base64
import os
import os.path as osp
import sys
import warnings
import zlib

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import _mask as coco_mask

# 確保能 import 本方案的 models（觸發 InternImage 註冊）
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from mmdet.apis import inference_detector, init_detector  # noqa: E402

TILE = 512  # 提交固定 512x512


# ---------------------------------------------------------------------------
# TTA：6 視角的「正向變換」與對應「逆變換」（作用在方形影像/遮罩上）
# ---------------------------------------------------------------------------
def _tta_forward(img, mode):
    if mode == 'orig':
        return img
    if mode == 'hflip':
        return np.ascontiguousarray(img[:, ::-1])
    if mode == 'vflip':
        return np.ascontiguousarray(img[::-1, :])
    if mode == 'rot90':
        return np.ascontiguousarray(np.rot90(img, k=1))
    if mode == 'rot180':
        return np.ascontiguousarray(np.rot90(img, k=2))
    if mode == 'rot270':
        return np.ascontiguousarray(np.rot90(img, k=3))
    raise ValueError(mode)


def _tta_inverse_mask(mask, mode):
    """把某視角預測出的 mask 還原回原始方位。"""
    if mode == 'orig':
        return mask
    if mode == 'hflip':
        return np.ascontiguousarray(mask[:, ::-1])
    if mode == 'vflip':
        return np.ascontiguousarray(mask[::-1, :])
    if mode == 'rot90':      # 正向 k=1，逆向 k=-1(=3)
        return np.ascontiguousarray(np.rot90(mask, k=-1))
    if mode == 'rot180':
        return np.ascontiguousarray(np.rot90(mask, k=-2))
    if mode == 'rot270':     # 正向 k=3，逆向 k=-3(=1)
        return np.ascontiguousarray(np.rot90(mask, k=-3))
    raise ValueError(mode)


TTA_MODES = ['orig', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']


# ---------------------------------------------------------------------------
# 後處理：形態學開運算 + 移除過小連通域
# ---------------------------------------------------------------------------
def morphological_opening(mask, ksize=3, iterations=1):
    """先腐蝕(Erosion)再膨脹(Dilation) = 開運算。
    去除集成後遮罩邊緣的毛刺與孤立的小噪點，邊界更乾淨。"""
    m = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(m, kernel, iterations=iterations)
    opened = cv2.dilate(eroded, kernel, iterations=iterations)
    return opened.astype(bool)


def remove_small(mask, min_area=10):
    """丟掉面積過小的連通域（開運算後可能殘留碎片）。"""
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m, dtype=bool)
    for i in range(1, num):  # 0 是背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = True
    return keep


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                    dtype=np.float32)


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms_instances(instances, iou_thr=0.5):
    """以 bbox IoU 對「跨 TTA 視角彙整後」的 instance 做去重，保留高分者。"""
    instances = sorted(instances, key=lambda x: -x['score'])
    kept = []
    for inst in instances:
        dup = False
        for k in kept:
            if k['label'] == inst['label'] and \
                    box_iou(k['bbox'], inst['bbox']) > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(inst)
    return kept


# ---------------------------------------------------------------------------
# RLE 編碼（與專案 train.py / HuBMAP 官方一致：COCO RLE → zlib → base64）
# ---------------------------------------------------------------------------
def encode_binary_mask(mask: np.ndarray) -> str:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    mask = np.squeeze(mask)
    m = mask.reshape(mask.shape[0], mask.shape[1], 1).astype(np.uint8)
    m = np.asfortranarray(m)
    encoded = coco_mask.encode(m)[0]['counts']
    compressed = zlib.compress(encoded, zlib.Z_BEST_COMPRESSION)
    return base64.b64encode(compressed).decode('ascii')


# ---------------------------------------------------------------------------
# 模型載入（含 torch.compile + fallback）
# ---------------------------------------------------------------------------
def build_model(config, checkpoint, device='cuda', use_compile=True):
    model = init_detector(config, checkpoint, device=device)
    # 推理不需梯度檢查點；遞迴關閉各 layer 的 with_cp（InternImage 下放到 layer）
    for m in model.modules():
        if hasattr(m, 'with_cp'):
            m.with_cp = False
    # 改成乾淨的 ndarray pipeline（去掉 LoadAnnotations）
    model.cfg.test_dataloader.dataset.pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor')),
    ]
    if use_compile:
        try:
            model = torch.compile(model, mode='max-autotune')
            print('[infer] torch.compile 已啟用 (max-autotune)')
        except Exception as e:  # noqa: BLE001
            warnings.warn(f'torch.compile 失敗，退回 eager 模式：{e}')
    return model


@torch.no_grad()
def infer_one_tile(model, img, score_thr=0.001, mask_thr=0.5, min_area=10,
                   device='cuda'):
    """對單張 512x512 tile 做 6 視角 TTA 推理 + 集成 + 後處理。
    回傳 list[ {mask(bool 512x512), score, box[x1,y1,x2,y2] 512px} ]，僅 blood_vessel(label 0)。

    注意：score_thr 預設 0.001（不是 0.3）。HuBMAP 是 mAP@IoU0.6 按 confidence 積分
    PR 曲線，丟掉低信心 TP 會直接砍掉高 recall 區段的分數。WBF/下游再依需要過濾。"""
    pooled = []
    for mode in TTA_MODES:
        timg = _tta_forward(img, mode)
        # FP16 autocast 前向（DCNv3_pytorch 內部以 grid_sample 為主，autocast 安全）
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            result = inference_detector(model, timg)
        inst = result.pred_instances
        scores = inst.scores.detach().float().cpu().numpy()
        labels = inst.labels.detach().cpu().numpy()
        if 'masks' not in inst:
            continue
        masks = inst.masks.detach().cpu().numpy()  # (N, H, W) bool，於該視角方位
        for s, lb, mk in zip(scores, labels, masks):
            if s < score_thr or lb != 0:   # 只取 blood_vessel
                continue
            inv = _tta_inverse_mask(mk.astype(np.uint8), mode).astype(bool)
            bbox = mask_to_bbox(inv)
            if bbox is None:
                continue
            pooled.append(dict(mask=inv, score=float(s), label=int(lb),
                               bbox=bbox))

    # 跨視角去重
    kept = nms_instances(pooled, iou_thr=0.5)

    # 形態學後處理 + 移除小碎片
    out = []
    for inst in kept:
        m = morphological_opening(inst['mask'], ksize=3, iterations=1)
        if min_area > 0:
            m = remove_small(m, min_area=min_area)
        if m.sum() == 0:
            continue
        box = mask_to_bbox(m)            # 後處理後重算 box（512px）
        if box is None:
            continue
        out.append(dict(mask=m, score=inst['score'], box=box.tolist()))
    return out


def _mask_to_coco_rle(mask):
    """bool/uint8 512x512 → COCO 壓縮 RLE dict（counts 為 bytes，pickle 友善）。
    下游 fusion notebook 用 pycocotools.mask.decode(rle) 還原成 512x512 uint8。

    注意：coco_mask 是 pycocotools 低階 `_mask` 模組，encode 要 3D (H,W,N) 輸入、
    回傳 list（對齊本檔 encode_binary_mask 的寫法），不能傳 2D。"""
    m = mask.reshape(mask.shape[0], mask.shape[1], 1).astype(np.uint8)
    m = np.asfortranarray(m)
    return coco_mask.encode(m)[0]   # 取 [0]；counts 維持 bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/stage2_dataset1_finetune.py')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--img-dir', required=True, help='測試 tile 影像資料夾')
    ap.add_argument('--out', default='submission.csv')
    # 預設 0.001：AP 指標要保留低信心 TP，不可設 0.3（會砍掉高 recall 區段）
    ap.add_argument('--score-thr', type=float, default=0.001)
    ap.add_argument('--min-area', type=int, default=10,
                    help='後處理移除小於此面積的連通域；設 0 不過濾')
    ap.add_argument('--pkl-out', default=None,
                    help='另存 ensemble 用 pkl（keyed by img_id），給跨版本 WBF 用')
    ap.add_argument('--no-compile', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.config, args.checkpoint, device=device,
                        use_compile=(not args.no_compile))

    exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    files = sorted(f for f in os.listdir(args.img_dir)
                   if f.lower().endswith(exts))
    rows = []
    pkl_dict = {} if args.pkl_out else None
    for fname in files:
        tid = osp.splitext(fname)[0]
        img = cv2.imread(osp.join(args.img_dir, fname))  # BGR；DetDataPreprocessor 內 bgr_to_rgb
        preds = infer_one_tile(model, img, score_thr=args.score_thr,
                               min_area=args.min_area, device=device)
        parts = []
        for p in preds:
            parts.append(f"0 {p['score']:.4f} {encode_binary_mask(p['mask'])}")
        rows.append(dict(id=tid, height=TILE, width=TILE,
                         prediction_string=' '.join(parts)))
        if pkl_dict is not None:
            # ensemble pkl：mask 存 COCO RLE（精簡），box 512px，score 原值
            pkl_dict[tid] = [
                dict(rle=_mask_to_coco_rle(p['mask']),
                     score=p['score'], box=p['box'])
                for p in preds
            ]
        print(f'{tid}: {len(parts)} instances')

    pd.DataFrame(rows, columns=['id', 'height', 'width',
                                'prediction_string']).to_csv(
        args.out, index=False)
    print(f'寫入 {args.out}（{len(rows)} 列）')

    if pkl_dict is not None:
        import pickle
        with open(args.pkl_out, 'wb') as f:
            pickle.dump(pkl_dict, f)
        n_inst = sum(len(v) for v in pkl_dict.values())
        print(f'寫入 {args.pkl_out}（{len(pkl_dict)} 圖，共 {n_inst} instances）')


if __name__ == '__main__':
    main()
