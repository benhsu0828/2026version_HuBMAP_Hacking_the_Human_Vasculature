"""HuBMAP Vasculature — 訓練 / 推理 / 提交

三個子命令：
    uv run python train.py train   --fold 0 --weights yolo26x-seg.pt
    uv run python train.py predict --weights runs/seg/fold0/weights/best.pt --out preds/fold0
    uv run python train.py submit  --preds preds/fold0 --out submission.csv

設計要點：
- 訓練：imgsz=768（對應 dataprocess.py 的 128px padding 後尺寸）、固定 LR（lrf=1.0）、
  Ultralytics 內建 ModelEMA 預設開啟、重度幾何增強、close_mosaic=15
- 推理：對 test tile 同樣做 128px padding -> 預測 -> center crop 回 512×512 -> RLE

推理與提交範例：

uv run python train.py predict \
    --weights runs/seg/fold3/weights/best.pt \
    --src dataset/test --out preds/single --tta
uv run python train.py submit --preds preds/single --out submission.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataprocess import TILE, build_padded_tile, load_polygons

PAD = 128
IMG_SIZE = TILE + 2 * PAD  # 768


# ---------- TRAIN ----------

def cmd_train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO, settings as ul_settings

    # Ultralytics 內建 wandb 整合：開啟 settings.wandb=True 後訓練自動 log
    if args.wandb:
        ul_settings.update({"wandb": True})
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"fold{args.fold}",
            config=vars(args),
            reinit=True,
        )
    else:
        ul_settings.update({"wandb": False})

    data_yaml = args.data or f"yolo_data/fold_{args.fold}.yaml"
    model = YOLO(args.weights)
    model.train(
        data=data_yaml,
        imgsz=IMG_SIZE,
        epochs=args.epochs,
        batch=args.batch,
        optimizer="AdamW",
        lr0=1e-3, lrf=1.0,           # 固定學習率（不退火），對應 ds2 雜訊
        cos_lr=False,
        warmup_epochs=3,
        close_mosaic=15,              # 最後 15 epochs 關 mosaic，配合固定 LR 更穩
        # 重度幾何增強（前 3 名 + 使用者指定）
        degrees=180.0, scale=0.5, fliplr=0.5, flipud=0.5,
        translate=0.1, shear=0.0, perspective=0.0,
        mosaic=1.0, mixup=0.1, copy_paste=0.3,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        # 加速與顯存
        amp=True, cache=args.cache,
        workers=args.workers, device=args.device,
        # 輸出（絕對路徑，繞過 Ultralytics 全域 settings.runs_dir）
        project=str(Path("runs/seg").resolve()), name=f"fold{args.fold}",
        patience=20, save_period=10,
        # mask / NMS
        overlap_mask=False, mask_ratio=2,
        iou=0.6, conf=0.001,
        # EMA 內建啟用（ultralytics.utils.torch_utils.ModelEMA）— 無需顯式參數
    )


# ---------- PREDICT ----------

def pad_test_tile(tile_id: str, test_meta: pd.DataFrame, test_dir: Path) -> np.ndarray:
    """對測試 tile 做 padding。若鄰居存在於 test_meta 則拼接，否則用 reflect。"""
    row = test_meta.loc[test_meta["id"] == tile_id]
    if len(row) and {"i", "j", "source_wsi"}.issubset(row.columns):
        # 走鄰居拼接（用 dataprocess.build_padded_tile 邏輯，無 polygons）
        canvas, _ = build_padded_tile(tile_id, test_meta, {}, test_dir, PAD)
        return canvas
    # 回退：reflect padding
    img = cv2.imread(str(test_dir / f"{tile_id}.tif"))
    if img is None:
        img = cv2.imread(str(test_dir / f"{tile_id}.png"))
    return cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)


def cmd_predict(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    test_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 嘗試讀 test 的 tile_meta（若存在）
    meta_path = Path(args.meta) if args.meta else test_dir.parent / "tile_meta.csv"
    if meta_path.exists():
        test_meta = pd.read_csv(meta_path)
        # 過濾出 test 集 id
        test_ids = [p.stem for p in test_dir.glob("*.tif")] + [p.stem for p in test_dir.glob("*.png")]
        test_meta = test_meta[test_meta["id"].isin(test_ids)].reset_index(drop=True)
    else:
        test_meta = pd.DataFrame(columns=["id", "source_wsi", "i", "j"])
        test_ids = [p.stem for p in test_dir.glob("*.tif")] + [p.stem for p in test_dir.glob("*.png")]

    for tid in tqdm(test_ids, desc="predict"):
        padded = pad_test_tile(tid, test_meta, test_dir)
        res = model.predict(
            padded, imgsz=IMG_SIZE,
            conf=args.conf, iou=args.iou,
            retina_masks=True, augment=args.tta,
            verbose=False, device=args.device,
        )[0]
        # masks: (N, H, W) — 已是 padded 圖大小
        if res.masks is None or len(res.masks.data) == 0:
            np.savez_compressed(out_dir / f"{tid}.npz", masks=np.zeros((0, TILE, TILE), np.uint8),
                                scores=np.zeros((0,), np.float32),
                                classes=np.zeros((0,), np.int32))
            continue

        masks = res.masks.data.cpu().numpy().astype(np.uint8)   # (N, H, W)
        # center crop 回 512×512
        masks = masks[:, PAD : PAD + TILE, PAD : PAD + TILE]
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        classes = res.boxes.cls.cpu().numpy().astype(np.int32)

        # 過濾：只留 blood_vessel（class 0），且 mask 面積 >= 50px
        keep = (classes == 0) & (masks.reshape(len(masks), -1).sum(1) >= 50)
        np.savez_compressed(
            out_dir / f"{tid}.npz",
            masks=masks[keep], scores=scores[keep], classes=classes[keep],
        )


# ---------- SUBMIT ----------

def encode_binary_mask(mask: np.ndarray) -> str:
    """HuBMAP 官方格式：COCO RLE → zlib → base64。"""
    import base64, zlib
    from pycocotools import _mask as coco_mask
    if mask.dtype != bool:
        mask = mask.astype(bool)
    mask = np.squeeze(mask)
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)
    encoded = coco_mask.encode(mask_to_encode)[0]["counts"]
    compressed = zlib.compress(encoded, zlib.Z_BEST_COMPRESSION)
    return base64.b64encode(compressed).decode("ascii")


def cmd_submit(args: argparse.Namespace) -> None:
    preds_dir = Path(args.preds)
    rows = []
    for npz_path in sorted(preds_dir.glob("*.npz")):
        tid = npz_path.stem
        data = np.load(npz_path)
        masks, scores = data["masks"], data["scores"]
        # 競賽提交格式：每張圖一行，prediction_string = "score1 rle1 score2 rle2 ..."
        parts = []
        for m, s in zip(masks, scores):
            if m.sum() == 0:
                continue
            # HuBMAP 格式：每個 instance "0 {conf} {base64_mask}"
            parts.append(f"0 {s:.4f} {encode_binary_mask(m)}")
        rows.append({"id": tid, "height": TILE, "width": TILE,
                     "prediction_string": " ".join(parts)})

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "height", "width", "prediction_string"])
        w.writeheader()
        w.writerows(rows)
    print(f"寫入 {out} （{len(rows)} 列）")


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--fold", type=int, required=True)
    p_tr.add_argument("--weights", default="yolo26x-seg.pt")
    p_tr.add_argument("--data", default=None)
    p_tr.add_argument("--epochs", type=int, default=80)
    p_tr.add_argument("--batch", type=int, default=8)
    p_tr.add_argument("--workers", type=int, default=8)
    p_tr.add_argument("--device", default="0")
    p_tr.add_argument("--cache", default="ram", choices=["ram", "disk", "False"])
    p_tr.add_argument("--wandb", action="store_true", help="開啟 W&B 訓練紀錄")
    p_tr.add_argument("--wandb-project", default="hubmap-vasculature", help="W&B project 名稱")
    p_tr.set_defaults(func=cmd_train)

    p_pr = sub.add_parser("predict")
    p_pr.add_argument("--weights", required=True)
    p_pr.add_argument("--src", default="dataset/test")
    p_pr.add_argument("--meta", default=None, help="test 的 tile_meta.csv 路徑；無則用 reflect padding")
    p_pr.add_argument("--out", required=True)
    p_pr.add_argument("--conf", type=float, default=0.15)
    p_pr.add_argument("--iou", type=float, default=0.6)
    p_pr.add_argument("--tta", action="store_true")
    p_pr.add_argument("--device", default="0")
    p_pr.set_defaults(func=cmd_predict)

    p_sb = sub.add_parser("submit")
    p_sb.add_argument("--preds", required=True)
    p_sb.add_argument("--out", default="submission.csv")
    p_sb.set_defaults(func=cmd_submit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
