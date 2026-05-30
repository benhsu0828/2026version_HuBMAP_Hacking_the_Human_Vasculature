"""HuBMAP Vasculature — 離線資料前處理

職責：
1. 讀 tile_meta.csv + polygons.jsonl
2. 對每張有標註的 tile 做 128px 鄰居拼接 padding（512 -> 768）
3. 將相鄰 tile 的 polygons 平移合併進來，解決血管橫跨邊界的截斷問題
4. 依「WSI + spatial」做 4-fold 切分（嚴格防 leak）
5. 輸出 Ultralytics YOLO-seg 格式（images/ + labels/ + foldK.yaml）

用法：
    uv run python dataprocess.py --out yolo_data --pad 128
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

TILE = 512
# 評分只看 blood_vessel；保留 glomerulus 作為輔助多任務訊號
CLASS_MAP = {"blood_vessel": 0, "glomerulus": 1}
NAMES = {0: "blood_vessel", 1: "glomerulus"}


# ---------- I/O ----------

def load_polygons(jsonl_path: Path) -> dict[str, list[dict]]:
    """讀 polygons.jsonl，回傳 {tile_id: [{type, coords:np.ndarray(N,2) int}, ...]}"""
    out: dict[str, list[dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            anns = []
            for a in rec["annotations"]:
                # coordinates shape: [[ [x,y], [x,y], ... ]]  (外層多一層 list)
                pts = np.asarray(a["coordinates"][0], dtype=np.int32)
                anns.append({"type": a["type"], "coords": pts})
            out[rec["id"]] = anns
    return out


# ---------- 鄰居拼接 padding ----------

def build_padded_tile(
    tile_id: str,
    meta: pd.DataFrame,
    polygons: dict[str, list[dict]],
    tile_dir: Path,
    pad: int = 128,
) -> tuple[np.ndarray, list[dict]]:
    """產出 (pad*2+TILE) 方形圖 + 合併後的 polygons（座標已平移至新圖座標系）。

    座標約定：tile_meta 的 (i, j) 為「tile 左上角」在 WSI 中的像素位置（i=row, j=col）。
    我們以中心 tile 的左上角 (i0, j0) 為基準，padded 圖上 (0,0) 對應 WSI 的 (i0-pad, j0-pad)。
    因此任何 polygon 點 (x_wsi, y_wsi) 在 padded 圖上的座標為：
        x_pad = x_wsi - (j0 - pad)   # x 對應 column = j
        y_pad = y_wsi - (i0 - pad)   # y 對應 row    = i
    對中心 tile，其 polygon 原本就是 tile 內局部座標 (0..TILE)，故直接 +pad。
    對鄰居 tile，其局部座標需先轉回 WSI 全域，再轉到 padded 圖。
    """
    row = meta.loc[meta["id"] == tile_id].iloc[0]
    wsi, i0, j0 = int(row["source_wsi"]), int(row["i"]), int(row["j"])
    size = TILE + 2 * pad

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    # 中心圖
    center_img = cv2.imread(str(tile_dir / f"{tile_id}.tif"))
    if center_img is None:
        center_img = cv2.imread(str(tile_dir / f"{tile_id}.png"))
    canvas[pad : pad + TILE, pad : pad + TILE] = center_img

    # 建 WSI 內 (i,j)->id 查詢表
    wsi_tiles = meta[meta["source_wsi"] == wsi]
    pos_index = {(int(r.i), int(r.j)): r.id for r in wsi_tiles.itertuples()}

    merged_polys: list[dict] = []
    # 先加中心 tile 的 polygons（+pad 偏移）
    for ann in polygons.get(tile_id, []):
        shifted = ann["coords"].copy()
        shifted[:, 0] += pad  # x
        shifted[:, 1] += pad  # y
        merged_polys.append({"type": ann["type"], "coords": shifted})

    # 8 個鄰居方向
    neighbors = [
        (-TILE, -TILE), (-TILE, 0), (-TILE, TILE),
        (0,     -TILE),             (0,     TILE),
        (TILE,  -TILE), (TILE,  0), (TILE,  TILE),
    ]
    for di, dj in neighbors:
        ni, nj = i0 + di, j0 + dj
        nid = pos_index.get((ni, nj))
        if nid is None:
            continue
        nb_path = tile_dir / f"{nid}.tif"
        nb_img = cv2.imread(str(nb_path))
        if nb_img is None:
            nb_img = cv2.imread(str(tile_dir / f"{nid}.png"))
        if nb_img is None:
            continue

        # 鄰居 tile 在 padded 畫布上的位置
        # padded 畫布 (0,0) <-> WSI (i0-pad, j0-pad)
        # 鄰居 tile 左上 <-> WSI (ni, nj)
        # -> 在畫布上的左上 = (ni-(i0-pad), nj-(j0-pad)) = (di+pad, dj+pad)
        y_off = di + pad  # row
        x_off = dj + pad  # col
        # 計算相交區域
        y1, y2 = max(0, y_off), min(size, y_off + TILE)
        x1, x2 = max(0, x_off), min(size, x_off + TILE)
        if y1 >= y2 or x1 >= x2:
            continue
        sy1, sx1 = y1 - y_off, x1 - x_off
        sy2, sx2 = sy1 + (y2 - y1), sx1 + (x2 - x1)
        canvas[y1:y2, x1:x2] = nb_img[sy1:sy2, sx1:sx2]

        # 合併鄰居 polygons
        for ann in polygons.get(nid, []):
            shifted = ann["coords"].copy()
            shifted[:, 0] += x_off  # 局部 x + 鄰居在 canvas 上的 x_off
            shifted[:, 1] += y_off
            merged_polys.append({"type": ann["type"], "coords": shifted})

    # 邊界 polygon clipping：clip 到 [0, size)
    for p in merged_polys:
        np.clip(p["coords"][:, 0], 0, size - 1, out=p["coords"][:, 0])
        np.clip(p["coords"][:, 1], 0, size - 1, out=p["coords"][:, 1])

    return canvas, merged_polys


# ---------- YOLO 標籤 ----------

def polygons_to_yolo_txt(polys: list[dict], size: int) -> list[str]:
    """將 polygons 轉成 Ultralytics seg TXT 行（class + 歸一化多邊形）。"""
    lines: list[str] = []
    for p in polys:
        if p["type"] == "unsure":
            continue  # 忽略不確定區域
        cls = CLASS_MAP.get(p["type"])
        if cls is None:
            continue
        coords = p["coords"].astype(np.float32)
        # 至少 3 個點才能成多邊形
        if len(coords) < 3:
            continue
        coords[:, 0] /= size
        coords[:, 1] /= size
        flat = " ".join(f"{v:.6f}" for v in coords.reshape(-1))
        lines.append(f"{cls} {flat}")
    return lines


# ---------- 4-Fold split ----------

def make_splits(meta: pd.DataFrame, annotated_ids: set[str]) -> list[tuple[list[str], list[str]]]:
    """4-fold by WSI + spatial。回傳 [(train_ids, val_ids), ...] 長度 4。

    val 一律從 dataset==1（高品質）抽；train 用同 fold 之外的 ds1 + 全部 ds2，
    並排除掉「同 WSI 且空間落在該 fold val block 內」的 ds2 tile（防 leak）。
    WSI 3, 4（只在 ds2）每 fold 都加入訓練。
    """
    ds1 = meta[(meta["dataset"] == 1) & (meta["id"].isin(annotated_ids))]
    ds2 = meta[(meta["dataset"] == 2) & (meta["id"].isin(annotated_ids))]

    # 對 WSI 1 / WSI 2 各取 j 中位數
    blocks: list[tuple[int, str, int]] = []  # (wsi, side, j_median)
    for wsi in (1, 2):
        jm = int(ds1[ds1["source_wsi"] == wsi]["j"].median())
        blocks.append((wsi, "left", jm))   # j < jm
        blocks.append((wsi, "right", jm))  # j >= jm

    folds: list[tuple[list[str], list[str]]] = []
    for wsi, side, jm in blocks:
        if side == "left":
            val_mask_ds1 = (ds1["source_wsi"] == wsi) & (ds1["j"] < jm)
            leak_mask_ds2 = (ds2["source_wsi"] == wsi) & (ds2["j"] < jm)
        else:
            val_mask_ds1 = (ds1["source_wsi"] == wsi) & (ds1["j"] >= jm)
            leak_mask_ds2 = (ds2["source_wsi"] == wsi) & (ds2["j"] >= jm)

        val_ids = ds1.loc[val_mask_ds1, "id"].tolist()
        train_ids = (
            ds1.loc[~val_mask_ds1, "id"].tolist()
            + ds2.loc[~leak_mask_ds2, "id"].tolist()
        )
        folds.append((train_ids, val_ids))
    return folds


# ---------- 主流程 ----------

def write_fold(
    fold_idx: int,
    train_ids: list[str],
    val_ids: list[str],
    meta: pd.DataFrame,
    polygons: dict[str, list[dict]],
    tile_dir: Path,
    out_root: Path,
    pad: int,
) -> None:
    size = TILE + 2 * pad
    fold_name = f"fold_{fold_idx}"
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = out_root / "images" / fold_name / split
        lbl_dir = out_root / "labels" / fold_name / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for tid in tqdm(ids, desc=f"{fold_name}/{split}"):
            canvas, polys = build_padded_tile(tid, meta, polygons, tile_dir, pad)
            cv2.imwrite(str(img_dir / f"{tid}.png"), canvas)
            lines = polygons_to_yolo_txt(polys, size)
            (lbl_dir / f"{tid}.txt").write_text("\n".join(lines))

    yaml_path = out_root / f"{fold_name}.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/{fold_name}/train\n"
        f"val: images/{fold_name}/val\n"
        f"names:\n"
        + "".join(f"  {k}: {v}\n" for k, v in NAMES.items())
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--out", default="yolo_data")
    ap.add_argument("--pad", type=int, default=128)
    ap.add_argument("--folds", type=int, default=4, help="只跑前 N folds (debug 用)")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)

    meta = pd.read_csv(root / "tile_meta.csv")
    polygons = load_polygons(root / "polygons.jsonl")
    annotated_ids = set(polygons.keys())

    folds = make_splits(meta, annotated_ids)
    print(f"產生 {len(folds)} folds：")
    for k, (tr, va) in enumerate(folds):
        print(f"  fold {k}: train={len(tr)}, val={len(va)}")

    tile_dir = root / "train"
    for k, (tr, va) in enumerate(folds[: args.folds]):
        write_fold(k, tr, va, meta, polygons, tile_dir, out_root, args.pad)

    print(f"\n完成 → {out_root.resolve()}")


if __name__ == "__main__":
    main()
