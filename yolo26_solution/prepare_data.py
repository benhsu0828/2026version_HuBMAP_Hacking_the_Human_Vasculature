"""YOLO26 兩階段資料前處理 — 重用 root dataprocess.py 的 helper

第三名兩階段策略:
    Stage1: 全部 dataset2(雜訊標註,1211 張)→ 粗練(pretrain)
    Stage2: dataset1(乾淨標註,422 張)4-fold → 精練(finetune);train 僅 ds1,不含 ds2

影像維持 512 + 2*128 = 768(鄰居拼接 padding,與 root dataprocess / pad_test_tile 一致)。
類別維持 2 類(blood_vessel=0, glomerulus=1),unsure 兩階段皆丟棄(沿用 polygons_to_yolo_txt)。

用法:
    uv run python yolo26_solution/prepare_data.py \
        --data-root data --out yolo26_solution/yolo26_data --folds 4 --pad 128
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

# 讓 import 找得到 repo 根的 dataprocess.py(本檔在 yolo26_solution/ 底下)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dataprocess import (  # noqa: E402
    TILE,
    NAMES,
    load_polygons,
    build_padded_tile,
    polygons_to_yolo_txt,
    make_splits,
)


def write_split(ids, split, fold_name, meta, polygons, tile_dir, out_root, pad):
    """產 images/<fold_name>/<split>/*.png + labels/<fold_name>/<split>/*.txt"""
    size = TILE + 2 * pad
    img_dir = out_root / "images" / fold_name / split
    lbl_dir = out_root / "labels" / fold_name / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for tid in tqdm(ids, desc=f"{fold_name}/{split}"):
        canvas, polys = build_padded_tile(tid, meta, polygons, tile_dir, pad)
        cv2.imwrite(str(img_dir / f"{tid}.png"), canvas)
        lines = polygons_to_yolo_txt(polys, size)
        (lbl_dir / f"{tid}.txt").write_text("\n".join(lines))


def write_yaml(out_root: Path, fold_name: str) -> None:
    (out_root / f"{fold_name}.yaml").write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/{fold_name}/train\n"
        f"val: images/{fold_name}/val\n"
        f"names:\n"
        + "".join(f"  {k}: {v}\n" for k, v in NAMES.items())
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="含 tile_meta.csv / polygons.jsonl / train/")
    ap.add_argument("--out", default="yolo26_solution/yolo26_data")
    ap.add_argument("--pad", type=int, default=128)
    ap.add_argument("--folds", type=int, default=4, help="Stage2 ds1 fold 數(1~4)")
    ap.add_argument("--val-frac", type=float, default=0.05, help="Stage1 ds2 的 val 比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data_root)
    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    tile_dir = root / "train"

    meta = pd.read_csv(root / "tile_meta.csv")
    polygons = load_polygons(root / "polygons.jsonl")
    annotated = set(polygons.keys())

    # ---- Stage 1: 全部 ds2(雜訊)----
    ds2_ids = meta[(meta["dataset"] == 2) & (meta["id"].isin(annotated))]["id"].tolist()
    rng = random.Random(args.seed)
    rng.shuffle(ds2_ids)
    n_val = max(1, int(len(ds2_ids) * args.val_frac))
    s1_val, s1_train = ds2_ids[:n_val], ds2_ids[n_val:]
    print(f"[Stage1] ds2 train={len(s1_train)}, val={len(s1_val)}")
    write_split(s1_train, "train", "stage1", meta, polygons, tile_dir, out_root, args.pad)
    write_split(s1_val, "val", "stage1", meta, polygons, tile_dir, out_root, args.pad)
    write_yaml(out_root, "stage1")

    # ---- Stage 2: ds1 4-fold(train 只留 ds1,丟掉 ds2)----
    ds1_ids = set(meta[(meta["dataset"] == 1) & (meta["id"].isin(annotated))]["id"])
    folds = make_splits(meta, annotated)  # 重用 WSI×左右 切分(val 本就純 ds1)
    for k, (tr, va) in enumerate(folds[: args.folds]):
        tr_ds1 = [i for i in tr if i in ds1_ids]
        fold_name = f"stage2_fold{k}"
        print(f"[Stage2] fold{k}: train(ds1)={len(tr_ds1)}, val(ds1)={len(va)}")
        write_split(tr_ds1, "train", fold_name, meta, polygons, tile_dir, out_root, args.pad)
        write_split(va, "val", fold_name, meta, polygons, tile_dir, out_root, args.pad)
        write_yaml(out_root, fold_name)

    print(f"\n完成 → {out_root.resolve()}")


if __name__ == "__main__":
    main()
