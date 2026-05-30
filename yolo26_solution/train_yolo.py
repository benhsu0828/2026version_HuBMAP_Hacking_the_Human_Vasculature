"""YOLO26-seg 兩階段訓練入口(薄包裝 ultralytics YOLO.train)

仿第三名兩階段策略(見 HubMap-2023-3rd-Place README「How to use dataset 2」):
    Stage1: ds2 粗練 — 載入 COCO 預訓練 yolo26x-seg.pt,epoch 少、lr 高、輕度增強
    Stage2: ds1 精練 — 載入 Stage1 best.pt,epoch 多、lr 低退火到極小、重度增強

影像尺寸固定 768(512 + 2*128 padding),與 prepare_data / root pad_test_tile 一致。

用法:
    # Stage1(全 ds2)
    uv run python yolo26_solution/train_yolo.py --stage 1 \
        --data yolo26_solution/yolo26_data/stage1.yaml \
        --weights yolo26x-seg.pt --name pretrain

    # Stage2(ds1 fold0,載入 Stage1 權重)
    uv run python yolo26_solution/train_yolo.py --stage 2 --fold 0 \
        --data yolo26_solution/yolo26_data/stage2_fold0.yaml \
        --weights yolo26_solution/runs/stage1/pretrain/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMG_SIZE = 768  # 512 + 2*128;與 dataprocess.TILE/PAD、root train.IMG_SIZE 對齊

# 兩階段超參預設(可被 CLI 覆寫)。lr 走 AdamW + cosine。
STAGE_DEFAULTS = {
    1: dict(  # ds2 粗練:高 lr、少 epoch、輕度 aug
        epochs=15, lr0=2e-3, lrf=0.1,  # cos → 2e-4
        degrees=30.0, scale=0.3, mixup=0.0, copy_paste=0.0, close_mosaic=5,
        default_weights="yolo26x-seg.pt",
    ),
    2: dict(  # ds1 精練:低 lr→極小、多 epoch、重度 aug(對齊 root train.py 重增強)
        epochs=25, lr0=5e-4, lrf=1e-4,  # cos → ~5e-8
        degrees=180.0, scale=0.5, mixup=0.1, copy_paste=0.3, close_mosaic=15,
        default_weights=None,  # 必須由 --weights 指定 Stage1 best.pt
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--data", required=True, help="YOLO data yaml")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--weights", default=None, help="初始權重;Stage1 預設 yolo26x-seg.pt")
    ap.add_argument("--epochs", type=int, default=None, help="覆寫 stage 預設 epoch")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "False"])
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="hubmap-yolo26")
    args = ap.parse_args()

    d = STAGE_DEFAULTS[args.stage]
    weights = args.weights or d["default_weights"]
    if weights is None:
        raise SystemExit("Stage2 需要 --weights 指定 Stage1 的 best.pt")
    epochs = args.epochs if args.epochs is not None else d["epochs"]
    project = args.project or f"yolo26_solution/runs/stage{args.stage}"
    if args.name:
        name = args.name
    elif args.fold is not None:
        name = f"fold{args.fold}"
    else:
        name = "pretrain" if args.stage == 1 else "run"

    from ultralytics import YOLO, settings as ul_settings

    if args.wandb:
        ul_settings.update({"wandb": True})
        import wandb
        wandb.init(project=args.wandb_project, name=f"s{args.stage}_{name}",
                   config=vars(args), reinit=True)
    else:
        ul_settings.update({"wandb": False})

    print(f">>> Stage{args.stage} | data={args.data} | init={weights} | epochs={epochs} "
          f"| lr0={d['lr0']} lrf={d['lrf']} | out={project}/{name}")

    model = YOLO(weights)
    model.train(
        data=args.data,
        imgsz=IMG_SIZE,
        epochs=epochs,
        batch=args.batch,
        optimizer="AdamW",
        lr0=d["lr0"], lrf=d["lrf"], cos_lr=True,
        warmup_epochs=3,
        close_mosaic=d["close_mosaic"],
        # 幾何 / 顏色增強(stage 差異化)
        degrees=d["degrees"], scale=d["scale"],
        fliplr=0.5, flipud=0.5, translate=0.1, shear=0.0, perspective=0.0,
        mosaic=1.0, mixup=d["mixup"], copy_paste=d["copy_paste"],
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        # 加速 / 顯存
        amp=True, cache=args.cache, workers=args.workers, device=args.device,
        # 輸出(絕對路徑,繞過 ultralytics 全域 runs_dir)
        # exist_ok=True:固定寫進 pretrain / fold{K},不讓 ultralytics 自動加序號
        # (否則 run_all.sh 接力與 package_for_kaggle.sh 會找不到目錄)
        project=str(Path(project).resolve()), name=name, exist_ok=True,
        patience=20, save_period=10,
        # mask / NMS(對齊 root train.py)
        overlap_mask=False, mask_ratio=2,
        iou=0.6, conf=0.001,
    )


if __name__ == "__main__":
    main()
