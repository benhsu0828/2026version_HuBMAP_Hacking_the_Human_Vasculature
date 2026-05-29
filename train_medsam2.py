"""MedSAM-2 LoRA + decoder full FT — HuBMAP vasculature

訓練策略：
    image encoder: PEFT LoRA (rank 16) on attn.qkv / attn.proj
    prompt encoder: 凍結
    mask decoder: 全參數微調 (no_obj_score head 也跟著訓)

資料：
    讀 yolo_data/fold_K/{train,val} 的 PNG + YOLO-seg polygon TXT
    過濾 dataset==1 (透過 dataset/tile_meta.csv)
    每張 tile 隨機挑 1 個 blood_vessel instance 當 sample
    bbox prompt = mask 緊湊外接矩形 + 隨機 jitter (±5%)
    val 用該 fold 的 val split (純 ds1)

Loss：Dice + BCE (sum) + IoU prediction MSE

正式訓練 4 個 fold（每個約 30 epoch × 10s ≈ 5 分鐘 on RTX 6000）
用法：
    for f in 0 1 2 3; do uv run python train_medsam2.py --fold $f --epochs 30 --batch 4; done

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_THIRD = Path(__file__).resolve().parent / "third_party" / "MedSAM2"
if str(_THIRD) not in sys.path:
    sys.path.insert(0, str(_THIRD))

import sam2  # noqa: F401 (hydra init)
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms


# ---------- Dataset ----------

def parse_yolo_seg_txt(txt_path: Path, size: int) -> list[tuple[int, np.ndarray]]:
    """回傳 [(class_id, polygon_pixels (N,2) float32), ...]。"""
    if not txt_path.exists():
        return []
    out: list[tuple[int, np.ndarray]] = []
    for line in txt_path.read_text().strip().splitlines():
        toks = line.split()
        if len(toks) < 7:
            continue
        cls = int(toks[0])
        coords = np.array([float(x) for x in toks[1:]], dtype=np.float32).reshape(-1, 2)
        coords *= size
        out.append((cls, coords))
    return out


def polygon_to_mask(poly: np.ndarray, size: int) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m


def jitter_box(box: np.ndarray, size: int, jitter: float = 0.05) -> np.ndarray:
    """box: (4,) xyxy. ±jitter*邊長 隨機擾動。"""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx1, dy1, dx2, dy2 = (np.random.uniform(-jitter, jitter, 4) * np.array([bw, bh, bw, bh]))
    nx1 = float(np.clip(x1 + dx1, 0, size - 1))
    ny1 = float(np.clip(y1 + dy1, 0, size - 1))
    nx2 = float(np.clip(x2 + dx2, nx1 + 1, size - 1))
    ny2 = float(np.clip(y2 + dy2, ny1 + 1, size - 1))
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


class MedSAMDataset(Dataset):
    def __init__(
        self,
        fold: int,
        split: str,
        yolo_data_root: Path,
        meta_csv: Path,
        size: int = 768,
        cls_keep: int = 0,            # blood_vessel only
        ds1_only: bool = True,
        jitter: float = 0.05,
        aug: bool = True,
    ) -> None:
        self.size = size
        self.cls_keep = cls_keep
        self.jitter = jitter
        self.aug = aug

        img_dir = yolo_data_root / "images" / f"fold_{fold}" / split
        lbl_dir = yolo_data_root / "labels" / f"fold_{fold}" / split

        meta = pd.read_csv(meta_csv) if meta_csv.exists() else pd.DataFrame()
        ds1_ids = set(meta.loc[meta["dataset"] == 1, "id"]) if ds1_only and len(meta) else None

        samples: list[tuple[Path, list[np.ndarray]]] = []
        for img_path in sorted(img_dir.glob("*.png")):
            tid = img_path.stem
            if ds1_ids is not None and tid not in ds1_ids:
                continue
            polys = parse_yolo_seg_txt(lbl_dir / f"{tid}.txt", size)
            polys = [p for c, p in polys if c == cls_keep and len(p) >= 3]
            if len(polys) == 0:
                continue
            samples.append((img_path, polys))
        self.samples = samples
        print(f"[MedSAMDataset] fold={fold} split={split} ds1_only={ds1_only}: "
              f"{len(samples)} tiles 有 ≥1 個 blood_vessel")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, polys = self.samples[idx]
        img = cv2.imread(str(img_path))                # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # RGB uint8 HWC
        # 隨機挑一個 instance
        poly = polys[np.random.randint(len(polys))]
        mask = polygon_to_mask(poly, self.size)

        # 幾何增強（image + mask 同步）
        if self.aug:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy(); mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :].copy(); mask = mask[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k).copy(); mask = np.rot90(mask, k).copy()

        # 從 mask 求 tight bbox，再 jitter
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            # 退化：給整張圖
            box = np.array([0, 0, self.size - 1, self.size - 1], dtype=np.float32)
        else:
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            if self.aug:
                box = jitter_box(box, self.size, self.jitter)

        return {
            "image": img,                         # HWC uint8 RGB
            "mask": mask.astype(np.float32),      # (H, W) {0,1}
            "box": box,                            # (4,) xyxy in size coords
        }


def collate(batch: list[dict]) -> dict:
    return {
        "image": [b["image"] for b in batch],
        "mask": torch.from_numpy(np.stack([b["mask"] for b in batch])),
        "box": torch.from_numpy(np.stack([b["box"] for b in batch])),
    }


# ---------- Loss ----------

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """pred, target: (B, 1, H, W)。pred 為 sigmoid 後機率。"""
    p = pred.flatten(1); t = target.flatten(1)
    inter = (p * t).sum(1)
    den = p.sum(1) + t.sum(1)
    return 1.0 - (2.0 * inter + eps) / (den + eps)


# ---------- 訓練核心 ----------

class MedSAMTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = args.device

        model = build_sam2(args.cfg, args.base_weights, device=self.device, mode="train")
        self.img_size = int(model.image_size)
        self.transforms = SAM2Transforms(resolution=self.img_size, mask_threshold=0.0)

        # --- 凍結 prompt_encoder ---
        for p in model.sam_prompt_encoder.parameters():
            p.requires_grad = False

        # --- LoRA on image_encoder ---
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none",
            target_modules=["qkv", "attn.proj"],
        )
        model.image_encoder = get_peft_model(model.image_encoder, lora_cfg)
        # 凍結 image_encoder 非 LoRA 參數已由 peft 處理

        # --- mask decoder 全參數可訓 ---
        for p in model.sam_mask_decoder.parameters():
            p.requires_grad = True

        self.model = model

        # 參數統計
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

        # Optimizer：LoRA + decoder 分開 lr
        lora_params = [p for n, p in model.image_encoder.named_parameters() if p.requires_grad]
        dec_params = [p for p in model.sam_mask_decoder.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW([
            {"params": lora_params, "lr": args.lr_lora},
            {"params": dec_params, "lr": args.lr_dec},
        ], weight_decay=1e-4)

    # ---------- 單 step ----------
    def forward_loss(self, images: list[np.ndarray], boxes: torch.Tensor,
                     gt_masks: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """images: list of HWC uint8 RGB；boxes: (B, 4) in image coords (size=tile_size);
        gt_masks: (B, H, W) {0,1}。"""
        B = len(images)
        tile_size = gt_masks.shape[-1]

        # SAM2Transforms 接 list[np.ndarray] HWC，輸出 (B,3,model.image_size,model.image_size)
        img_t = self.transforms.forward_batch(images).to(self.device)

        # box → model 座標 (model.image_size)
        scale = self.img_size / tile_size
        box_in_model = (boxes.to(self.device).float() * scale)  # (B,4)

        # image encoder forward
        backbone_out = self.model.forward_image(img_t)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # 還原 spatial: bb_feat_sizes [hires//4, //8, //16] from 1024→256? actually image_size//4
        hires = self.img_size // 4
        bb_sizes = [(hires // (2**k),) * 2 for k in range(3)]
        feats = [
            f.permute(1, 2, 0).view(B, -1, *sz)
            for f, sz in zip(vision_feats[::-1], bb_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]                # (B, 256, 32, 32) for size=512
        high_res = feats[:-1]                  # [(B,32,128,128), (B,64,64,64)] for size=512

        # prompt encoder（凍結但 forward）
        sparse, dense = self.model.sam_prompt_encoder(
            points=None, boxes=box_in_model.view(B, 1, 4), masks=None,
        )

        # mask decoder
        low_res, iou_pred, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        # low_res: (B, 1, h, w); upsample to tile_size
        pred = F.interpolate(low_res, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(pred)
        gt = gt_masks.to(self.device).unsqueeze(1).float()

        loss_bce = F.binary_cross_entropy_with_logits(pred, gt)
        loss_dice = dice_loss(prob, gt).mean()
        # IoU 監督
        with torch.no_grad():
            bin_pred = (prob > 0.5).float()
            inter = (bin_pred * gt).sum((1, 2, 3))
            union = (bin_pred + gt).clamp(0, 1).sum((1, 2, 3)).clamp_min(1e-6)
            real_iou = inter / union
        loss_iou = F.mse_loss(iou_pred.squeeze(-1), real_iou)

        loss = loss_bce + loss_dice + 0.05 * loss_iou
        return loss, {
            "loss": loss.item(),
            "bce": loss_bce.item(),
            "dice": loss_dice.item(),
            "iou_mse": loss_iou.item(),
            "iou": real_iou.mean().item(),
        }

    # ---------- 訓 / 驗 ----------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        sums = {"loss": 0, "dice": 0, "iou": 0, "n": 0}
        pbar = tqdm(loader, desc=f"train e{epoch}")
        for batch in pbar:
            loss, log = self.forward_loss(batch["image"], batch["box"], batch["mask"])
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0,
            )
            self.opt.step()
            for k in ("loss", "dice", "iou"):
                sums[k] += log[k]
            sums["n"] += 1
            pbar.set_postfix(loss=f"{sums['loss']/sums['n']:.4f}", iou=f"{sums['iou']/sums['n']:.3f}")
        return {k: sums[k] / max(sums["n"], 1) for k in ("loss", "dice", "iou")}

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> dict:
        self.model.eval()
        sums = {"loss": 0, "dice": 0, "iou": 0, "n": 0}
        for batch in tqdm(loader, desc=f"val e{epoch}"):
            loss, log = self.forward_loss(batch["image"], batch["box"], batch["mask"])
            for k in ("loss", "dice", "iou"):
                sums[k] += log[k]
            sums["n"] += 1
        return {k: sums[k] / max(sums["n"], 1) for k in ("loss", "dice", "iou")}

    def save(self, out_dir: Path, tag: str = "best") -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # LoRA adapter
        self.model.image_encoder.save_pretrained(out_dir / f"lora_{tag}")
        # mask decoder full state
        torch.save(self.model.sam_mask_decoder.state_dict(), out_dir / f"decoder_{tag}.pt")


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--yolo-data", default="yolo_data")
    ap.add_argument("--meta", default="dataset/tile_meta.csv")
    ap.add_argument("--base-weights", default="pretrained/MedSAM2_latest.pt")
    ap.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml")
    ap.add_argument("--tile-size", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-dec", type=float, default=5e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="預設 runs/medsam2/fold{K}")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(f"runs/medsam2/fold{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    yolo_root = Path(args.yolo_data)
    meta = Path(args.meta)

    train_ds = MedSAMDataset(args.fold, "train", yolo_root, meta, size=args.tile_size, aug=True)
    val_ds = MedSAMDataset(args.fold, "val", yolo_root, meta, size=args.tile_size, aug=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate, pin_memory=True)

    trainer = MedSAMTrainer(args)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.opt, T_max=args.epochs)

    best_iou = -1.0
    for ep in range(args.epochs):
        tr = trainer.train_one_epoch(train_loader, ep)
        vl = trainer.validate(val_loader, ep)
        sched.step()
        print(f"ep{ep}: train_loss={tr['loss']:.4f} train_iou={tr['iou']:.4f} | "
              f"val_loss={vl['loss']:.4f} val_iou={vl['iou']:.4f} val_dice={vl['dice']:.4f}")
        if vl["iou"] > best_iou:
            best_iou = vl["iou"]
            trainer.save(out_dir, tag="best")
            print(f"  ↳ new best val_iou={best_iou:.4f} (saved)")
        trainer.save(out_dir, tag="last")


if __name__ == "__main__":
    main()
