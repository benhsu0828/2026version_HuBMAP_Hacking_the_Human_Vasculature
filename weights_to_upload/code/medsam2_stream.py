"""MedSAM-2 Heavy Stream — bbox-prompt 驅動的 mask refinement

封裝 vendored third_party/MedSAM2 的 SAM2ImagePredictor，提供：
    MedSAM2Refiner(model_cfg, ckpt, lora_dir=None, fp16=True)
      .embed(image_np_hwc_rgb)        # 對單張 768x768 padded tile 算一次 image embedding
      .decode(boxes_xyxy_in_canvas)   # 對多個 bbox 批次解 mask，回傳 (N, H, W) uint8 + ious

bbox 座標系：使用 padded canvas (e.g. 768x768) 的像素座標，xyxy 格式。
SAM2 內部會 resize 到 model.image_size (=512)；輸出 mask 會還原回 set_image 的 orig_hw。

"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# vendored MedSAM2 — 把 third_party/MedSAM2 放進 sys.path
_THIRD_PARTY = Path(__file__).resolve().parent / "third_party" / "MedSAM2"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

# 觸發 sam2/__init__.py 的 hydra initialize_config_module("sam2")
import sam2  # noqa: F401
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEFAULT_CFG = "configs/sam2.1_hiera_t512.yaml"
DEFAULT_CKPT = "pretrained/MedSAM2_latest.pt"


class MedSAM2Refiner:
    def __init__(
        self,
        ckpt: str = DEFAULT_CKPT,
        cfg: str = DEFAULT_CFG,
        lora_dir: str | None = None,
        decoder_ckpt: str | None = None,
        device: str = "cuda",
        fp16: bool = True,
        mask_threshold: float = 0.0,
    ) -> None:
        self.device = device
        self.fp16 = fp16

        model = build_sam2(cfg, ckpt, device=device, mode="eval")

        if lora_dir is not None:
            self._load_lora(model, lora_dir)
        if decoder_ckpt is not None:
            self._load_decoder(model, decoder_ckpt)

        self.predictor = SAM2ImagePredictor(model, mask_threshold=mask_threshold)
        # device 由 predictor.model.device 自動推出，無需手動設定

        # 自動推得 model.image_size（512）— LoRA 訓練時也用同一個
        self.image_size = int(model.image_size)

    # ---------- LoRA / decoder weight loading ----------

    @staticmethod
    def _load_lora(model: torch.nn.Module, lora_dir: str) -> None:
        """PEFT LoRA adapter 已在 train_medsam2.py 用 model.image_encoder 為 base 儲存。
        這裡用 peft.PeftModel.from_pretrained 把 adapter 套上 image_encoder。"""
        from peft import PeftModel
        model.image_encoder = PeftModel.from_pretrained(
            model.image_encoder, lora_dir, is_trainable=False,
        )

    @staticmethod
    def _load_decoder(model: torch.nn.Module, decoder_ckpt: str) -> None:
        state = torch.load(decoder_ckpt, map_location="cpu")
        model.sam_mask_decoder.load_state_dict(state, strict=True)

    # ---------- 推理介面 ----------

    @torch.inference_mode()
    def embed(self, image: np.ndarray) -> None:
        """image: (H, W, 3) uint8 RGB，例如 768x768 padded tile。"""
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            self.predictor.set_image(image)

    @torch.inference_mode()
    def decode(self, boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """boxes_xyxy: (N, 4) float32，像素座標於 set_image 時的原始尺寸。
        回傳 (masks (N,H,W) uint8, ious (N,))。
        """
        assert self.predictor._is_image_set, "需先呼叫 embed(image)"
        if len(boxes_xyxy) == 0:
            h, w = self.predictor._orig_hw[0]
            return np.zeros((0, h, w), np.uint8), np.zeros((0,), np.float32)

        h, w = self.predictor._orig_hw[0]
        masks_out = np.zeros((len(boxes_xyxy), h, w), np.uint8)
        ious_out = np.zeros((len(boxes_xyxy),), np.float32)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            for i, box in enumerate(boxes_xyxy.astype(np.float32)):
                masks, ious, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False,
                    return_logits=False,
                )
                # masks shape: (1, H, W) bool/float — 取第一個
                m = masks[0] if masks.ndim == 3 else masks
                masks_out[i] = (m > 0).astype(np.uint8)
                ious_out[i] = float(ious[0] if ious.ndim else ious)

        return masks_out, ious_out

    def reset(self) -> None:
        self.predictor.reset_predictor()


# ---------- 快速 smoke test ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--cfg", default=DEFAULT_CFG)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--decoder", default=None)
    args = ap.parse_args()

    refiner = MedSAM2Refiner(
        ckpt=args.ckpt, cfg=args.cfg,
        lora_dir=args.lora, decoder_ckpt=args.decoder,
    )
    img = (np.random.rand(768, 768, 3) * 255).astype(np.uint8)
    refiner.embed(img)
    boxes = np.array([[100, 100, 400, 400], [300, 300, 600, 600]], dtype=np.float32)
    masks, ious = refiner.decode(boxes)
    print(f"OK — masks shape={masks.shape}, ious={ious}")
