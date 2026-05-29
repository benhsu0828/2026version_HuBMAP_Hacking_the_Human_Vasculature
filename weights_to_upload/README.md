# HuBMAP Dual-Stream Weights

YOLO Fast Stream + MedSAM-2 (LoRA + decoder FT) Heavy Stream.

## 內容
- `yolo/foldK.pt` — YOLOv8x-seg 4-fold
- `medsam2/base/MedSAM2_latest.pt` — MedSAM-2 base checkpoint (Ma Lab)
- `medsam2/foldK/lora_best/` — PEFT LoRA adapter (image encoder, rank 16)
- `medsam2/foldK/decoder_best.pt` — full-FT mask decoder
- `third_party_MedSAM2.tar.gz` — vendored MedSAM2 source (commit 332f30d)
- `code/*.py` — 推理腳本

## 在 Kaggle Notebook 用法
見 `hubmap_dual_stream_infer.ipynb`。
