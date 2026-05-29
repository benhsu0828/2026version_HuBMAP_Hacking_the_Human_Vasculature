"""Dump the strongest EMA copy from an m0c (Cascade Mask R-CNN + Swin-L) run.

Usage:
    cd kaggle-hubmap-hacking-the-human-vasculature
    python tools/dump_m0c_ckpt.py \
        --src runs/m0c_fold0/iter_22600.pth \
        --dst weights_to_upload/cascade/m0c_fold0.pth \
        --ema-id 2

`--ema-id` picks among the 3 MultiEMA copies (momentums 0.001 / 0.0005 / 0.00025).
ema-id=2 (slowest momentum, strongest smoothing) is usually the best — confirm
from the val log of each fold before dumping.
"""
import argparse
import os

import torch
import mmengine
from mmengine.runner import load_checkpoint
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS

import custom_modules  # noqa: F401  registers MultiEMADetector etc.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='path to iter_*.pth from training')
    parser.add_argument('--dst', required=True, help='output ckpt path')
    parser.add_argument('--cfg', default='configs/m0c.py')
    parser.add_argument('--ema-id', type=int, default=2,
                        help='index into MultiEMA copies (0..len(momentums)-1)')
    args = parser.parse_args()

    register_all_modules()
    cfg = mmengine.Config.fromfile(args.cfg)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.src)

    if not hasattr(model, 'ema_models'):
        raise RuntimeError(
            'Loaded model is not a MultiEMADetector — did you point --src at the wrong file?')
    if args.ema_id >= len(model.ema_models):
        raise IndexError(
            f'ema-id {args.ema_id} out of range (have {len(model.ema_models)} EMA copies)')

    os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
    torch.save(
        dict(state_dict=model.ema_models[args.ema_id].state_dict()),
        args.dst,
    )
    print(f'Wrote {args.dst} (ema-id={args.ema_id})')


if __name__ == '__main__':
    main()
