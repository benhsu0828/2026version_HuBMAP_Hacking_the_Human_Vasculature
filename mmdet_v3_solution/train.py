# Copyright (c) OpenMMLab. All rights reserved.
# 標準 MMDetection v3 訓練入口（沿用官方 tools/train.py），供 run_all.sh 呼叫。
# 支援 --amp（混合精度）、--cfg-options（覆寫 load_from / max_epochs 等）。
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


# === HuBMAP monkey-patch：修 mmdet 3.3.0 Albu._postprocess_results 順序錯的 bug ===
# 上游：idx_mapper 空時先 `results['masks'][0]` → IndexError，再也走不到後面的 skip。
# 補：先 skip → 再重建 masks；對空 idx_mapper 用 ori_masks H/W fallback。
# 寫成 monkey-patch 跟著專案走，不必動 site-packages，環境重建後依然生效。
def _apply_mmdet_albu_patch():
    import numpy as np
    from mmdet.datasets.transforms.transforms import Albu
    from mmdet.structures.bbox import HorizontalBoxes

    def _postprocess_results(self, results, ori_masks=None):
        if 'gt_bboxes_labels' in results and isinstance(
                results['gt_bboxes_labels'], list):
            results['gt_bboxes_labels'] = np.array(
                results['gt_bboxes_labels'], dtype=np.int64)
        if 'gt_ignore_flags' in results and isinstance(
                results['gt_ignore_flags'], list):
            results['gt_ignore_flags'] = np.array(
                results['gt_ignore_flags'], dtype=bool)
        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)
            results['bboxes'] = HorizontalBoxes(results['bboxes'])
            if self.filter_lost_elements:
                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                # PATCH: 空 idx_mapper 提前 skip，避免 [0] 噴 IndexError
                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None
                if 'masks' in results:
                    assert ori_masks is not None
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    if len(results['masks']) == 0:
                        results['masks'] = ori_masks.__class__(
                            results['masks'], ori_masks.height,
                            ori_masks.width)
                    else:
                        results['masks'] = ori_masks.__class__(
                            results['masks'],
                            results['masks'][0].shape[0],
                            results['masks'][0].shape[1])
            elif 'masks' in results:
                results['masks'] = ori_masks.__class__(
                    results['masks'], ori_masks.height, ori_masks.width)
        return results

    Albu._postprocess_results = _postprocess_results


_apply_mmdet_albu_patch()
# === END monkey-patch ===


# === HuBMAP monkey-patch：torch 2.6+ 把 torch.load 預設改成 weights_only=True，
# mmengine 0.10.x 的 load_from_local 沒傳 weights_only=False，會擋下 ckpt 裡
# 自家的 HistoryBuffer 等物件。我們載的都是自己訓的 ckpt（trusted），預設改回 False。
def _apply_torch_load_unsafe_default():
    import torch
    if getattr(torch.load, '_hubmap_patched', False):
        return
    _orig_torch_load = torch.load

    def _torch_load_patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_torch_load(*args, **kwargs)

    _torch_load_patched._hubmap_patched = True
    torch.load = _torch_load_patched


_apply_torch_load_unsafe_default()
# === END monkey-patch ===


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # 減少 torch.compile / dynamo 重複編譯，提升訓練速度
    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == '__main__':
    main()
