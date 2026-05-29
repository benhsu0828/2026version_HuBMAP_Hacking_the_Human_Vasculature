# Cascade Mask R-CNN swap — workflow

End-to-end recipe to (1) train Cascade Mask R-CNN + Swin-L on HuBMAP fold0 + fold1, (2) dump EMA weights, (3) package for Kaggle, (4) run [hubmap_cascade_swap_infer.ipynb](../hubmap_cascade_swap_infer.ipynb) to swap into the WBF.

All commands assume cwd `kaggle-hubmap-hacking-the-human-vasculature/` unless stated.

## 0a. Install training deps (once)

The mmdet stack is split into two pieces because mmcv's CUDA wheels can't go through PyPI / uv:

```bash
cd /home/ben/nycu_hw/2026version_HuBMAP_Hacking_the_Human_Vasculature
# Pure-python pieces resolved by uv (mmengine, ensemble-boxes, openmim itself)
uv sync --extra training
# CUDA-aware pieces installed via mim (which knows the torch+cuda+python triple)
bash scripts/install_mm.sh
```

If `bash scripts/install_mm.sh` fails because torch is too new for openmmlab's
pre-built wheel server, you'll need `nvcc` on PATH so mim can build from source.

## 0b. Prepare COCO annotations (once)

`tools/prepare_data.py` reads `../data/polygons.jsonl` and `../data/tile_meta.csv`, but the raw files live at the project root. Symlink them in:

```bash
cd /home/ben/nycu_hw/2026version_HuBMAP_Hacking_the_Human_Vasculature
mkdir -p data
ln -s ../polygons.jsonl data/polygons.jsonl
ln -s ../tile_meta.csv data/tile_meta.csv
ln -s ../train data/train

cd kaggle-hubmap-hacking-the-human-vasculature
python tools/prepare_data.py
python tools/drop_duplicates.py
```

After this you should have:
- `data/dtrain0i.json`, `data/dval0i.json`
- `data/dtrain1i.json`, `data/dval1i.json`
- `data/dtrain_dataset2.json`, `data/dtrain_dataset2_dropdup.json`
- `data/dtrainval.json`
- `data/train/<id>.tif` images reachable

## 1. Sanity-check the new config compiles and pretrained ckpt loads

```bash
python -c "
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
import custom_modules  # noqa
register_all_modules()
cfg = Config.fromfile('configs/m0c.py')
m = MODELS.build(cfg.model)
print(type(m).__name__, sum(p.numel() for p in m.parameters())/1e6, 'M params')
"
```

Expect: `MultiEMADetector`, ~210M params (3× from EMA copies → ~630M state dict).

First training iter should also be tested:
```bash
python train.py configs/m0c.py --work-dir runs/m0c_sanity --cfg-options train_cfg.max_iters=10 train_cfg.val_interval=10
```

If OOM at batch 4: lower further to batch 2 + `optim_wrapper.accumulative_counts=2` via `--cfg-options`.

## 2. Train fold 0

```bash
python train.py configs/m0c.py --work-dir runs/m0c_fold0
```

Defaults from m0c.py: 200×113 = 22600 iters, val every 9 epochs. AdamW lr=1e-4 with auto-scale halving (base_batch_size=16, you're at batch=4).

`MultiEMAValLoop` will print val mAP for all 3 EMA copies — note which `ema-id` has the highest segm mAP.

## 3. Train fold 1

```bash
python train.py configs/m0c.py \
    --work-dir runs/m0c_fold1 \
    --cfg-options \
        train_dataloader.dataset.datasets.0.ann_file=dtrain1i.json \
        val_dataloader.dataset.ann_file=dval1i.json \
        val_evaluator.0.ann_file=../data/dval1i.json
```

## 4. Dump EMA weights

Use `tools/dump_m0c_ckpt.py` (parameterized; the original `dump_ckpt.py` is hard-coded to r0):

```bash
python tools/dump_m0c_ckpt.py \
    --src runs/m0c_fold0/iter_22600.pth \
    --dst ../weights_to_upload/cascade/m0c_fold0.pth \
    --ema-id 2

python tools/dump_m0c_ckpt.py \
    --src runs/m0c_fold1/iter_22600.pth \
    --dst ../weights_to_upload/cascade/m0c_fold1.pth \
    --ema-id 2
```

(`--ema-id 2` = slowest momentum = strongest smoothing. Override based on val log.)

Also copy the inference config:
```bash
cp configs/m0ci.py ../weights_to_upload/cascade/m0ci.py
```

## 5. Upload as Kaggle dataset

Bundle the three files into a new Kaggle dataset named **`hubmap-cascade-ckpts`** (the notebook expects this slug):

```
hubmap-cascade-ckpts/
  m0ci.py
  m0c_fold0.pth
  m0c_fold1.pth
```

```bash
cd /home/ben/nycu_hw/2026version_HuBMAP_Hacking_the_Human_Vasculature/weights_to_upload/cascade
# Use kaggle CLI:
kaggle datasets create -p . -m "Cascade Mask R-CNN + Swin-L 2-fold for HuBMAP"
# Or zip and upload via web UI.
```

## 6. Submit on Kaggle

Open [hubmap_cascade_swap_infer.ipynb](../hubmap_cascade_swap_infer.ipynb) on Kaggle, attach the datasets listed in its first markdown cell (`hubmap-cascade-ckpts` is the new one; the rest are reused from the 2023 release), enable 2× GPUs and Internet OFF, and run all.

Three `ABLATION_MODE` values are available in the WBF cell:

| Mode | What it tests |
|---|---|
| `cascade_swap` | 8 legacy + 2 cascade — primary submission |
| `mmdet_only_legacy` | 10 legacy mmdet — reproduces 0.589 baseline |
| `cascade_only` | 2 cascade only — single-model floor sanity |

Run all three on separate submissions to isolate cascade's contribution.

## Common issues

- **`KeyError: 'ema_models'` when dumping**: the source ckpt isn't from a MultiEMA training run. Confirm `runs/m0c_fold0/last_checkpoint` points at an `iter_*.pth` saved by `train.py configs/m0c.py`.
- **Pretrained mismatch warning on first iter**: expected — bbox_head & mask_head get reinit because COCO has 80 classes, HuBMAP has 3. Backbone/FPN/RPN should load cleanly.
- **`AttributeError: 'CascadeRoIHead' object has no attribute 'predict_mask'`**: requires mmdet 3.1+. Confirm version in cell 2 install.
- **GPU OOM**: drop batch to 2, set `optim_wrapper.accumulative_counts=2`, keep `with_cp=True`.
- **Different fold for mask repredict**: by default predict_mask.py uses fold0. To try fold1, edit the `load_checkpoint(...)` path in the predict_mask cell of the notebook.
