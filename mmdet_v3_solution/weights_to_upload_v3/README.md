# HuBMAP InternImage-T Cascade（mmdet v3 單模）

## 內容
| 路徑 | 用途 |
|---|---|
| `code/kaggle_infer.py` | 推理入口（CLI: `--config --checkpoint --img-dir --out --no-compile`） |
| `code/models/` | InternImage backbone（觸發 mmdet registry） |
| `configs/` | 三層 `_base_` 鏈，Config.fromfile 用 |
| `weights/best_coco_segm_mAP_epoch_X.pth` | stage 2 best ckpt |
| `wheels/` | mmdet 全家桶 + 配套（不含 mmcv）|
| `mmcv_src/mmcv-2.1.0.tar.gz` | mmcv 原始碼，Kaggle 上 source build 對齊新 ABI（離線可跑） |

## Kaggle 環境鎖定（必須）
target: **Python 3.11** / **torch 2.1+cu121**

Kaggle latest 環境用 py3.12 + torch 2.10，預編 wheel 全部 ABI 不合。
解法是 fork 一個 2024 年的舊 notebook 繼承凍結環境（py3.10 + torch 2.1）。

## 用法
見 `hubmap_internimage_infer.ipynb`。cell 3 同時做兩件事：
1. 白名單裝 wheel（科學 stack 用 Kaggle 預裝版，不蓋）
2. 從 `mmcv_src/` 本地 tarball source build mmcv（`--no-build-isolation --no-deps` → 完全離線可跑）

## 重新打包
```bash
bash mmdet_v3_solution/package_for_kaggle.sh                   # 全自動
bash mmdet_v3_solution/package_for_kaggle.sh --skip-wheels     # 只更新 code/configs/ckpt
bash mmdet_v3_solution/package_for_kaggle.sh --python-version 311  # Kaggle Python 漂移時調整
```
