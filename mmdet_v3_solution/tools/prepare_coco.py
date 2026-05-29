# -*- coding: utf-8 -*-
"""把 HuBMAP 原始標註（polygons.jsonl + tile_meta.csv）轉成 COCO 格式 json。

產生本方案訓練所需的三個檔（與 kaggle-hubmap-.../tools/prepare_data.py 一致）：
  * dtrain0i.json / dval0i.json  : Dataset 1（乾淨）依 i 座標 0.2 分位切 train/val
  * dtrain_dataset2.json         : Dataset 2（弱標註，量大）全部

類別：blood_vessel(0) / glomerulus(1) / unsure(2)。訓練 config 的 metainfo 只取
前兩類，unsure 會被 CocoDataset 自動略過。

用法：
  python tools/prepare_coco.py \
      --polygons ../polygons.jsonl --tile-meta ../tile_meta.csv \
      --out-dir ../data
"""
import argparse
import json
import os.path as osp

import mmengine
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils


def load_annotations(ann_file):
    ret = {}
    with open(ann_file) as f:
        for line in f:
            ann = json.loads(line)
            ret[ann['id']] = ann['annotations']
    return ret


def decode_coords(coords):
    rles = mask_utils.frPyObjects(
        [_.flatten().tolist() for _ in np.asarray(coords)], 512, 512)
    rle = mask_utils.merge(rles)
    bbox = mask_utils.toBbox(rle)
    rle['counts'] = rle['counts'].decode()
    return bbox, rle


def df2coco(df, annotations):
    coco = {
        'info': {},
        'categories': [
            {'id': 0, 'name': 'blood_vessel'},
            {'id': 1, 'name': 'glomerulus'},
            {'id': 2, 'name': 'unsure'},
        ],
    }
    img_infos, ann_infos = [], []
    img_id = ann_id = 0
    cat_map = {'blood_vessel': 0, 'glomerulus': 1, 'unsure': 2}
    for _, row in df.iterrows():
        _id = row['id']
        if _id not in annotations:
            continue
        img_infos.append(dict(id=img_id, width=512, height=512,
                              file_name=f'{_id}.tif'))
        for ann in annotations[_id]:
            cat_id = cat_map[ann['type']]
            xs = np.asarray(ann['coordinates'])
            assert xs.shape[0] == 1
            xmin, ymin = xs[0].min(0)
            xmax, ymax = xs[0].max(0)
            w, h = xmax - xmin, ymax - ymin
            _bbox, _rle = decode_coords(ann['coordinates'])
            ann_infos.append(dict(
                id=ann_id, image_id=img_id, category_id=cat_id, iscrowd=0,
                segmentation=xs.reshape(1, -1).tolist(),
                area=float(w * h),
                bbox=[float(xmin), float(ymin), float(w), float(h)]))
            ann_id += 1
        img_id += 1
    coco['images'] = img_infos
    coco['annotations'] = ann_infos
    return coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--polygons', default='../polygons.jsonl')
    ap.add_argument('--tile-meta', default='../tile_meta.csv')
    ap.add_argument('--out-dir', default='../data')
    args = ap.parse_args()

    annotations = load_annotations(args.polygons)
    df = pd.read_csv(args.tile_meta)

    # Dataset 1：source_wsi 1 & 2
    ds1 = pd.concat([
        df.query('(dataset == 1) and (source_wsi == 1)'),
        df.query('(dataset == 1) and (source_wsi == 2)'),
    ], axis=0)

    # 依 i 座標 0.2 分位做 train/val 切分（空間切分，避免相鄰 tile 洩漏）
    q = ds1['i'].quantile(0.2)
    val0 = ds1[ds1['i'] < q]
    train0 = ds1[ds1['i'] >= q]
    mmengine.dump(df2coco(train0, annotations),
                  osp.join(args.out_dir, 'dtrain0i.json'))
    mmengine.dump(df2coco(val0, annotations),
                  osp.join(args.out_dir, 'dval0i.json'))

    # Dataset 2（弱標註）
    ds2 = df.query('(dataset == 2)')
    mmengine.dump(df2coco(ds2, annotations),
                  osp.join(args.out_dir, 'dtrain_dataset2.json'))

    print('已輸出 dtrain0i.json / dval0i.json / dtrain_dataset2.json 至',
          args.out_dir)


if __name__ == '__main__':
    main()
