# =============================================================================
# Stage 1（粗練）：在 Dataset 2（髒數據 / 弱標註，量大）上預訓練
# -----------------------------------------------------------------------------
# 策略（沿用 HuBMAP-2023 3rd-place 的兩階段思路）：
#   * 用較高 lr + 較長 cosine 衰減，把大量噪聲標註當「正則化」吸收，學到泛化特徵。
#   * 只用「輕度」數據增強（Resize + 翻轉 + 小幅仿射），避免在髒標註上過度形變。
#   * backbone 載 ImageNet 預訓練（見骨架 init_cfg），偵測頭從頭學。
# 跑法：python train.py configs/stage1_dataset2_pretrain.py --work-dir work_dirs/stage1 --amp
# =============================================================================
_base_ = ['cascade_mask_rcnn_internimage_t_fpn.py']

# ---- 資料路徑（請依實際情況調整）------------------------------------------
# 可用 kaggle-hubmap-.../tools/prepare_data.py 產生 COCO json。
data_root = '../data/'
img_prefix = '../data/train'

dataset_type = {{_base_.dataset_type}}
metainfo = {{_base_.metainfo}}
image_size = {{_base_.image_size}}
backend_args = None

# ---- 輕度增強 pipeline -----------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=image_size, keep_ratio=False),
    # 水平 + 垂直翻轉（內建 transform 原生支援 mask，無需經 Albu）
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    # 小幅平移/縮放/旋轉（Albu）；mask 經 keymap 一併變換
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.15,
                rotate_limit=15,
                interpolation=1,
                p=0.4),
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
        skip_img_without_anno=True),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_size, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ---- DataLoader：Stage1 單來源 = Dataset 2 --------------------------------
train_dataloader = dict(
    batch_size=2,            # 1024 解析度 + Cascade，單卡先用 2，視顯存調整
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='dtrain_dataset2.json',     # ← Dataset 2（弱標註）
        data_prefix=dict(img=img_prefix),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='dval0i.json',              # 在乾淨資料的 val 上監看泛化
        data_prefix=dict(img=img_prefix),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'dval0i.json',
    metric=['bbox', 'segm'],
    classwise=True,
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# ---- 訓練排程：10 個 epoch，高 lr + Cosine ---------------------------------
max_epochs = 10
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# AdamW + 換算後的偏高 lr（transformer/InternImage 用 SGD 0.02 易發散，
# 故改 AdamW 2e-4 作為「相對高」的粗練 lr）。
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={'backbone': dict(lr_mult=0.5)}),  # backbone 微調用較小 lr
    clip_grad=dict(max_norm=35, norm_type=2))

# 前 500 iter 線性 warmup，之後 CosineAnnealing 衰減到 eta_min=2e-5
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=2e-5,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

auto_scale_lr = dict(enable=False, base_batch_size=16)
