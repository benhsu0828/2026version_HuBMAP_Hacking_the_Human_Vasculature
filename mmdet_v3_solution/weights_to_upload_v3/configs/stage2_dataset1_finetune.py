# =============================================================================
# Stage 2（精練）：在 Dataset 1（乾淨數據，量小）上微調
# -----------------------------------------------------------------------------
# 策略：
#   * 載入 Stage 1 權重（由 run_all.sh 以 --cfg-options load_from=... 帶入，
#     或在此手動指定）。
#   * 用較低 lr（AdamW 1e-4）+ Cosine 衰減到極低 eta_min=1e-7，慢慢逼近最優。
#   * 開啟「重度」數據增強：RandomFlip(H/V/對角) + AutoAugment(Shear/Translate/
#     Color/Equalize/Rotate) + Albu(RandomRotate90 + ElasticTransform)。
#     ElasticTransform 對細胞/血管這類非剛性結構的分割特別有用。
# 跑法：python train.py configs/stage2_dataset1_finetune.py --work-dir work_dirs/stage2 \
#         --amp --cfg-options load_from=work_dirs/stage1/best_coco_segm_mAP_epoch_X.pth
# =============================================================================
_base_ = ['cascade_mask_rcnn_internimage_t_fpn.py']

data_root = '../data/'
img_prefix = '../data/train'

dataset_type = {{_base_.dataset_type}}
metainfo = {{_base_.metainfo}}
image_size = {{_base_.image_size}}
backend_args = None

# ---- 重度增強 pipeline -----------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=image_size, keep_ratio=False),
    # 水平 / 垂直 / 對角翻轉
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    # AutoAugment：每次隨機抽一條 sub-policy（幾何 + 色彩），皆會同步變換 mask
    dict(
        type='AutoAugment',
        policies=[
            [dict(type='Color', prob=0.6, level=6)],
            [dict(type='Equalize', prob=0.6)],
            [
                dict(type='ShearX', prob=0.5, level=4),
                dict(type='ShearY', prob=0.5, level=4)
            ],
            [
                dict(type='TranslateX', prob=0.5, level=4),
                dict(type='TranslateY', prob=0.5, level=4)
            ],
            [dict(type='Rotate', prob=0.6, level=6)],
            [
                dict(type='Brightness', prob=0.5, level=4),
                dict(type='Contrast', prob=0.5, level=4)
            ],
        ]),
    # AutoAugment 的 Rotate/Shear/Translate 可能把 bbox 推到圖邊 clip 成 zero-height。
    # albumentations 1.3+ 的 check_bbox 會對 y_max<=y_min 直接 raise，先把退化框濾掉。
    # 注意：keep_empty=True 才是「全濾完→return None 跳這張」；
    #       keep_empty=False 反而會把空 annotation 餵給下游 Albu，Albu 內部
    #       _check_args 做 masks[0] 會 IndexError。
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(2, 2),  # 任一邊 < 2 px 就丟（含 zero-height 邊界框）
        keep_empty=True),        # 全濾完 → return None，mmengine 自動跳過該圖
    # Albu 封裝：RandomRotate90 + ElasticTransform（彈性形變，細胞分割關鍵）
    # 註：ElasticTransform 後 bbox 由角點仿射近似，alpha 設較溫和避免框過鬆；
    #     mask 經 keymap('masks') 一併彈性變換，保留邊界細節。
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomRotate90', p=0.5),
            dict(
                type='ElasticTransform',
                alpha=120,
                sigma=6.0,
                alpha_affine=0,
                interpolation=1,
                border_mode=0,
                p=0.3),
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.1,
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

# ---- DataLoader：Stage2 單來源 = Dataset 1（乾淨）-------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='dtrain0i.json',            # ← Dataset 1（乾淨）
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
        ann_file='dval0i.json',
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

# ---- 訓練排程：20 個 epoch，低 lr + Cosine 衰減到 1e-7 ---------------------
max_epochs = 20
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={'backbone': dict(lr_mult=0.5)}),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0,
         end=250),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-7,                # 最低降至 1e-7
        begin=0,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

auto_scale_lr = dict(enable=False, base_batch_size=16)

# load_from 預設留空，由 run_all.sh 動態填入 Stage1 權重；
# 若手動執行，可在此直接指定：
# load_from = 'work_dirs/stage1/best_coco_segm_mAP_epoch_10.pth'
