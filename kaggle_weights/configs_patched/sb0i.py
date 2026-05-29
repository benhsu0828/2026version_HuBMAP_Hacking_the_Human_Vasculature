norm_cfg = dict(type='GN', num_groups=32)
model = dict(
    type='RTMDetWithMaskHead',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    mask_head=dict(type='FCNMaskHead',
        num_convs=7,
        in_channels=320,
        conv_out_channels=256,
        num_classes=1),
    backbone=dict(
        type='mmpretrain.SwinTransformer',
        arch=dict(
            embed_dims=128,
            depths=[2, 2, 18, 2, 1],
            num_heads=[4, 8, 16, 32, 64],
        ),
        img_size=384,
        drop_path_rate=0.2,
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        out_indices=(1, 2, 3, 4),
        with_cp=False,),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=320,
        num_csp_blocks=4,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=3,
        in_channels=320,
        stacked_convs=2,
        feat_channels=320,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32, 64]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        mask_pos_mode='weighted_sum',
        mask_roi_size=28,
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/kaggle/working/'
img_prefix = '/kaggle/input/competitions/hubmap-hacking-the-human-vasculature/test/'
metainfo = dict(classes=('blood_vessel', 'glomerulus', 'unsure'))
backend_args = None

img_scale = (1280, 1280)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=img_prefix),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_evaluator = dict(
        type='CocoMetric',
        ann_file=data_root + test_dataloader['dataset']['ann_file'],
        metric=['bbox'],
        classwise=True,
        format_only=False,
        backend_args=backend_args)
test_cfg = dict(type='TestLoop')

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
default_scope = 'mmdet'
custom_imports = dict(imports=['hubmap_modules'], allow_failed_imports=False)
