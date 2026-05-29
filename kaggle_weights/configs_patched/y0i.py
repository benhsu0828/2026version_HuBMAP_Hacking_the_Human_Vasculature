# model settings
model = dict(
    type='YOLOXWithMaskHead',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32),
    mask_head=dict(type='FCNMaskHead',
        num_convs=7,
        in_channels=320,
        conv_out_channels=256,
        num_classes=1),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=1.33,
        widen_factor=1.25,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=3,
        in_channels=320,
        feat_channels=320,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(
        mask_pos_mode='weighted_sum',
        mask_roi_size=28,
        assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.001,
                  max_per_img=300,
                  nms=dict(type='nms', iou_threshold=0.65))
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/kaggle/working/'
img_prefix = '/kaggle/input/competitions/hubmap-hacking-the-human-vasculature/test/'
metainfo = dict(classes=('blood_vessel', 'glomerulus', 'unsure'))
backend_args = None

img_scale = (768, 768)
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
