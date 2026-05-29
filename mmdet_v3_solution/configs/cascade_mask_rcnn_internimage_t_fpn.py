# =============================================================================
# Part 1 — 模型骨架：Cascade Mask R-CNN + InternImage-T + FPN
# -----------------------------------------------------------------------------
# 這份檔只負責「模型 + 共用 metainfo」，資料/增強/優化器/排程由 stage1、stage2
# 兩份 config 以 `_base_` 繼承後各自覆寫。
#
# 設計重點：
#   * Backbone 預設 InternImage-T（core_op='DCNv3_pytorch'，零編譯，見
#     models/intern_image.py）。下方附 ConvNeXt-V2-Tiny 切換區塊。
#   * Neck 用 FPN 做多尺度融合。
#   * Head 用 3-stage CascadeRoIHead（HTC 概念的精簡版：逐階段提高 IoU 門檻、
#     漸進精修 bbox），Mask Head 的 loss_mask 權重調高為 2.0 以強化分割細節。
# =============================================================================
_base_ = ['_base_/default_runtime.py']

# ---- 共用超參數（stage config 可覆寫）-------------------------------------
num_classes = 2          # blood_vessel(0) / glomerulus(1)；unsure 於資料前處理過濾
image_size = (1024, 1024)  # 訓練解析度

# InternImage-T 各 stage 輸出通道（FPN in_channels 用）
backbone_out_channels = [64, 128, 256, 512]

model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    # -------------------------------------------------------------------------
    # Backbone（預設 InternImage-T）
    # core_op='DCNv3_pytorch'：純 PyTorch，免編譯，工作站(Blackwell)+Kaggle 通吃。
    # with_cp=True：開梯度檢查點，1024 解析度在單卡/T4 省記憶體。
    # -------------------------------------------------------------------------
    backbone=dict(
        type='InternImage',
        core_op='DCNv3_pytorch',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            # InternImage-T ImageNet-1k 預訓練 backbone。先下載到本地再填路徑，
            # 或直接用此 URL（需可連網）。
            checkpoint='https://huggingface.co/OpenGVLab/InternImage/'
                       'resolve/main/internimage_t_1k_224.pth')),
    # ====== 切換成 ConvNeXt-V2-Tiny（零編譯 fallback）時，改用下面這段 ======
    # 1) pip/uv 裝好 mmpretrain；2) 把上面整段 backbone 換成：
    # backbone=dict(
    #     type='mmpretrain.ConvNeXt',
    #     arch='tiny',                 # dims=[96,192,384,768], depths=[3,3,9,3]
    #     out_indices=[0, 1, 2, 3],
    #     drop_path_rate=0.2,
    #     layer_scale_init_value=0.,   # ConvNeXt-V2 用 GRN，關掉 layer-scale
    #     use_grn=True,                # ConvNeXt-V2 的 Global Response Norm
    #     gap_before_final_norm=False,
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='https://download.openmmlab.com/mmclassification/'
    #                    'v1/convnext_v2/convnext-v2-tiny_3rdparty-fcmae_in1k_'
    #                    '20230104-80513adc.pth',
    #         prefix='backbone.')),
    # 3) 同步把下方 neck 的 in_channels 改成 [96, 192, 384, 768]。
    # =======================================================================
    neck=dict(
        type='FPN',
        in_channels=backbone_out_channels,  # InternImage-T: [64,128,256,512]
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # ---- Mask Head：loss_weight 調高為 2.0，強化分割邊界細節 ----
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=2.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)))

# ---- 資料集共用設定 --------------------------------------------------------
dataset_type = 'CocoDataset'
metainfo = dict(classes=('blood_vessel', 'glomerulus'))
backend_args = None
