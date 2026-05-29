# MMDet v3 通用 runtime 設定（hooks / 視覺化 / 環境）。
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 每個 epoch 存檔，保留最近 3 個，並依 val segm mAP 另存 best
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# 匯入自訂 InternImage backbone（models/intern_image.py）。
# 注意：須從 mmdet_v3_solution/ 目錄執行訓練，讓 `models` 在 sys.path 上。
custom_imports = dict(imports=['models'], allow_failed_imports=False)
