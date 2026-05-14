default_scope = 'mmseg'

custom_imports = dict(
    imports=[
        'mmseg.structures.dual_task_seg_data_sample',
        'mmseg.datasets.dual_task_dataset',
        'mmseg.datasets.transforms.pack_seg_inputs_with_label',
        'mmseg.datasets.transforms.map_cls_after_flip',
        'mmseg.models.segmentors.dual_task_segmentor',
        'mmseg.models.cls_heads.simple_cls_head',
        'mmseg.models.backbones.bisenetv1',
        'mmseg.models.backbones.resnet',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='DualTaskSegmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),
        size_divisor=None,
    ),
    backbone=dict(
        type='BiSeNetV1',
        backbone_cfg=dict(
            type='ResNetV1c',
            depth=50,
            in_channels=3,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'),
        ),
        in_channels=3,
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_indices=(0, 1, 2, 3),
        out_channels=1024,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=None,
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=0,
        channels=1024,
        num_classes=2,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
    ),
    cls_head=dict(
        type='SimpleClsHead',
        in_channels=1024,
        num_classes=3,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

dataset_type = 'automine1dDual'
data_root = '/automine1d_cls/'
resize_512 = dict(type='Resize', scale=(512, 512), keep_ratio=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    resize_512,
    dict(type='RandomRotate', prob=0.5, degree=5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='MapClsAfterFlip',
        map_if_flipped={0: 2, 1: 1, 2: 0},
        str_map_if_flipped={'izquierda': 'derecha', 'recta': 'recta', 'derecha': 'izquierda'},
        field_name='gt_label',
    ),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomCLAHE', prob=0.5, clip_limit=3.0, tile_grid_size=(7, 7)),
    dict(type='PackSegInputsWithLabel'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    resize_512,
    dict(type='PackSegInputsWithLabel'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        cls_labels_file='cls_labels_train.txt',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val_aug', seg_map_path='ann_dir/val_aug'),
        cls_labels_file='cls_labels_val_aug_fixed.txt',
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='seg'),
    dict(type='ClsAccuracy', topk=(1,), prefix='cls'),
]
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'cls_head': dict(lr_mult=3.0, decay_mult=1.0),
            'decode_head': dict(lr_mult=1.5, decay_mult=1.0),
            'backbone': dict(lr_mult=1.0, decay_mult=1.0),
        }
    ),
)

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-5, power=0.9, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = []

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=200,
        max_keep_ckpts=3,
        save_last=True,
        save_best=['seg/mIoU', 'cls/acc_cls_top1'],
        rule=['greater', 'greater'],
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

env_cfg = dict(cudnn_benchmark=False)
log_level = 'INFO'
load_from = None
resume = False

work_dir = './work_dirs/multitask_val_fix_wloss'
