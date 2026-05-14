default_scope = 'mmseg'

custom_imports = dict(
    imports=[
        # Módulos base
        'mmseg.structures.dual_task_seg_data_sample',
        'mmseg.models.cls_heads.simple_cls_head',
        'mmseg.models.backbones.bisenetv1',
        'mmseg.models.backbones.resnet',

        # Dataset/pack exclusivos scheduler
        'mmseg.datasets.scheduler_video_pair_dual_task_dataset',
        'mmseg.datasets.transforms.pack_scheduler_video_pair_inputs',

        # Transforms para pares
        'mmseg.datasets.transforms.load_video_pair',
        'mmseg.datasets.transforms.resize_video_pair',
        'mmseg.datasets.transforms.random_flip_video_pair',
        'mmseg.datasets.transforms.pair_random_clahe_video_pair',
        'mmseg.datasets.transforms.pair_photometric_distortion_video_pair',
        'mmseg.datasets.transforms.map_cls_after_flip',

        # Modelo low-latency
        'mmseg.models.data_preprocessors.pair_seg_data_preprocessor',
        'mmseg.models.video_modules.adaptive_feature_propagation',
        'mmseg.models.video_modules.adaptive_keyframe_selector',
        'mmseg.models.segmentors.bisenet_adaptive_video_segmentor',
    ],
    allow_failed_imports=False,
)

# ============================================================
# ROUTES
# ============================================================

DATA_ROOT_TRAIN = (
    '/0_secuencias_ann/1a_secuencia'
)

DATA_ROOT_VAL = (
    '/0_secuencias_ann/1a_secuencia_val_mixed'
)

TRAIN_PAIR_FILE = 'train_pairs_k30_devtargets.txt'
VAL_PAIR_FILE = 'val_pairs_mixed_devtargets.txt'

TRAIN_CLS_FILE = 'cls_labels_train_k30_sparsegt.txt'
VAL_CLS_FILE = 'cls_labels_val_mixed.txt'

dataset_type = 'SchedulerVideoPairDualTaskDataset'

# ============================================================
# CHECKPOINT OF 02
# ============================================================

load_from = (
    '/mmsegmentation/'
    'work_dirs/bisenet_adaptive_propagation_seqaug_pretrain/'
    'best_cls_acc_cls_top1_iter_2200.pth'
)

resume = False

# ============================================================
# MODELO
# ============================================================

model = dict(
    type='BiSeNetAdaptiveVideoSegmentor',
    train_mode='scheduler',
    freeze_base=True,

    data_preprocessor=dict(
        type='PairSegDataPreProcessor',
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_val=0,
        seg_pad_val=255,
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

    feature_propagation=dict(
        type='AdaptiveFeaturePropagation',
        spatial_channels=512,
        context_channels=512,
        hidden_channels=128,
        kernel_size=3,
        refine=True,
        norm_cfg=dict(type='BN', requires_grad=True),
    ),

    keyframe_selector=dict(
        type='AdaptiveKeyFrameSelector',
        in_channels=512,
        hidden_channels=128,
        norm_cfg=dict(type='BN', requires_grad=True),
    ),

    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=0,
        channels=1024,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.5,
        ),
    ),

    cls_head=dict(
        type='SimpleClsHead',
        in_channels=1024,
        num_classes=3,
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ============================================================
# PIPELINES
# ============================================================

train_pipeline = [
    dict(type='LoadVideoPairFromFile'),
    dict(type='ResizeVideoPair', scale=(512, 512)),

    dict(
        type='RandomFlipVideoPair',
        prob=0.5,
        direction='horizontal',
    ),

    dict(
        type='MapClsAfterFlip',
        map_if_flipped={0: 2, 1: 1, 2: 0},
        str_map_if_flipped={
            'izquierda': 'derecha',
            'recta': 'recta',
            'derecha': 'izquierda',
            'LEFT': 'RIGHT',
            'STRAIGHT': 'STRAIGHT',
            'RIGHT': 'LEFT',
        },
        field_name='gt_label',
    ),

    dict(
        type='PairRandomCLAHEVideoPair',
        prob=0.3,
        clip_limit=2.0,
        tile_grid_size=(7, 7),
    ),

    dict(type='PairPhotoMetricDistortionVideoPair'),

    dict(type='PackSchedulerVideoPairInputs'),
]

val_pipeline = [
    dict(type='LoadVideoPairFromFile'),
    dict(type='ResizeVideoPair', scale=(512, 512)),
    dict(type='PackSchedulerVideoPairInputs'),
]

# ============================================================
# DATALOADERS
# ============================================================

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT_TRAIN,
        data_prefix=dict(
            img_path='img',
            seg_map_path='ann',
        ),
        pair_file=TRAIN_PAIR_FILE,
        cls_labels_file=TRAIN_CLS_FILE,
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
        data_root=DATA_ROOT_VAL,
        data_prefix=dict(
            img_path='img',
            seg_map_path='ann',
        ),
        pair_file=VAL_PAIR_FILE,
        cls_labels_file=VAL_CLS_FILE,
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

# ============================================================
# EVALUADORES
# ============================================================

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='seg'),
    dict(type='ClsAccuracy', topk=(1,), prefix='cls'),
]

test_evaluator = val_evaluator

# ============================================================
# OPTIMIZADOR
# ============================================================

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            # Entrenar solo scheduler
            'keyframe_selector': dict(lr_mult=1.0, decay_mult=1.0),

            # Congelar lo aprendido en 02
            'feature_propagation': dict(lr_mult=0.0, decay_mult=0.0),
            'backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'decode_head': dict(lr_mult=0.0, decay_mult=0.0),
            'cls_head': dict(lr_mult=0.0, decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-5,
        power=0.9,
        by_epoch=False,
    ),
]

# ============================================================
# LOOPS
# ============================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=200,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================
# HOOKS
# ============================================================

custom_hooks = []

default_hooks = dict(
    timer=dict(type='IterTimerHook'),

    logger=dict(
        type='LoggerHook',
        interval=50,
    ),

    param_scheduler=dict(type='ParamSchedulerHook'),

    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=200,
        max_keep_ckpts=3,
        save_last=True,
        save_best=[
            'seg/mIoU',
            'cls/acc_cls_top1',
        ],
        rule=[
            'greater',
            'greater',
        ],
    ),

    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

env_cfg = dict(cudnn_benchmark=False)
log_level = 'INFO'

work_dir = './work_dirs/bisenet_adaptive_scheduler_from_pretrain02'
