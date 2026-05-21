default_scope = 'mmseg'

custom_imports = dict(
    imports=[
        # ====================================================
        # Módulos base tuyos
        # ====================================================
        'mmseg.structures.dual_task_seg_data_sample',
        'mmseg.models.cls_heads.simple_cls_head',
        'mmseg.models.backbones.bisenetv1',
        'mmseg.models.backbones.resnet',

        # ====================================================
        # Dataset y transforms de pares low-latency
        # ====================================================
        'mmseg.datasets.video_pair_dual_task_dataset',
        'mmseg.datasets.transforms.load_video_pair',
        'mmseg.datasets.transforms.resize_video_pair',
        'mmseg.datasets.transforms.pack_video_pair_inputs',

        # Augmentation sincronizada para pares
        # Deben existir en mmsegmentation/mmseg/datasets/transforms/
        'mmseg.datasets.transforms.random_crop_video_pair',
        'mmseg.datasets.transforms.random_flip_video_pair',
        'mmseg.datasets.transforms.pair_random_clahe_video_pair',
        'mmseg.datasets.transforms.pair_photometric_distortion_video_pair',

        # Remapeo de clasificación después de flip
        'mmseg.datasets.transforms.map_cls_after_flip',

        # ====================================================
        # Modelo low-latency
        # ====================================================
        'mmseg.models.data_preprocessors.pair_seg_data_preprocessor',
        'mmseg.models.video_modules.adaptive_feature_propagation',
        'mmseg.models.video_modules.adaptive_keyframe_selector',
        'mmseg.models.segmentors.bisenet_adaptive_video_segmentor',
    ],
    allow_failed_imports=False,
)

# ============================================================
# RUTAS
# ============================================================

DATA_ROOT_TRAIN = (
    '/0_secuencias_ann/1a_secuencia'
)

DATA_ROOT_VAL = (
    '/0_secuencias_ann/1a_secuencia_val_mixed'
)

TRAIN_PAIR_FILE = 'train_pairs_k30_sparsegt.txt'
TRAIN_CLS_FILE = 'cls_labels_train_k30_sparsegt.txt'

VAL_PAIR_FILE = 'val_pairs_mixed.txt'
VAL_CLS_FILE = 'cls_labels_val_mixed.txt'

dataset_type = 'VideoPairDualTaskDataset'
crop_size = (512, 512)

# ============================================================
# CHECKPOINT BASE FRAME-BY-FRAME
# ============================================================

# IMPORTANTE:
# Para entrenamiento real, reemplaza None por tu checkpoint base multitarea.
# Ejemplo:

load_from = '/mmsegmentation/work_dirs/multitask_val_fix_wloss/best_cls_acc_cls_top1_iter_25600.pth'
resume = False

# ============================================================
# MODELO
# ============================================================

model = dict(
    type='BiSeNetAdaptiveVideoSegmentor',
    train_mode='propagation',
    freeze_base=True,

    data_preprocessor=dict(
        type='PairSegDataPreProcessor',

        # Mantengo True porque tu config base frame-by-frame usaba bgr_to_rgb=True.
        # Si cargas un checkpoint entrenado con esa config, debe mantenerse igual.
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

    # Se construye, aunque en train_mode='propagation' no se usa la pérdida scheduler.
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

# TRAIN:
# Similar a tu DFF:
# Resize grande -> crop 512 -> flip -> CLAHE -> PhotoMetric.
# Importante: estos transforms deben aplicar la misma operación a:
# key_img, img y gt_seg_map.
train_pipeline = [
    dict(type='LoadVideoPairFromFile'),

    # Igual filosofía que DFF: ampliar primero para permitir crop aleatorio.
    dict(type='ResizeVideoPair', scale=(1024, 1024)),

    dict(
        type='RandomCropVideoPair',
        crop_size=crop_size,
        cat_max_ratio=0.75,
    ),

    dict(
        type='RandomFlipVideoPair',
        prob=0.5,
        direction='horizontal',
    ),

    # Después del flip, remapear clasificación:
    # LEFT <-> RIGHT, STRAIGHT queda igual.
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

    dict(type='PackVideoPairSegInputs'),
]

# VAL:
# Sin augmentation aleatoria. Solo resize fijo.
val_pipeline = [
    dict(type='LoadVideoPairFromFile'),
    dict(type='ResizeVideoPair', scale=(512, 512)),
    dict(type='PackVideoPairSegInputs'),
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
            # En esta etapa el objetivo principal es entrenar propagación.
            # Si freeze_base=True funciona como corresponde, backbone/decode/cls
            # quedan congelados, pero estos multiplicadores no molestan.
            'feature_propagation': dict(lr_mult=1.0, decay_mult=1.0),
            'keyframe_selector': dict(lr_mult=0.0, decay_mult=1.0),
            'backbone': dict(lr_mult=0.0, decay_mult=1.0),
            'decode_head': dict(lr_mult=0.0, decay_mult=1.0),
            'cls_head': dict(lr_mult=0.0, decay_mult=1.0),
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
    max_iters=40000,
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

        # Guarda mejor checkpoint por segmentación y clasificación,
        # igual que tu config DFF.
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

# ============================================================
# ENTORNO
# ============================================================

env_cfg = dict(cudnn_benchmark=False)
log_level = 'INFO'

work_dir = './work_dirs/bisenet_adaptive_propagation_seqaug_pretrain'
