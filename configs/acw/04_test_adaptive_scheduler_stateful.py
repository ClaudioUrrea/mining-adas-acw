# ============================================================
# 04_test_adaptive_scheduler_stateful.py
# Config para notebooks/scripts de evaluación stateful:
# - video simulado
# - benchmark GPU-only
# - benchmark E2E
# - comparación con Clockwork/DFF
#
# IMPORTANTE:
# Este config NO implementa por sí solo el loop temporal.
# El notebook debe leer:
#   adaptive_stateful_cfg['tau']
#   adaptive_stateful_cfg['first_frame_fire']
#   adaptive_stateful_cfg['max_hold_frames']
# y aplicar FIRE/HOLD frame por frame.
# ============================================================

default_scope = 'mmseg'

custom_imports = dict(
    imports=[
        # ====================================================
        # Módulos base
        # ====================================================
        'mmseg.structures.dual_task_seg_data_sample',
        'mmseg.models.cls_heads.simple_cls_head',
        'mmseg.models.backbones.bisenetv1',
        'mmseg.models.backbones.resnet',

        # ====================================================
        # Módulos low-latency/adaptive
        # ====================================================
        'mmseg.models.data_preprocessors.pair_seg_data_preprocessor',
        'mmseg.models.video_modules.adaptive_feature_propagation',
        'mmseg.models.video_modules.adaptive_keyframe_selector',
        'mmseg.models.segmentors.bisenet_adaptive_video_segmentor',
    ],
    allow_failed_imports=False,
)

# ============================================================
# CHECKPOINT FINAL DEL 03
# ============================================================
# Cambia este path si quieres usar best_seg_mIoU o best_cls_acc_cls_top1.
# Para notebooks puedes leer CKPT_PATH directamente desde cfg.CKPT_PATH,
# o pasar este mismo path como argumento checkpoint al inicializar el modelo.

CKPT_PATH = (
    '/home/maximilianovelez/00_openmmlab_respaldo/probando_low_latency/mmsegmentation/work_dirs/bisenet_adaptive_scheduler_from_pretrain02/best_cls_acc_cls_top1_iter_16200.pth'
)

load_from = CKPT_PATH
resume = False

# ============================================================
# RUTAS DE EVALUACIÓN POR DEFECTO
# ============================================================
# Estas rutas son solo referencias para tus notebooks.
# El notebook puede reemplazarlas por INPUT_VIDEO, IMG_DIR, ANN_DIR, etc.

DATA_ROOT_EVAL = (
    '/home/maximilianovelez/00_openmmlab_respaldo/'
    'probando_low_latency/0_secuencias_ann/1a_secuencia_val_mixed'
)

# Variante por defecto dentro del val_mixed.
# Puedes cambiar a:
#   'orig'
#   'clahe'
#   'photo'
#   'flip'
EVAL_VARIANT = 'orig'

IMG_DIR = DATA_ROOT_EVAL + '/img/' + EVAL_VARIANT
ANN_DIR = DATA_ROOT_EVAL + '/ann/' + EVAL_VARIANT

CLS_LABELS_FILE = DATA_ROOT_EVAL + '/cls_labels_val_mixed.txt'

# Si usas una secuencia no mixed, por ejemplo:
# /0_secuencias_ann/1a_secuencia/img
# puedes sobrescribir estas rutas desde el notebook.

# ============================================================
# PARÁMETROS STATEFUL PARA NOTEBOOK/BENCHMARK
# ============================================================

adaptive_stateful_cfg = dict(
    # Umbral del scheduler:
    #   dev_pred > tau  -> FIRE
    #   dev_pred <= tau -> HOLD
    #
    # Valor inicial razonable. Después conviene hacer sweep:
    # 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08
    tau=0.02,

    # Primer frame siempre debe ser keyframe.
    first_frame_fire=True,

    # Si quieres forzar un FIRE cada cierto número máximo de HOLD,
    # coloca por ejemplo max_hold_frames=30.
    # Si quieres que solo decida el scheduler, deja None.
    max_hold_frames=None,

    # Si quieres evitar decisiones muy nerviosas, puedes usar persistencia
    # en el notebook. Por defecto queda desactivada.
    persistence=1,

    # Tamaño de entrada usado durante entrenamiento.
    input_size=(512, 512),

    # Normalización igual al modelo base.
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],

    # Nombres de clases de clasificación.
    cls_classes=['LEFT', 'STRAIGHT', 'RIGHT'],

    # Clases de segmentación.
    seg_classes=['background', 'road'],

    # Para reportes.
    method_name='bisenet_adaptive_scheduler_stateful',
)

# ============================================================
# MODELO
# ============================================================

model = dict(
    type='BiSeNetAdaptiveVideoSegmentor',

    # No es entrenamiento, pero dejamos train_mode='scheduler'
    # porque este checkpoint final incluye keyframe_selector entrenado.
    train_mode='scheduler',

    # En inferencia no importa para gradientes, pero mantiene la intención:
    # backbone/decode/cls/propagation ya vienen desde el checkpoint.
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
            init_cfg=None,
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

    # Estos parámetros no ejecutan la lógica por sí solos.
    # El notebook debe leerlos y aplicar FIRE/HOLD.
    test_cfg=dict(
        mode='whole',
        adaptive_threshold=adaptive_stateful_cfg['tau'],
        first_frame_fire=adaptive_stateful_cfg['first_frame_fire'],
        max_hold_frames=adaptive_stateful_cfg['max_hold_frames'],
        persistence=adaptive_stateful_cfg['persistence'],
    ),
)

# ============================================================
# TEST PIPELINE SIMPLE
# ============================================================
# Este pipeline es solo auxiliar. Para el benchmark stateful probablemente
# cargarás frames con OpenCV y preprocesarás manualmente para medir tiempos.

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackSegInputs'),
]

# ============================================================
# DATALOADER OPCIONAL
# ============================================================
# No es el test final stateful, pero queda disponible si quieres hacer
# comprobaciones rápidas con tools/test.py sobre imágenes independientes.
#
# Para la evaluación stateful real, usa tu notebook/script de benchmark.

dataset_type = 'BaseSegDataset'

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT_EVAL,
        data_prefix=dict(
            img_path='img/' + EVAL_VARIANT,
            seg_map_path='ann/' + EVAL_VARIANT,
        ),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='seg'),
]

test_cfg = dict(type='TestLoop')

env_cfg = dict(cudnn_benchmark=False)
log_level = 'INFO'

work_dir = './work_dirs/test_bisenet_adaptive_scheduler_stateful'
