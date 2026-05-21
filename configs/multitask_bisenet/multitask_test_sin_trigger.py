# ==========================================================
#  TEST /
#  Modelo actual:
#   - DualTaskSegmentor
#   - BiSeNetV1 + ResNet50
#   - decode_head (SEG)
#   - cls_head = SimpleClsHead
# ==========================================================

default_scope = 'mmseg'

custom_imports = dict(
    imports=[
        'mmseg.structures.dual_task_seg_data_sample',
        'mmseg.datasets.dual_task_dataset',
        'mmseg.datasets.transforms.pack_seg_inputs_with_label',
        'mmseg.datasets.transforms.preresize',
        'mmseg.models.segmentors.dual_task_segmentor',
        'mmseg.models.cls_heads.simple_cls_head',
        'mmseg.models.backbones.bisenetv1',
        'mmseg.models.backbones.resnet',
    ],
    allow_failed_imports=False
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
            init_cfg=None,
        ),
        in_channels=3,
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_indices=(0, 1, 2, 3),
        out_channels=1024,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=None
    ),

    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=0,
        channels=1024,
        num_classes=2,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),

    cls_head=dict(
        type='SimpleClsHead',
        in_channels=1024,
        num_classes=3,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

dataset_type = 'automine1dDual'
data_root = '/automine1d_cls/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='FixOriToImgShape'),
    dict(type='PackSegInputsWithLabel'),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'
        ),
        cls_labels_file='cls_labels_val.txt',
        pipeline=test_pipeline,
    )
)

test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='seg'),
    dict(type='ClsAccuracy', topk=(1,), prefix='cls'),
]

test_cfg = dict(type='TestLoop')

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)

work_dir = './work_dirs/multitask_test_sin_trigger_clean'
