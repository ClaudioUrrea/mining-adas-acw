"""Single import point that registers every custom MMSegmentation module.

MMSegmentation's ``MODELS``/``DATASETS``/``TRANSFORMS`` registries only know
about classes that have been imported. The configs under ``configs/`` use a
``custom_imports`` block to pull in the right modules; this file makes sure
the same registrations also happen for plain Python users (notebooks, unit
tests, scripts under ``scripts/``).

Just do::

    import src  # noqa: F401

at the top of your script and every backbone, head, dataset, transform,
preprocessor, and segmentor below will be available by name in MMSeg's
registries.
"""
from __future__ import annotations

# Structures
from .structures import dual_task_seg_data_sample  # noqa: F401

# Models — backbones, heads, segmentors, video modules
from .models import bisenetv1                              # noqa: F401
from .models import resnet                                 # noqa: F401
from .models import simple_cls_head                        # noqa: F401
from .models import dual_task_segmentor                    # noqa: F401
from .models import cls_only_segmentor                     # noqa: F401
from .models import adaptive_feature_propagation           # noqa: F401
from .models import adaptive_keyframe_selector             # noqa: F401
from .models import bisenet_adaptive_video_segmentor       # noqa: F401

# Data preprocessors
from .preprocessors import pair_seg_data_preprocessor  # noqa: F401

# Datasets
from .data import dual_task_dataset                            # noqa: F401
from .data import video_pair_dual_task_dataset                 # noqa: F401
from .data import scheduler_video_pair_dual_task_dataset       # noqa: F401

# Transforms
from .data.transforms import pack_seg_inputs_with_label                  # noqa: F401
from .data.transforms import pack_video_pair_inputs                      # noqa: F401
from .data.transforms import pack_scheduler_video_pair_inputs            # noqa: F401
from .data.transforms import load_video_pair                             # noqa: F401
from .data.transforms import resize_video_pair                           # noqa: F401
from .data.transforms import random_flip_video_pair                      # noqa: F401
from .data.transforms import random_crop_video_pair                      # noqa: F401
from .data.transforms import pair_random_clahe_video_pair                # noqa: F401
from .data.transforms import pair_photometric_distortion_video_pair      # noqa: F401
from .data.transforms import map_cls_after_flip                          # noqa: F401
from .data.transforms import preresize                                   # noqa: F401

__all__: list = []
