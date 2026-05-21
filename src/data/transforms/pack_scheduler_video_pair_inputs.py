# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import numpy as np
import torch

from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData, LabelData
from mmseg.structures import SegDataSample
from mmseg.registry import TRANSFORMS


def _img_to_tensor(img):
    if img.ndim == 2:
        img = img[:, :, None]

    if not img.flags.c_contiguous:
        img = np.ascontiguousarray(img)

    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def _seg_to_tensor(seg):
    if not seg.flags.c_contiguous:
        seg = np.ascontiguousarray(seg)

    tensor = torch.from_numpy(seg).long()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    return tensor


def _label_to_int(label):
    if isinstance(label, str):
        m = {
            'LEFT': 0,
            'STRAIGHT': 1,
            'RIGHT': 2,
            'izquierda': 0,
            'recta': 1,
            'derecha': 2,
        }

        if label not in m:
            raise KeyError(f'Etiqueta de clasificación desconocida: {label}')

        return m[label]

    if torch.is_tensor(label):
        return int(label.item())

    return int(label)


@TRANSFORMS.register_module()
class PackSchedulerVideoPairInputs(BaseTransform):
    """Pack exclusivo para 03_train_adaptive_scheduler.py.

    Guarda dev_target solamente en data_sample.metainfo['dev_target'],
    que es donde lo busca BiSeNetAdaptiveVideoSegmentor.
    """

    def __init__(self):
        super().__init__()

    def transform(self, results):
        if 'key_img' not in results:
            raise KeyError('Falta key_img.')

        if 'img' not in results:
            raise KeyError('Falta img.')

        if 'gt_seg_map' not in results:
            raise KeyError('Falta gt_seg_map.')

        if 'dev_target' not in results:
            raise KeyError(
                'Falta dev_target en results. '
                'Usa SchedulerVideoPairDualTaskDataset y un pair_file con 4 columnas.'
            )

        key_tensor = _img_to_tensor(results['key_img'])
        cur_tensor = _img_to_tensor(results['img'])
        seg_tensor = _seg_to_tensor(results['gt_seg_map'])

        data_sample = SegDataSample()
        data_sample.gt_sem_seg = PixelData(data=seg_tensor)

        if 'gt_label' in results:
            label = _label_to_int(results['gt_label'])
            data_sample.set_field(
                LabelData(label=torch.tensor(label, dtype=torch.long)),
                'gt_label',
                dtype=LabelData,
            )

        metainfo = dict(
            img_path=results.get('img_path', None),
            key_img_path=results.get('key_img_path', None),
            cur_img_path=results.get('cur_img_path', results.get('img_path', None)),
            seg_map_path=results.get('seg_map_path', None),

            key_img_rel=results.get('key_img_rel', None),
            cur_img_rel=results.get('cur_img_rel', None),
            seg_map_rel=results.get('seg_map_rel', None),

            ori_shape=results.get('ori_shape', results['img'].shape[:2]),
            img_shape=results.get('img_shape', results['img'].shape[:2]),
            key_img_shape=results.get('key_img_shape', results['key_img'].shape[:2]),
            pad_shape=results.get('pad_shape', results['img'].shape[:2]),
            scale_factor=results.get('scale_factor', (1.0, 1.0)),

            # Clave que usa BiSeNetAdaptiveVideoSegmentor._get_dev_targets()
            dev_target=float(results['dev_target']),
        )

        data_sample.set_metainfo(metainfo)

        packed_results = dict(
            inputs=dict(
                key=key_tensor,
                cur=cur_tensor,
            ),
            data_samples=data_sample,
        )

        return packed_results
