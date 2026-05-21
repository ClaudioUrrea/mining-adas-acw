# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import mmcv
from mmengine.fileio import get
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadVideoPairFromFile(BaseTransform):
    """Carga keyframe, frame actual y máscara GT para entrenamiento por pares."""

    def __init__(
        self,
        color_type='color',
        imdecode_backend='cv2',
        ignore_empty=False,
    ):
        super().__init__()
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty

    def _load_image(self, filename):
        try:
            img_bytes = get(filename)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                backend=self.imdecode_backend,
            )
            if img is None:
                raise ValueError(f'No se pudo decodificar imagen: {filename}')
            return img
        except Exception:
            if self.ignore_empty:
                return None
            raise

    def _load_seg(self, filename):
        seg_bytes = get(filename)
        seg = mmcv.imfrombytes(
            seg_bytes,
            flag='unchanged',
            backend=self.imdecode_backend,
        )
        if seg is None:
            raise ValueError(f'No se pudo decodificar máscara: {filename}')

        if seg.ndim == 3:
            seg = seg[:, :, 0]

        return seg

    def transform(self, results):
        key_path = results.get('key_img_path', None)
        cur_path = results.get('cur_img_path', None)
        if cur_path is None:
            cur_path = results.get('img_path', None)

        seg_path = results.get('seg_map_path', None)

        if key_path is None:
            raise KeyError('Falta key_img_path en results.')
        if cur_path is None:
            raise KeyError('Falta img_path o cur_img_path en results.')
        if seg_path is None:
            raise KeyError('Falta seg_map_path en results.')

        key_img = self._load_image(key_path)
        cur_img = self._load_image(cur_path)
        gt_seg_map = self._load_seg(seg_path)

        if key_img is None or cur_img is None:
            return None

        results['key_img'] = key_img
        results['img'] = cur_img
        results['gt_seg_map'] = gt_seg_map

        results['ori_shape'] = cur_img.shape[:2]
        results['img_shape'] = cur_img.shape[:2]
        results['key_ori_shape'] = key_img.shape[:2]
        results['key_img_shape'] = key_img.shape[:2]

        results['filename'] = cur_path
        results['img_path'] = cur_path
        results['key_filename'] = key_path
        results['key_img_path'] = key_path
        results['seg_map_path'] = seg_path

        results['seg_fields'] = ['gt_seg_map']

        return results
