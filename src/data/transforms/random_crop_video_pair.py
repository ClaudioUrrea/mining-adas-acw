# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import random
import numpy as np

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomCropVideoPair(BaseTransform):
    """Random crop sincronizado para key_img, img y gt_seg_map.

    Espera:
        results['key_img']
        results['img']
        results['gt_seg_map']

    Aplica el mismo crop espacial a las dos imágenes y a la máscara.
    """

    def __init__(self, crop_size=(512, 512), cat_max_ratio=1.0, ignore_index=255, num_retry=10):
        super().__init__()
        self.crop_size = tuple(crop_size)  # (H, W)
        self.cat_max_ratio = float(cat_max_ratio)
        self.ignore_index = int(ignore_index)
        self.num_retry = int(num_retry)

    def _get_crop_bbox(self, img_shape):
        h, w = img_shape[:2]
        crop_h, crop_w = self.crop_size

        if h <= crop_h:
            y1 = 0
        else:
            y1 = random.randint(0, h - crop_h)

        if w <= crop_w:
            x1 = 0
        else:
            x1 = random.randint(0, w - crop_w)

        y2 = min(y1 + crop_h, h)
        x2 = min(x1 + crop_w, w)
        return y1, y2, x1, x2

    def _crop(self, arr, bbox):
        y1, y2, x1, x2 = bbox
        return arr[y1:y2, x1:x2, ...]

    def _crop_seg(self, arr, bbox):
        y1, y2, x1, x2 = bbox
        return arr[y1:y2, x1:x2]

    def _is_valid_crop(self, seg_crop):
        if self.cat_max_ratio >= 1.0:
            return True

        labels, counts = np.unique(seg_crop, return_counts=True)
        valid = labels != self.ignore_index
        counts = counts[valid]

        if len(counts) <= 1:
            return True

        return counts.max() / counts.sum() < self.cat_max_ratio

    def transform(self, results):
        if 'key_img' not in results or 'img' not in results or 'gt_seg_map' not in results:
            raise KeyError('RandomCropVideoPair requiere key_img, img y gt_seg_map.')

        img = results['img']
        seg = results['gt_seg_map']

        bbox = self._get_crop_bbox(img.shape)

        for _ in range(self.num_retry):
            candidate = self._get_crop_bbox(img.shape)
            seg_crop = self._crop_seg(seg, candidate)
            if self._is_valid_crop(seg_crop):
                bbox = candidate
                break

        results['key_img'] = self._crop(results['key_img'], bbox)
        results['img'] = self._crop(results['img'], bbox)
        results['gt_seg_map'] = self._crop_seg(results['gt_seg_map'], bbox)

        results['img_shape'] = results['img'].shape[:2]
        results['key_img_shape'] = results['key_img'].shape[:2]
        results['pad_shape'] = results['img'].shape[:2]
        results['crop_bbox'] = bbox

        return results
