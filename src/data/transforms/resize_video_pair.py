# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import mmcv
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ResizeVideoPair(BaseTransform):
    """Resize para key_img, img y gt_seg_map.

    scale debe venir como (W, H), por ejemplo (512, 512).
    """

    def __init__(
        self,
        scale=(512, 512),
        img_interpolation='bilinear',
        seg_interpolation='nearest',
    ):
        self.scale = tuple(scale)
        self.img_interpolation = img_interpolation
        self.seg_interpolation = seg_interpolation

    def transform(self, results):
        if 'key_img' not in results:
            raise KeyError('Falta key_img. Revisa LoadVideoPairFromFile.')
        if 'img' not in results:
            raise KeyError('Falta img. Revisa LoadVideoPairFromFile.')
        if 'gt_seg_map' not in results:
            raise KeyError('Falta gt_seg_map. Revisa LoadVideoPairFromFile.')

        results['key_img'] = mmcv.imresize(
            results['key_img'],
            self.scale,
            interpolation=self.img_interpolation,
        )
        results['img'] = mmcv.imresize(
            results['img'],
            self.scale,
            interpolation=self.img_interpolation,
        )
        results['gt_seg_map'] = mmcv.imresize(
            results['gt_seg_map'],
            self.scale,
            interpolation=self.seg_interpolation,
        )

        results['img_shape'] = results['img'].shape[:2]
        results['key_img_shape'] = results['key_img'].shape[:2]
        results['pad_shape'] = results['img'].shape[:2]
        results['scale'] = self.scale
        results['scale_factor'] = (1.0, 1.0)

        return results
