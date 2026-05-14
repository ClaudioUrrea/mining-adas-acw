# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import random
import mmcv

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomFlipVideoPair(BaseTransform):
    """Flip sincronizado para key_img, img y gt_seg_map."""

    def __init__(self, prob=0.5, direction='horizontal'):
        super().__init__()
        self.prob = float(prob)
        self.direction = direction
        assert self.direction in ['horizontal', 'vertical']

    def transform(self, results):
        do_flip = random.random() < self.prob

        results['flip'] = do_flip
        results['flip_direction'] = self.direction if do_flip else None

        if not do_flip:
            return results

        results['key_img'] = mmcv.imflip(results['key_img'], direction=self.direction)
        results['img'] = mmcv.imflip(results['img'], direction=self.direction)
        results['gt_seg_map'] = mmcv.imflip(results['gt_seg_map'], direction=self.direction)

        return results
