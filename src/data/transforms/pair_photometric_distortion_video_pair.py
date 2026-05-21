# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import random
import cv2
import numpy as np

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PairPhotoMetricDistortionVideoPair(BaseTransform):
    """PhotoMetricDistortion sincronizado para key_img e img.

    Usa los mismos parámetros aleatorios en ambos frames para no romper
    la coherencia temporal del par.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        super().__init__()
        self.brightness_delta = int(brightness_delta)
        self.contrast_range = tuple(contrast_range)
        self.saturation_range = tuple(saturation_range)
        self.hue_delta = int(hue_delta)

    def _distort(self, img, params):
        img = img.astype(np.float32)

        # brightness
        if params['do_brightness']:
            img += params['brightness_delta']

        # contrast antes
        if params['contrast_mode'] == 0 and params['do_contrast']:
            img *= params['contrast_alpha']

        img = np.clip(img, 0, 255).astype(np.uint8)

        # saturation/hue en HSV
        if params['do_saturation'] or params['do_hue']:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

            if params['do_saturation']:
                hsv[:, :, 1] *= params['saturation_alpha']

            if params['do_hue']:
                hsv[:, :, 0] += params['hue_delta']
                hsv[:, :, 0][hsv[:, :, 0] > 179] -= 180
                hsv[:, :, 0][hsv[:, :, 0] < 0] += 180

            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        img = img.astype(np.float32)

        # contrast después
        if params['contrast_mode'] == 1 and params['do_contrast']:
            img *= params['contrast_alpha']

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _sample_params(self):
        return dict(
            do_brightness=random.randint(0, 1) == 1,
            brightness_delta=random.uniform(-self.brightness_delta, self.brightness_delta),

            contrast_mode=random.randint(0, 1),
            do_contrast=random.randint(0, 1) == 1,
            contrast_alpha=random.uniform(*self.contrast_range),

            do_saturation=random.randint(0, 1) == 1,
            saturation_alpha=random.uniform(*self.saturation_range),

            do_hue=random.randint(0, 1) == 1,
            hue_delta=random.uniform(-self.hue_delta, self.hue_delta),
        )

    def transform(self, results):
        params = self._sample_params()

        results['key_img'] = self._distort(results['key_img'], params)
        results['img'] = self._distort(results['img'], params)
        results['photometric_params'] = params

        return results
