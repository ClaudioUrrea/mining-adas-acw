# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import random
import cv2
import numpy as np

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PairRandomCLAHEVideoPair(BaseTransform):
    """CLAHE sincronizado para key_img e img.

    Aplica CLAHE con la misma decisión aleatoria a ambos frames.
    La máscara no se modifica.
    """

    def __init__(self, prob=0.3, clip_limit=2.0, tile_grid_size=(7, 7)):
        super().__init__()
        self.prob = float(prob)
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)

    def _apply_clahe_bgr(self, img):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if img.ndim == 2:
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
            return clahe.apply(img)

        # mmcv/cv2 carga normalmente en BGR.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        l2 = clahe.apply(l)

        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out

    def transform(self, results):
        do_clahe = random.random() < self.prob
        results['clahe'] = do_clahe

        if not do_clahe:
            return results

        results['key_img'] = self._apply_clahe_bgr(results['key_img'])
        results['img'] = self._apply_clahe_bgr(results['img'])
        return results
