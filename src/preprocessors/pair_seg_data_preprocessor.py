# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Dict, List, Sequence, Union

import torch
from torch import Tensor

from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module()
class PairSegDataPreProcessor(BaseDataPreprocessor):
    """Preprocesador para pares de imágenes.

    Recibe ``inputs=dict(key=..., cur=...)`` y devuelve el mismo formato,
    pero normalizado, convertido a RGB si corresponde y en batch.

    Esta primera versión asume que las imágenes ya vienen con tamaño fijo
    desde ``ResizeVideoPair``; por eso no implementa padding complejo.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        pad_val: int = 0,
        seg_pad_val: int = 255,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert not (bgr_to_rgb and rgb_to_bgr), (
            'bgr_to_rgb y rgb_to_bgr no pueden ser True simultáneamente')
        self.channel_conversion = bgr_to_rgb or rgb_to_bgr
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.register_buffer(
            'mean', torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1), False)
        self.register_buffer(
            'std', torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1), False)

    @staticmethod
    def _stack_if_list(x: Union[Tensor, List[Tensor], tuple]) -> Tensor:
        if isinstance(x, Tensor):
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return x
        if isinstance(x, (list, tuple)):
            return torch.stack(list(x), dim=0)
        raise TypeError(f'Tipo de input no soportado: {type(x)}')

    def _preprocess_img(self, x: Union[Tensor, List[Tensor], tuple]) -> Tensor:
        x = self._stack_if_list(x).float()
        if self.channel_conversion and x.size(1) == 3:
            x = x[:, [2, 1, 0], ...]
        x = (x - self.mean) / self.std
        return x

    def forward(self, data: dict, training: bool = False) -> Dict:
        data = self.cast_data(data)
        inputs = data['inputs']

        if not isinstance(inputs, dict):
            raise TypeError(
                'PairSegDataPreProcessor espera inputs=dict(key=..., cur=...).')

        key = self._preprocess_img(inputs['key'])
        cur = self._preprocess_img(inputs['cur'])
        data['inputs'] = dict(key=key, cur=cur)
        return data
