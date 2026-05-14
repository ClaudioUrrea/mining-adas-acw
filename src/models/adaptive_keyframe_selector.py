# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class AdaptiveKeyFrameSelector(BaseModule):
    """Predice la desviación semántica entre keyframe y frame actual.

    Entrada: S_k y S_t, normalmente salidas del Spatial Path.
    Salida: dev_pred en [0, 1], shape [B].
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.reduce_key = ConvModule(
            in_channels, hidden_channels, kernel_size=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.reduce_cur = ConvModule(
            in_channels, hidden_channels, kernel_size=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv = nn.Sequential(
            ConvModule(
                hidden_channels, hidden_channels, kernel_size=3, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                hidden_channels, hidden_channels, kernel_size=3, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, spatial_key: torch.Tensor, spatial_cur: torch.Tensor) -> torch.Tensor:
        if spatial_key.shape[-2:] != spatial_cur.shape[-2:]:
            spatial_key = torch.nn.functional.interpolate(
                spatial_key, size=spatial_cur.shape[-2:], mode='bilinear', align_corners=False)
        k = self.reduce_key(spatial_key)
        t = self.reduce_cur(spatial_cur)
        x = torch.abs(t - k)
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        dev = torch.sigmoid(self.fc(x)).squeeze(1)
        return dev
