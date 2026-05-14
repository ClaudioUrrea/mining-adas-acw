# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.registry import MODELS


class SpatiallyVariantConv2d(nn.Module):
    """Convolución espacialmente variable compartida entre canales.

    Args:
        kernel_size: tamaño del kernel local. Debe ser impar.

    Inputs:
        x: Tensor [B, C, H, W]
        weights: Tensor [B, K*K, H, W], normalizado por softmax.

    Output:
        Tensor [B, C, H, W]
    """

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size debe ser impar')
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k2 = self.kernel_size * self.kernel_size
        if weights.shape[1] != k2:
            raise ValueError(
                f'weights debe tener {k2} canales, recibido {weights.shape[1]}')
        if weights.shape[-2:] != (h, w):
            weights = F.interpolate(weights, size=(h, w), mode='bilinear', align_corners=False)
            weights = F.softmax(weights, dim=1)

        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        patches = patches.view(b, c, k2, h, w)
        out = (patches * weights.unsqueeze(1)).sum(dim=2)
        return out


@MODELS.register_module()
class KernelPredictor(BaseModule):
    """Predice kernels locales W(k,t) desde features espaciales S_k y S_t."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.kernel_size = int(kernel_size)
        k2 = self.kernel_size * self.kernel_size
        self.reduce_key = ConvModule(
            in_channels, hidden_channels, kernel_size=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.reduce_cur = ConvModule(
            in_channels, hidden_channels, kernel_size=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.body = nn.Sequential(
            ConvModule(
                hidden_channels * 3, hidden_channels, kernel_size=3, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                hidden_channels, hidden_channels, kernel_size=3, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Conv2d(hidden_channels, k2, kernel_size=1),
        )

    def forward(self, spatial_key: torch.Tensor, spatial_cur: torch.Tensor) -> torch.Tensor:
        if spatial_key.shape[-2:] != spatial_cur.shape[-2:]:
            spatial_key = F.interpolate(
                spatial_key, size=spatial_cur.shape[-2:], mode='bilinear', align_corners=False)
        k = self.reduce_key(spatial_key)
        t = self.reduce_cur(spatial_cur)
        x = torch.cat([k, t, torch.abs(t - k)], dim=1)
        weights = self.body(x)
        weights = F.softmax(weights, dim=1)
        return weights


@MODELS.register_module()
class AdaptiveFeaturePropagation(BaseModule):
    """Propaga C_k hacia C_t usando kernels estimados desde S_k y S_t."""

    def __init__(
        self,
        spatial_channels: int,
        context_channels: int,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        refine: bool = True,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.kernel_predictor = KernelPredictor(
            in_channels=spatial_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.svconv = SpatiallyVariantConv2d(kernel_size=kernel_size)
        self.refine = ConvModule(
            context_channels, context_channels, kernel_size=3, padding=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg) if refine else None

    def forward(
        self,
        context_key: torch.Tensor,
        spatial_key: torch.Tensor,
        spatial_cur: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.kernel_predictor(spatial_key, spatial_cur)
        context_prop = self.svconv(context_key, weights)
        if self.refine is not None:
            context_prop = self.refine(context_prop)
        return context_prop
