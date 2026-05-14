# Copyright (c) OpenMMLab. All rights reserved.
# mmseg/models/cls_heads/simple_cls_head.py
from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmengine.structures import LabelData

Tensor = torch.Tensor


@MODELS.register_module()
class SimpleClsHead(BaseModule):
    """Cabeza de clasificación simple: GAP -> Dropout -> FC.

    Puede usarse tanto para la clasificación global multitarea como para un
    gatillo binario barato. En este último caso, ``label_map`` permite derivar
    la etiqueta binaria directamente desde ``gt_label`` sin tocar dataset ni
    pipeline, y ``roi_ratios`` permite recortar un ROI antes del GAP.

    Args:
        in_channels (int): Canales del feature map de entrada.
        num_classes (int): Número de clases de salida.
        dropout (float, optional): Dropout antes del clasificador.
        topk (Tuple[int, ...]): Métricas top-k a reportar en train/val.
        label_map (dict[int, int], optional): Remapeo opcional de ``gt_label``.
            Ejemplo para gatillo curva/recta con etiquetas globales
            [izquierda, recta, derecha] = [0, 1, 2]:
            ``{0: 1, 1: 0, 2: 1}``.
        roi_ratios (dict, optional): ROI normalizado para recortar el feature map
            antes del GAP. Formato:
            ``dict(x0=0.07, x1=0.93, y0=0.37, y1=0.79)``.
            Solo se usa si el tensor tiene forma (N, C, H, W).
        loss (dict, optional): Config de pérdida.
        init_cfg (dict, optional): Inicialización base.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: Optional[float] = 0.0,
        topk: Tuple[int, ...] = (1,),
        label_map: Optional[Dict[int, int]] = None,
        roi_ratios: Optional[Dict[str, float]] = None,
        loss: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        assert isinstance(in_channels, int) and in_channels > 0
        assert isinstance(num_classes, int) and num_classes >= 2

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.topk = tuple(sorted(set(int(k) for k in topk if int(k) >= 1))) or (1,)
        self.label_map = {int(k): int(v) for k, v in (label_map or {}).items()}
        self.roi_ratios = self._validate_roi_ratios(roi_ratios)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else None
        self.fc = nn.Linear(in_channels, num_classes, bias=True)

        loss_cfg = loss or dict(type='CrossEntropyLoss', loss_weight=1.0)
        self.loss_module = MODELS.build(loss_cfg)

        self.init_weights()

    def init_weights(self) -> None:  # type: ignore[override]
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    @staticmethod
    def _validate_roi_ratios(roi_ratios: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if roi_ratios is None:
            return None

        required = ('x0', 'x1', 'y0', 'y1')
        missing = [k for k in required if k not in roi_ratios]
        if missing:
            raise KeyError(f'roi_ratios incompleto. Faltan claves: {missing}')

        out = {k: float(roi_ratios[k]) for k in required}
        if not (0.0 <= out['x0'] < out['x1'] <= 1.0):
            raise ValueError(f'ROI inválido en X: {out["x0"]}, {out["x1"]}')
        if not (0.0 <= out['y0'] < out['y1'] <= 1.0):
            raise ValueError(f'ROI inválido en Y: {out["y0"]}, {out["y1"]}')
        return out

    def _crop_roi(self, x: Tensor) -> Tensor:
        if self.roi_ratios is None:
            return x

        if x.dim() != 4:
            raise ValueError(f'ROI solo puede aplicarse a tensores 4D. Recibido: {tuple(x.shape)}')

        _, _, h, w = x.shape
        x0 = max(0, min(w - 1, int(math.floor(self.roi_ratios['x0'] * w))))
        x1 = max(x0 + 1, min(w, int(math.ceil(self.roi_ratios['x1'] * w))))
        y0 = max(0, min(h - 1, int(math.floor(self.roi_ratios['y0'] * h))))
        y1 = max(y0 + 1, min(h, int(math.ceil(self.roi_ratios['y1'] * h))))

        x_roi = x[:, :, y0:y1, x0:x1]
        if x_roi.numel() == 0:
            raise ValueError(
                f'ROI vacío después del recorte. '
                f'H={h}, W={w}, x0={x0}, x1={x1}, y0={y0}, y1={y1}')
        return x_roi

    def _flatten_feat(
        self,
        x: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    ) -> Tensor:
        if isinstance(x, (list, tuple)):
            x = x[-1]
        assert torch.is_tensor(x), 'x debe ser Tensor'

        if x.dim() == 4:
            x = self._crop_roi(x)
            x = self.gap(x).flatten(1)
        elif x.dim() != 2:
            raise ValueError(f'Esperaba x con dim 2 o 4, recibido: {tuple(x.shape)}')

        if self.drop is not None and self.training:
            x = self.drop(x)
        return x

    def _maybe_map_label(self, value: int) -> int:
        if self.label_map:
            return self.label_map.get(int(value), int(value))
        return int(value)

    def _gather_gt_labels(self, data_samples) -> Tensor:
        gt_list = []
        for sample in data_samples:
            if not hasattr(sample, 'gt_label'):
                raise KeyError('Falta gt_label en un data_sample')

            label = getattr(sample, 'gt_label')

            if isinstance(label, LabelData):
                if label.get('label', None) is not None:
                    value = label.label
                    value = int(value.item()) if torch.is_tensor(value) else int(value)
                    gt_list.append(self._maybe_map_label(value))
                elif label.get('score', None) is not None:
                    score = label.score
                    if torch.is_tensor(score):
                        value = int(torch.argmax(score).item())
                    else:
                        value = int(max(range(len(score)), key=lambda i: score[i]))
                    gt_list.append(self._maybe_map_label(value))
                else:
                    raise ValueError('LabelData sin label ni score')
                continue

            if torch.is_tensor(label):
                gt_list.append(self._maybe_map_label(int(label.item())))
                continue

            gt_list.append(self._maybe_map_label(int(label)))

        device = next(self.parameters()).device
        return torch.tensor(gt_list, dtype=torch.long, device=device)

    def forward(self, x: Union[Tensor, List[Tensor], Tuple[Tensor, ...]]) -> Tensor:
        x = self._flatten_feat(x)
        logits = self.fc(x)
        return logits

    def loss(
        self,
        x: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        data_samples,
    ) -> dict:
        logits = self.forward(x)
        gt = self._gather_gt_labels(data_samples)
        losses = {}

        loss = self.loss_module(logits, gt)
        losses['loss_cls'] = loss

        with torch.no_grad():
            maxk = min(max(self.topk), logits.size(1))
            _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(gt.view(1, -1).expand_as(pred))
            for k in self.topk:
                k_eff = min(k, logits.size(1))
                correct_k = correct[:k_eff].reshape(-1).float().sum(0)
                acc = correct_k * (100.0 / logits.size(0))
                losses[f'acc_top{k}'] = acc
        return losses

    def predict(
        self,
        x: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        data_samples=None,
    ) -> Tensor:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs
