# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import PixelData, LabelData
from mmseg.registry import MODELS
from mmseg.models.segmentors import BaseSegmentor
from mmseg.structures import SegDataSample
from mmseg.utils import add_prefix


@MODELS.register_module()
class BiSeNetAdaptiveVideoSegmentor(BaseSegmentor):
    """BiSeNetV1 con propagación adaptativa de contexto.

    Está diseñado para usar tu BiSeNetV1 existente, que expone:
        feats[0] = x_fuse
        feats[1] = x_context8
        feats[2] = x_context16
        feats[3] = x_spatial

    Entrenamiento de propagación:
        I_k -> Spatial + Context completo
        I_t -> Spatial
        C_k se propaga hacia C_t_prop
        FFM(S_t, C_t_prop) -> decode_head / cls_head

    Entrenamiento de scheduler:
        dev_pred = selector(S_k, S_t)
        loss_dev = SmoothL1(dev_pred, dev_target)
    """

    def __init__(
        self,
        backbone: dict,
        decode_head: dict,
        cls_head: Optional[dict] = None,
        trigger_head: Optional[dict] = None,
        auxiliary_head: Optional[dict] = None,
        feature_propagation: Optional[dict] = None,
        keyframe_selector: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        train_mode: str = 'propagation',
        freeze_base: bool = True,
        scheduler_loss_weight: float = 1.0,
        pretrained: Optional[str] = None,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.decode_head = MODELS.build(decode_head)
        self.cls_head = MODELS.build(cls_head) if cls_head else None
        self.trigger_head = MODELS.build(trigger_head) if trigger_head else None
        self.auxiliary_head = MODELS.build(auxiliary_head) if auxiliary_head else None

        if feature_propagation is None:
            raise ValueError('Debes definir feature_propagation')
        self.feature_propagation = MODELS.build(feature_propagation)

        self.keyframe_selector = MODELS.build(keyframe_selector) if keyframe_selector else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.train_mode = train_mode
        self.freeze_base = bool(freeze_base)
        self.scheduler_loss_weight = float(scheduler_loss_weight)

        if self.freeze_base:
            self._freeze_base_network()

    def _freeze_module(self, module) -> None:
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def _freeze_base_network(self) -> None:
        self._freeze_module(self.backbone)
        self._freeze_module(self.decode_head)
        if self.cls_head is not None:
            self._freeze_module(self.cls_head)
        if self.trigger_head is not None:
            self._freeze_module(self.trigger_head)
        if self.auxiliary_head is not None:
            self._freeze_module(self.auxiliary_head)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if mode and self.freeze_base:
            self._freeze_base_network()
            self.feature_propagation.train(True)
            if self.keyframe_selector is not None:
                self.keyframe_selector.train(True)
        return self

    def _full_bisenet_feats(self, x: Tensor):
        """Ejecuta BiSeNet completo usando módulos internos."""
        x_context8, x_context16 = self.backbone.context_path(x)
        x_spatial = self.backbone.spatial_path(x)
        x_fuse = self.backbone.ffm(x_spatial, x_context8)
        return x_fuse, x_context8, x_context16, x_spatial

    def _spatial_only(self, x: Tensor) -> Tensor:
        return self.backbone.spatial_path(x)

    def _propagated_feats(self, key: Tensor, cur: Tensor):
        """Features del frame actual usando contexto propagado desde keyframe."""
        if self.freeze_base:
            with torch.no_grad():
                _, context8_key, context16_key, spatial_key = self._full_bisenet_feats(key)
                spatial_cur = self._spatial_only(cur)
        else:
            _, context8_key, context16_key, spatial_key = self._full_bisenet_feats(key)
            spatial_cur = self._spatial_only(cur)

        context8_prop = self.feature_propagation(
            context_key=context8_key,
            spatial_key=spatial_key,
            spatial_cur=spatial_cur)
        x_fuse = self.backbone.ffm(spatial_cur, context8_prop)
        feats = (x_fuse, context8_prop, context16_key, spatial_cur)
        return feats, dict(
            spatial_key=spatial_key,
            spatial_cur=spatial_cur,
            context8_key=context8_key,
            context8_prop=context8_prop)

    def extract_feat(self, inputs) -> List[Tensor]:
        if isinstance(inputs, dict):
            feats, _ = self._propagated_feats(inputs['key'], inputs['cur'])
            return list(feats)
        return list(self._full_bisenet_feats(inputs))

    def _get_fuse_feat(self, feats):
        return feats[0]

    def _get_trigger_feat(self, feats):
        if len(feats) < 4:
            raise ValueError('Se esperaba x_spatial en feats[3].')
        return feats[3]

    def _get_dev_targets(self, data_samples: List[SegDataSample], device) -> Tensor:
        values = []
        for sample in data_samples:
            if 'dev_target' not in sample.metainfo:
                raise KeyError(
                    'Falta dev_target en data_sample.metainfo. '
                    'Genera un pair_file con 4 columnas usando generate_dev_targets_from_teacher.py')
            values.append(float(sample.metainfo['dev_target']))
        return torch.tensor(values, dtype=torch.float32, device=device)

    def loss(self, inputs: Dict[str, Tensor], data_samples: List[SegDataSample]) -> dict:
        key = inputs['key']
        cur = inputs['cur']
        losses = {}

        feats, aux = self._propagated_feats(key, cur)

        if self.train_mode in ('propagation', 'joint'):
            seg_losses = self.decode_head.loss(feats, data_samples, train_cfg=self.train_cfg)
            losses.update(add_prefix(seg_losses, 'decode'))

            if self.auxiliary_head is not None:
                aux_losses = self.auxiliary_head.loss(feats, data_samples, train_cfg=self.train_cfg)
                losses.update(add_prefix(aux_losses, 'aux'))

            if self.cls_head is not None:
                cls_losses = self.cls_head.loss(self._get_fuse_feat(feats), data_samples)
                losses.update(add_prefix(cls_losses, 'cls'))

            if self.trigger_head is not None:
                trigger_losses = self.trigger_head.loss(self._get_trigger_feat(feats), data_samples)
                losses.update(add_prefix(trigger_losses, 'trigger'))

        if self.train_mode in ('scheduler', 'joint'):
            if self.keyframe_selector is None:
                raise ValueError('train_mode requiere keyframe_selector')
            dev_pred = self.keyframe_selector(aux['spatial_key'], aux['spatial_cur'])
            dev_target = self._get_dev_targets(data_samples, dev_pred.device)
            loss_dev = F.smooth_l1_loss(dev_pred, dev_target)
            losses['scheduler/loss_dev'] = loss_dev * self.scheduler_loss_weight
            with torch.no_grad():
                losses['scheduler/mae_dev'] = torch.mean(torch.abs(dev_pred - dev_target))
                losses['scheduler/dev_pred_mean'] = torch.mean(dev_pred)
                losses['scheduler/dev_target_mean'] = torch.mean(dev_target)

        return losses

    def encode_decode(self, inputs, data_samples: Optional[List[SegDataSample]] = None) -> Tensor:
        feats = self.extract_feat(inputs)
        return self.decode_head.forward(feats)

    def _forward(self, inputs, data_samples: Optional[List[SegDataSample]] = None):
        feats = self.extract_feat(inputs)
        outputs = dict(seg_logits=self.decode_head.forward(feats))
        if self.cls_head is not None:
            outputs['cls_logits'] = self.cls_head.forward(self._get_fuse_feat(feats))
        if self.trigger_head is not None:
            outputs['trigger_logits'] = self.trigger_head.forward(self._get_trigger_feat(feats))
        if isinstance(inputs, dict) and self.keyframe_selector is not None:
            _, aux = self._propagated_feats(inputs['key'], inputs['cur'])
            outputs['dev_pred'] = self.keyframe_selector(aux['spatial_key'], aux['spatial_cur'])
        return outputs

    def predict(self, inputs, data_samples: Optional[List[SegDataSample]] = None) -> List[SegDataSample]:
        feats = self.extract_feat(inputs)

        if data_samples is None:
            if isinstance(inputs, dict):
                batch_size = inputs['cur'].shape[0]
            else:
                batch_size = inputs.shape[0]
            data_samples = [SegDataSample() for _ in range(batch_size)]

        batch_img_metas = [sample.metainfo for sample in data_samples]
        seg_logits_list = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)

        cls_probs = None
        if self.cls_head is not None:
            cls_probs = self.cls_head.predict(self._get_fuse_feat(feats))

        trigger_probs = None
        if self.trigger_head is not None:
            trigger_probs = self.trigger_head.predict(self._get_trigger_feat(feats))

        dev_pred = None
        if isinstance(inputs, dict) and self.keyframe_selector is not None:
            _, aux = self._propagated_feats(inputs['key'], inputs['cur'])
            dev_pred = self.keyframe_selector(aux['spatial_key'], aux['spatial_cur'])

        results: List[SegDataSample] = []
        for i, sample in enumerate(data_samples):
            out = sample.clone()
            seg_pred = seg_logits_list[i]
            if seg_pred.dim() == 3 and seg_pred.shape[0] > 1:
                seg_pred = seg_pred.argmax(dim=0, keepdim=True)
            elif seg_pred.dim() == 2:
                seg_pred = seg_pred.unsqueeze(0)
            out.pred_sem_seg = PixelData(data=seg_pred)

            if cls_probs is not None:
                cls_data = LabelData(
                    label=cls_probs[i].argmax(dim=-1).reshape(1),
                    score=cls_probs[i])
                out.set_field(cls_data, 'pred_label', dtype=LabelData)

            if trigger_probs is not None:
                trigger_data = LabelData(
                    label=trigger_probs[i].argmax(dim=-1).reshape(1),
                    score=trigger_probs[i])
                out.set_field(trigger_data, 'pred_trigger_label', dtype=LabelData)
                out.set_field(trigger_data, 'pred_trigger', dtype=LabelData)

            if dev_pred is not None:
                out.set_metainfo({'dev_pred': float(dev_pred[i].detach().cpu())})

            results.append(out)
        return results
