# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import PixelData, LabelData
from mmseg.registry import MODELS
from mmseg.models.segmentors import BaseSegmentor
from mmseg.structures import SegDataSample
from mmseg.utils import add_prefix


@MODELS.register_module()
class DualTaskSegmentor(BaseSegmentor):
    """Segmentor multitarea extendido.

    Ramas:
    - decode_head: segmentación desde x_fuse
    - cls_head: clasificación global (3 clases) desde x_fuse
    - trigger_head: gatillo binario curva/recta desde x_spatial
    """

    def __init__(self,
                 backbone: dict,
                 decode_head: dict,
                 cls_head: Optional[dict] = None,
                 trigger_head: Optional[dict] = None,
                 auxiliary_head: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        self.decode_head = MODELS.build(decode_head)
        self.cls_head = MODELS.build(cls_head) if cls_head else None
        self.trigger_head = MODELS.build(trigger_head) if trigger_head else None
        self.auxiliary_head = MODELS.build(auxiliary_head) if auxiliary_head else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        return self.backbone(inputs)

    def _get_fuse_feat(self, feats):
        if len(feats) < 1:
            raise ValueError('El backbone no devolvió features.')
        return feats[0]

    def _get_trigger_feat(self, feats):
        if len(feats) < 4:
            raise ValueError(
                'Se esperaba la salida x_spatial en feats[3]. '
                'Asegúrate de usar BiSeNetV1 con out_indices que incluyan 3.')
        return feats[3]

    def encode_decode(self,
                      inputs: Tensor,
                      data_samples: Optional[List[SegDataSample]] = None) -> Tensor:
        feats = self.extract_feat(inputs)
        seg_logits = self.decode_head.forward(feats)
        return seg_logits

    def _forward(self,
                 inputs: Tensor,
                 data_samples: Optional[List[SegDataSample]] = None):
        feats = self.extract_feat(inputs)
        outputs = dict(seg_logits=self.decode_head.forward(feats))

        if self.cls_head is not None:
            outputs['cls_logits'] = self.cls_head.forward(self._get_fuse_feat(feats))
        if self.trigger_head is not None:
            outputs['trigger_logits'] = self.trigger_head.forward(self._get_trigger_feat(feats))
        return outputs

    def loss(self,
             inputs: Tensor,
             data_samples: List[SegDataSample]) -> dict:
        feats = self.extract_feat(inputs)
        losses = {}

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

        return losses


    def predict(self,
                inputs: Tensor,
                data_samples: Optional[List[SegDataSample]] = None) -> List[SegDataSample]:
        feats = self.extract_feat(inputs)

        if data_samples is None:
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
                label_i = cls_probs[i].argmax(dim=-1).reshape(1)
                score_i = cls_probs[i]
                cls_data = LabelData(label=label_i, score=score_i)
                out.set_field(cls_data, 'pred_label', dtype=LabelData)

            if trigger_probs is not None:
                trigger_i = trigger_probs[i].argmax(dim=-1).reshape(1)
                trigger_score_i = trigger_probs[i]
                trigger_data = LabelData(label=trigger_i, score=trigger_score_i)
                out.set_field(trigger_data, 'pred_trigger_label', dtype=LabelData)
                # Alias opcional por robustez para métricas/scripts.
                out.set_field(trigger_data, 'pred_trigger', dtype=LabelData)

            results.append(out)

        return results
