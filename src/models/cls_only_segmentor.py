from typing import List, Optional

import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import LabelData
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import add_prefix


@MODELS.register_module()
class ClsOnlySegmentor(BaseSegmentor):
    """Strict classification-only model using a segmentation backbone.

    - Builds only backbone + cls_head
    - No decode_head
    - No segmentation loss/prediction
    - Uses feats[0] (BiSeNet fused feature) as input to cls_head
    """

    def __init__(self,
                 backbone: dict,
                 cls_head: dict,
                 data_preprocessor: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        # Some MMSeg/MMEngine configs still pass a top-level `pretrained` field
        # when building the segmentor. This strict cls-only model does not use a
        # model-level pretrained argument, but we accept it here for compatibility.
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments for ClsOnlySegmentor: {list(kwargs.keys())}')
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, inputs: Tensor):
        return self.backbone(inputs)

    def _get_fuse_feat(self, feats):
        if not isinstance(feats, (list, tuple)) or len(feats) < 1:
            raise RuntimeError(
                'Expected the backbone to return a non-empty list/tuple of features.')
        return feats[0]

    def encode_decode(self,
                      inputs: Tensor,
                      data_samples: Optional[List[SegDataSample]] = None):
        feats = self.extract_feat(inputs)
        x_cls = self._get_fuse_feat(feats)
        return self.cls_head.forward(x_cls)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: Optional[List[SegDataSample]] = None):
        cls_logits = self.encode_decode(inputs, data_samples)
        return dict(cls_logits=cls_logits)

    def loss(self,
             inputs: Tensor,
             data_samples: List[SegDataSample]) -> dict:
        feats = self.extract_feat(inputs)
        x_cls = self._get_fuse_feat(feats)
        cls_losses = self.cls_head.loss(x_cls, data_samples)
        return add_prefix(cls_losses, 'cls')

    def predict(self,
                inputs: Tensor,
                data_samples: Optional[List[SegDataSample]] = None) -> List[SegDataSample]:
        feats = self.extract_feat(inputs)
        x_cls = self._get_fuse_feat(feats)
        cls_logits = self.cls_head.forward(x_cls)
        cls_probs = F.softmax(cls_logits, dim=1)

        if data_samples is None:
            batch_size = inputs.shape[0]
            data_samples = [SegDataSample() for _ in range(batch_size)]

        results = []
        for i, sample in enumerate(data_samples):
            out = sample.clone()
            label_i = cls_probs[i].argmax(dim=-1).reshape(1)
            score_i = cls_probs[i]
            out.set_field(
                LabelData(label=label_i, score=score_i),
                'pred_label',
                dtype=LabelData)
            results.append(out)
        return results
