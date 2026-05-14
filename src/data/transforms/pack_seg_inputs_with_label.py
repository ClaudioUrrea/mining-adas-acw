"""PackSegInputsWithLabel transform.

Extends ``PackSegInputs`` from MMSegmentation to also include the
classification ``gt_label`` field inside the produced ``SegDataSample``.
"""
from __future__ import annotations

import torch
from mmengine.structures import LabelData
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.formatting import PackSegInputs as _Pack


@TRANSFORMS.register_module()
class PackSegInputsWithLabel(_Pack):
    """Same as ``PackSegInputs`` but also packs ``gt_label`` as ``LabelData``."""

    def transform(self, results):
        packed = super().transform(results)
        ds = packed['data_samples']

        def _assign(sample, value):
            if torch.is_tensor(value):
                value = int(value)
            sample.set_field(
                LabelData(label=int(value)), 'gt_label', dtype=LabelData)

        if 'gt_label' in results:
            if isinstance(ds, list):
                for i, dsi in enumerate(ds):
                    _assign(dsi, results['gt_label'][i])
            else:
                _assign(ds, results['gt_label'])

        return packed
