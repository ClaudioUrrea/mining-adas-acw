from mmseg.structures import SegDataSample
from mmengine.structures import LabelData

class DualTaskSegDataSample(SegDataSample):
    @property
    def gt_label(self) -> LabelData:
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData) -> None:
        self.set_field(value, '_gt_label', dtype=LabelData)

    @property
    def pred_label(self) -> LabelData:
        return self._pred_label

    @pred_label.setter
    def pred_label(self, value: LabelData) -> None:
        self.set_field(value, '_pred_label', dtype=LabelData)
