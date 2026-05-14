from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class FixOriToImgShape(BaseTransform):
    def transform(self, results):
        s = results.get('img_shape', None)
        if s is not None:
            results['ori_shape'] = s
            results['scale_factor'] = (1.0, 1.0)
        return results
