from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class MapClsAfterFlip(BaseTransform):
    """Remapea gt_label SOLO si hubo flip horizontal antes en el pipeline."""
    def __init__(self,
                 map_if_flipped=None,
                 str_map_if_flipped=None,
                 field_name='gt_label'):
        # Tu orden: ['izquierda','recta','derecha'] -> ids [0,1,2]
        self.map_if_flipped = map_if_flipped or {0: 2, 1: 1, 2: 0}
        self.str_map_if_flipped = str_map_if_flipped or {
            "izquierda": "derecha", "recta": "recta", "derecha": "izquierda"
        }
        self.field_name = field_name

    def transform(self, results):
        # 1) Si NO hay flip o no es horizontal, no hacemos nada:
        if not results.get('flip', False):
            return results
        if results.get('flip_direction', None) != 'horizontal':
            return results

        # 2) Solo entonces remapear la clase:
        key = self.field_name  # 'gt_label'
        if key in results:
            lbl = results[key]
            if isinstance(lbl, int):
                results[key] = self.map_if_flipped.get(lbl, lbl)
            elif isinstance(lbl, str):
                results[key] = self.str_map_if_flipped.get(lbl, lbl)
        return results
