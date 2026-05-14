import os
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class automine1dDual(BaseSegDataset):
    """Dataset multitarea: segmentación binaria + clasificación global.

    - Segmentación: 0=background, 1=road
    - Clasificación: {izquierda, recta, derecha} desde un txt (uno por split)
    """

    METAINFO = {
        'classes': ['background', 'road'],
        'palette': [[0, 0, 0], [0, 128, 0]],  # mismo palette que usabas
        'cls_labels': ['izquierda', 'recta', 'derecha']
    }

    def __init__(self,
                 cls_labels_file=None,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs):
        # archivo de etiquetas de clasificación (p. ej. cls_labels_train.txt)
        self.cls_labels_file = cls_labels_file
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         **kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()

        # Si no pediste clasificación, queda solo segmentación
        if not self.cls_labels_file:
            return data_list

        # Lee mapa: nombre_de_imagen -> clase ('izquierda'|'recta'|'derecha')
        label_path = os.path.join(self.data_root, self.cls_labels_file)
        if not os.path.isfile(label_path):
            raise FileNotFoundError(
                f'No existe el archivo de etiquetas de clasificación: {label_path}')

        label_map = {}
        with open(label_path, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # Permite espacios extra: split en 2 partes
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f'Formato inválido en {self.cls_labels_file} (línea {ln}): "{line}" '
                        'Debe ser: "<nombre_archivo> <etiqueta>"')
                name, label = parts[0], parts[1]
                label_map[name] = label

        name2id = {name: i for i, name in enumerate(self.METAINFO['cls_labels'])}

        # Inyecta gt_label (int) por cada muestra
        for item in data_list:
            fname = os.path.basename(item['img_path'])
            if fname not in label_map:
                raise KeyError(
                    f'La imagen "{fname}" no aparece en {self.cls_labels_file}. '
                    'Asegurate de que los nombres (con extensión) coincidan.')
            cls_name = label_map[fname]
            if cls_name not in name2id:
                raise KeyError(
                    f'Clase "{cls_name}" no está en cls_labels {list(name2id.keys())} '
                    f'({self.cls_labels_file}).')
            item['gt_label'] = name2id[cls_name]

        return data_list
