# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import os.path as osp
from typing import Optional, Dict

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class SchedulerVideoPairDualTaskDataset(BaseDataset):
    """Dataset exclusivo para 03_train_adaptive_scheduler.py.

    Espera pair_file con 4 columnas:

        key_img current_img ann_img dev_target

    Ejemplo:

        1661922815.000000.png 1661922818.000000.png 1661922818.000000.png 0.084371
        orig/1661922897.000000.png orig/1661922898.000000.png orig/1661922898.000000.png 0.024075
    """

    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [0, 128, 0]],
        cls_classes=('LEFT', 'STRAIGHT', 'RIGHT'),
    )

    LABEL_MAP = {
        'LEFT': 0,
        'STRAIGHT': 1,
        'RIGHT': 2,
        'izquierda': 0,
        'recta': 1,
        'derecha': 2,
        0: 0,
        1: 1,
        2: 2,
    }

    def __init__(
        self,
        data_root: str,
        data_prefix: Dict[str, str],
        pair_file: str,
        cls_labels_file: Optional[str] = None,
        pipeline=None,
        test_mode: bool = False,
        **kwargs,
    ):
        self.pair_file = pair_file
        self.cls_labels_file = cls_labels_file

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs,
        )

    def _full_path(self, *parts):
        return osp.join(self.data_root, *parts)

    def _read_cls_labels(self):
        label_by_relpath = {}

        if self.cls_labels_file is None:
            return label_by_relpath

        labels_path = self._full_path(self.cls_labels_file)

        if not osp.exists(labels_path):
            raise FileNotFoundError(f'No existe cls_labels_file: {labels_path}')

        with open(labels_path, 'r') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                parts = line.split()

                if len(parts) < 2:
                    raise ValueError(
                        f'Línea inválida en {labels_path}:{line_no}: {line}'
                    )

                relpath = parts[0]
                label_raw = parts[1]

                if label_raw in self.LABEL_MAP:
                    label = self.LABEL_MAP[label_raw]
                else:
                    try:
                        label = int(label_raw)
                    except Exception as e:
                        raise ValueError(
                            f'Etiqueta desconocida en {labels_path}:{line_no}: {label_raw}'
                        ) from e

                label_by_relpath[relpath] = int(label)
                label_by_relpath[osp.basename(relpath)] = int(label)

        return label_by_relpath

    def load_data_list(self):
        cls_label_by_relpath = self._read_cls_labels()

        pair_path = self._full_path(self.pair_file)

        if not osp.exists(pair_path):
            raise FileNotFoundError(f'No existe pair_file: {pair_path}')

        img_prefix = self.data_prefix.get('img_path', '')
        seg_prefix = self.data_prefix.get('seg_map_path', '')

        data_list = []

        with open(pair_path, 'r') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                parts = line.split()

                if len(parts) < 4:
                    raise ValueError(
                        f'Para scheduler se requieren 4 columnas: '
                        f'key_img current_img ann_img dev_target. '
                        f'Error en {pair_path}:{line_no}: {line}'
                    )

                key_rel = parts[0]
                cur_rel = parts[1]
                ann_rel = parts[2]

                try:
                    dev_target = float(parts[3])
                except ValueError as e:
                    raise ValueError(
                        f'dev_target debe ser numérico. '
                        f'Error en {pair_path}:{line_no}: {parts[3]}'
                    ) from e

                key_img_path = self._full_path(img_prefix, key_rel)
                cur_img_path = self._full_path(img_prefix, cur_rel)
                seg_map_path = self._full_path(seg_prefix, ann_rel)

                if not osp.exists(key_img_path):
                    raise FileNotFoundError(
                        f'No existe key_img_path en línea {line_no}: {key_img_path}'
                    )

                if not osp.exists(cur_img_path):
                    raise FileNotFoundError(
                        f'No existe cur_img_path en línea {line_no}: {cur_img_path}'
                    )

                if not osp.exists(seg_map_path):
                    raise FileNotFoundError(
                        f'No existe seg_map_path en línea {line_no}: {seg_map_path}'
                    )

                if cur_rel in cls_label_by_relpath:
                    gt_label = cls_label_by_relpath[cur_rel]
                elif osp.basename(cur_rel) in cls_label_by_relpath:
                    gt_label = cls_label_by_relpath[osp.basename(cur_rel)]
                else:
                    raise KeyError(
                        f'No encontré gt_label para current frame "{cur_rel}" '
                        f'en {self.cls_labels_file}'
                    )

                data_info = dict(
                    key_img_path=key_img_path,
                    cur_img_path=cur_img_path,
                    img_path=cur_img_path,
                    seg_map_path=seg_map_path,

                    key_img_rel=key_rel,
                    cur_img_rel=cur_rel,
                    seg_map_rel=ann_rel,

                    gt_label=int(gt_label),
                    dev_target=float(dev_target),

                    label_map=self.LABEL_MAP,
                    reduce_zero_label=False,
                    seg_fields=[],
                )

                data_list.append(data_info)

        return data_list
