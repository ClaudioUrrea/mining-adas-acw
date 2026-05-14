# Copyright (c) OpenMMLab. All rights reserved.
"""Dataset de pares de frames para BiSeNet adaptativo.

Formato esperado del archivo de pares:

    key_img current_img target_ann
    key_img current_img target_ann FIRE|HOLD
    key_img current_img target_ann dev_target_float

Ejemplos:
    1661922815.000000.png 1661922818.000000.png 1661922818.000000.png
    orig/1661922898.000000.png orig/1661922901.000000.png orig/1661922901.000000.png
    1661922815.000000.png 1661922818.000000.png 1661922818.000000.png 0.1832

La etiqueta de clasificación se toma desde current_img usando cls_labels_file.
"""
from __future__ import annotations

import os.path as osp
from typing import Dict, List, Optional

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class VideoPairDualTaskDataset(BaseDataset):
    """Dataset para entrenamiento/evaluación con pares (keyframe, frame actual).

    Args:
        data_root: raíz del dataset.
        data_prefix: dict con ``img_path`` y ``seg_map_path``.
        pair_file: archivo con pares relativos a ``img_path``/``seg_map_path``.
        cls_labels_file: archivo con etiquetas de clasificación. Cada línea:
            ``relative_img_path LABEL`` o ``filename.png LABEL``.
        pipeline: pipeline de transforms.
        label_map: mapeo string->int para clasificación.
    """

    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [0, 128, 0]],
        cls_classes=('izquierda', 'recta', 'derecha'),
    )

    def __init__(
        self,
        data_root: str,
        data_prefix: Dict[str, str],
        pair_file: str,
        cls_labels_file: Optional[str] = None,
        pipeline: Optional[list] = None,
        label_map: Optional[Dict[str, int]] = None,
        test_mode: bool = False,
        **kwargs,
    ) -> None:
        self.pair_file = pair_file
        self.cls_labels_file = cls_labels_file
        self.label_map = label_map or dict(
            LEFT=0,
            STRAIGHT=1,
            RIGHT=2,
            izquierda=0,
            recta=1,
            derecha=2,
        )
        self.cls_label_by_relpath = {}
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs,
        )

    def _resolve_under_root(self, maybe_rel_path: str) -> str:
        if osp.isabs(maybe_rel_path):
            return maybe_rel_path
        return osp.join(self.data_root, maybe_rel_path)

    def _read_cls_labels(self) -> Dict[str, int]:
        labels: Dict[str, int] = {}
        if not self.cls_labels_file:
            return labels

        labels_path = self._resolve_under_root(self.cls_labels_file)
        if not osp.exists(labels_path):
            raise FileNotFoundError(f'No existe cls_labels_file: {labels_path}')

        with open(labels_path, 'r', encoding='utf-8') as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(
                        f'Línea inválida en {labels_path}:{line_no}: {raw!r}')
                rel_img = parts[0]
                lab_str = parts[1]
                if lab_str not in self.label_map:
                    raise KeyError(
                        f'Etiqueta de clasificación desconocida {lab_str!r} '
                        f'en {labels_path}:{line_no}')
                lab = int(self.label_map[lab_str])
                labels[rel_img] = lab
                labels[osp.basename(rel_img)] = lab
        return labels

    def load_data_list(self) -> List[dict]:
        self.cls_label_by_relpath = self._read_cls_labels()

        img_prefix = self.data_prefix.get('img_path', '')
        seg_prefix = self.data_prefix.get('seg_map_path', '')
        img_root = self._resolve_under_root(img_prefix)
        seg_root = self._resolve_under_root(seg_prefix)
        pair_path = self._resolve_under_root(self.pair_file)

        if not osp.exists(pair_path):
            raise FileNotFoundError(f'No existe pair_file: {pair_path}')

        data_list: List[dict] = []
        with open(pair_path, 'r', encoding='utf-8') as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) not in (3, 4):
                    raise ValueError(
                        f'pair_file debe tener 3 o 4 columnas. '
                        f'Error en {pair_path}:{line_no}: {raw!r}')

                key_rel, cur_rel, ann_rel = parts[:3]
                extra = parts[3] if len(parts) == 4 else None

                gt_label = self.cls_label_by_relpath.get(cur_rel)
                if gt_label is None:
                    gt_label = self.cls_label_by_relpath.get(osp.basename(cur_rel))
                if gt_label is None:
                    raise KeyError(
                        f'No encontré etiqueta cls para current_img={cur_rel!r}. '
                        f'Revisa {self.cls_labels_file}.')

                item = dict(
                    key_img_path=osp.join(img_root, key_rel),
                    img_path=osp.join(img_root, cur_rel),
                    seg_map_path=osp.join(seg_root, ann_rel),
                    key_img_rel=key_rel,
                    img_rel=cur_rel,
                    seg_map_rel=ann_rel,
                    gt_label=int(gt_label),
                    reduce_zero_label=False,
                    seg_fields=[],
                )

                if extra is not None:
                    if extra.upper() in ('FIRE', 'HOLD'):
                        item['frame_mode'] = extra.upper()
                    else:
                        try:
                            item['dev_target'] = float(extra)
                        except ValueError as exc:
                            raise ValueError(
                                f'Cuarta columna inválida en {pair_path}:{line_no}: {extra!r}. '
                                'Debe ser FIRE, HOLD o un float dev_target.') from exc

                data_list.append(item)

        if len(data_list) == 0:
            raise RuntimeError(f'No se cargó ningún par desde {pair_path}')
        return data_list
