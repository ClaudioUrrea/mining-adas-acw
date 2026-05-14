"""Utility shim that re-exports symbols from ``mmseg.models.utils``.

``src/models/bisenetv1.py`` does ``from ..utils import resize`` and
``src/models/resnet.py`` does ``from ..utils import ResLayer`` because both
files were written to live under ``mmseg/models/backbones/`` (where ``..utils``
resolves to ``mmseg.models.utils``). Re-exporting the same symbols here lets
the files work unmodified both as a standalone package and as a drop-in to an
existing MMSegmentation tree.
"""
from __future__ import annotations
from mmseg.models.utils import resize, ResLayer  # noqa: F401
__all__ = ["resize", "ResLayer"]
