"""Mining ADAS — Adaptive Clockwork (A-CW).

Reference implementation accompanying:
    Urrea, C.; Vélez, M. (2026) "Efficient Video-Based Multitask Scene
    Perception from Onboard Remote Sensing Imagery for Open-Pit Mining ADAS
    Using Classification-Guided Adaptive Inference." Remote Sensing (MDPI).

Importing this package registers every custom MMSegmentation module into the
global ``mmseg.registry``. After ``import src`` the configs under ``configs/``
can be built with ``mmengine.config.Config`` and ``MODELS.build`` directly.
"""
from __future__ import annotations

__version__ = "1.0.0"

# Side-effect import: registers every custom module with MMSegmentation.
from . import registry  # noqa: F401
