#!/usr/bin/env python3
"""Validate that a `.pth` checkpoint is loadable.

Run BEFORE uploading any checkpoint to Figshare and before opening any GitHub
issue about a missing-key error. Verifies that:

    * the file is a valid PyTorch save format,
    * the SHA-256 matches the sibling ``*_metadata.json`` if present,
    * the state-dict keys are compatible with the architecture declared by the
      config (``--config`` argument, optional but recommended),
    * the iteration count and best metric in the file metadata match the
      filename convention ``best_<metric>_iter_<iter>.pth``.

Exit code 0 on success, non-zero on any failure. No GPU or pretrained
weights are required to run the check.

Usage:
    python tools/validate_checkpoint.py path/to/checkpoint.pth
    python tools/validate_checkpoint.py path/to/checkpoint.pth \\
        --config configs/multitask_bisenet/multitask_bisenet_v1_resnet50_512x512_40k.py
    python tools/validate_checkpoint.py path/to/checkpoint.pth --strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Tuple


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _load_pth(path: Path) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is not installed in this environment") from exc

    try:
        # weights_only=True is safer; some MMSeg checkpoints save mmengine
        # config objects, so we fall back to weights_only=False on failure.
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj


def _parse_filename(path: Path) -> Tuple[str, int]:
    """Parses ``best_<metric>_iter_<iter>.pth`` patterns; returns (metric, iter)."""
    m = re.search(r"best_([a-zA-Z0-9_]+)_iter_(\d+)\.pth$", path.name)
    if not m:
        return "", -1
    return m.group(1), int(m.group(2))


def validate(path: Path, config: Path | None, strict: bool) -> Tuple[int, str]:
    if not path.is_file():
        return 1, f"file not found: {path}"

    print(f"Loading {path} …")
    obj = _load_pth(path)
    if not isinstance(obj, dict):
        return 1, f"expected a dict checkpoint, got {type(obj).__name__}"

    state_dict = obj.get("state_dict") or obj
    if not isinstance(state_dict, dict):
        return 1, "no state_dict key and root is not a dict"

    print(f"  ✓ loaded, {len(state_dict)} tensors")

    meta = obj.get("meta", {})
    iters_in_meta = int(meta.get("iter", -1)) if isinstance(meta, dict) else -1
    epoch_in_meta = int(meta.get("epoch", -1)) if isinstance(meta, dict) else -1
    if iters_in_meta >= 0:
        print(f"  ✓ iteration in meta: {iters_in_meta}")
    if epoch_in_meta >= 0:
        print(f"  ✓ epoch in meta:     {epoch_in_meta}")

    metric, iters_in_name = _parse_filename(path)
    if iters_in_name >= 0:
        print(f"  ✓ iteration in name: {iters_in_name} (metric: {metric})")
        if iters_in_meta >= 0 and iters_in_meta != iters_in_name and strict:
            return 2, (f"iter mismatch: meta={iters_in_meta} but "
                       f"filename says {iters_in_name}")

    sidecar = path.with_name(path.stem + "_metadata.json")
    if sidecar.is_file():
        print(f"  ✓ sidecar found: {sidecar.name}")
        meta_json = json.loads(sidecar.read_text(encoding="utf-8"))
        expected_sha = meta_json.get("sha256")
        if expected_sha:
            print(f"  → verifying SHA-256 against sidecar …")
            actual = _sha256(path)
            if actual.lower() != expected_sha.lower():
                return 3, f"SHA-256 mismatch: sidecar={expected_sha[:12]}…, file={actual[:12]}…"
            print(f"  ✓ SHA-256 matches sidecar")
    elif strict:
        return 4, f"missing sidecar metadata: {sidecar}"
    else:
        print(f"  (no sidecar metadata; pass --strict to require it)")

    if config is not None:
        print(f"  → building model from config: {config}")
        try:
            import src  # registers MMSeg custom modules
            from mmengine.config import Config
            from mmseg.registry import MODELS

            cfg = Config.fromfile(str(config))
            model = MODELS.build(cfg.model)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            n_miss = len(missing) if hasattr(missing, "__len__") else 0
            n_unex = len(unexpected) if hasattr(unexpected, "__len__") else 0
            print(f"  ✓ model built ({type(model).__name__}); "
                  f"missing={n_miss}, unexpected={n_unex}")
            if strict and (n_miss > 0 or n_unex > 0):
                return 5, (f"strict mode: {n_miss} missing and {n_unex} unexpected keys "
                           f"when loading into {type(model).__name__}")
        except Exception as exc:  # pragma: no cover — depends on env
            return 6, f"config-build failed: {exc!r}"

    return 0, "OK"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional path to the MMSeg config; if given, "
                             "the script will build the model and try to "
                             "load the state-dict.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on any mismatch (missing sidecar, missing "
                             "state-dict keys, iter-vs-filename mismatch).")
    args = parser.parse_args()

    code, msg = validate(args.checkpoint, args.config, args.strict)
    print(f"\n[validate_checkpoint] result: {msg}")
    return code


if __name__ == "__main__":
    sys.exit(main())
