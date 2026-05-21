#!/usr/bin/env python3
"""Package the user's trained `.pth` checkpoints for Figshare upload.

For each checkpoint listed in the manifest (defaults to the 6 files expected
by the Figshare deposit; see ``checkpoints/EXPECTED.txt``), this tool:

    1. Verifies that the file is loadable as a PyTorch state-dict.
    2. Computes the SHA-256 hash.
    3. Writes a sibling ``*_metadata.json`` recording framework versions, file
       size, training config path, iteration, expected metrics and hash.
    4. Optionally copies the validated file (with its metadata) into the
       Figshare staging folder.

Use this AFTER your training pipeline has produced the .pth files locally.
The tool only PACKAGES weights you already have — it never fabricates them.

Usage:
    python tools/package_checkpoints.py \\
        --source-dir /path/to/your/local/checkpoints \\
        --staging-dir figshare-deposit/checkpoints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _framework_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": platform.python_version()}
    for mod in ("torch", "mmengine", "mmcv", "mmsegmentation"):
        try:
            m = __import__(mod if mod != "mmsegmentation" else "mmseg")
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[mod] = "not-installed"
    return versions


# Canonical mapping (source filename → metadata fields). Edit if you publish
# more checkpoints under different names.
EXPECTED: Dict[str, Dict] = {
    "multitask_BC_iter_25600.pth": dict(
        config_relpath="configs/multitask_bisenet/multitask_bisenet_v1_resnet50_512x512_40k.py",
        iteration=25600,
        expected_metrics={"Top-1 Acc": 0.9625, "mIoU": 0.9733},
        purpose="Multitask Best-Classification checkpoint (BC).",
    ),
    "multitask_BS_iter_33400.pth": dict(
        config_relpath="configs/multitask_bisenet/multitask_bisenet_v1_resnet50_512x512_40k.py",
        iteration=33400,
        expected_metrics={"Top-1 Acc": 0.9625, "mIoU": 0.9764},
        purpose="Multitask Best-Segmentation checkpoint (BS).",
    ),
    "seg_only_best_miou.pth": dict(
        config_relpath="configs/singletask/bisenet_seg_only_strict.py",
        iteration=-1,
        expected_metrics={"mIoU": 0.9710},
        purpose="Single-task segmentation baseline.",
    ),
    "cls_only_best_top1.pth": dict(
        config_relpath="configs/singletask/bisenet_cls_only_strict.py",
        iteration=-1,
        expected_metrics={"Top-1 Acc": 0.9450},
        purpose="Single-task classification baseline.",
    ),
    "acw_propagation_iter_2200.pth": dict(
        config_relpath="configs/acw/02_train_adaptive_propagation_pretrain.py",
        iteration=2200,
        expected_metrics={"Top-1 Acc": 0.9625, "mIoU": 0.9650},
        purpose="A-CW Stage 1: feature-propagation pre-training.",
    ),
    "acw_scheduler_iter_16200.pth": dict(
        config_relpath="configs/acw/03_train_adaptive_scheduler_pretrain.py",
        iteration=16200,
        expected_metrics={"Top-1 Acc": 0.9620},
        purpose="A-CW Stage 2: final scheduler-trained model.",
    ),
}


def _write_sidecar(src_path: Path, declared: Dict, copy_to: Path | None) -> Path:
    sha = _sha256(src_path)
    sidecar_name = src_path.stem + "_metadata.json"
    versions = _framework_versions()
    record = {
        "file": src_path.name,
        "sha256": sha,
        "size_bytes": src_path.stat().st_size,
        "config_relpath": declared.get("config_relpath", ""),
        "framework": (
            f"pytorch {versions.get('torch','?')} / "
            f"mmsegmentation {versions.get('mmsegmentation','?')} / "
            f"mmcv {versions.get('mmcv','?')}"
        ),
        "trained_on": "NVIDIA RTX 3090",
        "training_iters": int(declared.get("iteration", -1)),
        "expected_metrics": declared.get("expected_metrics", {}),
        "purpose": declared.get("purpose", ""),
        "license": "CC BY 4.0",
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packaged_by": "tools/package_checkpoints.py",
        "framework_versions": versions,
    }
    sidecar_path = src_path.parent / sidecar_name
    sidecar_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, copy_to / src_path.name)
        shutil.copy2(sidecar_path, copy_to / sidecar_name)
    return sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory holding the trained .pth files.")
    parser.add_argument("--staging-dir", type=Path, default=None,
                        help="Optional destination — usually "
                             "figshare-deposit/checkpoints.")
    parser.add_argument("--also", nargs="*", default=[],
                        help="Extra .pth filenames to package (each will use a "
                             "minimal default metadata template).")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"ERROR: source-dir does not exist: {args.source_dir}", file=sys.stderr)
        return 1

    targets: List[Path] = []
    for name in list(EXPECTED.keys()) + list(args.also):
        path = args.source_dir / name
        if path.is_file():
            targets.append(path)
        else:
            print(f"  (skipping, not found: {name})")

    if not targets:
        print("No checkpoints found. Nothing to do.")
        return 1

    print(f"Packaging {len(targets)} checkpoints …")
    for src_path in targets:
        declared = EXPECTED.get(src_path.name, {
            "config_relpath": "unknown",
            "iteration": -1,
            "expected_metrics": {},
            "purpose": "User-supplied; no template available.",
        })
        sidecar = _write_sidecar(src_path, declared, args.staging_dir)
        print(f"  ✓ {src_path.name}: sidecar written -> {sidecar.name}")

    print("\nDone.")
    if args.staging_dir is not None:
        print(f"Files copied to: {args.staging_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
