#!/usr/bin/env python3
"""Verify the paper-figure PNGs satisfy Remote Sensing (MDPI) submission rules.

For every entry in ``tools/expected_figures.txt`` (one path per line, ``#``
comments allowed) this script verifies:

    * the file exists,
    * the file is at least 50 KB (catches accidental empty / 1-pixel PNGs),
    * the dpi is >= 600 along both axes,
    * the image mode is RGB or RGBA (no indexed-colour PNGs),
    * the file can actually be opened and pixel-decoded by Pillow.

Exits with code 0 if every check passes, 1 otherwise. Designed to be the
gating step before ``tools/build_figures_zip.sh``.

Usage:
    python tools/verify_figures.py [--list tools/expected_figures.txt] [--root .]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

MIN_BYTES = 50 * 1024     # 50 KB
MIN_DPI = 600
ALLOWED_MODES = {"RGB", "RGBA"}


def _read_list(path: Path) -> List[Path]:
    rels: List[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("##"):
            continue
        rels.append(Path(line))
    return rels


def _check_one(path: Path) -> Tuple[bool, str]:
    """Return (ok, message) for a single file."""
    if not path.exists():
        return False, "missing"

    size = path.stat().st_size
    if size < MIN_BYTES:
        return False, f"too small ({size} B < {MIN_BYTES} B)"

    try:
        from PIL import Image
    except ImportError:
        return False, "Pillow is not installed; pip install Pillow"

    try:
        with Image.open(path) as im:
            im.verify()  # raises on corruption
        with Image.open(path) as im:  # re-open after verify
            im.load()
            dpi = im.info.get("dpi", (0, 0))
            mode = im.mode
            width, height = im.size
    except Exception as exc:  # pragma: no cover — IO / decode
        return False, f"open/decode failed: {exc!r}"

    if mode not in ALLOWED_MODES:
        return False, f"mode {mode!r} not in {ALLOWED_MODES}"

    if not isinstance(dpi, tuple) or len(dpi) < 2:
        return False, f"dpi tag missing or malformed: {dpi!r}"

    dpix, dpiy = float(dpi[0]), float(dpi[1])
    if dpix < MIN_DPI or dpiy < MIN_DPI:
        return False, f"dpi {dpix:.0f}x{dpiy:.0f} < {MIN_DPI}"

    return True, f"OK ({width}x{height}, {dpix:.0f}x{dpiy:.0f} dpi, {mode}, {size//1024} KB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", default="tools/expected_figures.txt", type=Path)
    parser.add_argument("--root", default=".", type=Path)
    args = parser.parse_args()

    if not args.list.exists():
        print(f"ERROR: figure list not found: {args.list}", file=sys.stderr)
        return 1

    rels = _read_list(args.list)
    if not rels:
        print(f"ERROR: figure list is empty: {args.list}", file=sys.stderr)
        return 1

    ok_count = 0
    fail_count = 0
    print(f"Checking {len(rels)} figure files (relative to {args.root.resolve()}):")
    print("-" * 78)
    for rel in rels:
        path = (args.root / rel).resolve()
        ok, msg = _check_one(path)
        marker = "  ✓" if ok else "  ✗"
        print(f"{marker}  {rel}")
        if not ok:
            print(f"        {msg}")
            fail_count += 1
        else:
            ok_count += 1
    print("-" * 78)
    print(f"OK: {ok_count} / Failed: {fail_count} / Total: {len(rels)}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
