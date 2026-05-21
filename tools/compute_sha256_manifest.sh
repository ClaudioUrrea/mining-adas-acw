#!/usr/bin/env bash
# =============================================================================
#  tools/compute_sha256_manifest.sh
#
#  Generates a deterministic per-file SHA-256 manifest for a directory tree.
#  The manifest is sorted by path so it is reproducible across machines and
#  filesystems. The manifest file itself is excluded.
#
#  Usage:
#      bash tools/compute_sha256_manifest.sh figshare-deposit
#
#  Verify with:
#      cd figshare-deposit
#      sha256sum --check MANIFEST.sha256
# =============================================================================
set -euo pipefail

ROOT="${1:-figshare-deposit}"
if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: not a directory: $ROOT" >&2
    exit 1
fi

cd "$ROOT"
OUT="MANIFEST.sha256"

# Sorted, NUL-delimited list of regular files, excluding the manifest itself.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
find . -type f ! -name "$OUT" -print0 | sort -z > "$TMP"

# Empty-tree guard
if [[ ! -s "$TMP" ]]; then
    echo "ERROR: no files under $ROOT — nothing to hash" >&2
    exit 1
fi

# Hash each file. GNU coreutils sha256sum is required.
: > "$OUT"
xargs -0 sha256sum < "$TMP" >> "$OUT"

# Strip the leading "./" so the manifest is identical from inside or outside.
sed -i 's|  \./|  |' "$OUT"

n=$(wc -l < "$OUT")
echo "Wrote $OUT with $n entries."
echo "Verify with:  cd $ROOT && sha256sum --check $OUT"
