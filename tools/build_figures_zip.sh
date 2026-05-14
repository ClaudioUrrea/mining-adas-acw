#!/usr/bin/env bash
# =============================================================================
#  tools/build_figures_zip.sh
#
#  Builds the single figures ZIP required by the Remote Sensing (MDPI)
#  submission system. Workflow:
#      1. Run tools/verify_figures.py (≥600 dpi, RGB, ≥50 KB, openable).
#      2. Abort if any check fails.
#      3. Pack every file from tools/expected_figures.txt into the ZIP.
#      4. Emit a SHA-256 next to the ZIP for integrity reference.
#
#  Output:
#      figures_zip_for_submission.zip
#      figures_zip_for_submission.sha256
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

OUT_ZIP="figures_zip_for_submission.zip"
LIST="tools/expected_figures.txt"

if [[ ! -f "$LIST" ]]; then
    echo "ERROR: expected-figures list not found: $LIST" >&2
    exit 1
fi

echo "Step 1/3 — verifying every PNG against MDPI requirements"
python "$SCRIPT_DIR/verify_figures.py" --list "$LIST" --root "$REPO_ROOT"

echo ""
echo "Step 2/3 — packing $OUT_ZIP"
rm -f "$OUT_ZIP"

# Collect non-comment paths into a tmp file (zip -@ reads paths from stdin)
TMPLIST="$(mktemp)"
trap 'rm -f "$TMPLIST"' EXIT
grep -v '^\s*#' "$LIST" | grep -v '^\s*$' > "$TMPLIST"

# Use `zip` if available; otherwise fall back to Python.
if command -v zip >/dev/null 2>&1; then
    zip -q -@ "$OUT_ZIP" < "$TMPLIST"
else
    python - "$OUT_ZIP" "$TMPLIST" <<'PY'
import sys, zipfile
out, listfile = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    with open(listfile, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                z.write(p, p)
PY
fi

echo ""
echo "Step 3/3 — hashing"
sha256sum "$OUT_ZIP" > "${OUT_ZIP%.zip}.sha256"

echo ""
echo "Done."
ls -lh "$OUT_ZIP" "${OUT_ZIP%.zip}.sha256"
