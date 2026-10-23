#!/bin/bash
# Symlink KillBench's sitecustomize.py into the currently-active Python env's
# site-packages, so worker subprocesses (loky/joblib) auto-apply the Kill
# Switch shims at interpreter startup.
#
# Usage:
#   conda activate AutoGuard
#   bash KillBench/bootstrap/install.sh
#
# Re-run only when you change envs. Edits to sitecustomize.py itself are
# picked up automatically because we install a symlink, not a copy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/sitecustomize.py"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: source not found at $SRC" >&2
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: no python on PATH — did you forget to 'conda activate'?" >&2
    exit 1
fi

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
DEST="$SITE/sitecustomize.py"

# If an existing non-symlink file is there (e.g. a stale copy from a previous
# install), back it up before overwriting so the user can diff.
if [[ -e "$DEST" && ! -L "$DEST" ]]; then
    BACKUP="$DEST.bak.$(date +%Y%m%d_%H%M%S)"
    echo "Existing non-symlink file detected — backing up to $BACKUP"
    mv "$DEST" "$BACKUP"
fi

ln -sfn "$SRC" "$DEST"
echo "Linked: $DEST -> $SRC"

# Sanity check: re-run python and confirm import resolves to our file.
RESOLVED=$(python -c "import sitecustomize; print(sitecustomize.__file__)")
echo "Verified import: sitecustomize -> $RESOLVED"
