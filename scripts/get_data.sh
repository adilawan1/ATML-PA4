#!/usr/bin/env bash
# Clone the course data repository at the commit used for this submission (no data is committed here).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_COMMIT="6a0a9ec9da7ef3d313ee3036cfec493886723644"
DEST="${PA4_DATA:-$ROOT/external/atml-pa4-datasets}"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/AbDu11aHHH/atml-pa4-datasets.git "$DEST"
fi
git -C "$DEST" fetch -q origin || true
git -C "$DEST" checkout -q "$DATA_COMMIT"
echo "data repo at $(git -C "$DEST" rev-parse HEAD) -> $DEST"
echo "Next: python $DEST/scripts/setup_data.py --public   (and --prefix-banks on a CUDA GPU)"
