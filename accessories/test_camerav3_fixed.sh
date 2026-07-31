#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <lens-position>" >&2
  echo "Example: $0 9.0" >&2
  exit 1
fi

LENS_POSITION="$1"

rpicam-hello \
  -t 0 \
  --width 1280 \
  --height 960 \
  --shutter 45000 \
  --gain 1 \
  --hdr off \
  --awbgains 1,1 \
  --autofocus-mode manual \
  --lens-position "$LENS_POSITION"
