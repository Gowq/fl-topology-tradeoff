#!/usr/bin/env bash
# Docker entrypoint: ensure datasets are present, then dispatch to run.sh.
# The target is the container's CMD / `docker compose run` argument:
#   (default) figures   — rebuild figures from shipped results (fast, CPU, no data needed for figures)
#   smoke               — fast end-to-end stack check (1 seed)
#   exp01|exp02|exp03|exp04 — reproduce one primary experiment
#   all|reproduce       — reproduce exp01-04 + figures (GPU host recommended)
#   defense-semantic | defense-losses [args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-figures}"

echo ">>> torch: $(python3 -c 'import torch;print(torch.__version__, "cuda="+str(torch.cuda.is_available()))' 2>/dev/null || echo 'n/a')"

# Datasets are only needed to actually run experiments; the figures path reads
# committed results and needs no data.
if [ "$TARGET" != "figures" ]; then
    bash "$ROOT/scripts/prepare_data.sh"
fi

exec bash "$ROOT/scripts/run.sh" "$@"
