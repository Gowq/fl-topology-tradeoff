#!/usr/bin/env bash
# Docker entrypoint: ensure datasets are present, then dispatch to run.sh.
# The target is the container's CMD / `docker compose run` argument:
#   (default) list      — show available experiment targets
#   figures             — rebuild figures from shipped results
#   smoke               — fast end-to-end stack check (1 seed)
#   exp01|exp02|exp03|exp04 — reproduce one primary experiment
#   all|reproduce       — reproduce exp01-04 + figures (GPU host recommended)
#   defense-semantic | defense-losses [args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-list}"

echo ">>> torch: $(python3 -c 'import torch;print(torch.__version__, "cuda="+str(torch.cuda.is_available()))' 2>/dev/null || echo 'n/a')"

# Cluster runs normally mount ./data as a volume. Set PREPARE_DATA=1 only when
# the container should download public datasets itself.
if [ "${PREPARE_DATA:-0}" = "1" ]; then
    bash "$ROOT/scripts/prepare_data.sh"
fi

exec bash "$ROOT/scripts/run.sh" "$@"
