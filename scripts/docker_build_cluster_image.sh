#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-fl-topology-tradeoff:cluster-cu130}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime}"

docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${IMAGE}" \
    .

if [ "${SAVE_TAR:-0}" = "1" ]; then
    safe_name="${IMAGE//[:\/]/_}"
    docker save "${IMAGE}" | gzip -c > "${safe_name}.tar.gz"
    echo "wrote ${safe_name}.tar.gz"
fi
