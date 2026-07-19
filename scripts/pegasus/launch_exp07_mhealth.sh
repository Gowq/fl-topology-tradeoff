#!/usr/bin/env bash
# Launch a disjoint acceleration slice on Pegasus in detached screen sessions.
set -euo pipefail

REMOTE_REPO="${EXP07_REMOTE_REPO:-$HOME/experimentos/fl-topology-tradeoff-exp07-pr3}"
IMAGE="${EXP07_PEGASUS_IMAGE:-exp05_pegasus:latest}"
CONTAINER_REPO="/work/$(basename "$REMOTE_REPO")"

if screen -list 2>/dev/null | grep -q '[.]exp07_pegasus_'; then
  echo "Exp. 07 Pegasus sessions already exist; refusing a duplicate launch." >&2
  screen -list 2>/dev/null | grep '[.]exp07_pegasus_' || true
  exit 3
fi

run_in_screen() {
  local session="$1"
  shift
  screen -dmS "$session" docker run --rm --gpus all --shm-size=8g \
    -v "$HOME/experimentos:/work" \
    -w "$CONTAINER_REPO" \
    "$IMAGE" \
    bash scripts/pegasus/run_exp07_mhealth.sh "$@"
}

run_in_screen exp07_pegasus_tune tune
run_in_screen exp07_pegasus_tail_a portion 380 444
run_in_screen exp07_pegasus_tail_b portion 445 509

echo "Launched Pegasus sessions:"
screen -list 2>/dev/null | grep '[.]exp07_pegasus_' || true
