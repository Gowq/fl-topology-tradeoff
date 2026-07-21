#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="$ROOT/experiments/exp07_mhealth_generalization/results"
HYPERPARAMETERS="$RESULTS/tuned_hyperparameters.json"
SMOKE_OUTPUT="${EXP07_BATCH_SMOKE_OUTPUT:-/tmp/exp07_mhealth_batch_smoke}"
BATCH_INDICES="${EXP07_BATCH_SMOKE_INDICES:-0,62,170,203,260,287,350,380,420,509}"

if [[ ! -s "$HYPERPARAMETERS" ]]; then
    echo "missing tuned hyperparameters: $HYPERPARAMETERS" >&2
    exit 2
fi

started=$(date +%s)
bash "$ROOT/scripts/run.sh" exp07 \
    --config-indices "$BATCH_INDICES" \
    --device cuda \
    --hyperparameters-file "$HYPERPARAMETERS" \
    --output-dir "$SMOKE_OUTPUT"
elapsed=$(( $(date +%s) - started ))

python3 - "$SMOKE_OUTPUT" "$BATCH_INDICES" "$elapsed" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected = len(sys.argv[2].split(","))
elapsed = int(sys.argv[3])
files = list(output.glob("*.json"))
if len(files) != expected:
    raise SystemExit(f"incomplete batch smoke: expected={expected}, observed={len(files)}")
for path in files:
    result = json.loads(path.read_text())
    if len(result["round_metrics"]) != 25 or result["device"] != "cuda":
        raise SystemExit(f"invalid batch smoke result: {path}")
print(f"EXP07 PEGASUS BATCH SMOKE PASSED: configs={expected}/{expected} rounds=25 cuda=true elapsed_s={elapsed}")
PY
