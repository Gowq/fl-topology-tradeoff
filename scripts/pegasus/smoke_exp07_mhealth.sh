#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="$ROOT/experiments/exp07_mhealth_generalization/results"
HYPERPARAMETERS="$RESULTS/tuned_hyperparameters.json"
SMOKE_OUTPUT="${EXP07_SMOKE_OUTPUT:-/tmp/exp07_mhealth_smoke}"

if [[ ! -s "$HYPERPARAMETERS" ]]; then
  echo "missing tuned hyperparameters: $HYPERPARAMETERS" >&2
  exit 2
fi

for config_index in $(seq 0 4); do
  bash "$ROOT/scripts/run.sh" exp07 \
    --smoke-only \
    --config-index "$config_index" \
    --device cuda \
    --output-dir "$SMOKE_OUTPUT/protocol"
done

bash "$ROOT/scripts/run.sh" exp07 \
  --config-index 0 \
  --rounds 2 \
  --device cuda \
  --hyperparameters-file "$HYPERPARAMETERS" \
  --output-dir "$SMOKE_OUTPUT/real"

python3 - "$SMOKE_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
protocol = list((root / "protocol").glob("*.json"))
real = list((root / "real").glob("*.json"))
if len(protocol) != 5 or len(real) != 1:
    raise SystemExit(f"incomplete smoke: protocol={len(protocol)}, real={len(real)}")
for path in protocol:
    result = json.loads(path.read_text())
    if len(result["round_metrics"]) != 1 or result["device"] != "cuda":
        raise SystemExit(f"invalid protocol smoke result: {path}")
result = json.loads(real[0].read_text())
if len(result["round_metrics"]) != 2 or result["device"] != "cuda":
    raise SystemExit(f"invalid real smoke result: {real[0]}")
print("EXP07 PEGASUS SMOKE PASSED: protocol=5/5 real=1/1 cuda=true")
PY
