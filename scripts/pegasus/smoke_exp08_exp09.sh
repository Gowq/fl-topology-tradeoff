#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${CROSS_DATASET_SMOKE_OUTPUT:-/tmp/exp08_exp09_smoke}"

for index in $(seq 0 3); do
  bash "$ROOT/scripts/run.sh" exp08 --smoke-only --config-index "$index" \
    --device cuda --output-dir "$OUTPUT/exp08"
  bash "$ROOT/scripts/run.sh" exp09 --smoke-only --config-index "$index" \
    --device cuda --output-dir "$OUTPUT/exp09"
done

python3 - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for experiment in ("exp08", "exp09"):
    files = list((root / experiment).glob("*.json"))
    if len(files) != 4:
        raise SystemExit(f"{experiment}: expected 4 results, found {len(files)}")
    for path in files:
        result = json.loads(path.read_text())
        if len(result["round_metrics"]) != 1 or result["device"] != "cuda":
            raise SystemExit(f"invalid smoke result: {path}")
print("EXP08+EXP09 PEGASUS SMOKE PASSED: 8/8 cuda=true")
PY
