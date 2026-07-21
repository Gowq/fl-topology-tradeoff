#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${EXP07_SMOKE_OUTPUT:-$ROOT/experiments/exp07_corrected_attacks/results/smoke}"
mkdir -p "$OUTPUT_ROOT"

for index in 0 1 2 3 4; do
    bash "$ROOT/scripts/run.sh" exp07 --smoke-only --config-index "$index" \
        --device cuda --output-dir "$OUTPUT_ROOT"
done

python3 - "$OUTPUT_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

paths = sorted(Path(sys.argv[1]).glob("*.json"))
assert len(paths) == 5, f"expected 5 smoke JSONs, found {len(paths)}"
for path in paths:
    row = json.loads(path.read_text())
    assert row["device"].startswith("cuda")
    assert math.isfinite(row["final"]["f1_macro"])
print("EXP07 CORRECTED PEGASUS SMOKE PASSED: 5/5 cuda=true")
PY

