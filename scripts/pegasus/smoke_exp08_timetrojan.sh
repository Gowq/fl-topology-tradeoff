#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${EXP08_SMOKE_OUTPUT:-$ROOT/experiments/exp08_timetrojan_topology/results/smoke}"
ARTIFACT_ROOT="${EXP08_SMOKE_ARTIFACTS:-$ROOT/experiments/exp08_timetrojan_topology/artifacts/smoke}"
mkdir -p "$OUTPUT_ROOT" "$ARTIFACT_ROOT"

for artifact_index in 0 5; do
    bash "$ROOT/scripts/run.sh" exp08 --prepare-artifact \
        --artifact-index "$artifact_index" --device cuda --artifact-dir "$ARTIFACT_ROOT"
done

for index in 0 1 2 3 4 5; do
    bash "$ROOT/scripts/run.sh" exp08 --smoke-only --config-index "$index" \
        --device cuda --output-dir "$OUTPUT_ROOT" --artifact-dir "$ARTIFACT_ROOT"
done

python3 - "$OUTPUT_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

paths = sorted(Path(sys.argv[1]).glob("*.json"))
assert len(paths) == 6, f"expected 6 smoke JSONs, found {len(paths)}"
hashes = {}
for path in paths:
    row = json.loads(path.read_text())
    assert row["device"].startswith("cuda")
    assert math.isfinite(row["final"]["clean_f1_macro"])
    if row["config"]["poison_rate"] > 0:
        key = (row["config"]["dataset"], row["config"]["seed"])
        hashes.setdefault(key, row["attack"]["artifact_hash"])
        assert hashes[key] == row["attack"]["artifact_hash"]
print("EXP08 TIMETROJAN PEGASUS SMOKE PASSED: 6/6 cuda=true")
PY
