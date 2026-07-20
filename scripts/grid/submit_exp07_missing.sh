#!/bin/bash
# Submit only missing Exp. 07 configs in resumable batches. The required smoke
# is executed on Pegasus before this script is invoked.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

batch_size="${EXP07_BATCH_SIZE:-10}"
if (( batch_size < 1 || batch_size > 10 )); then
    echo "EXP07_BATCH_SIZE must be in [1, 10]" >&2
    exit 2
fi

mkdir -p logs
manifest="$(pwd)/logs/exp07_batch_manifest_$(date +%Y%m%d_%H%M%S).txt"

python3 - "$manifest" "$batch_size" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
batch_size = int(sys.argv[2])
experiment = Path("experiments/exp07_mhealth_generalization")
results = experiment / "results"
sys.path.insert(0, str(experiment / "code"))
from protocol import build_protocol

missing = []
for index, config in enumerate(build_protocol()):
    path = results / f"{config.config_id}.json"
    if not path.exists():
        missing.append(index)
        continue
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        missing.append(index)
        continue
    if payload.get("config", {}).get("config_id") != config.config_id:
        missing.append(index)

if not missing:
    raise SystemExit("nothing to do: all 510 Grid/Pegasus results are present")

batches = [missing[start : start + batch_size] for start in range(0, len(missing), batch_size)]
manifest.write_text("".join(",".join(map(str, batch)) + "\n" for batch in batches))
print(f"missing={len(missing)} batches={len(batches)} batch_size={batch_size}")
PY

batch_count=$(wc -l < "$manifest")
last_task=$((batch_count - 1))

job_id=$(sbatch \
    --parsable \
    --array="0-${last_task}%2" \
    --export="ALL,EXP07_BATCH_MANIFEST=${manifest}" \
    --job-name=exp07_batches \
    scripts/grid/run_exp07_mhealth.sh)
echo "main=${job_id} array=0-${last_task}%2 manifest=${manifest}"
