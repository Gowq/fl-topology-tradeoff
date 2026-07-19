#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
import sys
from pathlib import Path

experiment = Path("experiments/exp07_mhealth_generalization")
results = experiment / "results"
sys.path.insert(0, str(experiment / "code"))
from protocol import build_protocol

tuned = json.loads((results / "tuned_hyperparameters.json").read_text())
if set(tuned.get("selected", {})) != {"hfl", "vfl"}:
    raise SystemExit("invalid tuned_hyperparameters.json")

expected = {config.config_id for config in build_protocol()[380:510]}
present = {path.stem for path in results.glob("*.json")}
missing = sorted(expected - present)
if missing:
    raise SystemExit(f"missing {len(missing)} Pegasus results in indices 380-509")

print("Validated tuned parameters and 130 Pegasus results.")
PY

mkdir -p logs

smoke_job=$(sbatch \
    --parsable \
    --array=0 \
    --time=01:00:00 \
    --job-name=exp07_resume_smoke \
    scripts/grid/run_exp07_mhealth.sh)
echo "smoke=${smoke_job} array=0"

portions=(
    "generalization:0-79"
    "scale:80-169"
    "robustness_a:170-259"
    "robustness_b:260-349"
    "robustness_vfl:350-379"
)

for portion in "${portions[@]}"; do
    name="${portion%%:*}"
    indices="${portion#*:}"
    job_id=$(sbatch \
        --parsable \
        --dependency="afterok:${smoke_job}" \
        --array="${indices}%2" \
        --job-name="exp07_${name}" \
        scripts/grid/run_exp07_mhealth.sh)
    echo "${name}=${job_id} array=${indices}%2 dependency=afterok:${smoke_job}"
done
