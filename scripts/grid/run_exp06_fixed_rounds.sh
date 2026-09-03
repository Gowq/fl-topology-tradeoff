#!/bin/bash
# Legacy compatibility wrapper. The fixed-round diagnostic was renamed to Exp05.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_exp05_fixed_rounds.sh" "$@"
