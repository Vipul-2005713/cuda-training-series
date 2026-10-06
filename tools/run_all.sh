#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec python3 tools/run_exercises.py --build --include-solutions --suite all "$@"
