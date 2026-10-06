#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_no_mpi.sh [total_N] [ranks] [reps] [executable] [launch_delay_seconds]
PROBLEM_SIZE=${1:-1048576}
NUM_RANKS=${2:-4}
REPETITIONS=${3:-100}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXECUTABLE=${4:-$SCRIPT_DIR/../../build/ubuntu/hw11_process}
LAUNCH_DELAY=${5:-5}
for value in "$PROBLEM_SIZE" "$NUM_RANKS" "$REPETITIONS" "$LAUNCH_DELAY"; do
  if [[ ! "$value" =~ ^[1-9][0-9]{0,9}$ ]]; then
    echo "Arguments must be positive decimal integers without leading zeros." >&2
    exit 2
  fi
done
if (( NUM_RANKS > 128 || REPETITIONS > 900 || PROBLEM_SIZE > 1000000000 || PROBLEM_SIZE < NUM_RANKS || LAUNCH_DELAY > 3600 )); then
  echo "Require ranks <= 128, reps <= 900, ranks <= N <= 1000000000, delay <= 3600." >&2
  exit 2
fi
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "Executable not found: $EXECUTABLE. Build with python3 tools/run_exercises.py --build --hw 11 --suite none." >&2
  exit 2
fi
COMMON_START_MS=$(( $(date +%s%3N) + LAUNCH_DELAY * 1000 ))
pids=()
for ((rank=0; rank<NUM_RANKS; rank++)); do
  "$EXECUTABLE" "$PROBLEM_SIZE" "$NUM_RANKS" "$REPETITIONS" "$rank" "$COMMON_START_MS" &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
