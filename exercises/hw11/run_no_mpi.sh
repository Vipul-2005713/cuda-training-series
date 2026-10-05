#!/usr/bin/env bash
set -euo pipefail
# Usage: ./run_no_mpi.sh [total_N=1048576] [ranks=4] [reps=100] [executable=./test]
PROBLEM_SIZE=${1:-1048576}
NUM_RANKS=${2:-4}
REPETITIONS=${3:-100}
EXECUTABLE=${4:-./test}
COMMON_START_MS=$(( ($(date +%s) + 5) * 1000 ))
pids=()
for ((rank=0; rank<NUM_RANKS; rank++)); do
  "$EXECUTABLE" "$PROBLEM_SIZE" "$NUM_RANKS" "$REPETITIONS" "$rank" "$COMMON_START_MS" &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
