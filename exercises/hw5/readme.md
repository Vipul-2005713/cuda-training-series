# HW5: Atomics, reductions, and warp shuffle

`reductions.cu` compares per-element atomics, block aggregation, and warp-shuffle
aggregation. `max_reduction.cu` uses a two-stage maximum reduction.
`matrix_sums.cu` applies cooperative reduction to row sums.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 5 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw5_reductions 8388608 3
./build/ubuntu/hw5_reductions 1003 3
./build/ubuntu/hw5_reductions 33554432 3
./build/ubuntu/hw5_max_reduction 8388608 10
./build/ubuntu/hw5_max_reduction 1003 5
./build/ubuntu/hw5_max_reduction 1 5
./build/ubuntu/hw5_matrix_sums 2048 10
./build/ubuntu/hw5_matrix_sums 257 5
```

## What to expect

- Reduction variants print `PASS`, timing, computed sum, expected sum, and absolute error.
- For N=33554432, per-element float atomics deliberately print `PASS_EXPECTED_FP32_SATURATION`: the result saturates at 16777216 because float cannot represent the next integer. The double-finish variant checks the exact sum. This is an expected precision demonstration.
- Maximum reduction prints `PASS max_reduction` for several input patterns, including negative values. A maximum of zero would be incorrect for all-negative data.
- Matrix sums print three `PASS` lines: `row_sums_naive`, `row_sums_block`, and `column_sums`. Compare against HW4 at matching sizes.
- Arguments: reductions `[elements=8388608] [repeats=3]`; maximum `[elements=8388608] [repeats=10]`; matrix `[side=2048] [repeats=10]`. The atomics experiment can be noticeably slower than other homework.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 5 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW5.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw5` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
