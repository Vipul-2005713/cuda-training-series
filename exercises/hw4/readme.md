# HW4: Memory access patterns: row and column sums

`matrix_sums.cu` computes both row sums and column sums. Adjacent threads access
memory differently in the two kernels, illustrating coalescing. Both results are
checked against CPU expectations.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 4 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw4_matrix_sums 2048 10
./build/ubuntu/hw4_matrix_sums 257 5
./build/ubuntu/hw4_matrix_sums 1 5
```

## What to expect

- Each command prints two `PASS` lines, one for `row_sums_naive` and one for `column_sums`.
- Each line contains the side length, repeat count, kernel time, bandwidth, and number of sums checked.
- Arguments: `[matrix_side=2048] [repeats=10]`. Side 257 tests a partial block.
- Compare timings for the two access patterns. In HW5, compare the cooperative row reduction at the same matrix side.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 4 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW4.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw4` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
