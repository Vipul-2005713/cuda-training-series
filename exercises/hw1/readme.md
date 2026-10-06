# HW1: CUDA basics: launch, copy, and validate

`hello.cu` launches one block with four threads. `vector_add.cu` computes
`C = A + B`. `matrix_mul.cu` computes a matrix product with a simple global-memory kernel.
Learn the host/device allocation, copy, launch, synchronization, and validation sequence.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 1 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw1_hello
./build/ubuntu/hw1_vector_add 1048576 20
./build/ubuntu/hw1_vector_add 1003 5
./build/ubuntu/hw1_matrix_mul 512 10
./build/ubuntu/hw1_matrix_mul 65 5
```

## What to expect

- Hello prints four `Hello from block: 0, thread: ...` lines, one each for threads 0–3. Their order is not guaranteed; success is exit code 0.
- Vector addition prints `PASS vector_add`, the first input/output values, `kernel_ms`, bandwidth, and the number of elements checked.
- Matrix multiplication prints `PASS matrix_mul_naive`, timing, GFLOP/s, and `checked=N*N`. The 65-side case checks a partial block at the matrix boundary.
- Vector arguments: `[elements=1048576] [repeats=20]`. Matrix arguments: `[side=512] [repeats=10]`.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 1 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW1.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw1` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
