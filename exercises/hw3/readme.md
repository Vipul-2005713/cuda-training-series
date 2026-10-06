# HW3: Grid-stride loops and launch configuration

`vector_add.cu` lets you change the number of blocks and threads without changing
which elements are processed. Compare a single thread, one full block, and many
blocks using the same input size.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 3 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw3_vector_add 262144 1 1 3
./build/ubuntu/hw3_vector_add 262144 1 1024 5
./build/ubuntu/hw3_vector_add 262144 160 1024 5
./build/ubuntu/hw3_vector_add 1003 3 37 5
```

## What to expect

- Every command prints `PASS grid_stride_vadd` and checks all elements.
- Output includes GPU name, SM count, block/thread configuration, `kernel_ms`, and effective bandwidth.
- Arguments: `[elements=262144] [blocks=160] [threads=1024] [repeats=5]`.
- The single-thread case demonstrates serial work on a GPU and should be much slower. Keep its input small on a WSL display GPU; use the populated grid for large vectors.
- Faster launch configurations expose more parallel work, but more threads do not guarantee a lower time.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 3 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW3.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw3` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
