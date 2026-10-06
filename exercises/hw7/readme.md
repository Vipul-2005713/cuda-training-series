# HW7: Streams, transfer overlap, and multiple GPUs

`overlap.cu` compares a sequential H2D → Gaussian kernel → D2H pipeline with
chunked work across streams. Both schedules run in one invocation; no build-time
`USE_STREAMS` switch is needed. `multi.cu` runs four independent jobs on one GPU
and, when available, on four GPUs.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 7 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw7_overlap --n 8388608 --chunks 32 --streams 8 --repeats 5
./build/ubuntu/hw7_overlap --n 1003 --chunks 7 --streams 3 --repeats 2
./build/ubuntu/hw7_overlap --streams 1
./build/ubuntu/hw7_multi --n 1048576 --repeats 5
```

## What to expect

- Overlap prints `PASS overlap`, `sequential_ms`, `streams_ms`, and their speedup ratio. Timings include transfers and synchronization.
- The 1003-element case checks uneven chunks. Changing stream count preserves the numerical result.
- Multi-GPU code prints `PASS multi devices=1 jobs=4`. With one GPU, it then prints `SKIP four-GPU experiment: requires 4 CUDA GPUs; found 1`.
- Overlap defaults: N=8388608, chunks=32, streams=8, repeats=5. Multi defaults: N per job=1048576, repeats=5.
- Actual overlap depends on copy engines, workload size, and driver scheduling. A speedup below 1 still passes if the results match.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 7 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW7.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw7` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
