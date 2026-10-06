# HW10: CPU threads, CUDA streams, and device selection

The runner builds `streams.cu` three ways: serial; `-DUSE_STREAMS`; and
`-DUSE_STREAMS -Xcompiler -fopenmp`. This compares serial submission, one CPU
thread feeding streams, and OpenMP threads feeding streams. No MPI is needed.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 10 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw10_serial 1048576 16 4 1
./build/ubuntu/hw10_streams 1048576 16 4 1
./build/ubuntu/hw10_openmp 1048576 16 4 1
./build/ubuntu/hw10_openmp 1003 7 3 1
./build/ubuntu/hw10_openmp 1048577 17 4 4
```

## What to expect

- Each executable prints `PASS: all ... elements checked` after comparing with a CPU reference.
- Stream variants also print streamed time and speedup. The OpenMP build prints `OpenMP=enabled`; the single-thread stream build prints `OpenMP=disabled`.
- Arguments: `[N=1048576] [chunks=16] [streams_per_GPU=4] [requested_GPUs=1]`.
- Asking for four GPUs on this laptop prints `MULTI_GPU_LIMITATION` and `used_gpus=1`, then checks the available-device path. This does not measure four-GPU performance.
- Compare equal workloads. More CPU threads can add overhead and need not improve a small submission workload.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 10 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW10.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw10` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
