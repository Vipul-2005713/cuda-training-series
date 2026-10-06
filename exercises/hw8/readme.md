# HW8: Matrix transpose and GPU profiling

The three task directories progress from a global-memory transpose to a shared
32×32 tile and then a padded shared tile that avoids bank conflicts. Each program
checks the transpose and measures a device-to-device copy for comparison.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 8 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw8_task1 --size 2048 --iterations 20
./build/ubuntu/hw8_task2 --size 2048 --iterations 20
./build/ubuntu/hw8_task3 --size 2048 --iterations 20
./build/ubuntu/hw8_task1 --size 33 --iterations 3
./build/ubuntu/hw8_task2 --size 33 --iterations 3
./build/ubuntu/hw8_task3 --size 33 --iterations 3
```

## What to expect

- Every task prints `PASS transpose variant=...`, `kernel_ms`, effective bandwidth, and device-copy measurements.
- Task 1 is the global-memory version; task 2 uses shared tiles; task 3 adds padding. Compare times at the same side length.
- Options: `--size` (default 2048) and `--iterations` (default 20). Size 33 tests a partial tile; 1031 is another useful boundary case.
- For optional profiling from the repository root, run `bash exercises/hw8/task3/profile --size 2048 --iterations 1`. The script uses Linux `ncu`; equivalent scripts exist for tasks 1 and 2.
- Each `build_nvcc` script now invokes the Ubuntu runner and writes to `build/ubuntu/`. For example: `bash exercises/hw8/task1/build_nvcc`.
- Profiling requires working Nsight Compute and driver counter access. Failure to collect counters is separate from numerical correctness; the ordinary runs do not require a profiler.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 8 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW8.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw8` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
