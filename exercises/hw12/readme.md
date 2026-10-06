# HW12: CUDA debugging and numerical validation

These are the corrected debugging exercises. `task1.cu` performs a tiled matrix
product with boundary handling and synchronization. `task2.cu` estimates ln(2)
using a shared-memory reduction and double-precision atomics. It checks against
a CPU reference and a series truncation bound. Running these files should succeed; the original bugs are
described in the archived lesson.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 12 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw12_matrix 32
./build/ubuntu/hw12_matrix 65
./build/ubuntu/hw12_transform 1
./build/ubuntu/hw12_transform 1003
./build/ubuntu/hw12_transform
```

## What to expect

- Matrix runs print `PASS: all ... outputs checked`, `max_abs_error`, and kernel time. Side 65 tests incomplete shared-memory tiles.
- The series reduction prints `estimate`, `cpu_reference`, `log2`, `cpu_abs_error`, `truncation_bound`, and `PASS`.
- Arguments: matrix `[side=128]`; series `[terms=1048576]`. A small number of terms can be a poor approximation to ln(2) while still correctly computing that finite series.
- More terms should reduce truncation error; floating-point differences remain subject to the printed tolerance.

## Check with Compute Sanitizer

From the repository root after building (the runner includes `-lineinfo`):

```bash
bash tools/compute_sanitizer.sh --tool memcheck --error-exitcode 1 ./build/ubuntu/hw12_matrix 65
bash tools/compute_sanitizer.sh --tool racecheck --error-exitcode 1 ./build/ubuntu/hw12_matrix 65
bash tools/compute_sanitizer.sh --tool synccheck --error-exitcode 1 ./build/ubuntu/hw12_matrix 65
bash tools/compute_sanitizer.sh --tool memcheck --error-exitcode 1 ./build/ubuntu/hw12_transform 1003
```

The wrapper sets the WSL driver path and locates the injection libraries in
Ubuntu's toolkit package, fixing `Unable to find injection library
libsanitizer-collection.so` on this installation. It passes your arguments to
Compute Sanitizer; see [NVIDIA's command-line options](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#command-line-options).

On this machine, Compute Sanitizer 2022.4.1 still exits with `Target application
terminated before first instrumented API call` after resolving the library paths.
The normal HW12 runs pass, but sanitizer validation remains **unverified** on
this installed tool/driver combination. This startup error is not a clean
sanitizer result.

With a working sanitizer installation, expect the program to pass and the tool to report zero errors/hazards. Sanitizer
availability depends on your installed tooling and WSL driver. For an interactive
`cuda-gdb` build, use `nvcc -std=c++17 -arch=sm_86 -G -g exercises/hw12/task1.cu -o build/ubuntu/hw12_matrix_debug`;
run `cuda-gdb --args ./build/ubuntu/hw12_matrix_debug 65`. Debug builds are not
suitable for performance comparisons.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 12 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW12.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw12` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
