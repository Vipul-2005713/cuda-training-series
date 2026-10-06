# HW13: CUDA graphs and cuBLAS graph capture

The stream-capture examples compare direct stream launches with a captured graph
for A → {B,C} → D. The timer example runs the direct-stream baseline. The cuBLAS
examples capture a library call inside a parent graph. The runner links cuBLAS
with `-lcublas`. Files named `with_fixme` already contain completed implementations.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 13 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw13_axpy_stream_capture_with_fixme 65536 100
./build/ubuntu/hw13_axpy_stream_capture_from_scratch 65536 100
./build/ubuntu/hw13_axpy_stream_capture_timer 65536 100
./build/ubuntu/hw13_axpy_cublas_with_fixme 65536 100
./build/ubuntu/hw13_axpy_cublas_from_scratch 65536 100
./build/ubuntu/hw13_axpy_stream_capture_from_scratch 1003 7
./build/ubuntu/hw13_axpy_cublas_from_scratch 1003 7
```

## What to expect

- Stream examples print `direct_streams PASS` and, except for the timer baseline, `captured_graph PASS`. They verify `y = y0 + 8*repetitions*x` for every element.
- cuBLAS examples print `direct_cublas PASS` and `cublas_child_graph PASS`. They verify `y = y0 + repetitions*(5*x + 2)`.
- Graph output includes node counts, capture/creation and instantiation costs, replay timings, and speedup. The cuBLAS parent has three nodes; internal library graph details can depend on toolkit version.
- All programs accept `[N=65536] [repetitions=100]`. N=1003 tests a partial final block.
- Compare steady-state replay and setup-plus-replay separately. Graph creation has a cost, and a fixed speedup is not guaranteed.
- Reference entry points under `Solutions/` share the corrected implementation. Build and run them with `python3 tools/run_exercises.py --build --hw 13 --include-solutions --suite default`.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 13 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW13.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw13` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
