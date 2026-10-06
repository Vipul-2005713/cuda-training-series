# CUDA training on Ubuntu / WSL2

Runnable CUDA homework for the [ORNL / NERSC CUDA Training Series](https://www.olcf.ornl.gov/cuda-training-series/).
The supported workflow is **Ubuntu Bash**, either inside WSL2 or on native Ubuntu,
using Linux CUDA binaries. Each homework has its own execution guide below.
The exercises are completed implementations with correctness checks; reference
`*_solution.cu` files share those implementations and accept the same arguments.

## Start here

Open your Ubuntu terminal and run:

```bash
cd /mnt/d/CUDA/cuda-training-series
source tools/ubuntu_env.sh
nvcc --version
nvidia-smi
python3 --version

# Check CUDA and run the first homework.
python3 tools/run_exercises.py --build --only device_info cuda_smoke hw1_ --suite default
```

You should see the GPU information, four greetings from each hello program,
`PASS` checks, and zero failures in the runner summary. Commands throughout this
repository assume you are in the repository root unless stated otherwise.

The required tools are the **Linux CUDA Toolkit**, a compatible C++ compiler, and
Python 3. The runner only uses Python's standard library. This workspace already
has CUDA Toolkit 12.0 and the RTX 3050 Laptop GPU (compute capability 8.6), so the
build default is `-arch=sm_86`. For a different GPU, pass `--arch sm_XX` or set
`CUDA_ARCH`. Use `nvcc --version` to identify the installed compiler; the CUDA
version shown by `nvidia-smi` describes driver capability.

If basic Ubuntu tools are missing:

```bash
sudo apt update
sudo apt install build-essential python3
```

If `nvcc` is missing, follow [NVIDIA's WSL CUDA Toolkit instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2).
WSL uses the Windows host's GPU driver; install the toolkit inside Ubuntu without
installing a Linux display driver. If your toolkit is installed under
`/usr/local/cuda`, add `/usr/local/cuda/bin` to your `PATH`. For a host-compiler
compatibility error, choose a compiler supported by your toolkit and pass it with
`--ccbin`, for example `--ccbin g++-12` if that compiler is installed and supported.
The checked local default compiler setup builds successfully as installed.

`source tools/ubuntu_env.sh` selects `/usr/lib/wsl/lib` first in this shell's
library search path when running in WSL. This fixes a conflict present on this
machine: cuBLAS otherwise loads an installed native Linux `libcuda` and reports
no CUDA device. The Python runner applies the same setting to its child processes
automatically. Source the script once per terminal before directly running
executables or profiling tools. Native Ubuntu keeps its existing driver path;
the script does not change system files.

## Run one homework or everything

```bash
# Build and run just HW2, including boundary cases.
python3 tools/run_exercises.py --build --hw 2 --suite all

# Build every exercise and solution, then run all registered checks.
bash tools/run_all.sh

# Build all entry points without using the GPU at runtime.
python3 tools/run_exercises.py --build --include-solutions --suite none

# List programs, or run existing HW6 binaries without rebuilding.
python3 tools/run_exercises.py --list
python3 tools/run_exercises.py --hw 6 --suite default
```

`--hw 1` selects exactly HW1; multiple numbers such as `--hw 1 2` are supported.
For one executable, use a target prefix such as `--only hw5_max_reduction`.
`--only hw1_` selects HW1; the broader prefix `hw1` also matches HW10–13.
`--include-solutions` adds the reference entry points. HW10 builds serial, stream,
and OpenMP variants; HW11 defaults to ordinary processes with no MPI dependency.
HW9's cooperative target uses `-rdc=true`, OpenMP uses `-Xcompiler -fopenmp`, and
HW13's library examples link `-lcublas` automatically.

Suites are `default` (normal workloads), `edge` (small/uneven boundaries),
`experiments` (comparisons and larger workloads), `all`, and `none` (build only).
The full build and run can take several minutes. Each build/run has a 180-second
timeout; use `--timeout 600` if needed. Rebuild after changing sources or CUDA
architecture. Programs and build logs are separate from the source directories.

## Individual homework guides

| Homework | Topic | Execution guide |
| --- | --- | --- |
| 1 | CUDA basics, vector addition, matrix multiplication | [HW1](exercises/hw1/readme.md) |
| 2 | Shared memory, stencil, tiled matrix multiplication | [HW2](exercises/hw2/readme.md) |
| 3 | Grid-stride loops and launch configuration | [HW3](exercises/hw3/readme.md) |
| 4 | Coalescing, row and column sums | [HW4](exercises/hw4/readme.md) |
| 5 | Atomics, reductions, warp shuffle | [HW5](exercises/hw5/readme.md) |
| 6 | Unified memory and explicit copies | [HW6](exercises/hw6/readme.md) |
| 7 | Streams, overlap, multiple GPUs | [HW7](exercises/hw7/readme.md) |
| 8 | Transpose and profiling | [HW8](exercises/hw8/readme.md) |
| 9 | Cooperative groups and compaction | [HW9](exercises/hw9/readme.md) |
| 10 | OpenMP and CUDA concurrency | [HW10](exercises/hw10/readme.md) |
| 11 | Multiple processes, optional MPI/MPS | [HW11](exercises/hw11/README.md) |
| 12 | Debugging and Compute Sanitizer | [HW12](exercises/hw12/readme.md) |
| 13 | CUDA graphs and cuBLAS | [HW13](exercises/hw13/README.md) |

## Results and expected limitations

Executables are in `build/ubuntu/`. Detailed compiler logs are in
`results/ubuntu/build/`, run output is in `results/ubuntu/runs/`, and
`summary.json` / `summary.csv` describe the latest invocation. A nonzero runner
exit code means a compilation failure, runtime failure, or timeout. Failed builds
never run an older executable. To preserve separate runs, supply an output
folder, e.g. `--output results/ubuntu/hw2`.

`PASS` means the available checks succeeded. `PASS_WITH_SKIPS` means supported
checks succeeded and the output explains a capability limitation. On this
one-GPU WSL laptop, expect managed-memory prefetch in HW6 and four-GPU comparisons
in HW7/HW10 to be unavailable. Basic managed memory and cooperative launches are
available. [NVIDIA documents WSL's unified-memory limits](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications).
HW11 runs ordinary processes without MPS; an MPS-on comparison needs separate
support and setup. Profiling/debugging tools are optional for the ordinary suite.

Timings, bandwidth, and speedups depend on GPU, workload, driver, and load. Passing
requires correct results, not reproducing lecture timings or a minimum speedup.
The 32-million-element HW5 atomics case intentionally demonstrates float precision
saturation and labels it `PASS_EXPECTED_FP32_SATURATION`.

The original assignments are retained as `LESSON.md` inside each homework.
Existing `REPORT.md` files and `results/windows/` contain historical measurements
from the prior environment. Current Ubuntu execution instructions are in the
linked homework READMEs. Local binaries, traces, and generated Ubuntu logs are
not source files; rebuild them for your current toolkit.

## Validation on the current WSL machine

Checked on 2026-10-06 with Ubuntu 24.04.3, CUDA Toolkit 12.0, and the RTX 3050
Laptop GPU (4 GiB, `sm_86`):

| Check | Result |
| --- | --- |
| All 57 build targets, covering all 53 CUDA source files | Passed |
| Complete exercise and solution suite | 167 runs: 155 PASS, 12 PASS_WITH_SKIPS, 0 failures |
| HW11, four processes, normal and uneven sizes | All ranks passed |
| HW8 task 3, Nsight Compute profile | Captured successfully |
| Runner regression checks | 7 passed |
| HW12 Compute Sanitizer 2022.4.1 | Tool exits before instrumentation; unverified |

The 12 partial skips concern managed-memory prefetch and four-GPU comparisons.
Detailed full-suite results are in `results/ubuntu/validation/`; extra launcher,
profiler, and sanitizer logs are in `results/ubuntu/extra-checks/`. The [HW12
guide](exercises/hw12/readme.md) records the sanitizer limitation and includes a
wrapper for Ubuntu's split injection-library layout. MPI, MPS-on, and interactive
`cuda-gdb` are optional workflows and were not validated here.

To rerun the CPU-only runner checks:

```bash
python3 -m unittest discover -s tools -p 'test_*.py'
```

## Compile a file directly

The runner is the simplest way to apply each exercise's required flags. To build
one program manually from the repository root:

```bash
mkdir -p build/ubuntu
nvcc -std=c++17 -O3 -lineinfo -arch=sm_86 exercises/hw1/vector_add.cu -o build/ubuntu/hw1_vector_add
./build/ubuntu/hw1_vector_add 1003 5
```

Reference solutions compile independently with the same flags. Do not pass an
exercise and its solution to the same `nvcc` command: each is a program with its
own `main`. Bash scripts use LF line endings, enforced by `.gitattributes`.
