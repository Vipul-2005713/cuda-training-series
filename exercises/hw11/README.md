# HW11: Multiple processes and the MPS experiment

`test.cu` splits a total vector across processes and repeatedly doubles each
process's slice. The standard runner builds with `-DNO_MPI`, so the WSL homework
runs without an MPI installation. MPS is a separate optional experiment.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 11 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw11_process 1048576 1 100 0
bash exercises/hw11/run_no_mpi.sh 1048576 4 100
bash exercises/hw11/run_no_mpi.sh 1003 4 10
```

## What to expect

- A one-process run prints one line ending in `PASS`; the launcher prints one `PASS` line for each rank 0–3, in any order.
- Output includes `N_total`, `N_local`, `elapsed_ms`, and `wall_ms_per_kernel`. With N=1048576 and four ranks, each rank checks 262144 elements.
- The initial value is `2^-repetitions`; after doubling, every result must equal 1. This avoids overflow during verification.
- Executable arguments: `[total_N=1048576] [rank_count=1] [repetitions=100] [rank_index=0] [optional_start_epoch_ms]`.
- Launcher arguments: `[total_N=1048576] [ranks=4] [repetitions=100] [executable=build/ubuntu/hw11_process] [launch_delay_seconds=5]`. The default executable path is resolved relative to the script.
- The launcher waits for every process and returns nonzero if any fails. If a rank warns that it missed the common start, increase the delay, e.g. `bash exercises/hw11/run_no_mpi.sh 1048576 4 100 ./build/ubuntu/hw11_process 15`.

## Optional MPI build on Ubuntu

If you choose to use MPI, install `libopenmpi-dev` and `openmpi-bin` with Ubuntu's
package manager. Then, from the repository root:

```bash
mkdir -p build/ubuntu
nvcc -O3 -std=c++17 -arch=sm_86 -ccbin mpicxx exercises/hw11/test.cu -o build/ubuntu/hw11_mpi
mpirun --oversubscribe -np 4 ./build/ubuntu/hw11_mpi 1048576 4 100
```

MPI supplies rank count and index; retain the second positional placeholder so
100 remains the repetition count. This optional build is not part of the default
suite and requires an MPI wrapper using a CUDA-compatible host compiler.

## Optional MPS comparison

One versus four ordinary processes works without MPS. Do not treat that as an
MPS-on result. MPS requires a supported driver, GPU, and execution environment;
Ubuntu inside WSL alone does not establish support. Follow [NVIDIA's MPS deployment
guide](https://docs.nvidia.com/deploy/mps/index.html) on a supported setup, verify
that clients actually connect, then compare the same total N and repetitions
with and without the MPS server. The runner does not start or reconfigure MPS.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 11 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW11.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw11` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
