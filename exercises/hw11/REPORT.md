> Historical report from the earlier Windows environment. These measurements are not Ubuntu/WSL results. Use [README.md](README.md) for current Ubuntu commands and expected behavior; fresh logs go to `results/ubuntu/`.

﻿# Homework 11: multi-process GPU execution and MPS

`test.cu` repairs the lecture benchmark and supports both MPI and ordinary processes. `run_no_mpi.ps1` is the Windows launcher; `run_no_mpi.sh` serves Linux. The native Windows machine can measure one versus four processes without MPS. MPS-on experiments and the minimum beneficial problem size require a supported MPS environment and cannot be inferred from the Windows measurements.

## Build and run

In a CUDA/MSVC developer shell:

```powershell
nvcc -std=c++17 -arch=sm_86 -lineinfo -DNO_MPI test.cu -o test.exe
.\test.exe 1048576 1 100 0
.\run_no_mpi.ps1 -Executable .\test.exe -N 1048576 -Ranks 4 -Repetitions 100
```

The arguments to the executable are `total_N`, `rank_count`, `repetitions`, `rank_index`, and an optional common future Unix start time in milliseconds. The rank count/index are provided by MPI when built without `NO_MPI`. With MPI, retain the positional placeholder for rank count, for example `mpirun -np 4 ./test 1048576 4 100`; the real MPI rank count wins. Build that version with an MPI-compatible host compiler and link flags, as described in the original README.

The non-MPI launchers start every rank as a separate operating-system process, pass distinct rank indices, and check every exit status. A shared start time five seconds into the future keeps ordinary CUDA initialization from serializing short timed loops. A missed start prints a warning; rerun with a larger `-LaunchDelaySeconds` if this occurs. This is a practical launch rendezvous, not an MPI barrier.

## Correctness fixes

The original benchmark doubled uninitialized device memory and did not inspect results or CUDA return values. It also defaulted to roughly 8 GiB of input storage, too large for this GPU. The repaired default is 1,048,576 total doubles, approximately 8 MiB shared across the requested ranks.

Each rank receives the interval `[N*rank/ranks, N*(rank+1)/ranks)`, so no remainder is lost. Every local element starts at exactly `2^(-repetitions)`. After the requested number of doublings it must equal exactly `1.0`. Repetitions are limited to 900 to avoid representational overflow/underflow in this construction. All elements are inspected, all CUDA calls are checked, and memory is released. A warm-up kernel is followed by a reset, keeping warm-up arithmetic out of the answer.

## Exercise 1: reproducing the lecture's comparison

Use the same **total** `N` for one and four ranks. Each of four ranks therefore owns approximately `N/4` elements; this is not the same as running four copies of the one-rank allocation. The measured interval preserves the original `kernel -> cudaDeviceSynchronize` repetition. It includes launch/synchronization overhead, scheduling delays, and contention. The printed `wall_ms_per_kernel` must not be described as isolated kernel execution time.

Without MPS, multiple processes contend for one GPU and context scheduling can increase latency. The exact outcome depends on workload size, driver mode, GPU utilization, and timing variation. The lecture's numerical ratios are not universal. Report all rank times, their spread, and the slowest rank rather than selecting only the fastest rank. The printed local effective bandwidth counts one 8-byte read and one 8-byte write per element; it is an application-level rate including synchronization, not a hardware memory-bandwidth measurement.

NVIDIA documents MPS support on Linux and QNX; native Windows is not a supported MPS environment. See [NVIDIA: when to use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html). There is therefore no measured MPS-on result on this machine, and no claim that enabling MPS reproduces the lecture.

## Exercise 2: finding the useful problem size

Sweep total sizes such as `65536`, `1048576`, and `8388608` at one and four ranks, holding the repetition count and GPU constant. On a supported host, repeat the exact sweep with and without MPS, then run the no-MPS case again to check reversibility. Collect at least five repetitions per condition and compare medians and spread. A credible crossover is the smallest size at which the improvement is repeatable and larger than the run-to-run spread, not simply the first single faster sample.

An isolated Linux test session using the traditional MPS control interface can follow this sequence:

```bash
nvcc -std=c++17 -arch=sm_86 -DNO_MPI test.cu -o test
./run_no_mpi.sh 1048576 1 100 ./test
./run_no_mpi.sh 1048576 4 100 ./test
nvidia-cuda-mps-control -d
./run_no_mpi.sh 1048576 4 100 ./test
# Stop only the MPS daemon owned by this dedicated experiment.
echo quit | nvidia-cuda-mps-control
./run_no_mpi.sh 1048576 4 100 ./test
```

Use the control interface appropriate for the installed driver and an isolated user-owned test environment. See [NVIDIA MPS common tasks](https://docs.nvidia.com/deploy/mps/common-tasks.html). Nsight Systems can record CUDA/NVTX activity on the supported host as in the original README. A Windows timing run alone neither establishes an MPS crossover nor proves the context-scheduling mechanism in a timeline.

## Key takeaways

- Initialize benchmark inputs and verify outputs even when the main aim is timing.
- Fix the total problem size and align starts before comparing process counts.
- Host synchronization time is not pure kernel time.
- MPS is useful when concurrent processes leave capacity unused; saturation and overhead can limit its benefit.
- The MPS-on/off crossover is explicitly unmeasured here because native Windows cannot supply that experimental condition.
