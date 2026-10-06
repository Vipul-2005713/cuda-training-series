> Historical report from the earlier Windows environment. These measurements are not Ubuntu/WSL results. Use [readme.md](readme.md) for current Ubuntu commands and expected behavior; fresh logs go to `results/ubuntu/`.

﻿# Homework 10: streams, OpenMP, and multiple GPUs

`streams.cu` completes all three tasks. It uses the original 22-sample Gaussian calculation, a fully checked serial GPU baseline, pinned host memory, asynchronous chunk processing, optional OpenMP, and device-aware stream ownership. Historical `streams_solution.cu` is retained as a reference; build the repaired starter for the validated implementation.

## Build and run

Run from this directory in a CUDA/MSVC developer shell. The repository runner supplies the installed host compiler automatically.

```powershell
nvcc -std=c++17 -arch=sm_86 -lineinfo streams.cu -o streams_serial.exe
nvcc -std=c++17 -arch=sm_86 -lineinfo -DUSE_STREAMS streams.cu -o streams.exe
nvcc -std=c++17 -arch=sm_86 -lineinfo -DUSE_STREAMS -Xcompiler /openmp streams.cu -o streams_openmp.exe
.\streams_serial.exe 1048576 16 4 1
.\streams.exe 1048576 16 4 1
.\streams_openmp.exe 1048576 16 4 1
.\streams_openmp.exe 1048577 17 4 4
```

On Linux, replace `/openmp` with `-fopenmp` and omit `.exe`. Arguments are `N`, number of chunks, streams per GPU, and requested GPUs. Defaults are `1048576 16 4 1`. A requested device count above the available count prints a limitation and uses the available devices. It does not claim a multi-GPU benchmark was performed.

## Task 1: streams review

The serial path copies the entire vector to device 0, runs one kernel, and copies the entire result back. The streamed path creates a private input/output buffer pair for each stream. Each stream queues `H2D -> kernel -> D2H` for its assigned chunks; stream ordering also prevents the next chunk from overwriting those private buffers too early. Independent streams can overlap when hardware resources and the driver permit it.

The input and output arrays use `cudaHostAllocPortable`, so transfers can use pinned storage across every participating CUDA device. Chunk endpoints are `N*chunk/chunks` and `N*(chunk+1)/chunks`; integer division here covers all elements even when the dimensions are not divisible. Empty chunks are skipped.

For every element, the serial GPU answer is compared with an independent host calculation, including a finite-value check. The streamed answer is then compared against that verified baseline. The absolute tolerance is `3e-6`, allowing CPU/GPU floating-point differences without hiding unprocessed chunks. A failure returns a nonzero status.

## Task 2: OpenMP and CUDA

The OpenMP loop distributes stream owners across CPU threads. Each owner alone submits commands for its private stream and buffers. This is safer than letting several threads reuse a common scratch buffer while interleaving copy/kernel submissions. Every worker calls `cudaSetDevice` because the current CUDA device is host-thread-specific. The CPU thread team is warmed outside the measured region.

Additional CPU threads do not create extra GPU arithmetic throughput. For this simple producer, one CPU thread often enqueues work quickly enough; thread scheduling and API contention can cost more than parallel submission saves. The original homework's approximately 2x streams speedup and profiling slowdown are historical observations, not requirements or promises for a modern Windows laptop.

## Task 3: multi-GPU bonus

Stream owner `j` is assigned to device `j % available_selected_devices`. Its stream and buffers are created while that device is current. Portable pinned host arrays provide a common source and destination. Every stream is synchronized on its own device before results are inspected or elapsed time is stopped. No peer access is needed: the workload has no communication between chunks.

The code implements real work distribution across multiple GPUs when they are present. This workstation exposes one RTX 3050 Laptop GPU, so requesting four exercises the selection and fallback path; simultaneous work on four devices and multi-GPU scaling remain unmeasured. A four-GPU host can run the exact same binary with its final argument set to `4`.

## Performance interpretation

Both `serial_end_to_end_ms` and `streamed_end_to_end_ms` are monotonic host wall times that include H2D transfer, launches, computation, D2H transfer, and completion. Allocation, first-use warmup, and CPU verification are excluded. Speedup is serial time divided by streamed time. For `N` floats, the two transfers move `8*N` bytes; the Gaussian kernel performs 22 exponential evaluations per element, so this is not a pure transfer benchmark.

A stream count of one is a useful control. Compare 1, 2, 4, and 8 streams and larger chunks after checking correctness. Small chunks increase launch overhead; too many streams do not increase the number of copy engines. Timing alone cannot prove actual transfer/compute overlap; a Nsight Systems trace is needed to see the overlap directly.

The central run logs contain measured outputs for the installed GPU. Any table added below is drawn from those outputs, not the lecture's example hardware.

## Key takeaways

- Pinned memory, asynchronous APIs, independent storage, and correct ordering are all needed for a useful streaming pipeline.
- A stream belongs to a device; the host thread's current device must match it.
- Synchronize every participating stream/device before timing or verifying the result.
- OpenMP and more GPUs are correctness and decomposition choices first; measure whether they improve the actual workload.
