> Historical report from the earlier Windows environment. These measurements are not Ubuntu/WSL results. Use [readme.md](readme.md) for current Ubuntu commands and expected behavior; fresh logs go to `results/ubuntu/`.

﻿# Homework 3 — grid-stride loops and launch-size experiments

Completed source: [vector_add.cu](vector_add.cu). Assignment: [readme.md](readme.md).

## Build and reproduce all three launch configurations

```powershell
nvcc -O3 -std=c++17 -arch=sm_86 vector_add.cu -o vector_add.exe
.\vector_add.exe 262144 1 1 3
.\vector_add.exe 262144 1 1024 5
.\vector_add.exe 262144 160 1024 5
.\vector_add.exe 1003 3 37 5
```

Arguments: `[elements=262144] [blocks=160] [threads=1024] [repeats=5]`. All three comparison cases must use the same N. The original N is 33,554,432; it is available as the first argument, but the one-thread case can run long enough to trigger the Windows display-driver watchdog. The smaller default permits the complete teaching experiment safely on an interactive display GPU. Large-N tests with well-populated grids are useful for bandwidth comparisons.

## 1. Completed vector addition

The allocation, copies, and element addition are the same as homework 1. The difference is the loop:

```text
start  = blockIdx.x * blockDim.x + threadIdx.x
stride = gridDim.x * blockDim.x
for i = start; i < N; i += stride:
    C[i] = A[i] + B[i]
```

Every index has a unique remainder modulo the grid stride, so it belongs to exactly one thread. The same computation works with one thread or hundreds of thousands of threads. Full host verification uses varying signed binary-exact inputs; the irregular 1,003-element, 3-block, 37-thread case exercises arbitrary launch geometry and a tail.

## 2a. One block, one thread

One lane serially performs all N additions and memory operations. It cannot provide enough concurrent requests or independent warps to hide memory latency. Each access also uses only a small part of a memory sector. Kernel duration should be much larger and effective throughput much smaller than the parallel cases.

The process wall time includes CUDA context initialization, allocation, transfers, and CPU verification. The program's event interval excludes them and answers the homework's question about time actually associated with the kernel workload.

## 2b. One block, 1,024 threads

Adjacent lanes access adjacent floats, improving coalescing. Multiple warps can hide some latency. However, a block cannot be split across SMs: only one SM runs this grid. Other SMs remain idle. This usually improves substantially over one thread while leaving most of the GPU unused.

## 2c. 160 blocks, 1,024 threads

Many independent blocks can occupy multiple SMs, increasing memory-level parallelism and latency hiding. The assignment's 160-block example was motivated by 80 V100 SMs and 2,048 threads per SM. Those numbers must not be assumed for another GPU.

The executable prints this device's SM count and maximum resident threads per SM. Occupancy also depends on block size, registers, shared memory, and hardware limits; maximum resident threads alone does not guarantee maximum performance. In particular, an SM whose thread capacity is below 2,048 cannot host two 1,024-thread blocks just because the V100 could. A follow-up sweep of 128, 256, 512, and 1,024 threads can find a better configuration.

At N=262,144 the arrays total 3 MiB. Repeated runs may benefit from cache, so measured algorithmic throughput cannot automatically be interpreted as external DRAM bandwidth. Use a larger N for that question.

## Profiling and interpretation

The executable's metric is effective bandwidth:

```text
GB/s = 3 * N * sizeof(float) / (kernel_ms * 1e6)
```

This is logical traffic, not a measured hardware counter. CUDA events measure an average after one warm-up. Nsight Compute can independently measure duration and hardware memory throughput:

```powershell
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --launch-count 1 .\vector_add.exe 262144 1 1 1
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --launch-count 1 .\vector_add.exe 262144 1 1024 1
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --launch-count 1 .\vector_add.exe 262144 160 1024 1
```

The first kernel is the program's warm-up. Profiled timings can differ from unprofiled events because the profiler may replay work and control caches/clocks. Hardware-counter availability is an environment requirement; an event measurement is not a substitute for a missing counter.

## Results and takeaways

Execution logs provide the observed durations and correctness status; the local three-way timing comparison is added after the measured run.

Grid-stride loops separate correctness from launch geometry. More threads help only if they expose useful concurrency and memory access patterns. Coalescing and using multiple SMs explain why the three requested configurations perform differently.

