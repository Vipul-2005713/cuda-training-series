> Historical report from the earlier Windows environment. These measurements are not Ubuntu/WSL results. Use [readme.md](readme.md) for current Ubuntu commands and expected behavior; fresh logs go to `results/ubuntu/`.

﻿# Homework 4 — row/column sums and memory coalescing

Completed source: [matrix_sums.cu](matrix_sums.cu), with kernels in [matrix_sums_kernels.cuh](matrix_sums_kernels.cuh). Assignment: [readme.md](readme.md).

## Build and run

```powershell
nvcc -O3 -std=c++17 -arch=sm_86 matrix_sums.cu -o matrix_sums.exe
.\matrix_sums.exe 2048 10
.\matrix_sums.exe 257 5
.\matrix_sums.exe 1 5
```

Arguments are `[matrix_side=2048] [repeats=10]`. Both kernels run in one invocation. The original 16,384-side workload is available as `matrix_sums.exe 16384`; its matrix alone occupies 1 GiB, plus a matching host allocation.

## 1. Correct row and column sums

One thread owns one output. With row-major data:

| Output | Loop address |
|---|---|
| Row r | `A[r*N + col]`, with col changing |
| Column c | `A[row*N + c]`, with row changing |

The grid is rounded upward, and each thread checks its output index. The host copies back and validates all N row sums and all N column sums against independent CPU accumulation. Nonuniform signed integer values distinguish rows from columns while remaining exactly representable. This catches mistakes hidden by the original all-ones matrix.

## 2. Duration and memory-efficiency questions

Both kernels read the same N*N values and perform approximately the same additions. They need not have equal durations because **coalescing is determined across lanes of a warp at one instruction**, not by whether one thread walks contiguous elements over time.

For row sums, adjacent lanes own different rows. At a fixed loop iteration, their addresses are separated by N floats. A warp can request 32 different 32-byte sectors for its 32 scalar float loads.

For column sums, adjacent lanes own adjacent columns. At a fixed row, the warp's 32 float loads span 128 contiguous bytes: four 32-byte sectors when aligned. The ideal aligned sectors/request ratio is therefore about 32 for naive rows versus 4 for columns, an 8x difference in requested sector volume. Partial warps, alignment, and compiler instructions affect exact measurements.

Cache reuse can reduce later traffic to lower memory levels. Therefore an L1 sector/request ratio must not be equated directly to DRAM bytes or to an 8x runtime ratio. Both kernels also launch only `ceil(N/256)` blocks, limiting available parallelism at modest dimensions.

## Nsight Compute experiment

Duration appears in the profiler's GPU speed-of-light/kernel-duration output. The requested efficiency experiment uses:

```powershell
ncu --kernel-name-base function --kernel-name row_sums --launch-count 1 --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum .\matrix_sums.exe 2048 1
ncu --kernel-name-base function --kernel-name column_sums --launch-count 1 --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum .\matrix_sums.exe 2048 1
```

Use the corresponding pair from each kernel:

```text
sectors per request = l1tex sectors / l1tex requests
```

The numerator is sectors; the denominator is requests. A lower value is more efficient for these comparable 32-lane scalar-float requests. The source README's prose reverses the order once, so use the metric names and this equation.

The event-timing output also reports logical effective bandwidth, counting matrix reads and result writes: `4*(N*N + N)/(kernel_ms*1e6)` GB/s. It is not a measured sectors/request ratio.

## Results and answer to “can we improve this?”

Observed timings and profiling availability are recorded in execution logs and the measured-results section added after the run. Counter values will be labeled as unavailable if profiling permissions prevent collection; theoretical ratios are not substituted as measurements.

Yes: assign the lanes in a warp/block to neighboring columns of the **same row**, sum their partial results in parallel, then write one row result. Homework 5 implements one block per row. That improves coalescing and launches many more independent blocks.

## Key takeaways

- Count memory transactions across a warp, not along one thread's loop.
- Equal arithmetic counts and input sizes do not imply equal runtimes.
- Distinguish sectors/requests, algorithmic effective bandwidth, and actual DRAM throughput.
- A reduction can fix the row kernel's memory layout and its shortage of concurrent work together.

