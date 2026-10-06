> Historical report from the earlier Windows environment. These measurements are not Ubuntu/WSL results. Use [readme.md](readme.md) for current Ubuntu commands and expected behavior; fresh logs go to `results/ubuntu/`.

﻿# Homework 1 — CUDA execution, vector addition, and matrix multiplication

The completed programs are [hello.cu](hello.cu), [vector_add.cu](vector_add.cu), and [matrix_mul.cu](matrix_mul.cu). The original assignment remains in [readme.md](readme.md); the supplied `*_solution.cu` files are preserved as historical reference.

## Build and run

Run in a shell with CUDA and the MSVC x64 compiler environment configured. On the tested Ampere target, compile with `nvcc -O3 -std=c++17 -arch=sm_86`; use an architecture supported by your own GPU elsewhere. From this directory:

```powershell
nvcc -O3 -std=c++17 -arch=sm_86 hello.cu -o hello.exe
nvcc -O3 -std=c++17 -arch=sm_86 vector_add.cu -o vector_add.exe
nvcc -O3 -std=c++17 -arch=sm_86 matrix_mul.cu -o matrix_mul.exe
.\hello.exe
.\vector_add.exe 1048576 20
.\vector_add.exe 1003 5
.\matrix_mul.exe 512 10
.\matrix_mul.exe 65 5
```

Arguments are `[elements] [repeats]` for vectors and `[matrix_side] [repeats]` for multiplication. Defaults are the first benchmark commands above. The original vector size is available as `vector_add.exe 4096`; the original starter's matrix size is available as `matrix_mul.exe 4096`.

## 1. Hello world

Launch `hello<<<2,2>>>()`. The kernel prints `blockIdx.x` and `threadIdx.x`, producing exactly the four requested block/thread pairs. Their order is unspecified because CUDA schedules blocks and warps independently.

A launch returns before the device finishes. `cudaGetLastError()` detects launch errors, and `cudaDeviceSynchronize()` waits for completion, reports asynchronous faults, and makes device output visible before process exit. Timing device `printf` is not a useful computational benchmark.

## 2. Vector addition

Each thread computes the flattened index `blockIdx.x * blockDim.x + threadIdx.x`, then writes `C[i] = A[i] + B[i]` when `i < N`. Rounding the grid upward ensures the last partial block is covered; the guard prevents out-of-bounds accesses.

The host allocates three device arrays, copies both inputs to the GPU, launches the kernel, and copies the result back. Every result is compared with a host addition, using nonuniform positive and negative binary-exact inputs. This is stronger than printing only element zero. The 1,003-element case tests the partial block.

Each result logically requires two 4-byte reads and one 4-byte write. Reported effective bandwidth is `12*N / (kernel_ms*1e6)` GB/s. It measures algorithmic traffic divided by elapsed time, not a hardware DRAM counter. A 4,096-element problem is too small to characterize peak bandwidth: launch and timing overhead can dominate.

## 3. Naive matrix multiplication

Rows and columns map to the y and x launch dimensions. One thread computes:

```text
C[row,col] = sum over k of A[row,k] * B[k,col]
```

With row-major storage, the addresses are `A[row*N+k]`, `B[k*N+col]`, and `C[row*N+col]`. Both row and column bounds are checked. The 16x16 block contains 256 threads.

The inputs vary with row, column, and inner-product index. An independent expanded dot-product formula checks every output without an expensive cubic CPU calculation. For sides up to 129, a second direct CPU triple loop verifies every output too. The integer-valued data and bounded dimensions keep sums exactly representable in float, so exact comparison is intentional.

Useful work is approximately `2*N^3` floating-point operations; the program reports `2*N^3/(kernel_ms*1e6)` GFLOP/s. GPU allocation and copies are excluded. Compare it with homework 2 using the **same dimension, repeats, compiler flags, inputs, and 16x16 block**.

## Measurement and results

CUDA events measure an average over repeated kernels after one untimed warm-up. Synchronizing the stop event makes asynchronous execution part of the measured interval. For very short kernels, submission gaps and event resolution still affect the average. Full result checking and transfers occur after timing. Device memory and events are released by scope-bound helpers.

Measured execution evidence is recorded by the repository's run logs; this report's measured-results table is populated after the benchmark run. No performance number is inferred from the assignment's old V100 examples.

## Key takeaways

- Thread indexing and tail guards determine correctness before any optimization.
- A successful launch does not prove successful execution: check both launch and completion.
- Kernel duration and total application duration answer different questions.
- A complete correctness check with varying inputs catches mistakes that uniform data can conceal.
- Matrix multiplication exposes reuse opportunities that shared-memory tiling will address next.

