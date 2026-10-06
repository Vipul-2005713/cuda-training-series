> Historical report from the earlier Windows environment. These measurements are not Ubuntu/WSL results. Use [readme.md](readme.md) for current Ubuntu commands and expected behavior; fresh logs go to `results/ubuntu/`.

﻿# Homework 2 — shared-memory stencil and tiled matrix multiplication

Completed sources: [stencil_1d.cu](stencil_1d.cu) and [matrix_mul_shared.cu](matrix_mul_shared.cu). Assignment: [readme.md](readme.md). Original reference solutions are preserved.

## Build and run

From a configured CUDA/MSVC shell in this directory:

```powershell
nvcc -O3 -std=c++17 -arch=sm_86 stencil_1d.cu -o stencil_1d.exe
nvcc -O3 -std=c++17 -arch=sm_86 matrix_mul_shared.cu -o matrix_mul_shared.exe
.\stencil_1d.exe 1048576 20
.\stencil_1d.exe 4099 5
.\stencil_1d.exe 1 5
.\matrix_mul_shared.exe 512 10
.\matrix_mul_shared.exe 65 5
```

Arguments are `[interior_elements] [repeats]` for the stencil and `[matrix_side] [repeats]` for matrix multiplication. Original exercise sizes remain available with `stencil_1d.exe 4096` and `matrix_mul_shared.exe 8192`. Large naive matrix comparisons can take substantial time; the default 512-side problem makes the exercise practical on the local GPU.

## 1. Radius-three stencil

A block computes 16 outputs. Its shared cache needs `16 + 2*3 = 22` integers: the 16 central elements and three neighbors on each side. The local center index is `threadIdx.x + RADIUS`; each result sums the seven entries from offsets -3 through +3.

The host allocates `N + 2*RADIUS` elements. Passing `d_in + RADIUS` and `d_out + RADIUS` makes negative indices into the left input halo valid. Threads cooperatively load all 22 cache entries, synchronize, and then calculate output values.

Two changes extend the original skeleton safely:

1. The grid is rounded upward and final stores are guarded, supporting sizes not divisible by 16.
2. Shared-cache loads in the last partial block are bounded. All threads still reach the barrier, even if their output index is outside the interior.

A CPU stencil verifies every interior output with nonuniform signed inputs. Both output halos must retain their initial value of one. The one-element case checks both halos and an almost completely inactive block; 4,099 elements exercises a partial final block.

Shared memory allows neighboring outputs to reuse inputs. Each full block explicitly loads 22 values for 16 outputs, instead of issuing seven global loads per output. This does not imply a 7x speedup: global caches already reuse data, barriers cost time, and 16-thread blocks underfill 32-lane warps. Keeping the homework's 16-thread block emphasizes the original indexing exercise; larger blocks are a possible follow-up experiment.

## 2. Tiled matrix multiplication

A 16x16 block computes a 16x16 output tile. In each inner-dimension step, threads cooperatively load an A tile and a B tile into `as` and `bs`. Each loaded value is reused across 16 multiply-accumulate operations.

The first barrier ensures the tile is fully loaded before reads. The second prevents fast threads from overwriting it while slower threads still consume it. Out-of-range loads become zero, and out-of-range stores are suppressed. Critically, both barriers are outside per-thread output bounds conditions. This avoids the original reference's assumptions that dimensions are exact tile multiples.

The number of tile steps is `ceil(N/16)`, so the final partial tile is included. With a side of 65, five tile steps are required and the last contains only one valid inner-dimension entry.

## Comparison with homework 1

Both implementations use the same host driver, nonuniform inputs, complete verification, 16x16 block, event timing, and GFLOP/s calculation. Run equal-size inputs and calculate:

```text
speedup = naive_kernel_ms / tiled_kernel_ms
```

Tiling reduces repeated global input loads and increases data reuse. It adds shared-memory accesses and barriers, so a smaller runtime is an empirical result to verify, not a universal guarantee. At small sizes, cache effects and overhead can make the difference small or even reverse it. At larger sizes, explicit reuse should matter more, though these teaching kernels are not competitive with tuned library GEMM.

## Measurement and results

The reported time is a CUDA-event average after one warm-up; allocations, transfers, and verification are excluded. The shared and naive matrix drivers check every output using an independent expanded dot product and also use direct CPU dot products for sides up to 129.

Measured timings are supplied by the repository run logs after execution. The assignment's earlier timings on other GPUs are not local measurements.

## Key takeaways

- Halo capacity and pointer offsets must be designed together.
- Block barriers require uniform participation.
- Pad incomplete tiles with zero and round up the tile count.
- Shared memory helps when useful reuse pays for synchronization and extra instructions.
- Compare identical workloads and timing boundaries before attributing a speedup to an optimization.

