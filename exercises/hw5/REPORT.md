> Historical report from the earlier Windows environment. These measurements are not Ubuntu/WSL results. Use [readme.md](readme.md) for current Ubuntu commands and expected behavior; fresh logs go to `results/ubuntu/`.

﻿# Homework 5 — reductions, floating-point limits, and faster row sums

Completed programs: [reductions.cu](reductions.cu), [max_reduction.cu](max_reduction.cu), and [matrix_sums.cu](matrix_sums.cu). Assignment: [readme.md](readme.md). Historical reference solutions are unchanged.

## Build and reproduce the assignment

```powershell
nvcc -O3 -std=c++17 -arch=sm_86 reductions.cu -o reductions.exe
nvcc -O3 -std=c++17 -arch=sm_86 max_reduction.cu -o max_reduction.exe
nvcc -O3 -std=c++17 -arch=sm_86 matrix_sums.cu -o matrix_sums.exe
.\reductions.exe 8388608 3
.\reductions.exe 163840 5
.\reductions.exe 33554432 3
.\max_reduction.exe 8388608 10
.\max_reduction.exe 1003 5
.\max_reduction.exe 1 5
.\matrix_sums.exe 2048 10
.\matrix_sums.exe 257 5
```

Each executable takes `[N] [repeats]`; for matrix sums, N means the matrix side. The reduction inputs and block sizes are selected at runtime, so the small and bonus experiments need no source edits.

## 1. Three sum reductions

| Method | Within a thread/block | Global atomic additions |
|---|---|---|
| `atomic_red` | One input per thread | N |
| `reduce_a` | Grid-stride register sum, then shared-memory tree | 640 |
| `reduce_ws` | Grid-stride register sum, warp shuffles, one shared warp-partial array | 640 |

The classical tree halves active threads at each step and uses a block barrier between levels. Warp shuffles exchange register values within a warp. Only eight warp totals pass through shared memory for a 256-thread block; the first warp combines them after one block barrier. All 32 lanes of that first warp participate, with zero supplied for unused totals.

The global result is reset before **every** warm-up and timed launch. Reset time is outside the measured kernel interval. Each method is independently checked against the known mathematical all-ones sum.

### Why the atomic-only version is slow

All N updates contend on a single memory address. Even if input loads coalesce, the atomic destination serializes progress. Combining data locally reduces those global updates from millions to only 640.

### Why shared-tree and shuffle timings can be similar at 8 Mi elements

Both read the same 32 MiB input and perform the same grid-stride scan. If memory traffic dominates, removing some barriers and shared accesses changes only a small part of total time. Hardware, block count, cache, and atomic contention can also matter; the observed times determine whether this explanation fits this run.

For both methods, useful input bandwidth is:

```text
input_GB_s = N * 4 / (kernel_ms * 1e6)
```

This intentionally counts input bytes only, as the homework asks. Atomic transactions and other internal traffic are extra. A meaningful comparison to peak bandwidth needs the actual GPU's memory specification or a measured device-memory copy benchmark. Host-to-device PCIe bandwidth is a different limit.

### The 163,840-element experiment

This is exactly 640*256, so every thread loads one element. There is much less scanning work per block. The fixed reduction tree, barriers, shuffle instructions, and atomic finish become a larger fraction of execution time. Thus the percentage difference between shared-tree and shuffle methods can increase. The actual direction and magnitude remain empirical; launch/event overhead and cache residency can obscure a very small difference.

### Bonus: 32 Mi elements and numerical correctness

IEEE-754 float has 24 significant binary bits. Every integer up to 2^24 = 16,777,216 is representable. At that value, adding 1 produces a midpoint that rounds back to 2^24 under round-to-nearest, ties-to-even.

The naive atomic-only sum of 33,554,432 ones therefore stalls at 16,777,216. This is numerical precision loss, not a missing barrier or lost atomic update. The executable explicitly reports `PASS_EXPECTED_FP32_SATURATION` only when this known failure is reproduced, prints the exact mathematical sum and absolute error, and continues to run the optimized methods.

The block methods accumulate larger partials; for the three required input sizes, their integer partials and final sums are exactly representable. Arbitrary large odd sizes can accumulate round-off, so that separate optional regime uses a documented floating-point error bound. An additional shuffle reduction with a **double atomic finish** runs above 2^24 and must exactly equal N for these all-ones inputs. This is a corrected computation, not a claim that changing summation order guarantees arbitrary floating-point exactness.

## 2. Change sum into a maximum

The two-stage algorithm is retained:

1. 640 blocks reduce grid-stride subsets into 640 partial maxima.
2. One block reduces those partial maxima to the final output.

Replace both the per-thread `+` and tree `+` with `fmaxf`. The identity must change from zero to **negative infinity**. Initializing max reduction to zero, as in the historical reference, gives the wrong result for all-negative arrays.

Every invocation runs mixed-sign and all-negative cases. The known maximum is placed at the final input index, exercising tails and avoiding the reference's fixed index 100. A complete CPU scan obtains the expected maximum. N=1 and N=1,003 additionally exercise empty per-thread subsets and incomplete blocks.

The repeated benchmark writes to a separate output allocation rather than overwriting input element zero. This avoids corrupting the next timed iteration's input. Reported time covers both stages. These tests specify finite inputs; NaN propagation semantics would require an explicit separate policy.

## 3. Improve homework 4 row sums

The executable runs the original one-thread-per-row kernel, the new one-block-per-row kernel, and the original column kernel on the same matrix and checks every result.

For the optimized row kernel, block index selects the row and thread index selects a column within it. A block-stride loop visits columns `tid, tid+256, ...`. Neighboring lanes read neighboring floats. A shared-memory tree combines the 256 partial sums, and thread zero writes the row output; no atomic operation is needed because only one block owns each row.

This addresses both priorities from the notes:

- Efficient memory access: adjacent lanes load adjacent row elements.
- Enough threads: N blocks replace `ceil(N/256)` blocks.

It can outperform the column kernel as well. The column kernel coalesces loads but still gives each thread a serial N-element sum and offers far fewer blocks. The block-row version distributes both reading and addition across many threads. For very short rows, however, idle lanes and reduction/barrier overhead can make it less attractive.

Other valid approaches have different tradeoffs. One kernel launch per row pays excessive launch overhead; a warp per row reduces synchronization but exposes fewer workers for long rows; a block per row is a straightforward general teaching solution.

## Measurement and results

CUDA events report repeated kernel averages after an untimed warm-up. Atomic resets, allocation, data transfers, and host verification are excluded. The maximum reduction measures its pair of kernel launches together. The original all-ones input is intentionally retained for the floating-point experiment; matrix and maximum tests use varied inputs.

Actual timing comparisons and the bonus precision-loss observation are inserted from run logs after execution. Profiler counters, theoretical memory bandwidth, and event-derived input throughput must be kept distinct.

## Key takeaways

- Reduce contention by combining values locally before global atomics.
- Match the identity to the operation: zero for sum, negative infinity for max.
- Floating-point arithmetic is not associative; a race-free algorithm can still lose accuracy.
- Warp shuffles reduce synchronization overhead, but their benefit depends on what dominates runtime.
- Parallel reduction can simultaneously improve memory coalescing and available concurrency.

