# HW6: Unified memory and explicit copies

`linked_list.cu` allocates every list node in managed memory so CPU and GPU can
follow the same pointers. `array_inc.cu` compares explicit transfers, managed
memory, and prefetch when the device supports it.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 6 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw6_linked_list
./build/ubuntu/hw6_array_inc --n 4194304 --iterations 1 --mode all
./build/ubuntu/hw6_array_inc --n 1003 --iterations 3 --mode explicit
./build/ubuntu/hw6_array_inc --n 262144 --iterations 10000 --mode managed
```

## What to expect

- The linked-list program prints `key = 3` twice and `PASS linked_list: all 5 CPU/GPU keys match`.
- Array increment checks that every element equals the iteration count and prints `PASS mode=...` with kernel and host timing.
- Options: `--n` (default 4194304), `--iterations` (default 1), and `--mode all|explicit|managed|prefetch` (default `all`).
- On the checked RTX 3050 under WSL, `managedMemory=1` and `concurrentManagedAccess=0`: explicit and managed modes run, while prefetch prints `SKIP mode=prefetch`. The runner reports `PASS_WITH_SKIPS`.
- CPU reads happen after GPU synchronization. Do not infer full demand-paged unified-memory support from successful basic managed allocation. See [NVIDIA's WSL limitations](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications).

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 6 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW6.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw6` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
