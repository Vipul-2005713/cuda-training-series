# HW9: Cooperative groups and stream compaction

`task1.cu` demonstrates whole-block, 32-thread, and 16-thread group reductions.
`task2.cu` performs cooperative grid-wide compaction, comparing naive offsets,
linear offsets, and Thrust. The runner supplies `-rdc=true` for task 2.

## Build and run on Ubuntu / WSL2

Use an **Ubuntu Bash terminal**, with the Linux CUDA Toolkit, `g++`, and Python 3
available. Run all commands below from the **repository root** (`cuda-training-series`),
not this homework directory. See the [main setup guide](../../README.md) first.

```bash
source tools/ubuntu_env.sh
python3 tools/run_exercises.py --build --hw 9 --suite default
```

This builds the homework's programs into `build/ubuntu/` and runs their default
checks. To run the individual programs or change problem sizes after building:

```bash
./build/ubuntu/hw9_groups --group all
./build/ubuntu/hw9_groups --group 16
./build/ubuntu/hw9_compaction --n 256 --iterations 5 --mode both
./build/ubuntu/hw9_compaction --n 1003 --iterations 3 --mode both
./build/ubuntu/hw9_compaction --n 1048576 --iterations 3 --mode linear
```

## What to expect

- For all-ones inputs, the group example prints one sum of 256, eight sums of 32, and sixteen sums of 16. It also checks signed inputs and prints `PASS group_size=...` for each selected group.
- Compaction prints `PASS compaction variant=naive` and/or `variant=linear_offsets`, checks five input patterns, and reports custom and Thrust timings.
- Group option: `--group all|256|32|16` (default `all`). Compaction options: `--n` (256), `--iterations` (5), `--mode both|naive|linear` (default `both`).
- If `cooperativeLaunch=0`, task 2 prints `SKIP cooperative compaction`. The checked WSL RTX 3050 reports support and should execute it.
- The cooperative launch limits resident blocks using occupancy; increasing N does not justify launching an arbitrarily large cooperative grid.

## Check the complete homework

```bash
python3 tools/run_exercises.py --hw 9 --suite all
```

This runs the default, boundary, and any additional experiments defined for HW9.
Add `--build` after changing code. Add `--include-solutions --build` to check the
reference entry points too; they share the completed exercise implementations.
The runner reports `PASS`, `PASS_WITH_SKIPS`, or `FAIL`, and returns nonzero on
build errors, timeouts, or failed checks. Kernel timings vary with hardware and
system load; use correctness messages to judge success.

Detailed output is in `results/ubuntu/runs/`, compiler output is in
`results/ubuntu/build/`, and the latest invocation has `summary.json` and
`summary.csv`. Use `--output results/ubuntu/hw9` to keep this homework's logs
separate. For a different GPU, pass `--arch sm_XX`; the default `sm_86` matches
the RTX 3050. For compiler selection, use `--ccbin g++-12` when needed.

The [original lecture assignment](LESSON.md) is preserved for background. Its
cluster commands, FIXME locations, and historical timings do not describe the
current completed Ubuntu programs.
