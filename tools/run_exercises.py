"""Build and run the CUDA homework on Ubuntu (native or WSL2).

Requires Python 3 and the Linux CUDA Toolkit; no Python packages needed.
Example: python3 tools/run_exercises.py --build --hw 1 --suite all
"""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TARGETS = json.loads((ROOT / "tools/targets.json").read_text())
WSL_DRIVER_DIR = Path("/usr/lib/wsl/lib")


def runtime_environment():
    env = os.environ.copy()
    # cuBLAS may load a native Linux libcuda from its own library directory.
    # Under WSL, the host-provided driver must take precedence for every child.
    if "microsoft" in platform.release().lower() and (WSL_DRIVER_DIR / "libcuda.so.1").is_file():
        paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and p != str(WSL_DRIVER_DIR)]
        env["LD_LIBRARY_PATH"] = ":".join([str(WSL_DRIVER_DIR), *paths])
    return env


def invoke(command, log, timeout=180):
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    tick = time.perf_counter()
    try:
        p = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace", timeout=timeout, env=runtime_environment())
        code, output = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        output = e.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        code, output = 124, output + "\nTIMEOUT\n"
    except OSError as e:
        code, output = 127, str(e)
    elapsed = time.perf_counter() - tick
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"UTC: {started}\nCOMMAND: {shlex.join(command)}\n"
                   f"EXIT: {code}\nPROCESS_WALL_SECONDS: {elapsed:.6f}\n\n{output}", encoding="utf-8")
    return code, output, elapsed


def primary_cases(suite):
    if suite in ("default", "all"):
        for target in TARGETS:
            if not target.get("variant_of"):
                yield target["name"], "default", []
    if suite in ("edge", "all"):
        edge = {
            "hw1_vector_add": [["1"], ["1003"]],
            "hw1_matrix_mul": [["1"], ["17"], ["65"]],
            "hw2_matrix_mul_shared": [["1"], ["17"], ["65"]],
            "hw2_stencil": [["1"], ["19"], ["4099"]],
            "hw3_vector_add": [["1003", "3", "128", "3"]],
            "hw4_matrix_sums": [["1"], ["33"], ["257"]],
            "hw5_matrix_sums": [["1"], ["33"], ["257"]],
            "hw5_max_reduction": [["1"], ["1003"]],
            "hw5_reductions": [["1003"]],
            "hw6_array_inc": [["--n", "1003", "--iterations", "3"]],
            "hw7_overlap": [["--n", "1003", "--chunks", "7", "--streams", "3", "--repeats", "2"]],
            "hw7_multi": [["--n", "1003", "--repeats", "2"]],
            "hw8_task1": [["--size", "33", "--iterations", "3"], ["--size", "1031", "--iterations", "5"]],
            "hw8_task2": [["--size", "33", "--iterations", "3"], ["--size", "1031", "--iterations", "5"]],
            "hw8_task3": [["--size", "33", "--iterations", "3"], ["--size", "1031", "--iterations", "5"]],
            "hw9_compaction": [["--n", "1"], ["--n", "1003"]],
            "hw10_serial": [["1003", "7", "3", "1"]],
            "hw10_streams": [["1003", "7", "3", "1"]],
            "hw10_openmp": [["1003", "7", "3", "1"]],
            "hw12_matrix": [["32"], ["65"]],
            "hw12_transform": [["1"], ["1003"]],
        }
        for target in TARGETS:
            if target["name"].startswith("hw13_") and not target.get("variant_of"):
                edge[target["name"]] = [["1003", "7"]]
        for name, inputs in edge.items():
            for i, args in enumerate(inputs):
                yield name, f"edge_{i+1}", args
    if suite in ("experiments", "all"):
        experiments = [
            ("hw3_vector_add", "launch_1x1", ["262144", "1", "1", "3"]),
            ("hw3_vector_add", "launch_1x1024", ["262144", "1", "1024", "10"]),
            ("hw3_vector_add", "launch_1x256", ["262144", "1", "256", "10"]),
            ("hw3_vector_add", "launch_32x256", ["262144", "32", "256", "10"]),
            ("hw3_vector_add", "launch_160x256", ["262144", "160", "256", "10"]),
            ("hw3_vector_add", "launch_160x1024", ["262144", "160", "1024", "10"]),
            ("hw3_vector_add", "large_vector", ["16777216", "160", "256", "10"]),
            ("hw5_reductions", "small", ["163840", "3"]),
            ("hw5_reductions", "precision_32M", ["33554432", "3"]),
            ("hw6_array_inc", "repeated_10000", ["--n", "262144", "--iterations", "10000"]),
            ("hw7_overlap", "one_stream", ["--streams", "1"]),
            ("hw7_overlap", "eight_streams", ["--streams", "8"]),
            ("hw9_compaction", "large", ["--n", "1048576", "--iterations", "3"]),
            ("hw10_openmp", "four_gpu_bonus", ["1048576", "16", "4", "4"]),
        ]
        yield from experiments


def cases(suite):
    for name, case, parameters in primary_cases(suite):
        yield name, case, parameters
        for target in TARGETS:
            if target.get("variant_of") == name:
                yield target["name"], case, parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--suite", choices=["none", "default", "edge", "experiments", "all"], default="default")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only", nargs="+", help="Target name prefixes, e.g. hw1_ or hw9_compaction")
    selection.add_argument("--hw", type=int, choices=range(1, 14), nargs="+", help="Homework numbers, e.g. --hw 1 2")
    parser.add_argument("--include-solutions", action="store_true", help="Also build/run the *_solution.cu entry points")
    parser.add_argument("--list", action="store_true", help="List selected targets and exit")
    parser.add_argument("--arch", default=os.environ.get("CUDA_ARCH", "sm_86"), help="GPU architecture (default: CUDA_ARCH or sm_86 for RTX 3050)")
    parser.add_argument("--ccbin", default=os.environ.get("NVCC_CCBIN"), help="Optional CUDA-compatible host compiler, e.g. g++-12")
    parser.add_argument("--output", default="results/ubuntu", help="Log directory; summaries describe this invocation only")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this tool in Ubuntu/WSL2 using python3 and the Linux CUDA Toolkit")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    selected = [t for t in TARGETS
                if (args.include_solutions or not t.get("variant_of"))
                and (not args.hw or any(t["name"].startswith(f"hw{n}_") for n in args.hw))
                and (not args.only or any(t["name"].startswith(p) for p in args.only))]
    if not selected:
        parser.error("No targets matched; use --list or --include-solutions")
    if args.list:
        for target in selected:
            print(f"{target['name']}: {target['source']}")
        return 0
    build = ROOT / "build" / "ubuntu"
    build.mkdir(parents=True, exist_ok=True)
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    build_records = []
    failed_builds = set()
    if args.build:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            parser.error("nvcc is not on PATH; install the Linux CUDA Toolkit inside Ubuntu")
        host = ["-ccbin", args.ccbin] if args.ccbin else []
        for target in selected:
            command = [nvcc, *host, "-std=c++17", "-O3", "-lineinfo", f"-arch={args.arch}",
                       str(ROOT / target["source"]), "-o", str(build / target["name"]), *target["flags"]]
            if target.get("openmp"):
                command += ["-Xcompiler", "-fopenmp"]
            code, _, elapsed = invoke(command, output / "build" / (target["name"] + ".log"), args.timeout)
            print(f"BUILD {'PASS' if code == 0 else 'FAIL'} {target['name']} ({elapsed:.1f}s)", flush=True)
            if code:
                failed_builds.add(target["name"])
            build_records.append(dict(target=target["name"], status="FAIL" if code else "PASS", exit_code=code))
    selected_names = {t["name"] for t in selected}
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    for name, case, parameters in cases(args.suite):
        if name not in selected_names:
            continue
        log = output / "runs" / f"{name}__{case}.log"
        command = [str(build / name), *parameters]
        if name in failed_builds:
            code, text, elapsed = 125, "BUILD_FAILED; stale executable not run", 0
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(text + "\n", encoding="utf-8")
        else:
            code, text, elapsed = invoke(command, log, args.timeout)
        # Executables return failure on mismatches; skips are explicit feature-level limitations.
        status = "FAIL" if code else ("PASS_WITH_SKIPS" if "SKIP" in text or "MULTI_GPU_LIMITATION" in text else "PASS")
        records.append(dict(target=name, case=case, arguments=parameters, status=status,
                            exit_code=code, process_wall_seconds=round(elapsed, 6), log=os.path.relpath(log, ROOT)))
        print(f"RUN {status} {name}/{case} ({elapsed:.2f}s)", flush=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(dict(updated_utc=stamp, platform=platform.platform(), architecture=args.arch,
                                           builds=build_records, runs=records), indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "case", "status", "exit_code", "process_wall_seconds", "log", "arguments"])
        writer.writeheader()
        writer.writerows(records)
    failed_runs = sum(r["status"] == "FAIL" for r in records)
    print(f"Summary: {len(build_records)} builds, {len(records)} runs, {len(failed_builds)} build failures, {failed_runs} run failures")
    print(f"Logs and summaries: {output}")
    return int(bool(failed_builds) or bool(failed_runs))


if __name__ == "__main__":
    sys.exit(main())
