#!/usr/bin/env python3
"""Private stdlib-only support for sidedgpu shell entrypoints."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


class QueryError(RuntimeError):
    pass


def run_text(command):
    try:
        result = subprocess.run(
            command,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise QueryError(str(exc))
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise QueryError(detail or f"command exited with status {result.returncode}")
    return result.stdout


def gpu_inventory(smi):
    output = run_text(
        [smi, "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"]
    )
    gpus = []
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        if len(row) < 3:
            raise QueryError("malformed GPU inventory row")
        try:
            index = int(row[0].strip())
        except ValueError as exc:
            raise QueryError("non-numeric GPU index in inventory") from exc
        uuid = row[1].strip()
        name = ",".join(row[2:]).strip()
        if not uuid or not name:
            raise QueryError("GPU inventory omitted UUID or name")
        gpus.append({"index": index, "uuid": uuid, "name": name})
    if not gpus:
        raise QueryError("nvidia-smi reported no GPUs")
    return gpus


def selected_gpus(gpus, selector):
    if selector == "auto":
        return gpus
    matches = [
        gpu
        for gpu in gpus
        if selector == str(gpu["index"]) or selector == gpu["uuid"]
    ]
    if not matches:
        raise ValueError(
            f"GPU selector '{selector}' did not match a physical index or UUID"
        )
    return matches


def sample_stats(smi, index):
    output = run_text(
        [
            smi,
            "-i",
            str(index),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = list(csv.reader(output.splitlines()))
    if not rows or len(rows[0]) < 2:
        raise QueryError("malformed GPU telemetry row")
    try:
        return int(rows[0][0].strip()), int(rows[0][1].strip())
    except ValueError as exc:
        raise QueryError("non-numeric GPU telemetry") from exc


def compute_lines(smi, index):
    output = run_text(
        [
            smi,
            "-i",
            str(index),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def no_compute_processes(smi, index):
    try:
        lines = compute_lines(smi, index)
    except QueryError:
        return False
    # N/A and every other unknown/non-empty value are deliberately busy.
    return not lines or lines == ["No running processes found"]


def candidates(args):
    try:
        gpus = selected_gpus(gpu_inventory(args.nvidia_smi), args.selector)
    except ValueError as exc:
        print(f"[sidedgpu] error: {exc}", file=sys.stderr)
        return 2
    except QueryError as exc:
        print(f"[sidedgpu] error: failed to query GPU inventory: {exc}", file=sys.stderr)
        return 1

    state = {
        int(gpu["index"]): {"gpu": gpu, "eligible": True, "memory": 0, "util": 0}
        for gpu in gpus
    }
    for sample in range(args.samples):
        for index, item in state.items():
            if not item["eligible"]:
                continue
            if not no_compute_processes(args.nvidia_smi, index):
                item["eligible"] = False
                continue
            try:
                memory, util = sample_stats(args.nvidia_smi, index)
            except QueryError:
                item["eligible"] = False
                continue
            if memory > args.max_memory or util > args.max_util:
                item["eligible"] = False
                continue
            item["memory"] = max(int(item["memory"]), memory)
            item["util"] = max(int(item["util"]), util)
        if sample + 1 < args.samples:
            time.sleep(1)

    ranked = sorted(
        (item for item in state.values() if item["eligible"]),
        key=lambda item: (item["memory"], item["util"], item["gpu"]["index"]),
    )
    for item in ranked:
        gpu = item["gpu"]
        name = str(gpu["name"]).replace("\t", " ").replace("\n", " ")
        print(
            item["memory"],
            item["util"],
            gpu["index"],
            gpu["uuid"],
            name,
            sep="\t",
        )
    return 0


def diagnostics(args):
    try:
        gpus = gpu_inventory(args.nvidia_smi)
    except QueryError as exc:
        print(f"[sidedgpu] error: failed to query GPU inventory: {exc}", file=sys.stderr)
        return 1
    for gpu in gpus:
        index = int(gpu["index"])
        try:
            memory, util = sample_stats(args.nvidia_smi, index)
            memory_text, util_text = str(memory), str(util)
        except QueryError:
            memory_text = util_text = "unknown"
        try:
            lines = compute_lines(args.nvidia_smi, index)
            pids = ",".join(lines) if lines else "none"
        except QueryError:
            pids = "query-error"
        name = str(gpu["name"]).replace("\t", " ").replace("\n", " ")
        pids = pids.replace("\t", " ").replace("\n", ",")
        print(index, gpu["uuid"], name, memory_text, util_text, pids, sep="\t")
    return 0


def command_output(command):
    try:
        result = subprocess.run(
            command,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        return 127, str(exc)
    return result.returncode, (result.stdout + result.stderr).strip()


def git_metadata(root):
    status, _ = command_output(["git", "-C", root, "rev-parse", "--is-inside-work-tree"])
    if status != 0:
        return "", "unknown"
    status, commit = command_output(["git", "-C", root, "rev-parse", "HEAD"])
    if status != 0:
        commit = ""
    dirty = False
    for extra in ([], ["--cached"]):
        status, _ = command_output(
            ["git", "-C", root, "diff", *extra, "--quiet", "--ignore-submodules", "--"]
        )
        if status != 0:
            dirty = True
    return commit, "true" if dirty else "false"


def driver_version(smi, gpu_index):
    if not smi or not os.access(smi, os.X_OK):
        return ""
    command = [smi]
    if gpu_index:
        command += ["-i", gpu_index]
    command += ["--query-gpu=driver_version", "--format=csv,noheader,nounits"]
    try:
        first = run_text(command).splitlines()[0].strip()
    except (QueryError, IndexError):
        return ""
    return first if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", first) else ""


def optional_bool(value):
    return True if value == "true" else False if value == "false" else None


def optional_int(value):
    return int(value) if value else None


def write_metadata(args):
    commit, dirty = git_metadata(args.repo_root)
    nvcc = shutil.which("nvcc")
    nvcc_version = command_output([nvcc, "--version"])[1] if nvcc else ""
    metadata = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": args.command,
        "cwd": os.getcwd(),
        "repo_root": args.repo_root,
        "git_commit": commit,
        "git_dirty": dirty,
        "cuda_driver_version": driver_version(args.nvidia_smi, args.gpu_index),
        "nvcc_version": nvcc_version,
        "gpu": {
            "physical_index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "name": args.gpu_name,
            "arch": args.gpu_arch,
            "memory_used_mib": args.gpu_memory,
            "max_util_percent": args.gpu_util,
        },
        "idle_check": {
            "no_compute_processes": optional_bool(args.no_compute),
            "samples": args.samples,
            "max_memory_threshold_mib": args.max_memory,
            "max_util_threshold_percent": args.max_util,
            "max_memory_used_mib": optional_int(args.gpu_memory),
            "max_util_percent": optional_int(args.gpu_util),
            "lock_path": args.lock_path,
            "lock_held": optional_bool(args.lock_held),
        },
    }
    for item in args.extra:
        key, value = item.split("=", 1)
        metadata[key] = value
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def parser():
    root = argparse.ArgumentParser(add_help=False)
    subparsers = root.add_subparsers(dest="action")
    candidate_parser = subparsers.add_parser("gpu-snapshot")
    candidate_parser.add_argument("--nvidia-smi", required=True)
    candidate_parser.add_argument("--selector", required=True)
    candidate_parser.add_argument("--samples", type=int, required=True)
    candidate_parser.add_argument("--max-memory", type=int, required=True)
    candidate_parser.add_argument("--max-util", type=int, required=True)
    candidate_parser.set_defaults(handler=candidates)

    diagnostic_parser = subparsers.add_parser("gpu-diagnostics")
    diagnostic_parser.add_argument("--nvidia-smi", required=True)
    diagnostic_parser.set_defaults(handler=diagnostics)

    metadata_parser = subparsers.add_parser("write-metadata")
    for option in (
        "output",
        "repo-root",
        "command",
        "gpu-index",
        "gpu-uuid",
        "gpu-name",
        "gpu-arch",
        "gpu-memory",
        "gpu-util",
        "no-compute",
        "lock-path",
        "lock-held",
        "nvidia-smi",
    ):
        metadata_parser.add_argument(f"--{option}", required=True)
    metadata_parser.add_argument("--samples", type=int, required=True)
    metadata_parser.add_argument("--max-memory", type=int, required=True)
    metadata_parser.add_argument("--max-util", type=int, required=True)
    metadata_parser.add_argument("--extra", action="append", default=[])
    metadata_parser.set_defaults(handler=write_metadata)
    return root


def main():
    args = parser().parse_args()
    if not hasattr(args, "handler"):
        print("missing support action", file=sys.stderr)
        return 2
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
