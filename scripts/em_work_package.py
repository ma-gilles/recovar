"""Capture a work-package receipt or watch existing Slurm jobs without model polling.

Standard library only; this tool never submits, restarts, cancels or qualifies jobs.
Keep output outside the checkout. Large manifests stay on disk; stdout is concise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def snapshot(repo):
    tracked = git(repo, "ls-files", "-z").decode().split("\0")
    untracked = git(repo, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    files = {}
    for name in sorted(set(tracked + untracked) - {""}):
        path = repo / name
        if path.is_symlink():
            data = os.readlink(path).encode()
        elif path.is_file():
            data = path.read_bytes()
        else:
            files[name] = None  # Deleted tracked file; never silently omit it.
            continue
        files[name] = hashlib.sha256(data).hexdigest()
    return {
        "head": git(repo, "rev-parse", "HEAD").decode().strip(),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        "status": git(repo, "status", "--short", "--branch").decode(),
        "diff_stat": git(repo, "diff", "HEAD", "--stat").decode(),
        "diff_sha256": hashlib.sha256(git(repo, "diff", "HEAD", "--binary")).hexdigest(),
        "untracked": sorted(set(untracked) - {""}),
        "files": files,
    }


def run(args):
    if not args.command:
        raise ValueError("run requires a command after --")
    if "SLURM_JOB_ID" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise ValueError("Set CUDA_VISIBLE_DEVICES before launch (empty for CPU).")
    before = snapshot(args.repo)
    write_json(args.output / "before.json", before)
    start = time.monotonic()
    with (args.output / "command.log").open("w") as log:
        result = subprocess.run(args.command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
    after = snapshot(args.repo)
    write_json(args.output / "after.json", after)
    receipt = {
        "utc": utc(), "repo": str(args.repo), "head": before["head"],
        "diff_sha256": before["diff_sha256"], "command": args.command,
        "exit_code": result.returncode, "seconds": time.monotonic() - start,
        "source_unchanged": before == after,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "PYTHONNOUSERSITE",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "SLURM_JOB_ID", "TMPDIR",
            "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV",
            "RECOVAR_DISABLE_CUDA", "RECOVAR_CUDA_LIB", "RECOVAR_CUDA_CACHE_DIR",
            "RECOVAR_RELION_BIND_BUILD_DIR", "RECOVAR_JAX_CACHE_DIR",
            "JAX_COMPILATION_CACHE_DIR",
        )},
        "log": str(args.output / "command.log"),
        "qualification": "Command execution only; inspect results against applicable gates.",
    }
    write_json(args.output / "receipt.json", receipt)
    print(json.dumps(receipt))
    return 0 if result.returncode == 0 and receipt["source_unchanged"] else 1


TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED",
}


def job_states(job_ids):
    """Queue visibility wins over accounting, including requeued jobs."""
    selection = ",".join(job_ids)
    queue = subprocess.check_output(
        ["squeue", "-h", "-j", selection, "-o", "%i|%T"], text=True, timeout=30,
    )
    rows = {line.split("|", 1)[0]: line.split("|", 1)[1].strip()
            for line in queue.splitlines() if "|" in line}
    missing = set(job_ids) - rows.keys()
    if missing:
        accounting = subprocess.check_output(
            ["sacct", "-n", "-P", "-j", ",".join(sorted(missing)),
             "--format=JobIDRaw,State,ExitCode"], text=True, timeout=30,
        )
        for line in accounting.splitlines():
            fields = line.split("|")
            if len(fields) >= 3 and fields[0] in missing:
                rows[fields[0]] = fields[1].split()[0].rstrip("+") + "|" + fields[2]
    return {job: rows.get(job, "UNKNOWN") for job in job_ids}


def watch(args):
    previous = None
    while True:
        try:
            states = job_states(args.jobs)
            record = {"utc": utc(), "jobs": states}
            # Require terminal accounting with exit code, never absence alone.
            terminal = all("|" in state and state.split("|", 1)[0] in TERMINAL
                           for state in states.values())
            record["all_terminal"] = terminal
        except (subprocess.SubprocessError, OSError) as error:
            states = None
            record = {"utc": utc(), "observation_error": str(error), "all_terminal": False}
            terminal = False
        write_json(args.output / "jobs.json", record)
        if states != previous or states is None:
            with (args.output / "events.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            previous = states
        if terminal:
            print(json.dumps(record), flush=True)
            return 0
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True, help="new directory outside repository")
    commands = parser.add_subparsers(dest="mode", required=True)
    execution = commands.add_parser("run", help="record a command and source identity before/after")
    execution.add_argument("command", nargs=argparse.REMAINDER)
    watcher = commands.add_parser("watch", help="wait for exact existing job IDs; no job mutation")
    watcher.add_argument("jobs", nargs="+")
    watcher.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    args.repo = Path(git(args.repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.repo):
        parser.error("output must be outside the repository")
    if args.mode == "watch" and (args.interval < 30 or not all(j.isdecimal() for j in args.jobs)):
        parser.error("use explicit numeric job IDs and an interval >=30 seconds")
    if args.mode == "run" and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "SAFE_TO_DELETE").write_text("Disposable command/watch evidence; preserve while active.\n")
    return run(args) if args.mode == "run" else watch(args)


if __name__ == "__main__":
    raise SystemExit(main())
