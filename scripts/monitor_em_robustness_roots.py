#!/usr/bin/env python3
"""Monitor EM robustness matrix roots for a fixed wall-clock window."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True, help="Robustness matrix root.")
    parser.add_argument("--job-id", action="append", default=[], help="Slurm job ID to track. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    parser.add_argument("--once", action="store_true", help="Write one report and exit.")
    return parser.parse_args()


def run_text(cmd: list[str], *, cwd: Path = REPO_ROOT) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout


def compact_jobs(job_ids: list[str]) -> str:
    unique = []
    seen = set()
    for job_id in job_ids:
        for part in str(job_id).replace(",", " ").split():
            if part and part not in seen:
                seen.add(part)
                unique.append(part)
    return ",".join(unique)


def job_ids_from_summary_json(summary_json: Path) -> list[str]:
    try:
        payload = json.loads(summary_json.read_text())
    except Exception:
        return []

    jobs: list[str] = []
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        job_id = str(case.get("job_id") or "").strip()
        if job_id.isdigit():
            jobs.append(job_id)
    return jobs


def summarize_roots(roots: list[Path], output_dir: Path) -> tuple[int, str, Path, Path]:
    markdown = output_dir / "latest_robustness_summary.md"
    json_path = output_dir / "latest_robustness_summary.json"
    cmd = [
        sys.executable,
        "scripts/summarize_em_robustness_matrix.py",
        *[str(root) for root in roots],
        "--output-markdown",
        str(markdown),
        "--output-json",
        str(json_path),
        "--dedupe-case-reruns",
    ]
    code, output = run_text(cmd)
    return code, output, markdown, json_path


def root_summary_markdowns(roots: list[Path]) -> list[Path]:
    """Return one-off root summaries that are not represented as matrix cases."""

    summaries = []
    for root in roots:
        summary = root / "summary.md"
        try:
            is_summary = summary.is_file()
        except OSError:
            is_summary = False
        if is_summary:
            summaries.append(summary)
    return summaries


def write_report(args: argparse.Namespace, started: dt.datetime, deadline: dt.datetime) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    code, summary_out, summary_md, summary_json = summarize_roots(args.root, output_dir)
    jobs = compact_jobs([*args.job_id, *job_ids_from_summary_json(summary_json)])

    lines: list[str] = [
        "# EM Robustness 24h Monitor",
        "",
        f"Last update: {now.isoformat(timespec='seconds')}",
        f"Started: {started.isoformat(timespec='seconds')}",
        f"Deadline: {deadline.isoformat(timespec='seconds')}",
        f"Monitor root: `{output_dir}`",
        "",
        "## Roots",
        "",
    ]
    lines.extend(f"- `{root}`" for root in args.root)
    lines.extend(["", "## Jobs", "", f"`{jobs or '<none>'}`", ""])

    if jobs:
        squeue_cmd = ["squeue", "-j", jobs, "-o", "%18i %.9P %.40j %.10T %.10M %.10l %.6D %R"]
        _, squeue_out = run_text(squeue_cmd)
        lines.extend(["## Queue", "", "```text", "$ " + shlex.join(squeue_cmd), squeue_out.rstrip(), "```", ""])

        sacct_cmd = [
            "sacct",
            "-j",
            jobs,
            "-X",
            "-P",
            "-o",
            "JobIDRaw,JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList",
        ]
        _, sacct_out = run_text(sacct_cmd)
        lines.extend(["## Accounting", "", "```text", "$ " + shlex.join(sacct_cmd), sacct_out.rstrip(), "```", ""])

    lines.extend(
        [
            "## Robustness Summary",
            "",
            f"summary_exit_status: `{code}`",
            f"summary_markdown: `{summary_md}`",
            f"summary_json: `{summary_json}`",
            "",
        ]
    )
    if summary_out.strip():
        lines.extend(["```text", summary_out.rstrip(), "```", ""])
    if summary_md.exists():
        lines.append(summary_md.read_text())

    root_summaries = root_summary_markdowns(args.root)
    if root_summaries:
        lines.extend(["", "## Root Summaries", ""])
        for summary_path in root_summaries:
            lines.extend([f"### `{summary_path}`", "", summary_path.read_text().rstrip(), ""])

    tmp = output_dir / "latest_report.md.tmp"
    tmp.write_text("\n".join(lines).rstrip() + "\n")
    tmp.replace(output_dir / "latest_report.md")


def main() -> None:
    args = parse_args()
    started = dt.datetime.now(dt.timezone.utc).astimezone()
    deadline = started + dt.timedelta(hours=args.duration_hours)
    (args.output_dir / "SAFE_TO_DELETE").parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "SAFE_TO_DELETE").touch()

    while True:
        write_report(args, started, deadline)
        if args.once or dt.datetime.now(dt.timezone.utc).astimezone() >= deadline:
            break
        time.sleep(max(1.0, args.interval_seconds))


if __name__ == "__main__":
    main()
