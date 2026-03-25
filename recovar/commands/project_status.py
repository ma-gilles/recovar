"""Show project status: list all jobs, their types, and completion status.

Usage::

    recovar project_status [directory]
    recovar project_status --tree
"""

import argparse
import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


def _format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _format_time(iso_str):
    """Format an ISO timestamp into a short display string."""
    if not iso_str:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_str[:16]


def _status_symbol(status):
    """Return a status indicator character."""
    return {
        "completed": "+",
        "running": ">",
        "failed": "!",
    }.get(status, "?")


def show_status(project_dir):
    """Print a formatted table of all jobs in the project."""
    project_file = os.path.join(project_dir, "project.json")
    if not os.path.isfile(project_file):
        print(f"No project found at {project_dir}")
        print(f"Run: recovar init_project {project_dir}")
        return 1

    with open(project_file) as f:
        data = json.load(f)

    name = data.get("name", os.path.basename(project_dir))
    jobs = data.get("jobs", [])

    print(f"Project: {name}")
    print(f"  Path: {os.path.abspath(project_dir)}")
    print(f"  Jobs: {len(jobs)}")
    print()

    if not jobs:
        print("  No jobs yet. Run: recovar pipeline --particles <particles.star>")
        return 0

    # Table header
    print(f"  {'':1} {'UID':<30} {'Status':<11} {'Created':<18} {'Duration':<8} {'Parent'}")
    print(f"  {'-':->1} {'-' * 30} {'-' * 11} {'-' * 18} {'-' * 8} {'-' * 20}")

    for job in jobs:
        uid = job.get("uid", "?")
        status = job.get("status", "?")
        created = _format_time(job.get("created"))
        parent = ", ".join(job.get("parent_jobs", [])) or "-"

        # Try to get duration from the job's own job.json
        duration_str = ""
        job_json_path = os.path.join(project_dir, uid, "job.json")
        if os.path.isfile(job_json_path):
            try:
                with open(job_json_path) as jf:
                    jdata = json.load(jf)
                dur = jdata.get("timing", {}).get("duration_seconds")
                duration_str = _format_duration(dur)
            except (json.JSONDecodeError, IOError):
                pass

        sym = _status_symbol(status)
        print(f"  {sym} {uid:<30} {status:<11} {created:<18} {duration_str:<8} {parent}")

    # Show aliases
    aliases = data.get("aliases", {})
    if aliases:
        print()
        print("  Aliases:")
        for alias, target in aliases.items():
            print(f"    {alias} -> {target}")

    return 0


def show_tree(project_dir):
    """Print the job DAG as a simple tree."""
    project_file = os.path.join(project_dir, "project.json")
    if not os.path.isfile(project_file):
        print(f"No project found at {project_dir}")
        return 1

    with open(project_file) as f:
        data = json.load(f)

    jobs = data.get("jobs", [])
    if not jobs:
        print("  No jobs yet.")
        return 0

    # Build adjacency: parent -> children
    children = {}
    roots = []
    for job in jobs:
        uid = job["uid"]
        parents = job.get("parent_jobs", [])
        if not parents:
            roots.append(uid)
        for p in parents:
            children.setdefault(p, []).append(uid)

    def _print_tree(uid, indent=0):
        status = "?"
        for j in jobs:
            if j["uid"] == uid:
                status = j.get("status", "?")
                break
        sym = _status_symbol(status)
        prefix = "  " + "  " * indent + ("|- " if indent > 0 else "")
        print(f"{prefix}{sym} {uid} [{status}]")
        for child in children.get(uid, []):
            _print_tree(child, indent + 1)

    for root in roots:
        _print_tree(root)

    return 0


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Project directory (default: current directory).",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Show job dependency tree instead of table.",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()

    if args.tree:
        return show_tree(args.directory)
    return show_status(args.directory)


if __name__ == "__main__":
    main()
