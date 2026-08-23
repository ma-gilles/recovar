#!/usr/bin/env python3
"""Validate and summarize a complete exact-local BPref row-capture boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from recovar.em.bpref_contribution_replay import (
    load_bpref_contribution_bundle,
    summarize_bpref_contribution_bundle,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = load_bpref_contribution_bundle(args.inputs)
    summary = summarize_bpref_contribution_bundle(bundle)
    repo_root = Path(__file__).resolve().parents[1]
    summary["tool_provenance"] = {
        "repo_root": str(repo_root),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "git_status_porcelain": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).splitlines(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
