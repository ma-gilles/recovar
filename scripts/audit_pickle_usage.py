"""Audit pickle load/dump usage across tracked Python files.

This is a bootstrap tool for issue #34. It finds concrete pickle call sites
so migration work can be scoped from actual usage instead of ad hoc grep.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

_PICKLE_OPS = {"load", "dump", "loads", "dumps"}


@dataclass(frozen=True)
class PickleUse:
    path: str
    line: int
    operation: str
    snippet: str


def _resolve_pickle_aliases(tree: ast.AST) -> tuple[set[str], dict[str, str]]:
    module_aliases: set[str] = set()
    function_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pickle":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pickle":
            for alias in node.names:
                if alias.name in _PICKLE_OPS:
                    function_aliases[alias.asname or alias.name] = alias.name

    return module_aliases, function_aliases


def audit_path(path: Path, *, root: Path | None = None) -> list[PickleUse]:
    root = path.parent if root is None else root
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module_aliases, function_aliases = _resolve_pickle_aliases(tree)
    source_lines = source.splitlines()
    rel_path = str(path.relative_to(root))

    uses: list[PickleUse] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        operation = None
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in module_aliases
            and func.attr in _PICKLE_OPS
        ):
            operation = func.attr
        elif isinstance(func, ast.Name):
            operation = function_aliases.get(func.id)

        if operation is None:
            continue

        snippet = source_lines[node.lineno - 1].strip()
        uses.append(PickleUse(path=rel_path, line=node.lineno, operation=operation, snippet=snippet))

    return sorted(uses, key=lambda use: (use.path, use.line, use.operation))


def _fallback_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in {".git", ".pixi", "__pycache__"} for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def tracked_python_files(root: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--", "*.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return _fallback_python_files(root)

    tracked = [root / line for line in output.splitlines() if line.strip()]
    return sorted(path for path in tracked if path.exists())


def audit_paths(paths: Iterable[Path], *, root: Path) -> list[PickleUse]:
    uses: list[PickleUse] = []
    for path in paths:
        uses.extend(audit_path(path, root=root))
    return sorted(uses, key=lambda use: (use.path, use.line, use.operation))


def build_summary(uses: list[PickleUse]) -> dict:
    by_operation = Counter(use.operation for use in uses)
    by_file: dict[str, Counter] = defaultdict(Counter)
    for use in uses:
        by_file[use.path][use.operation] += 1

    return {
        "total_calls": len(uses),
        "total_files": len(by_file),
        "by_operation": dict(sorted(by_operation.items())),
        "by_file": {
            path: {op: counts.get(op, 0) for op in sorted(_PICKLE_OPS)} | {"total": sum(counts.values())}
            for path, counts in sorted(by_file.items())
        },
    }


def render_markdown(summary: dict, uses: list[PickleUse], *, details: bool = False) -> str:
    lines = [
        "# Pickle Usage Audit",
        "",
        f"Found {summary['total_calls']} pickle call sites across {summary['total_files']} tracked Python files.",
        "",
        "| File | dump | load | dumps | loads | total |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for path, counts in summary["by_file"].items():
        lines.append(
            f"| `{path}` | {counts['dump']} | {counts['load']} | {counts['dumps']} | {counts['loads']} | {counts['total']} |"
        )

    if details:
        lines.extend(["", "## Call Sites", ""])
        for use in uses:
            lines.append(f"- `{use.path}:{use.line}` `{use.operation}`: `{use.snippet}`")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to audit")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--details", action="store_true", help="Include per-callsite details in markdown output")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    uses = audit_paths(tracked_python_files(root), root=root)
    summary = build_summary(uses)

    if args.format == "json":
        payload = {
            "root": str(root),
            "summary": summary,
            "uses": [asdict(use) for use in uses],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(summary, uses, details=args.details))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
