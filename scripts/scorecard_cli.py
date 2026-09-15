"""Shared command handling for fixed historical scorecard renderers.

Each renderer retains its own validation rules, expected case inventory and
Markdown format. This module only selects paths and prints, checks or writes
that renderer's result.
"""

import argparse
from collections.abc import Callable
from pathlib import Path


def run_scorecard_cli(
    default_scorecard: Path,
    default_markdown: Path,
    load_and_validate: Callable[[Path], dict],
    render_markdown: Callable[[dict], str],
) -> None:
    """Run the historical print/write/check CLI with the supplied renderer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", type=Path, default=default_scorecard)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scorecard = load_and_validate(args.scorecard)
    rendered = render_markdown(scorecard)
    if args.check:
        target = default_markdown if args.output is None else args.output
        if target.read_text() != rendered:
            raise SystemExit(f"{target} is stale; regenerate it")
    elif args.output is not None:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
