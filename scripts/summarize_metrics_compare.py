#!/usr/bin/env python
"""Summarize local-vs-clean run_test_all_metrics score deltas.

Direction rules are based on how metrics are computed in
`recovar/commands/run_test_all_metrics.py`:
- `*_error`, `embedding_squared_error_*`, and `contrasts_*`/`constrasts_*`
  are absolute errors -> lower is better.
- `*_locres` comes from local resolution values (Angstrom) -> lower is better.
- `*fsc`, `*correlation`, and `pcs_relative_variance_*` are quality/correlation
  scores -> higher is better.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LOWER_IS_BETTER_TOKENS = (
    "error",
    "locres",
    "angle",
    "loss",
    "rmse",
    "mse",
    "bias",
    "constrast",
    "contrast",
)

HIGHER_IS_BETTER_TOKENS = (
    "fsc",
    "correlation",
    "variance_explained",
    "pcs_relative_variance",
)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def metric_direction(metric_name: str) -> str:
    name = metric_name.lower()
    if any(tok in name for tok in LOWER_IS_BETTER_TOKENS):
        return "lower"
    if any(tok in name for tok in HIGHER_IS_BETTER_TOKENS):
        return "higher"
    return "ignore"


def practical_significance(abs_directional_change: float, tol_frac: float) -> tuple[bool, str]:
    if not math.isfinite(abs_directional_change):
        return False, "non-finite"
    if abs_directional_change < 0.01:
        return False, "negligible"
    if abs_directional_change < tol_frac:
        return False, "small"
    if abs_directional_change < 0.10:
        return True, "moderate"
    return True, "large"


def load_scores(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def parse_score_paths(path: Path) -> tuple[Path, Path]:
    local_path = None
    clean_path = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() in ("current_scores", "local_scores"):
            local_path = Path(val.strip())
        if key.strip() == "clean_scores":
            clean_path = Path(val.strip())
    if local_path is None or clean_path is None:
        raise ValueError(f"Could not parse local/clean score paths from {path}")
    return local_path, clean_path


def fmt_num(x: float | None) -> str:
    if x is None:
        return "-"
    if not math.isfinite(x):
        return str(x)
    return f"{x:.10g}"


def summarize(
    local_scores: dict[str, Any],
    clean_scores: dict[str, Any],
    tol_frac: float,
) -> tuple[dict[str, int], list[dict[str, Any]], list[str], list[str]]:
    metrics = sorted(set(local_scores) | set(clean_scores))
    rows: list[dict[str, Any]] = []
    local_only: list[str] = []
    clean_only: list[str] = []
    counts = {
        "local_better": 0,
        "clean_better": 0,
        "tie": 0,
        "ignored": 0,
        "non_numeric": 0,
        "significant": 0,
    }

    for metric in metrics:
        local_val = local_scores.get(metric)
        clean_val = clean_scores.get(metric)
        if metric not in clean_scores:
            local_only.append(metric)
            continue
        if metric not in local_scores:
            clean_only.append(metric)
            continue
        if not (_is_number(local_val) and _is_number(clean_val)):
            counts["non_numeric"] += 1
            continue

        local_f = float(local_val)
        clean_f = float(clean_val)
        direction = metric_direction(metric)
        if direction == "ignore":
            counts["ignored"] += 1
            continue

        scale = max(abs(clean_f), 1e-12)
        signed_raw_change = (local_f - clean_f) / scale
        if direction == "lower":
            # Positive means local improved (lower than clean).
            directional_change = -signed_raw_change
        else:
            # Higher is better.
            directional_change = signed_raw_change

        if math.isclose(directional_change, 0.0, rel_tol=0.0, abs_tol=1e-12):
            better = "tie"
            counts["tie"] += 1
        elif directional_change > 0:
            better = "local"
            counts["local_better"] += 1
        else:
            better = "clean"
            counts["clean_better"] += 1

        sig, sig_label = practical_significance(abs(directional_change), tol_frac=tol_frac)
        if sig:
            counts["significant"] += 1

        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "local": local_f,
                "clean": clean_f,
                "better": better,
                "directional_change": directional_change,
                "signed_raw_change": signed_raw_change,
                "abs_diff": abs(local_f - clean_f),
                "significant": sig,
                "sig_label": sig_label,
            }
        )

    rows.sort(
        key=lambda r: (
            0 if r["significant"] else 1,
            -abs(r["directional_change"]),
            r["metric"],
        )
    )
    return counts, rows, local_only, clean_only


def render_markdown(
    *,
    local_scores_path: Path,
    clean_scores_path: Path,
    tol_frac: float,
    counts: dict[str, int],
    rows: list[dict[str, Any]],
    local_only: list[str],
    clean_only: list[str],
) -> str:
    out: list[str] = []
    out.append("# Metrics Comparison Summary")
    out.append("")
    out.append(f"- local scores: `{local_scores_path}`")
    out.append(f"- clean scores: `{clean_scores_path}`")
    out.append(f"- significance threshold: `{tol_frac:.2%}` directional change")
    out.append("")
    out.append("## Totals")
    out.append("")
    out.append(f"- local better: **{counts['local_better']}**")
    out.append(f"- clean better: **{counts['clean_better']}**")
    out.append(f"- ties: **{counts['tie']}**")
    out.append(f"- significant differences: **{counts['significant']}**")
    out.append(f"- ignored by direction rules: **{counts['ignored']}**")
    out.append(f"- non-numeric skipped: **{counts['non_numeric']}**")
    if local_only:
        out.append(f"- local-only metrics: **{len(local_only)}**")
    if clean_only:
        out.append(f"- clean-only metrics: **{len(clean_only)}**")
    out.append("")
    out.append("## Per-Metric (ranked by significance, then effect size)")
    out.append("")
    out.append("| metric | better | direction | local | clean | abs diff | directional change | significance |")
    out.append("|---|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        out.append(
            "| {metric} | {better} | {direction} | {local} | {clean} | {abs_diff} | {dc:+.2%} | {sig} |".format(
                metric=r["metric"],
                better=r["better"],
                direction=r["direction"],
                local=fmt_num(r["local"]),
                clean=fmt_num(r["clean"]),
                abs_diff=fmt_num(r["abs_diff"]),
                dc=r["directional_change"],
                sig=r["sig_label"],
            )
        )
    if local_only:
        out.append("")
        out.append("## Local-Only Metrics")
        out.append("")
        for m in local_only:
            out.append(f"- `{m}`")
    if clean_only:
        out.append("")
        out.append("## Clean-Only Metrics")
        out.append("")
        for m in clean_only:
            out.append(f"- `{m}`")
    out.append("")
    out.append("Note: significance here is practical significance from effect size, not a statistical hypothesis test.")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local/current vs clean all_scores.json and summarize which is better."
    )
    parser.add_argument("--local-scores", type=Path, default=None, help="Path to local/current all_scores.json")
    parser.add_argument("--clean-scores", type=Path, default=None, help="Path to clean all_scores.json")
    parser.add_argument(
        "--score-paths",
        type=Path,
        default=None,
        help="Path to score_paths.txt containing current_scores=... and clean_scores=...",
    )
    parser.add_argument(
        "--tol-frac",
        type=float,
        default=0.03,
        help="Directional relative change threshold for practical significance (default: 0.03).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional path to write markdown summary.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write machine-readable summary json.",
    )
    args = parser.parse_args()

    if args.score_paths is not None:
        local_scores_path, clean_scores_path = parse_score_paths(args.score_paths)
    else:
        if args.local_scores is None or args.clean_scores is None:
            raise SystemExit("Pass either --score-paths or both --local-scores and --clean-scores.")
        local_scores_path = args.local_scores
        clean_scores_path = args.clean_scores

    local_scores = load_scores(local_scores_path)
    clean_scores = load_scores(clean_scores_path)
    counts, rows, local_only, clean_only = summarize(
        local_scores=local_scores,
        clean_scores=clean_scores,
        tol_frac=args.tol_frac,
    )

    markdown = render_markdown(
        local_scores_path=local_scores_path,
        clean_scores_path=clean_scores_path,
        tol_frac=args.tol_frac,
        counts=counts,
        rows=rows,
        local_only=local_only,
        clean_only=clean_only,
    )
    print(markdown)

    if args.output_md is not None:
        args.output_md.write_text(markdown)

    if args.output_json is not None:
        payload = {
            "local_scores": str(local_scores_path),
            "clean_scores": str(clean_scores_path),
            "tol_frac": args.tol_frac,
            "counts": counts,
            "rows": rows,
            "local_only": local_only,
            "clean_only": clean_only,
        }
        args.output_json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
