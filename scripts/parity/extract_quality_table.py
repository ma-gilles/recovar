"""Print a markdown table comparing observed parity metrics against the baseline.

Wraps ``check_parity.py`` to produce a table suitable for pasting into a PR
description. One row per scenario, one column per metric.

Usage:
    pixi run python scripts/parity/extract_quality_table.py \\
        --baseline tests/baselines/parity/quality_baseline_5k_128.json \\
        --recovar-dump-root _agent_scratch/parity \\
        --relion-dump-dir _agent_scratch/parity/relion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/parity"))
from check_parity import extract_metrics  # noqa: E402


def _load(p: Path) -> dict:
    return dict(np.load(p, allow_pickle=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument(
        "--recovar-dump-root",
        type=Path,
        required=True,
        help="Dir containing per-scenario dump subdirs named <scenario>/iter_NNN.npz",
    )
    ap.add_argument(
        "--relion-dump-dir",
        type=Path,
        required=True,
        help="Dir containing iter_NNN.npz for the RELION reference",
    )
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text())
    scenarios = baseline.get("scenarios", {})

    cols = [
        "ave_pmax",
        "ave_pmax_relion_ref",
        "vol_corr_half1",
        "vol_corr_half2",
        "pp_hard_assign_match_lt_5deg_rate",
        "sigma_offset_a",
        "wall_time_s",
    ]
    print("### Parity quality (current vs baseline)")
    print()
    print("| Scenario | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")

    for name, scen in scenarios.items():
        if scen.get("optional"):
            continue
        cfg = scen["config"]
        relion_iter = int(cfg["init_iter"]) + int(cfg["max_iter"])
        rec_npz = args.recovar_dump_root / name / f"iter_{relion_iter:03d}.npz"
        rel_npz = args.relion_dump_dir / f"iter_{relion_iter:03d}.npz"
        if not rec_npz.exists():
            print(f"| {name} | (no recovar dump at {rec_npz}) |")
            continue
        if not rel_npz.exists():
            print(f"| {name} | (no relion dump at {rel_npz}) |")
            continue
        rec = _load(rec_npz)
        rel = _load(rel_npz)
        m = extract_metrics(rec, rel)
        cells = [name]
        for c in cols:
            v = m.get(c)
            if v is None:
                cells.append("n/a")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
