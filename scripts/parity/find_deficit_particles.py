#!/usr/bin/env python
"""Find per-particle deficit between recovar parity_dump and RELION reference data.star.

Requires: parity_dump iter_NNN.npz with the `half[12]_original_image_indices`
fields populated (added 2026-04-25). Match by image_name → identify which
particles have the largest recovar pmax << RELION pmax for targeted
single-particle diff² investigation.

Usage:
    pixi run python scripts/parity/find_deficit_particles.py \
        --recovar-dump _agent_scratch/match_test/iter_001.npz \
        --recovar-input-star /path/to/particles.star \
        --relion-data-star /path/to/run_it001_data.star \
        [--top-n 20]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import starfile


def stack_index_from_image_name(name: str) -> int:
    m = re.match(r"(\d+)@", str(name))
    return int(m.group(1)) - 1 if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovar-dump", required=True, type=Path)
    ap.add_argument("--recovar-input-star", required=True, type=Path)
    ap.add_argument("--relion-data-star", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    rec = dict(np.load(args.recovar_dump, allow_pickle=False))
    if "half1_original_image_indices" not in rec:
        raise SystemExit(
            "dump is missing original_image_indices field — need a dump from "
            "the parity branch with the 2026-04-25 instrumentation"
        )

    inp = starfile.read(str(args.recovar_input_star))
    inp_parts = inp["particles"] if isinstance(inp, dict) else inp
    inp_imgname = inp_parts["rlnImageName"].to_numpy(dtype="U")

    rdata = starfile.read(str(args.relion_data_star))
    rparts = rdata["particles"] if isinstance(rdata, dict) else rdata
    relion_pmax = rparts["rlnMaxValueProbDistribution"].to_numpy(dtype=np.float64)
    relion_imgname = rparts["rlnImageName"].to_numpy(dtype="U")
    relion_pmax_by_name = {n: p for n, p in zip(relion_imgname, relion_pmax)}

    rows = []
    for half in (1, 2):
        rec_pmax = rec[f"half{half}_max_posterior"]
        orig_idx = rec[f"half{half}_original_image_indices"]
        names = inp_imgname[orig_idx]
        rel_pmax = np.array([relion_pmax_by_name.get(n, np.nan) for n in names])
        for i in range(len(rec_pmax)):
            rows.append((half, int(orig_idx[i]), str(names[i]), float(rec_pmax[i]), float(rel_pmax[i])))

    arr = np.asarray(
        [(h, oi, rp, rl, rl - rp) for h, oi, _, rp, rl in rows],
        dtype=[("half", "i4"), ("orig_idx", "i4"), ("rec_pmax", "f8"), ("rel_pmax", "f8"), ("deficit", "f8")],
    )
    names = [r[2] for r in rows]
    print(f"matched {len(rows)} particles")
    h1 = arr[arr["half"] == 1]
    h2 = arr[arr["half"] == 2]
    for h in (h1, h2):
        if len(h):
            label = f"half {int(h['half'][0])}"
            print(
                f"  {label}: corr={np.corrcoef(h['rec_pmax'], h['rel_pmax'])[0, 1]:.4f}  "
                f"rec_mean={h['rec_pmax'].mean():.4f}  rel_mean={h['rel_pmax'].mean():.4f}  "
                f"mean_deficit={h['deficit'].mean():+.4f}"
            )

    print(f"\nTop {args.top_n} DEFICIT particles (recovar << RELION):")
    print(f"{'half':>4} {'orig_idx':>8} {'image_name':>22} {'rec':>8} {'rel':>8} {'deficit':>8}")
    top = np.argsort(arr["deficit"])[::-1][: args.top_n]
    for i in top:
        print(
            f"{int(arr['half'][i]):>4} {int(arr['orig_idx'][i]):>8} {names[i][:22]:>22} "
            f"{arr['rec_pmax'][i]:>8.4f} {arr['rel_pmax'][i]:>8.4f} {arr['deficit'][i]:>+8.4f}"
        )

    print("\nBottom 5 deficit (recovar > RELION):")
    bot = np.argsort(arr["deficit"])[:5]
    for i in bot:
        print(
            f"{int(arr['half'][i]):>4} {int(arr['orig_idx'][i]):>8} {names[i][:22]:>22} "
            f"{arr['rec_pmax'][i]:>8.4f} {arr['rel_pmax'][i]:>8.4f} {arr['deficit'][i]:>+8.4f}"
        )


if __name__ == "__main__":
    main()
