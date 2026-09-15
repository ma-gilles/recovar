#!/usr/bin/env python3
"""Compare native RELION accelerator fine-pass weights with RECOVAR dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_meta(path: Path) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        result[key] = int(value) if key in {"part_id", "iter", "weight_count"} else float(value)
    return result


def _read_relion(prefix: Path) -> dict[str, np.ndarray | dict[str, float | int]]:
    with prefix.with_name(prefix.name + "_raw_diff2.bin").open("rb") as stream:
        dims = np.fromfile(stream, dtype=np.int64, count=7)
        raw_diff2 = np.fromfile(stream, dtype=np.float64)
    count = int(dims[0])
    if raw_diff2.size != count:
        raise ValueError(f"{prefix}: raw diff2 count {raw_diff2.size} != {count}")

    with prefix.with_name(prefix.name + "_indices.bin").open("rb") as stream:
        index_dims = np.fromfile(stream, dtype=np.int64, count=7)
        indices = np.fromfile(stream, dtype=np.uint64)
    if not np.array_equal(index_dims, dims) or indices.size != 4 * count:
        raise ValueError(f"{prefix}: malformed compact-index capture")
    rot_id, rot_idx, trans_idx, ihidden_overs = indices.reshape(4, count)

    with prefix.with_name(prefix.name + "_weights.bin").open("rb") as stream:
        weight_count = int(np.fromfile(stream, dtype=np.int64, count=1)[0])
        weights = np.fromfile(stream, dtype=np.float64)
    if weight_count != count or weights.size != count:
        raise ValueError(f"{prefix}: posterior weight count differs from raw diff2")

    return {
        "dims": dims,
        "raw_diff2": raw_diff2,
        "weights": weights,
        "rot_id": rot_id,
        "rot_idx": rot_idx,
        "trans_idx": trans_idx,
        "ihidden_overs": ihidden_overs,
        "meta": _read_meta(prefix.with_name(prefix.name + "_meta.txt")),
    }


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), np.finfo(np.float64).tiny))


def _compare_one(recovar_path: Path, relion_prefix: Path) -> dict[str, object]:
    rec = np.load(recovar_path)
    rel = _read_relion(relion_prefix)
    meta = rel["meta"]
    assert isinstance(meta, dict)

    candidate_mask = np.asarray(rec["candidate_mask"], dtype=bool)
    shape = candidate_mask.shape
    rot_idx = np.asarray(rel["rot_idx"], dtype=np.int64)
    trans_idx = np.asarray(rel["trans_idx"], dtype=np.int64)
    if np.any(rot_idx >= shape[0]) or np.any(trans_idx >= shape[1]):
        raise ValueError(
            f"{relion_prefix}: compact index exceeds RECOVAR grid {shape}; "
            f"rot_max={rot_idx.max(initial=-1)}, trans_max={trans_idx.max(initial=-1)}"
        )
    linear = np.ravel_multi_index((rot_idx, trans_idx), shape)
    if np.unique(linear).size != linear.size:
        raise ValueError(f"{relion_prefix}: duplicate compact fine-pose indices")

    rel_candidate_mask = np.zeros(shape, dtype=bool)
    rel_candidate_mask[rot_idx, trans_idx] = True
    rel_weights = np.asarray(rel["weights"], dtype=np.float64)
    rel_probs = rel_weights / float(meta["sum_weight"])
    rel_probs_grid = np.zeros(shape, dtype=np.float64)
    rel_probs_grid[rot_idx, trans_idx] = rel_probs
    rec_probs = np.asarray(rec["probs"], dtype=np.float64)

    rel_reconstruction_mask = np.zeros(shape, dtype=bool)
    rel_reconstruction_mask[rot_idx, trans_idx] = (
        rel_weights >= float(meta["significant_weight"])
    ) & (rel_weights > 0.0)
    rec_reconstruction_mask = np.asarray(rec["reconstruction_mask"], dtype=bool)
    rec_reconstruction_probs = np.asarray(rec["reconstruction_probs"], dtype=np.float64)

    rel_raw = np.asarray(rel["raw_diff2"], dtype=np.float64)
    rec_raw_grid = (
        np.asarray(rec["relion_min_diff2"], dtype=np.float64)
        - np.asarray(rec["scores_pre_prior"], dtype=np.float64)
    )
    rec_raw = rec_raw_grid[rot_idx, trans_idx]
    raw_delta = rec_raw - rel_raw
    probability_delta = rec_probs - rel_probs_grid
    reconstruction_delta = rec_reconstruction_probs - np.where(
        rel_reconstruction_mask,
        rel_probs_grid,
        0.0,
    )

    return {
        "particle": int(rec["original_index"]),
        "half": int(rec["half"]),
        "candidate_count": int(candidate_mask.sum()),
        "relion_compact_count": int(linear.size),
        "candidate_mask_mismatch_count": int(np.count_nonzero(candidate_mask != rel_candidate_mask)),
        "reconstruction_mask_mismatch_count": int(
            np.count_nonzero(rec_reconstruction_mask != rel_reconstruction_mask)
        ),
        "recovar_reconstruction_count": int(rec_reconstruction_mask.sum()),
        "relion_reconstruction_count": int(rel_reconstruction_mask.sum()),
        "posterior": {
            "relative_l2": _relative_l2(rec_probs, rel_probs_grid),
            "max_absolute": float(np.max(np.abs(probability_delta))),
            "recovar_max": float(rec_probs.max()),
            "relion_max": float(rel_probs_grid.max()),
            "recovar_sum": float(rec_probs.sum()),
            "relion_sum": float(rel_probs_grid.sum()),
        },
        "reconstruction_posterior": {
            "relative_l2": _relative_l2(
                rec_reconstruction_probs,
                np.where(rel_reconstruction_mask, rel_probs_grid, 0.0),
            ),
            "max_absolute": float(np.max(np.abs(reconstruction_delta))),
            "recovar_mass": float(rec_reconstruction_probs.sum()),
            "relion_mass": float(rel_probs_grid[rel_reconstruction_mask].sum()),
        },
        "raw_diff2_matched": {
            "relative_l2": _relative_l2(rec_raw, rel_raw),
            "delta_mean": float(raw_delta.mean()),
            "delta_std": float(raw_delta.std()),
            "max_absolute": float(np.max(np.abs(raw_delta))),
            "max_absolute_after_mean": float(np.max(np.abs(raw_delta - raw_delta.mean()))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-dir", type=Path, required=True)
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for recovar_path in sorted(args.recovar_dir.glob("pass2_orig*_cs*.npz")):
        particle = int(recovar_path.name.split("orig", 1)[1].split("_", 1)[0])
        prefix = args.relion_dir / f"pass2_part{particle}"
        reports.append(_compare_one(recovar_path, prefix))
    if not reports:
        raise ValueError(f"no RECOVAR pass-2 dumps found in {args.recovar_dir}")

    payload = {"schema": "recovar.em.relion_pass2_panel.v1", "particles": reports}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
