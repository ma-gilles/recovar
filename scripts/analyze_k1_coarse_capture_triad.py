#!/usr/bin/env python3
"""Attribute one K=1 coarse-support crossing across exact/map/live captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

SCORE_FIELDS = (
    "scores_pre_prior_per_class",
    "scores_with_prior_per_class",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinate(flat_index: int, n_trans: int) -> tuple[int, int]:
    return divmod(int(flat_index), int(n_trans))


def _stable_descending(values: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(values), kind="stable")


def _arm_summary(
    payload: Mapping[str, np.ndarray],
    *,
    reference_top_count: int,
    tracked_flat_indices: Mapping[str, int],
) -> dict[str, object]:
    weights = np.asarray(payload["weights_full"], dtype=np.float64).reshape(-1)
    order = _stable_descending(weights)
    rank_by_flat = np.empty(order.size, dtype=np.int64)
    rank_by_flat[order] = np.arange(1, order.size + 1, dtype=np.int64)
    mask = np.asarray(payload["significant_mask"], dtype=bool).reshape(-1)
    own_count = int(np.asarray(payload["n_significant"]).item())
    top_reference = order[:reference_top_count]
    return {
        "pmax": float(np.asarray(payload["max_posterior"]).item()),
        "n_significant": own_count,
        "hard_assignment": int(np.asarray(payload["hard_assignment"]).item()),
        "own_significant_mass": float(weights[mask].sum(dtype=np.float64)),
        "own_top_n_mass": float(weights[order[:own_count]].sum(dtype=np.float64)),
        "top_reference_count_mass": float(weights[top_reference].sum(dtype=np.float64)),
        "tracked": {
            name: {
                "flat_index": int(flat_index),
                "rank": int(rank_by_flat[flat_index]),
                "posterior": float(weights[flat_index]),
                "selected": bool(mask[flat_index]),
            }
            for name, flat_index in tracked_flat_indices.items()
        },
    }


def _score_components(
    payload: Mapping[str, np.ndarray],
    *,
    flat_index: int,
    anchor_flat_index: int,
) -> dict[str, float]:
    raw = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64).reshape(-1)
    total = np.asarray(payload["scores_with_prior_per_class"], dtype=np.float64).reshape(-1)
    prior = total - raw
    return {
        "raw": float(raw[flat_index]),
        "prior": float(prior[flat_index]),
        "total": float(total[flat_index]),
        "raw_margin_to_exact_winner": float(raw[flat_index] - raw[anchor_flat_index]),
        "prior_margin_to_exact_winner": float(prior[flat_index] - prior[anchor_flat_index]),
        "total_margin_to_exact_winner": float(total[flat_index] - total[anchor_flat_index]),
    }


def _subtract_components(
    lhs: Mapping[str, float], rhs: Mapping[str, float]
) -> dict[str, float]:
    return {name: float(rhs[name] - lhs[name]) for name in lhs}


def _centered_score_metrics(
    control: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray], field: str
) -> dict[str, float]:
    lhs = np.asarray(control[field], dtype=np.float32).reshape(-1)
    rhs = np.asarray(candidate[field], dtype=np.float32).reshape(-1)
    finite = np.isfinite(lhs) & np.isfinite(rhs)
    if not np.any(finite):
        raise ValueError(f"no finite values in {field}")
    lhs_centered = lhs[finite] - np.max(lhs[finite])
    rhs_centered = rhs[finite] - np.max(rhs[finite])
    delta = rhs_centered.astype(np.float64) - lhs_centered.astype(np.float64)
    denominator = np.linalg.norm(lhs_centered.astype(np.float64))
    return {
        "max_abs_delta": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / denominator) if denominator else 0.0,
        "unequal_count": int(np.count_nonzero(lhs_centered != rhs_centered)),
    }


def analyze(*, exact_path: Path, map_path: Path, live_path: Path) -> dict[str, object]:
    paths = {"exact": exact_path, "map_only": map_path, "live": live_path}
    with (
        np.load(exact_path, allow_pickle=False) as exact,
        np.load(map_path, allow_pickle=False) as map_only,
        np.load(live_path, allow_pickle=False) as live,
    ):
        arms = {"exact": exact, "map_only": map_only, "live": live}
        scalar_fields = (
            "original_index",
            "current_size",
            "n_classes",
            "n_rot",
            "n_trans",
            "adaptive_fraction",
            "max_significants",
        )
        for field in scalar_fields:
            values = [np.asarray(payload[field]).item() for payload in arms.values()]
            if values[1:] != values[:-1]:
                raise ValueError(f"capture scalar {field} differs: {values}")
        for field in ("rotations", "translations", "class_indices", "rot_indices", "trans_indices"):
            if not all(np.array_equal(exact[field], payload[field]) for payload in (map_only, live)):
                raise ValueError(f"capture topology {field} differs")

        n_trans = int(np.asarray(exact["n_trans"]).item())
        exact_mask = np.asarray(exact["significant_mask"], dtype=bool).reshape(-1)
        map_mask = np.asarray(map_only["significant_mask"], dtype=bool).reshape(-1)
        live_mask = np.asarray(live["significant_mask"], dtype=bool).reshape(-1)
        exact_weights = np.asarray(exact["weights_full"], dtype=np.float64).reshape(-1)
        exact_order = _stable_descending(exact_weights)
        exact_count = int(np.asarray(exact["n_significant"]).item())
        exact_winner = int(np.asarray(exact["hard_assignment"]).item())
        exact_cutoff = int(exact_order[exact_count - 1])
        live_added = np.flatnonzero(live_mask & ~exact_mask)
        map_added = np.flatnonzero(map_mask & ~exact_mask)
        if live_added.size == 0:
            raise ValueError("live capture does not add a coarse-support hypothesis")
        live_weights = np.asarray(live["weights_full"], dtype=np.float64).reshape(-1)
        crossing = int(live_added[np.argmax(live_weights[live_added])])
        tracked = {
            "exact_winner": exact_winner,
            "exact_cutoff": exact_cutoff,
            "crossing_hypothesis": crossing,
        }

        summaries = {
            name: _arm_summary(
                payload,
                reference_top_count=exact_count,
                tracked_flat_indices=tracked,
            )
            for name, payload in arms.items()
        }
        components = {
            tracked_name: {
                arm_name: _score_components(
                    payload,
                    flat_index=flat_index,
                    anchor_flat_index=exact_winner,
                )
                for arm_name, payload in arms.items()
            }
            for tracked_name, flat_index in tracked.items()
        }
        component_deltas = {
            tracked_name: {
                "map_only_minus_exact": _subtract_components(values["exact"], values["map_only"]),
                "live_minus_map_only": _subtract_components(values["map_only"], values["live"]),
                "live_minus_exact": _subtract_components(values["exact"], values["live"]),
            }
            for tracked_name, values in components.items()
        }
        centered_score_metrics = {
            field: {
                "map_only_vs_exact": _centered_score_metrics(exact, map_only, field),
                "live_vs_map_only": _centered_score_metrics(map_only, live, field),
                "live_vs_exact": _centered_score_metrics(exact, live, field),
            }
            for field in SCORE_FIELDS
        }

        support = {
            "map_only_vs_exact_symmetric_difference": int(np.count_nonzero(map_mask != exact_mask)),
            "live_vs_exact_symmetric_difference": int(np.count_nonzero(live_mask != exact_mask)),
            "map_only_added_flat_indices": [int(value) for value in map_added],
            "live_added_flat_indices": [int(value) for value in live_added],
            "live_removed_flat_indices": [
                int(value) for value in np.flatnonzero(exact_mask & ~live_mask)
            ],
        }

    return {
        "schema": "recovar.em.k1_coarse_capture_triad.v1",
        "status": "complete",
        "coordinates": {
            name: {
                "flat_index": int(flat_index),
                "rotation": _coordinate(flat_index, n_trans)[0],
                "translation": _coordinate(flat_index, n_trans)[1],
            }
            for name, flat_index in tracked.items()
        },
        "support": support,
        "arms": summaries,
        "score_components": components,
        "score_component_deltas": component_deltas,
        "centered_score_metrics": centered_score_metrics,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--map-only", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(exact_path=args.exact, map_path=args.map_only, live_path=args.live)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
