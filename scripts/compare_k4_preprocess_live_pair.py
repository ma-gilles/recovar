#!/usr/bin/env python3
"""Compare paired live K=4 preprocessing score captures with RELION."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .validate_relion_fine_operand_capture import (
        load_fine_operand_capture,
        validate_capture,
    )
else:
    from validate_relion_fine_operand_capture import (  # type: ignore[no-redef]
        load_fine_operand_capture,
        validate_capture,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - np.mean(array)


def _residual_summary(
    relion_scores: np.ndarray,
    recovar_scores: np.ndarray,
) -> dict[str, object]:
    relion_centered = _center(relion_scores)
    recovar_centered = _center(recovar_scores)
    delta = recovar_centered - relion_centered
    energy = float(np.vdot(delta, delta).real)
    return {
        "relion_centered": relion_centered.tolist(),
        "recovar_centered": recovar_centered.tolist(),
        "delta_recovar_minus_relion": delta.tolist(),
        "residual_l2": float(np.sqrt(energy)),
        "residual_energy": energy,
        "residual_max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _load_pass2(
    path: Path,
    *,
    global_rotation: int,
    translations: np.ndarray,
) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    _require(int(values["original_index"]) == 42987, "target original index changed")
    _require(int(values["class_index"]) == 1, "target class changed")
    _require(int(values["current_size"]) == 74, "target current size changed")
    rotation_rows = np.flatnonzero(
        np.asarray(values["oversampled_rot_indices"], dtype=np.int64)
        == global_rotation
    )
    _require(rotation_rows.size == 1, "target global rotation is not unique")
    rotation = int(rotation_rows[0])
    _require(
        np.all(values["candidate_mask"][rotation, translations]),
        "target translations left active candidate support",
    )
    scores = np.asarray(
        values["scores_pre_prior"][rotation, translations],
        dtype=np.float64,
    )
    return values, rotation, scores


def compare(
    host_path: Path,
    candidate_path: Path,
    relion_path: Path,
    *,
    candidate_label: str = "jax_gpu",
    global_rotation: int,
    translations: np.ndarray,
) -> dict[str, object]:
    _require(
        candidate_label in {"jax_gpu", "relion_cuda"},
        f"unsupported candidate backend {candidate_label!r}",
    )
    capture = load_fine_operand_capture(relion_path)
    validation = validate_capture(capture)
    _require(validation["status"] == "accepted", "RELION operand capture was rejected")
    captured_translations = np.asarray(
        [candidate["translation_id"] for candidate in validation["candidates"]],
        dtype=np.int64,
    )
    _require(
        np.array_equal(captured_translations, translations),
        "RELION translation order changed",
    )
    relion_scores = -np.asarray(
        [candidate["production_raw_diff2"] for candidate in validation["candidates"]],
        dtype=np.float64,
    )
    exact_mask = np.asarray(
        [candidate["production_replay_exact"] for candidate in validation["candidates"]],
        dtype=bool,
    )
    host, host_rotation, host_scores = _load_pass2(
        host_path,
        global_rotation=global_rotation,
        translations=translations,
    )
    candidate, candidate_rotation, candidate_scores = _load_pass2(
        candidate_path,
        global_rotation=global_rotation,
        translations=translations,
    )
    _require(host_rotation == candidate_rotation, "live rotation-local mapping changed")
    topology_fields = (
        "fine_translations",
        "oversampled_rot_indices",
        "parent_map",
        "candidate_mask",
        "rotation_log_prior",
        "translation_log_prior",
    )
    topology_exact = {
        name: bool(np.array_equal(host[name], candidate[name]))
        for name in topology_fields
    }
    _require(all(topology_exact.values()), "live preprocessing changed candidate topology")

    host_all = _residual_summary(relion_scores, host_scores)
    candidate_all = _residual_summary(relion_scores, candidate_scores)
    host_exact = _residual_summary(relion_scores[exact_mask], host_scores[exact_mask])
    candidate_exact = _residual_summary(
        relion_scores[exact_mask],
        candidate_scores[exact_mask],
    )

    def energy_change(candidate: dict[str, object], baseline: dict[str, object]) -> float:
        baseline_energy = float(baseline["residual_energy"])
        return (
            float(float(candidate["residual_energy"]) / baseline_energy - 1.0)
            if baseline_energy > 0
            else 0.0
        )

    return {
        "schema": "em_k4_it10_live_preprocess_relion_comparison_v1",
        "status": "complete",
        "classification": (
            f"{candidate_label}_preprocessing_reduces_centered_data_score_residual"
        ),
        "inputs": {
            "host_numpy": {
                "path": str(host_path.resolve()),
                "sha256": _sha256(host_path),
            },
            candidate_label: {
                "path": str(candidate_path.resolve()),
                "sha256": _sha256(candidate_path),
            },
            "relion": {
                "path": str(relion_path.resolve()),
                "sha256": _sha256(relion_path),
            },
        },
        "scope": {
            "stack_index_one_based": capture.stack_index,
            "original_index_zero_based": 42987,
            "class_one_based": capture.class_one_based,
            "global_rotation": global_rotation,
            "recovar_rotation_local": host_rotation,
            "relion_rotation_local": int(capture.candidates[0]["rotation_local"]),
            "translations": translations.tolist(),
            "relion_production_replay_exact_mask": exact_mask.tolist(),
        },
        "topology_exact": topology_exact,
        "all_candidates": {
            "host_numpy": host_all,
            candidate_label: candidate_all,
            f"{candidate_label}_residual_energy_change_vs_host_numpy": energy_change(
                candidate_all, host_all
            ),
        },
        "production_exact_candidates": {
            "host_numpy": host_exact,
            candidate_label: candidate_exact,
            f"{candidate_label}_residual_energy_change_vs_host_numpy": energy_change(
                candidate_exact, host_exact
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    candidate_group = parser.add_mutually_exclusive_group(required=True)
    candidate_group.add_argument("--candidate", type=Path)
    candidate_group.add_argument("--jax", type=Path)
    parser.add_argument(
        "--candidate-label",
        choices=("jax_gpu", "relion_cuda"),
        default="jax_gpu",
    )
    parser.add_argument("--relion", type=Path, required=True)
    parser.add_argument("--global-rotation", type=int, required=True)
    parser.add_argument("--translations", default="56,57,58,59")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    translations = np.asarray(
        [int(value) for value in args.translations.split(",")],
        dtype=np.int64,
    )
    report = compare(
        args.host,
        args.candidate if args.candidate is not None else args.jax,
        args.relion,
        candidate_label=args.candidate_label,
        global_rotation=args.global_rotation,
        translations=translations,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
