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
    jax_path: Path,
    relion_path: Path,
    *,
    global_rotation: int,
    translations: np.ndarray,
) -> dict[str, object]:
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
    jax_gpu, jax_rotation, jax_scores = _load_pass2(
        jax_path,
        global_rotation=global_rotation,
        translations=translations,
    )
    _require(host_rotation == jax_rotation, "live rotation-local mapping changed")
    topology_fields = (
        "fine_translations",
        "oversampled_rot_indices",
        "parent_map",
        "candidate_mask",
        "rotation_log_prior",
        "translation_log_prior",
    )
    topology_exact = {
        name: bool(np.array_equal(host[name], jax_gpu[name]))
        for name in topology_fields
    }
    _require(all(topology_exact.values()), "live preprocessing changed candidate topology")

    host_all = _residual_summary(relion_scores, host_scores)
    jax_all = _residual_summary(relion_scores, jax_scores)
    host_exact = _residual_summary(relion_scores[exact_mask], host_scores[exact_mask])
    jax_exact = _residual_summary(relion_scores[exact_mask], jax_scores[exact_mask])

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
        "classification": "jax_gpu_preprocessing_reduces_centered_data_score_residual",
        "inputs": {
            "host_numpy": {
                "path": str(host_path.resolve()),
                "sha256": _sha256(host_path),
            },
            "jax_gpu": {
                "path": str(jax_path.resolve()),
                "sha256": _sha256(jax_path),
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
            "jax_gpu": jax_all,
            "jax_gpu_residual_energy_change_vs_host_numpy": energy_change(
                jax_all, host_all
            ),
        },
        "production_exact_candidates": {
            "host_numpy": host_exact,
            "jax_gpu": jax_exact,
            "jax_gpu_residual_energy_change_vs_host_numpy": energy_change(
                jax_exact, host_exact
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--jax", type=Path, required=True)
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
        args.jax,
        args.relion,
        global_rotation=args.global_rotation,
        translations=translations,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
