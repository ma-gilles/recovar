#!/usr/bin/env python3
"""Locate a VDAM fine-posterior mismatch at exact RELION float32 stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar import cuda_backproject  # noqa: E402
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (  # noqa: E402
    _relion_f32_fine_posterior,
)
from scripts.analyze_em_k1_native_fine_operands import _flat_memmap  # noqa: E402
from scripts.analyze_vdam_storewavg_boundary import _scalar  # noqa: E402
from scripts.compare_relion_recovar_estep_dump import (  # noqa: E402
    _nearest_rotation_rows_by_matrix,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool]:
    """Report exact and norm errors without hiding float32 bit differences."""

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError(
            f"posterior-stage metric requires aligned nonempty arrays: "
            f"{reference.shape} != {candidate.shape}"
        )
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    residual = candidate64 - reference64
    denominator = float(np.linalg.norm(reference64.reshape(-1)))
    return {
        "bitwise_equal": bool(np.array_equal(reference, candidate)),
        "exact_count": int(np.count_nonzero(reference == candidate)),
        "value_count": int(reference.size),
        "relative_l2": (
            float(np.linalg.norm(residual.reshape(-1)) / denominator)
            if denominator
            else float(np.linalg.norm(residual.reshape(-1)))
        ),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _scalar_metric(reference: float, candidate: float) -> dict[str, float | int | bool]:
    reference32 = np.asarray(reference, dtype=np.float32)
    candidate32 = np.asarray(candidate, dtype=np.float32)
    return {
        "bitwise_equal": bool(reference32.view(np.uint32) == candidate32.view(np.uint32)),
        "reference": float(reference32),
        "candidate": float(candidate32),
        "signed_error": float(candidate32.astype(np.float64) - reference32.astype(np.float64)),
        "reference_bits": int(reference32.view(np.uint32)),
        "candidate_bits": int(candidate32.view(np.uint32)),
    }


def _target_row(payload: dict[str, np.ndarray], original_index: int) -> int:
    matches = np.flatnonzero(
        np.asarray(payload["original_indices"], dtype=np.int64) == int(original_index)
    )
    if matches.size != 1:
        raise ValueError(
            f"expected one contribution row for original index {original_index}, "
            f"found {matches.size}"
        )
    return int(matches[0])


def _candidate_rotation_matrices(
    payload: dict[str, np.ndarray],
    *,
    particle_row: int,
) -> np.ndarray:
    actual_count = int(np.asarray(payload["actual_counts"])[particle_row])
    particle_rows = np.asarray(payload["active_particle_rows"], dtype=np.int64)
    selected = particle_rows == int(particle_row)
    local_rows = np.asarray(payload["active_rotation_rows"], dtype=np.int64)[selected]
    rotations = np.asarray(payload["active_rotations"], dtype=np.float32)[selected]
    if local_rows.size != actual_count or not np.array_equal(
        np.sort(local_rows), np.arange(actual_count, dtype=np.int64)
    ):
        raise ValueError("contribution capture does not contain one matrix per active rotation row")
    result = np.empty((actual_count, 3, 3), dtype=np.float32)
    result[local_rows] = rotations
    return result


def _map_native_candidates(
    *,
    native_rotation_ids: np.ndarray,
    native_translation_ids: np.ndarray,
    native_rotation_matrices: np.ndarray,
    payload: dict[str, np.ndarray],
    particle_row: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    candidate_rotations = _candidate_rotation_matrices(
        payload,
        particle_row=particle_row,
    )
    nearest, distance, orientation = _nearest_rotation_rows_by_matrix(
        np.asarray(native_rotation_matrices, dtype=np.float64),
        np.asarray(candidate_rotations, dtype=np.float64),
    )
    native_rotation_ids = np.asarray(native_rotation_ids, dtype=np.int64).reshape(-1)
    native_translation_ids = np.asarray(native_translation_ids, dtype=np.int64).reshape(-1)
    if native_rotation_ids.shape != native_translation_ids.shape:
        raise ValueError("native rotation and translation candidate vectors differ in shape")
    if np.any(native_rotation_ids < 0) or np.any(native_rotation_ids >= nearest.size):
        raise ValueError("native rotation id lies outside captured RELION Euler rows")
    mapped_rotation_ids = nearest[native_rotation_ids]
    n_translations = int(np.asarray(payload["fine_translations"]).shape[0])
    if np.any(native_translation_ids < 0) or np.any(native_translation_ids >= n_translations):
        raise ValueError("native translation id lies outside candidate translation rows")
    keys = np.stack((mapped_rotation_ids, native_translation_ids), axis=1)
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise ValueError("native candidate mapping produced duplicate RECOVAR keys")
    candidate_mask = np.asarray(payload["candidate_mask"], dtype=bool)[particle_row]
    if not np.all(candidate_mask[mapped_rotation_ids, native_translation_ids]):
        raise ValueError("native support maps outside RECOVAR's finite candidate mask")
    if int(np.count_nonzero(candidate_mask)) != int(keys.shape[0]):
        raise ValueError("native and RECOVAR fine candidate supports differ in size")
    return mapped_rotation_ids, native_translation_ids, {
        "rotation_matrix_orientation": orientation,
        "rotation_matrix_max_frobenius": float(np.max(distance)),
        "rotation_matrix_median_frobenius": float(np.median(distance)),
    }


def _replay(scores: np.ndarray, *, adaptive_fraction: float) -> dict[str, np.ndarray]:
    """Run the exact production CUDA exp/sort/scan/divide posterior stages."""

    scores32 = np.asarray(scores, dtype=np.float32)
    if scores32.ndim != 2 or scores32.shape[0] != 1:
        raise ValueError(f"posterior replay expects shape (1,N), got {scores32.shape}")
    scores_jax = jnp.asarray(scores32)
    finite = jnp.isfinite(scores_jax)
    best = jnp.max(jnp.where(finite, scores_jax, -jnp.inf), axis=1)
    exponent_add = jnp.float32(50.0) - best
    raw = jax.vmap(cuda_backproject.relion_exponentiate_f32)(
        jnp.where(finite, scores_jax, -jnp.inf),
        exponent_add,
    )
    sorted_weights, cumulative = jax.vmap(
        cuda_backproject.relion_cub_sort_scan_f32
    )(raw)
    (
        normalized,
        reconstruction,
        reconstruction_mask,
        n_significant,
        sum_weight,
        threshold,
    ) = _relion_f32_fine_posterior(
        scores_jax,
        adaptive_fraction=float(adaptive_fraction),
    )
    values = jax.block_until_ready(
        (
            best,
            exponent_add,
            raw,
            sorted_weights,
            cumulative,
            normalized,
            reconstruction,
            reconstruction_mask,
            n_significant,
            sum_weight,
            threshold,
        )
    )
    names = (
        "best",
        "exponent_add",
        "raw",
        "sorted",
        "cumulative",
        "normalized",
        "reconstruction",
        "reconstruction_mask",
        "n_significant",
        "sum_weight",
        "threshold",
    )
    return {name: np.asarray(value) for name, value in zip(names, values, strict=True)}


def analyze(
    native_dir: Path,
    contribution_path: Path,
    *,
    original_index: int,
) -> dict[str, object]:
    native_dir = Path(native_dir)
    contribution_path = Path(contribution_path)
    if jax.default_backend() != "gpu" or not cuda_backproject.custom_cuda_requested():
        raise RuntimeError("exact VDAM posterior audit requires the custom CUDA backend")
    with np.load(contribution_path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    particle_row = _target_row(payload, original_index)

    native_rotation_ids = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_rot_idx.bin", np.int32),
        dtype=np.int32,
    )
    native_translation_ids = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_trans_idx.bin", np.int32),
        dtype=np.int32,
    )
    native_rotation_matrices = np.asarray(
        _flat_memmap(native_dir / "pass1_class0_fine_eulers.bin"),
        dtype=np.float64,
    ).reshape(-1, 3, 3)
    mapped_rotations, mapped_translations, rotation_report = _map_native_candidates(
        native_rotation_ids=native_rotation_ids,
        native_translation_ids=native_translation_ids,
        native_rotation_matrices=native_rotation_matrices,
        payload=payload,
        particle_row=particle_row,
    )
    native_weights = np.asarray(
        _flat_memmap(native_dir / "pass1_exp_Mweight_posterior.bin"),
        dtype=np.float32,
    )
    native_sum_weight = np.float32(_scalar(native_dir / "pass1_exp_sum_weight.bin"))
    native_threshold = np.float32(_scalar(native_dir / "pass1_exp_significant_weight.bin"))
    native_mask = np.asarray(
        _flat_memmap(
            native_dir / "pass1_candidate_in_reconstruction_set.bin",
            np.int32,
        ),
        dtype=bool,
    )
    candidate_count = int(native_weights.size)
    if not (
        native_rotation_ids.size
        == native_translation_ids.size
        == native_mask.size
        == candidate_count
    ):
        raise ValueError("native posterior capture arrays differ in size")

    combined_scores = np.asarray(payload["candidate_combined_scores"])[particle_row]
    candidate_mask = np.asarray(payload["candidate_mask"], dtype=bool)[particle_row]
    compact_scores = np.asarray(
        combined_scores[mapped_rotations, mapped_translations],
        dtype=np.float32,
    )[None, :]
    dense_scores = np.where(candidate_mask, combined_scores, -np.inf).astype(np.float32)[
        None, ...
    ]
    dense_shape = dense_scores.shape
    dense_replay = _replay(dense_scores.reshape(1, -1), adaptive_fraction=0.999)
    compact_replay = _replay(compact_scores, adaptive_fraction=0.999)
    dense_indices = mapped_rotations * combined_scores.shape[1] + mapped_translations

    native_normalized = np.asarray(native_weights / native_sum_weight, dtype=np.float32)
    production_reconstruction = np.asarray(payload["reconstruction_probs"])[particle_row][
        mapped_rotations,
        mapped_translations,
    ]
    production_mask = np.asarray(payload["reconstruction_mask"], dtype=bool)[particle_row][
        mapped_rotations,
        mapped_translations,
    ]

    native_log_path = native_dir / "pass1_fine_log_weight_preexp.bin"
    log_weight_comparison: dict[str, object]
    if native_log_path.is_file():
        native_log_weights = np.asarray(_flat_memmap(native_log_path), dtype=np.float32)
        if native_log_weights.shape != compact_scores.reshape(-1).shape:
            raise ValueError("native pre-exp log-weight capture has an unexpected shape")
        candidate_log_weights = compact_scores.reshape(-1)
        native_centered = native_log_weights.astype(np.float64) - float(
            np.mean(native_log_weights, dtype=np.float64)
        )
        candidate_centered = candidate_log_weights.astype(np.float64) - float(
            np.mean(candidate_log_weights, dtype=np.float64)
        )
        log_weight_comparison = {
            "status": "captured",
            # RELION adds op.min_diff2 before exponentiation; RECOVAR's
            # combined scores omit that candidate-independent constant.  The
            # later max shift cancels it, so only centered spacing is causal.
            "candidate_minus_native_common_offset": float(
                np.mean(
                    candidate_log_weights.astype(np.float64)
                    - native_log_weights.astype(np.float64),
                    dtype=np.float64,
                )
            ),
            "candidate_compact_centered_vs_native": _metric(
                native_centered,
                candidate_centered,
            ),
        }
    else:
        log_weight_comparison = {"status": "native_capture_missing"}

    dense_raw_on_native = dense_replay["raw"].reshape(dense_shape)[
        0,
        mapped_rotations,
        mapped_translations,
    ]
    dense_normalized_on_native = dense_replay["normalized"].reshape(dense_shape)[
        0,
        mapped_rotations,
        mapped_translations,
    ]
    dense_reconstruction_on_native = dense_replay["reconstruction"].reshape(dense_shape)[
        0,
        mapped_rotations,
        mapped_translations,
    ]
    dense_mask_on_native = dense_replay["reconstruction_mask"].reshape(dense_shape)[
        0,
        mapped_rotations,
        mapped_translations,
    ]

    return {
        "schema": "recovar.vdam_f32_posterior_boundary.v1",
        "status": "complete",
        "device": str(jax.devices()[0]),
        "identity": {
            "original_index": int(original_index),
            "particle_row": particle_row,
            "candidate_count": candidate_count,
            "dense_slot_count": int(np.prod(dense_shape[1:])),
            **rotation_report,
        },
        "comparisons": {
            "log_weights": log_weight_comparison,
            "current_production_reconstruction_vs_native": _metric(
                native_normalized * native_mask,
                production_reconstruction,
            ),
            "current_production_mask_vs_native": {
                "equal": bool(np.array_equal(native_mask, production_mask)),
                "mismatch_count": int(np.count_nonzero(native_mask != production_mask)),
            },
            "compact_native_order_raw_exp_vs_native": _metric(
                native_weights,
                compact_replay["raw"].reshape(-1),
            ),
            "dense_padded_raw_exp_on_native_vs_native": _metric(
                native_weights,
                dense_raw_on_native,
            ),
            "compact_native_order_sum_weight_vs_native": _scalar_metric(
                native_sum_weight,
                compact_replay["sum_weight"].reshape(-1)[0],
            ),
            "dense_padded_sum_weight_vs_native": _scalar_metric(
                native_sum_weight,
                dense_replay["sum_weight"].reshape(-1)[0],
            ),
            "compact_native_order_threshold_vs_native": _scalar_metric(
                native_threshold,
                compact_replay["threshold"].reshape(-1)[0],
            ),
            "dense_padded_threshold_vs_native": _scalar_metric(
                native_threshold,
                dense_replay["threshold"].reshape(-1)[0],
            ),
            "compact_native_order_normalized_vs_native": _metric(
                native_normalized,
                compact_replay["normalized"].reshape(-1),
            ),
            "dense_padded_normalized_on_native_vs_native": _metric(
                native_normalized,
                dense_normalized_on_native,
            ),
            "compact_native_order_reconstruction_vs_native": _metric(
                native_normalized * native_mask,
                compact_replay["reconstruction"].reshape(-1),
            ),
            "dense_padded_reconstruction_on_native_vs_native": _metric(
                native_normalized * native_mask,
                dense_reconstruction_on_native,
            ),
            "compact_native_order_mask_vs_native": {
                "equal": bool(
                    np.array_equal(
                        native_mask,
                        compact_replay["reconstruction_mask"].reshape(-1),
                    )
                ),
                "mismatch_count": int(
                    np.count_nonzero(
                        native_mask
                        != compact_replay["reconstruction_mask"].reshape(-1)
                    )
                ),
            },
            "dense_padded_mask_on_native_vs_native": {
                "equal": bool(np.array_equal(native_mask, dense_mask_on_native)),
                "mismatch_count": int(np.count_nonzero(native_mask != dense_mask_on_native)),
            },
        },
        "artifacts": {
            "native_directory": str(native_dir.resolve()),
            "contribution": str(contribution_path.resolve()),
            "contribution_sha256": _sha256(contribution_path),
            "native_log_weight_capture": (
                str(native_log_path.resolve()) if native_log_path.is_file() else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--contribution", type=Path, required=True)
    parser.add_argument("--original-index", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.contribution,
        original_index=args.original_index,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
