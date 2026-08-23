#!/usr/bin/env python3
"""Compare one native RELION fine posterior with RECOVAR's fused local path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_em_k1_native_fine_operands import _flat_memmap  # noqa: E402
from scripts.analyze_vdam_storewavg_boundary import _scalar  # noqa: E402
from scripts.compare_relion_recovar_estep_dump import (  # noqa: E402
    _nearest_rotation_rows_by_matrix,
)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / denominator) if denominator else 0.0


def compare_posteriors(
    *,
    native_rotation_ids: np.ndarray,
    native_translation_ids: np.ndarray,
    native_rotation_matrices: np.ndarray,
    native_unnormalized_weights: np.ndarray,
    native_sum_weight: float,
    native_reconstruction_mask: np.ndarray,
    live: dict[str, np.ndarray],
) -> dict[str, object]:
    """Compare mapped candidate probabilities without assuming matching row order."""

    native_rotation_ids = np.asarray(native_rotation_ids, dtype=np.int64).reshape(-1)
    native_translation_ids = np.asarray(native_translation_ids, dtype=np.int64).reshape(-1)
    native_weights = np.asarray(native_unnormalized_weights, dtype=np.float64).reshape(-1)
    native_reconstruction_mask = np.asarray(native_reconstruction_mask, dtype=bool).reshape(-1)
    candidate_count = int(native_rotation_ids.size)
    if not (
        native_translation_ids.size
        == native_weights.size
        == native_reconstruction_mask.size
        == candidate_count
    ):
        raise ValueError("native fine-posterior arrays have inconsistent sizes")
    if not np.isfinite(native_sum_weight) or native_sum_weight <= 0.0:
        raise ValueError("native fine-posterior sum weight must be positive and finite")

    live_rotations = np.asarray(live["local_rotation_matrices"], dtype=np.float64)
    nearest, rotation_distance, orientation = _nearest_rotation_rows_by_matrix(
        np.asarray(native_rotation_matrices, dtype=np.float64),
        live_rotations,
    )
    if np.any(native_rotation_ids < 0) or np.any(native_rotation_ids >= nearest.size):
        raise ValueError("native candidate rotation id is outside the captured rotation table")

    posterior = np.asarray(live["posterior"], dtype=np.float64)
    if posterior.ndim == 3 and posterior.shape[0] == 1:
        posterior = posterior[0]
    if posterior.ndim != 2:
        raise ValueError(f"RECOVAR fused posterior must be 2D after unbatching, got {posterior.shape}")
    mapped_rotations = nearest[native_rotation_ids]
    if np.any(native_translation_ids < 0) or np.any(native_translation_ids >= posterior.shape[1]):
        raise ValueError("native candidate translation id is outside the RECOVAR table")
    recovar_on_native = posterior[mapped_rotations, native_translation_ids]
    native_posterior = native_weights / float(native_sum_weight)

    native_best = int(np.argmax(native_posterior))
    live_best_flat = int(np.argmax(posterior))
    live_best = tuple(int(value) for value in np.unravel_index(live_best_flat, posterior.shape))
    mapped_native_best = (
        int(mapped_rotations[native_best]),
        int(native_translation_ids[native_best]),
    )

    def _ranked_rows(values: np.ndarray, count: int = 5) -> list[dict[str, object]]:
        rows = np.argsort(np.asarray(values, dtype=np.float64))[-int(count) :][::-1]
        return [
            {
                "native_candidate_row": int(row),
                "mapped_key": [
                    int(mapped_rotations[row]),
                    int(native_translation_ids[row]),
                ],
                "native_probability": float(native_posterior[row]),
                "recovar_probability": float(recovar_on_native[row]),
                "signed_recovar_minus_native": float(
                    recovar_on_native[row] - native_posterior[row]
                ),
            }
            for row in rows
        ]

    recovar_reconstruction_mask = np.asarray(
        live.get("reconstruction_sample_mask", np.zeros((1,) + posterior.shape, dtype=bool)),
        dtype=bool,
    )
    if recovar_reconstruction_mask.ndim == 3 and recovar_reconstruction_mask.shape[0] == 1:
        recovar_reconstruction_mask = recovar_reconstruction_mask[0]
    if recovar_reconstruction_mask.shape != posterior.shape:
        raise ValueError("RECOVAR reconstruction mask differs from posterior shape")
    recovar_reconstruction_on_native = recovar_reconstruction_mask[
        mapped_rotations,
        native_translation_ids,
    ]

    return {
        "candidate_count": candidate_count,
        "rotation_matrix_orientation": orientation,
        "rotation_matrix_max_frobenius": float(np.max(rotation_distance)),
        "rotation_matrix_median_frobenius": float(np.median(rotation_distance)),
        "native_probability_sum": float(np.sum(native_posterior, dtype=np.float64)),
        "recovar_probability_sum": float(np.sum(posterior, dtype=np.float64)),
        "recovar_probability_on_native_support_sum": float(
            np.sum(recovar_on_native, dtype=np.float64)
        ),
        "probability_l1": float(
            np.sum(np.abs(recovar_on_native - native_posterior), dtype=np.float64)
        ),
        "probability_relative_l2": _relative_l2(native_posterior, recovar_on_native),
        "probability_max_abs": float(np.max(np.abs(recovar_on_native - native_posterior))),
        "probability_exact_count": int(np.count_nonzero(recovar_on_native == native_posterior)),
        "native_pmax": float(native_posterior[native_best]),
        "recovar_pmax": float(np.max(posterior)),
        "native_best_mapped_key": list(mapped_native_best),
        "recovar_best_key": list(live_best),
        "argmax_equal": bool(mapped_native_best == live_best),
        "native_top_candidates": _ranked_rows(native_posterior),
        "recovar_top_candidates_on_native_support": _ranked_rows(recovar_on_native),
        "largest_probability_residuals": _ranked_rows(
            np.abs(recovar_on_native - native_posterior)
        ),
        "native_reconstruction_count": int(np.count_nonzero(native_reconstruction_mask)),
        "recovar_reconstruction_count": int(np.count_nonzero(recovar_reconstruction_mask)),
        "reconstruction_mask_on_native_equal": bool(
            np.array_equal(native_reconstruction_mask, recovar_reconstruction_on_native)
        ),
        "reconstruction_mask_on_native_mismatch_count": int(
            np.count_nonzero(native_reconstruction_mask != recovar_reconstruction_on_native)
        ),
    }


def analyze(native_dir: Path, recovar_fused_posterior: Path) -> dict[str, object]:
    native_dir = Path(native_dir)
    recovar_fused_posterior = Path(recovar_fused_posterior)
    with np.load(recovar_fused_posterior, allow_pickle=False) as payload:
        live = {name: np.asarray(payload[name]) for name in payload.files}

    native_rotation_ids = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_rot_idx.bin", np.int32),
        dtype=np.int32,
    )
    native_translation_ids = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_trans_idx.bin", np.int32),
        dtype=np.int32,
    )
    native_weights = np.asarray(
        _flat_memmap(native_dir / "pass1_exp_Mweight_posterior.bin"),
        dtype=np.float64,
    )
    native_reconstruction_mask = np.asarray(
        _flat_memmap(native_dir / "pass1_candidate_in_reconstruction_set.bin", np.int32),
        dtype=bool,
    )
    native_rotation_matrices = np.asarray(
        _flat_memmap(native_dir / "pass1_class0_fine_eulers.bin"),
        dtype=np.float64,
    ).reshape(-1, 3, 3)
    comparison = compare_posteriors(
        native_rotation_ids=native_rotation_ids,
        native_translation_ids=native_translation_ids,
        native_rotation_matrices=native_rotation_matrices,
        native_unnormalized_weights=native_weights,
        native_sum_weight=float(_scalar(native_dir / "pass1_exp_sum_weight.bin")),
        native_reconstruction_mask=native_reconstruction_mask,
        live=live,
    )
    return {
        "schema": "recovar.vdam_fused_posterior_boundary.v1",
        "status": "complete",
        "comparison": comparison,
        "artifacts": {
            "native_directory": str(native_dir.resolve()),
            "recovar_fused_posterior": str(recovar_fused_posterior.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-fused-posterior", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(args.native_directory, args.recovar_fused_posterior)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
