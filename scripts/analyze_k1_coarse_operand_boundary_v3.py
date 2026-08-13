#!/usr/bin/env python3
"""Compare bounded native/RECOVAR K=1 coarse components and live operands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    POSTERIOR_TV_GATE,
    SCORE_MAX_GATE,
    SCORE_P95_GATE,
    _map_relion_table,
    _translation_permutation,
)
from scripts.analyze_em_k1_live_reference_counterfactual import (
    recovar_score_components,
    reference_swap_counterfactual,
    relion_reference_on_recovar_window,
    relion_values_on_recovar_window,
)
from scripts.validate_relion_coarse_operand_capture import load_artifact as load_operand
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
)
from scripts.validate_relion_coarse_pass1_components import (
    load_artifact as load_components,
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


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.complex128).reshape(-1)
    right = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    _require(left.shape == right.shape and left.size > 0, "operand topology mismatch")
    return float(np.linalg.norm(right - left) / max(np.linalg.norm(left), np.finfo(float).tiny))


def _stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    _require(absolute.size > 0, "cannot summarize an empty residual")
    _require(np.all(np.isfinite(absolute)), "residual contains non-finite values")
    return {
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _center_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.max(values)


def _component_decomposition(
    total_residual: np.ndarray,
    norm_residual: np.ndarray,
    cross_residual: np.ndarray,
) -> dict[str, Any]:
    total = np.asarray(total_residual, dtype=np.float64)
    norm = np.asarray(norm_residual, dtype=np.float64)
    cross = np.asarray(cross_residual, dtype=np.float64)
    _require(total.shape == norm.shape == cross.shape, "component topology mismatch")
    _require(total.ndim == 2 and total.size > 1, "component panel is too small")

    def center(values: np.ndarray) -> np.ndarray:
        return values - np.mean(values)

    centered_total = center(total)
    total_energy = float(np.sum(centered_total**2))
    _require(total_energy > 0.0, "component residual has zero centered energy")
    without_norm = float(np.sum(center(total - norm) ** 2))
    without_cross = float(np.sum(center(total - cross) ** 2))
    closure = centered_total - center(norm + cross)
    return {
        "total_centered_energy": total_energy,
        "counterfactual_energy_removal_fraction": {
            "reference_norm": float(1.0 - without_norm / total_energy),
            "cross": float(1.0 - without_cross / total_energy),
        },
        "closure": {
            "p95_abs": float(np.percentile(np.abs(closure), 95)),
            "max_abs": float(np.max(np.abs(closure))),
        },
    }


def _rotation_key_to_recovar(rotation_key: int, n_directions: int, n_psi: int) -> int:
    direction, psi = divmod(int(rotation_key), int(n_psi))
    _require(direction < n_directions, "native rotation key is out of range")
    return int(psi * n_directions + direction)


def _load_recovar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "original_index",
            "current_size",
            "scores_pre_prior_per_class",
            "scores_with_prior_per_class",
            "weights_per_class",
            "significant_mask",
            "translations",
            "window_indices",
            "shifted_data",
            "ctf2_data",
            "half_weights",
            "projected_reference_rotation_ids",
            "projected_reference_per_class",
            "projected_reference_norm_score_per_class",
            "projected_cross_score_per_class",
        }
        _require(required <= set(payload.files), f"missing RECOVAR fields: {path}")
        return {
            "path": path.resolve(),
            "sha256": _sha256(path),
            "original_index": int(payload["original_index"]),
            "current_size": int(payload["current_size"]),
            "scores": np.asarray(payload["scores_pre_prior_per_class"][0], dtype=np.float64),
            "scores_with_prior": np.asarray(
                payload["scores_with_prior_per_class"][0], dtype=np.float64
            ),
            "weights": np.asarray(payload["weights_per_class"][0], dtype=np.float64),
            "significant_mask": np.asarray(payload["significant_mask"], dtype=bool),
            "translations": np.asarray(payload["translations"], dtype=np.float64),
            "window_indices": np.asarray(payload["window_indices"], dtype=np.int64),
            "shifted": np.asarray(payload["shifted_data"], dtype=np.complex128),
            "ctf2": np.asarray(payload["ctf2_data"][0], dtype=np.float64),
            "half_weights": np.asarray(payload["half_weights"], dtype=np.float64),
            "rotation_ids": np.asarray(payload["projected_reference_rotation_ids"], dtype=np.int64),
            "references": np.asarray(payload["projected_reference_per_class"][0], dtype=np.complex128),
            "norms": np.asarray(payload["projected_reference_norm_score_per_class"][0], dtype=np.float64),
            "crosses": np.asarray(payload["projected_cross_score_per_class"][0], dtype=np.float64),
        }


def _compare(
    components_path: Path,
    operand_path: Path,
    recovar_path: Path,
    *,
    physical_iteration: int,
) -> dict[str, Any]:
    components = load_components(components_path)
    operand = load_operand(operand_path)
    recovar = _load_recovar(recovar_path)
    _require(components.stack_index == operand.stack_index, "native stack mismatch")
    _require(components.part_id == operand.part_id, "native part mismatch")
    _require(components.stack_index - 1 == recovar["original_index"], "cross-engine identity mismatch")
    _require(
        components.header[5] == int(physical_iteration),
        "native component physical iteration mismatch",
    )
    _require(
        operand.header[5] == int(physical_iteration),
        "native operand physical iteration mismatch",
    )
    _require(components.header[27] == recovar["current_size"], "component/RECOVAR current-size mismatch")
    _require(operand.header[12] == recovar["current_size"], "operand/RECOVAR current-size mismatch")
    n_directions, n_psi, _ = components.header[10:13]
    translation_permutation, translation_mapping = _translation_permutation(
        components.translations,
        recovar["translations"],
    )
    mapped_raw = _map_relion_table(
        components.raw_diff2,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    )
    mapped_norm = _map_relion_table(
        components.reference_norms,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    )
    mapped_cross = _map_relion_table(
        components.cross_terms,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    )
    mapped_weights = _map_relion_table(
        components.weights,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(np.float64)
    mapped_significant = _map_relion_table(
        components.significant_mask,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(bool)
    _require(mapped_raw.shape == recovar["scores"].shape, "candidate topology mismatch")
    relion_prior_support = mapped_raw != RELION_INVALID_DIFF2
    recovar_prior_support = np.isfinite(recovar["scores_with_prior"])
    common_prior_support = relion_prior_support & recovar_prior_support
    _require(np.any(common_prior_support), "no common finite-prior support")
    raw_residual = (
        _center_max(recovar["scores"][common_prior_support])
        - _center_max(-mapped_raw[common_prior_support])
    )
    raw_stats = _stats(raw_residual)
    relion_probabilities = mapped_weights / np.sum(mapped_weights)
    recovar_probabilities = recovar["weights"].reshape(mapped_weights.shape)
    recovar_probabilities = recovar_probabilities / np.sum(recovar_probabilities)
    posterior_tv = float(
        0.5 * np.sum(np.abs(recovar_probabilities - relion_probabilities))
    )
    relion_with_prior = np.full(mapped_weights.shape, -np.inf, dtype=np.float64)
    positive = mapped_weights > 0.0
    relion_with_prior[positive] = np.log(mapped_weights[positive])
    common_positive = positive & np.isfinite(recovar["scores_with_prior"])
    _require(np.any(common_positive), "no common positive posterior support")
    with_prior_stats = _stats(
        _center_max(recovar["scores_with_prior"][common_positive])
        - _center_max(relion_with_prior[common_positive])
    )
    recovar_significant = recovar["significant_mask"].reshape(mapped_significant.shape)
    significant_mismatch_count = int(
        np.count_nonzero(mapped_significant != recovar_significant)
    )
    parent_mismatch_count = int(
        np.count_nonzero(
            np.any(mapped_significant, axis=1)
            != np.any(recovar_significant, axis=1)
        )
    )
    raw_pass = raw_stats["p95_abs"] <= SCORE_P95_GATE and raw_stats["max_abs"] < SCORE_MAX_GATE
    prior_pass = with_prior_stats["p95_abs"] <= SCORE_P95_GATE and with_prior_stats["max_abs"] < SCORE_MAX_GATE
    stage_exact = {
        "current_size": True,
        "prior_support": bool(np.array_equal(relion_prior_support, recovar_prior_support)),
        "raw_scores": bool(raw_pass),
        "scores_with_prior": bool(prior_pass),
        "posterior": bool(posterior_tv <= POSTERIOR_TV_GATE),
        "significant_support": significant_mismatch_count == 0,
        "winner": int(np.argmax(mapped_weights)) == int(np.argmax(recovar["weights"])),
    }
    first_unequal = next((name for name, equal in stage_exact.items() if not equal), "coarse_boundary_exact")
    rotation_ids = recovar["rotation_ids"]
    active = np.all(mapped_raw[rotation_ids] != RELION_INVALID_DIFF2, axis=1)
    _require(np.any(active), "no active requested rotations")
    selected_ids = rotation_ids[active]
    recovar_norm = recovar["norms"][active]
    recovar_cross = recovar["crosses"][active]
    recovar_total = recovar["scores"][selected_ids]
    decomposition = _component_decomposition(
        recovar_total + mapped_raw[selected_ids],
        recovar_norm + mapped_norm[selected_ids],
        recovar_cross + mapped_cross[selected_ids],
    )

    mapped_operand_ids = np.asarray(
        [
            _rotation_key_to_recovar(key, n_directions, n_psi)
            for key in operand.rotation_keys
        ],
        dtype=np.int64,
    )
    operand_index = {int(key): index for index, key in enumerate(mapped_operand_ids)}
    _require(set(rotation_ids.tolist()) == set(operand_index), "captured rotation sets differ")
    operand_order = np.asarray([operand_index[int(key)] for key in rotation_ids], dtype=np.int64)
    native_reference = relion_reference_on_recovar_window(
        (
            operand.reference_real.astype(np.float64)
            + 1j * operand.reference_imag.astype(np.float64)
        )[operand_order],
        recovar["window_indices"],
        full_image_size=128,
        current_size=recovar["current_size"],
    )
    native_shifted = relion_values_on_recovar_window(
        operand.shifted_real.astype(np.float64)
        + 1j * operand.shifted_imag.astype(np.float64),
        recovar["window_indices"],
        full_image_size=128,
        current_size=recovar["current_size"],
    )
    native_shifted_ordered = np.empty_like(native_shifted)
    native_shifted_ordered[translation_permutation] = native_shifted
    native_correction = relion_values_on_recovar_window(
        operand.correction[np.newaxis, :],
        recovar["window_indices"],
        full_image_size=128,
        current_size=recovar["current_size"],
    )[0].real
    image_normalization = float(128**2)
    native_weighted_shifted = (
        -native_shifted_ordered
        * native_correction[np.newaxis, :]
        / (image_normalization * recovar["half_weights"][np.newaxis, :])
    )
    native_ctf2 = native_correction / (
        image_normalization**2 * recovar["half_weights"]
    )
    configurations = {
        "projected_reference": (native_reference, recovar["shifted"], recovar["ctf2"]),
        "weighted_shifted_image": (recovar["references"], native_weighted_shifted, recovar["ctf2"]),
        "correction": (recovar["references"], recovar["shifted"], native_ctf2),
        "all_native": (native_reference, native_weighted_shifted, native_ctf2),
    }
    total_residual = recovar_total + mapped_raw[rotation_ids]
    counterfactuals = {}
    for label, (reference, shifted, ctf2) in configurations.items():
        norm, cross = recovar_score_components(
            reference,
            shifted,
            ctf2,
            recovar["half_weights"],
        )
        counterfactuals[label] = reference_swap_counterfactual(
            total_residual,
            norm + cross + mapped_raw[rotation_ids],
        )
    replay_norm, replay_cross = recovar_score_components(
        recovar["references"],
        recovar["shifted"],
        recovar["ctf2"],
        recovar["half_weights"],
    )
    replay_error = replay_norm + replay_cross - recovar_total
    return {
        "stack_index_one_based": components.stack_index,
        "original_index_zero_based": recovar["original_index"],
        "relion_part_id": components.part_id,
        "active_requested_rotation_count": int(np.count_nonzero(active)),
        "requested_rotation_count": int(rotation_ids.size),
        "complete_coarse_boundary": {
            "first_unequal_stage": first_unequal,
            "stage_exact": stage_exact,
            "prior_support_mismatch_count": int(
                np.count_nonzero(relion_prior_support != recovar_prior_support)
            ),
            "raw_centered_score_diff": raw_stats,
            "with_prior_centered_score_diff": with_prior_stats,
            "posterior_total_variation": posterior_tv,
            "posterior_max_abs": float(
                np.max(np.abs(recovar_probabilities - relion_probabilities))
            ),
            "significant_candidate_mismatch_count": significant_mismatch_count,
            "significant_parent_mismatch_count": parent_mismatch_count,
            "relion_significant_count": int(np.count_nonzero(mapped_significant)),
            "recovar_significant_count": int(np.count_nonzero(recovar_significant)),
        },
        "component_decomposition": decomposition,
        "recovar_component_replay": {
            "p95_abs": float(np.percentile(np.abs(replay_error), 95)),
            "max_abs": float(np.max(np.abs(replay_error))),
        },
        "operand_relative_l2": {
            "projected_reference": _relative_l2(native_reference, recovar["references"]),
            "weighted_shifted_image": _relative_l2(native_weighted_shifted, recovar["shifted"]),
            "correction": _relative_l2(native_ctf2, recovar["ctf2"]),
        },
        "counterfactuals": counterfactuals,
        "translation_mapping": translation_mapping,
        "artifacts": {
            "components": str(components_path.resolve()),
            "components_sha256": components.sha256,
            "operands": str(operand_path.resolve()),
            "operands_sha256": operand.sha256,
            "recovar": str(recovar["path"]),
            "recovar_sha256": recovar["sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text())
    physical_iteration = int(selection["physical_iteration"])
    rows = []
    for target in selection["targets"]:
        stack = int(target["stack_index_one_based"])
        original = int(target["original_index_zero_based"])
        component_paths = list(args.native_directory.glob(f"part*_stack{stack}.p1-v2.bin"))
        operand_paths = list(args.native_directory.glob(f"part*_stack{stack}.p1-op-v2.bin"))
        recovar_paths = list(args.recovar_directory.glob(f"significance_orig{original:06d}*_cs*.npz"))
        _require(len(component_paths) == len(operand_paths) == len(recovar_paths) == 1, f"artifact lookup failed for stack {stack}")
        rows.append(
            _compare(
                component_paths[0],
                operand_paths[0],
                recovar_paths[0],
                physical_iteration=physical_iteration,
            )
        )
    validation_dir = args.native_directory.resolve().parent / "analysis"
    validation_paths = {
        "components": validation_dir / "components_validation.json",
        "operands": validation_dir / "operand_validation.json",
    }
    capture_validation: dict[str, Any] = {}
    classification_ready = True
    for label, path in validation_paths.items():
        if path.is_file():
            payload = json.loads(path.read_text())
            ready = bool(payload.get("classification_ready", False))
            capture_validation[label] = {
                "path": str(path),
                "sha256": _sha256(path),
                "status": payload.get("status"),
                "classification_ready": ready,
            }
            classification_ready = classification_ready and ready
        else:
            capture_validation[label] = {
                "path": str(path),
                "status": "missing",
                "classification_ready": False,
            }
            classification_ready = False
    report = {
        "schema": "recovar.em.k1_coarse_operand_boundary.v3",
        "case_id": int(selection["case_id"]),
        "physical_iteration": physical_iteration,
        "metric_policy": "scale-sensitive relative-L2 and centered residual energy; no correlation",
        "classification_ready": classification_ready,
        "capture_validation": capture_validation,
        "particles": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
