#!/usr/bin/env python3
"""Localize K=1 coarse residuals across the post-optics score-weight transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.analyze_em_k1_live_reference_counterfactual import (
    _load_recovar,
    recovar_score_components,
    reference_swap_counterfactual,
    relion_values_on_recovar_window,
)
from scripts.validate_relion_coarse_operand_capture import (
    CoarseOperandCapture,
)
from scripts.validate_relion_coarse_operand_capture import (
    validate_directory as validate_operands,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
)
from scripts.validate_relion_coarse_pass1_components import (
    validate_directory as validate_components,
)
from scripts.validate_relion_preprocess_capture import (
    validate_directory as validate_preprocess,
)

EXPECTED_PARTICLES = 14
COMPONENT_DOMINANCE_FRACTION = 0.5
HYBRID_BASE_MATERIAL_RELATIVE_L2 = 1.0e-6
CTF_ZERO_THRESHOLD = 1.0e-8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_l2(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source)
    target = np.asarray(target)
    _require(source.shape == target.shape, "relative-L2 shapes differ")
    denominator = float(np.linalg.norm(target))
    _require(denominator > 0.0, "relative-L2 target has zero norm")
    return float(np.linalg.norm(source - target) / denominator)


def _load_ctf_half(
    path: Path,
    *,
    full_image_size: int,
) -> tuple[np.ndarray, float]:
    """Evaluate the repository CTF convention from a fixed cryoDRGN CTF pickle."""

    with Path(path).open("rb") as stream:
        raw = np.asarray(pickle.load(stream))
    _require(
        raw.ndim == 2 and raw.shape[1] == 9 and raw.shape[0] > 0,
        "CTF pickle must have shape (N, 9)",
    )
    _require(
        np.issubdtype(raw.dtype, np.number) and np.all(np.isfinite(raw)),
        "CTF pickle must contain finite numeric values",
    )
    _require(
        np.all(raw[:, 0] == full_image_size),
        "CTF pickle image size differs from the fixed full image size",
    )
    voxel_size = float(raw[0, 1])
    _require(
        voxel_size > 0.0 and np.all(raw[:, 1] == raw[0, 1]),
        "CTF pickle voxel size must be positive and constant",
    )
    evaluator_params = np.concatenate(
        (
            raw[:, 2:].astype(np.float32, copy=False),
            np.zeros((raw.shape[0], 1), dtype=np.float32),
            np.ones((raw.shape[0], 1), dtype=np.float32),
        ),
        axis=1,
    )
    from recovar.core.ctf import CTFEvaluator

    ctf_half = np.asarray(
        CTFEvaluator()(
            evaluator_params,
            (full_image_size, full_image_size),
            voxel_size,
            half_image=True,
        ),
        dtype=np.float64,
    )
    expected_shape = (
        raw.shape[0],
        full_image_size * (full_image_size // 2 + 1),
    )
    _require(
        ctf_half.shape == expected_shape,
        f"evaluated CTF shape mismatch: {ctf_half.shape} != {expected_shape}",
    )
    _require(np.all(np.isfinite(ctf_half)), "evaluated CTF contains non-finite values")
    return ctf_half, voxel_size


def recovar_weighted_base_from_relion_postoptics(
    relion_postoptics: np.ndarray,
    *,
    window_indices: np.ndarray,
    ctf_half: np.ndarray,
    ctf2_data: np.ndarray,
    full_image_size: int,
    current_size: int,
) -> np.ndarray:
    """Apply RECOVAR's captured CTF/noise weight to RELION post-optics data."""

    relion_postoptics = np.asarray(relion_postoptics)
    _require(
        relion_postoptics.shape
        == (current_size, current_size // 2 + 1),
        "RELION post-optics topology mismatch",
    )
    window_indices = np.asarray(window_indices, dtype=np.int64)
    ctf_half = np.asarray(ctf_half, dtype=np.float64).reshape(-1)
    ctf2_data = np.asarray(ctf2_data, dtype=np.float64)
    _require(
        ctf_half.shape
        == (full_image_size * (full_image_size // 2 + 1),),
        "full CTF half-spectrum topology mismatch",
    )
    _require(
        ctf2_data.shape == window_indices.shape,
        "captured RECOVAR CTF/noise topology mismatch",
    )
    ctf_window = ctf_half[window_indices]
    zero_ctf = np.abs(ctf_window) <= CTF_ZERO_THRESHOLD
    _require(
        not np.any(np.abs(ctf2_data[zero_ctf]) > CTF_ZERO_THRESHOLD),
        "RECOVAR CTF/noise is nonzero at a zero-CTF pixel",
    )
    score_transfer = np.zeros_like(ctf_window)
    np.divide(
        ctf2_data,
        ctf_window,
        out=score_transfer,
        where=~zero_ctf,
    )
    selected_native = relion_values_on_recovar_window(
        relion_postoptics.reshape(1, -1),
        window_indices,
        full_image_size=full_image_size,
        current_size=current_size,
    )[0]
    selected_centered = selected_native * float(full_image_size**2)
    return selected_centered * score_transfer


def relion_live_score_base(
    operand: CoarseOperandCapture,
    recovar: dict[str, Any],
    *,
    full_image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert captured production Fimg_/corr_img and translations to score space."""

    current_size = int(recovar["current_size"])
    image_native = operand.image_real.astype(np.float64) + (
        np.complex128(1j) * operand.image_imag.astype(np.float64)
    )
    correction = relion_values_on_recovar_window(
        operand.correction.reshape(1, -1),
        recovar["window_indices"],
        full_image_size=full_image_size,
        current_size=current_size,
    )[0].real
    half_weights = np.asarray(recovar["half_weights"], dtype=np.float64)
    _require(np.all(half_weights > 0.0), "RECOVAR half weights must be positive")
    image_selected = relion_values_on_recovar_window(
        image_native.reshape(1, -1),
        recovar["window_indices"],
        full_image_size=full_image_size,
        current_size=current_size,
    )[0]
    live_base = (
        -image_selected
        * correction
        / (float(full_image_size**2) * half_weights)
    )
    shifted_native = relion_values_on_recovar_window(
        operand.shifted_real.astype(np.float64)
        + np.complex128(1j) * operand.shifted_imag.astype(np.float64),
        recovar["window_indices"],
        full_image_size=full_image_size,
        current_size=current_size,
    )
    live_shifted = (
        -shifted_native
        * correction[np.newaxis, :]
        / (float(full_image_size**2) * half_weights[np.newaxis, :])
    )
    return live_base, live_shifted


def _recover_base_and_phase(
    *,
    live_base: np.ndarray,
    live_shifted: np.ndarray,
    recovar_shifted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover RECOVAR's common base using the captured RELION translation phase."""

    live_power = np.abs(live_base) ** 2
    live_nonzero = live_power > float(np.max(live_power)) * 1.0e-16
    _require(np.count_nonzero(live_nonzero) > 1, "live base image is empty")
    live_phase = np.ones_like(live_shifted)
    live_phase[:, live_nonzero] = (
        live_shifted[:, live_nonzero]
        * np.conj(live_base[np.newaxis, live_nonzero])
        / live_power[np.newaxis, live_nonzero]
    )
    live_phase[:, live_nonzero] /= np.abs(live_phase[:, live_nonzero])
    recovar_bases = recovar_shifted / live_phase
    recovar_base = np.mean(recovar_bases, axis=0)
    recovar_power = np.abs(recovar_base) ** 2
    recovar_nonzero = (
        recovar_power > float(np.max(recovar_power)) * 1.0e-16
    )
    _require(np.count_nonzero(recovar_nonzero) > 1, "RECOVAR base image is empty")
    recovar_phase = np.ones_like(recovar_shifted)
    recovar_phase[:, recovar_nonzero] = (
        recovar_shifted[:, recovar_nonzero]
        / recovar_base[np.newaxis, recovar_nonzero]
    )
    return recovar_base, recovar_phase


def classify_postoptics_transfer(
    *,
    qualified: bool,
    actual_live_dominated: int,
    hybrid_live_dominated: int,
    hybrid_within_material_threshold: int,
    expected_particles: int,
) -> str:
    """Classify the fixed post-optics score-transfer intervention."""

    if not qualified:
        return "postoptics_inputs_not_qualified"
    if (
        actual_live_dominated == expected_particles
        and hybrid_live_dominated == 0
        and hybrid_within_material_threshold == expected_particles
    ):
        return (
            "raw_coarse_residual_is_postoptics_score_weight_transfer_"
            "dominated_not_preprocessing"
        )
    return "raw_coarse_residual_has_mixed_postoptics_score_transfer_effect"


def capture_inputs_qualified(
    *,
    preprocess_validation: dict[str, Any],
    operand_validation: dict[str, Any],
) -> bool:
    """Gate capture integrity without treating the target residual as corruption."""

    return (
        preprocess_validation.get("status") == "pass"
        and operand_validation.get("status") == "pass"
    )


def _validate_preprocess_analysis(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    _require(report.get("status") == "pass", "preprocessing analysis did not pass")
    _require(
        report.get("classification")
        == "all_preprocessing_boundaries_within_fixed_material_threshold",
        "preprocessing boundary is not qualified",
    )
    fixed = report.get("fixed_metric", {})
    _require(
        fixed.get("evaluated_particles") == EXPECTED_PARTICLES
        and fixed.get("expected_particles") == EXPECTED_PARTICLES,
        "preprocessing analysis denominator mismatch",
    )
    for key in (
        "normalized_material_gap",
        "masked_material_gap",
        "unmasked_fft_material_gap",
        "masked_fft_material_gap",
        "masked_post_optics_material_gap",
    ):
        _require(fixed.get(key) == 0, f"preprocessing material gap remains: {key}")
    return report


def build_report(
    *,
    cohort_json: Path,
    preprocess_capture_directory: Path,
    operand_capture_directory: Path,
    recovar_directory: Path,
    ctf_pickle: Path,
    preprocess_analysis_json: Path,
    full_image_size: int,
) -> dict[str, Any]:
    """Build the fixed-cohort post-optics score-transfer report."""

    _require(
        full_image_size > 0 and full_image_size % 2 == 0,
        "full image size must be positive and even",
    )
    cohort = json.loads(Path(cohort_json).read_text())
    _require(
        cohort.get("selected_particle_count") == EXPECTED_PARTICLES,
        "cohort denominator must be 14",
    )
    expected_stacks = np.asarray(
        cohort["selected_stack_indices_one_based"],
        dtype=np.int64,
    )
    expected_parts = np.asarray(
        [row["relion_part_id"] for row in cohort["rows"]],
        dtype=np.int64,
    )
    preprocess, preprocess_validation = validate_preprocess(
        preprocess_capture_directory,
        expected_particles=EXPECTED_PARTICLES,
        expected_part_ids=expected_parts,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=int(cohort["mpi_rank"]),
        expected_iteration=int(cohort["iteration"]),
    )
    operands, operand_validation = validate_operands(
        operand_capture_directory,
        expected_particles=EXPECTED_PARTICLES,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=int(cohort["mpi_rank"]),
    )
    components, component_validation = validate_components(
        operand_capture_directory,
        expected_particles=EXPECTED_PARTICLES,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=int(cohort["mpi_rank"]),
    )
    prior_preprocess = _validate_preprocess_analysis(preprocess_analysis_json)
    preprocess_by_stack = {item.stack_index: item for item in preprocess}
    operands_by_stack = {item.stack_index: item for item in operands}
    components_by_stack = {item.stack_index: item for item in components}

    recovar_items = [
        _load_recovar(path)
        for path in sorted(Path(recovar_directory).glob("*.npz"))
    ]
    _require(
        len(recovar_items) == EXPECTED_PARTICLES,
        "RECOVAR denominator must be 14",
    )
    recovar_by_index = {item["original_index"]: item for item in recovar_items}
    _require(
        len(recovar_by_index) == EXPECTED_PARTICLES,
        "duplicate RECOVAR original index",
    )
    ctf_half, voxel_size = _load_ctf_half(
        ctf_pickle,
        full_image_size=full_image_size,
    )

    particles = []
    for cohort_row in cohort["rows"]:
        stack_index = int(cohort_row["stack_index_one_based"])
        original_index = int(cohort_row["original_index_zero_based"])
        _require(
            stack_index == original_index + 1,
            "fixed stack/original identity relationship changed",
        )
        preprocess_item = preprocess_by_stack[stack_index]
        operand = operands_by_stack[stack_index]
        component = components_by_stack[stack_index]
        recovar = recovar_by_index[original_index]
        _require(
            preprocess_item.part_id == operand.part_id == component.part_id,
            "cross-capture RELION particle identity mismatch",
        )
        _require(
            int(recovar["current_size"])
            == preprocess_item.masked_fourier_post_optics.shape[-2]
            == operand.header[12],
            "cross-engine current-size mismatch",
        )
        live_base, live_shifted = relion_live_score_base(
            operand,
            recovar,
            full_image_size=full_image_size,
        )
        recovar_base, recovar_phase = _recover_base_and_phase(
            live_base=live_base,
            live_shifted=live_shifted,
            recovar_shifted=recovar["shifted_data"],
        )
        hybrid_base = recovar_weighted_base_from_relion_postoptics(
            preprocess_item.masked_fourier_post_optics[0],
            window_indices=recovar["window_indices"],
            ctf_half=ctf_half[original_index],
            ctf2_data=recovar["ctf2_data"],
            full_image_size=full_image_size,
            current_size=int(recovar["current_size"]),
        )

        translation_permutation, translation_mapping = _translation_permutation(
            component.translations,
            recovar["translations"],
        )
        mapped_raw = _map_relion_table(
            component.raw_diff2,
            n_directions=component.header[10],
            n_psi=component.header[11],
            relion_to_recovar_translation=translation_permutation,
        )
        selected_raw = mapped_raw[recovar["rotation_ids"]]
        _require(
            np.all(selected_raw != RELION_INVALID_DIFF2),
            "captured rotation has inactive RELION score",
        )
        baseline_residual = (
            recovar["scores"][recovar["rotation_ids"]] + selected_raw
        )

        counterfactuals = {}
        for label, base in (
            ("actual_relion_live_base", live_base),
            ("relion_postoptics_recovar_score_transfer", hybrid_base),
        ):
            shifted = base[np.newaxis, :] * recovar_phase
            swapped_norm, swapped_cross = recovar_score_components(
                recovar["references"],
                shifted,
                recovar["ctf2_data"],
                recovar["half_weights"],
            )
            counterfactuals[label] = reference_swap_counterfactual(
                baseline_residual,
                swapped_norm + swapped_cross + selected_raw,
            )

        live_relative_l2 = _relative_l2(live_base, recovar_base)
        hybrid_relative_l2 = _relative_l2(hybrid_base, recovar_base)
        transfer_delta_relative_l2 = _relative_l2(live_base, hybrid_base)
        particles.append(
            {
                "group": cohort_row["group"],
                "stack_index_one_based": stack_index,
                "original_index_zero_based": original_index,
                "actual_relion_base_relative_l2": live_relative_l2,
                "postoptics_recovar_transfer_relative_l2": hybrid_relative_l2,
                "score_transfer_delta_relative_l2": transfer_delta_relative_l2,
                "postoptics_recovar_transfer_within_material_threshold": bool(
                    hybrid_relative_l2 <= HYBRID_BASE_MATERIAL_RELATIVE_L2
                ),
                "counterfactuals": counterfactuals,
                "translation_mapping": translation_mapping,
                "artifact_paths": {
                    "preprocess": str(preprocess_item.path.resolve()),
                    "operand": str(operand.path.resolve()),
                    "component": str(component.path.resolve()),
                    "recovar": recovar["path"],
                },
                "artifact_sha256": {
                    "preprocess": preprocess_item.sha256,
                    "operand": operand.sha256,
                    "component": component.sha256,
                    "recovar": recovar["sha256"],
                },
            }
        )

    actual_live_dominated = sum(
        row["counterfactuals"]["actual_relion_live_base"][
            "live_reference_dominated"
        ]
        for row in particles
    )
    hybrid_live_dominated = sum(
        row["counterfactuals"]["relion_postoptics_recovar_score_transfer"][
            "live_reference_dominated"
        ]
        for row in particles
    )
    hybrid_within = sum(
        row["postoptics_recovar_transfer_within_material_threshold"]
        for row in particles
    )
    qualified = capture_inputs_qualified(
        preprocess_validation=preprocess_validation,
        operand_validation=operand_validation,
    )
    classification = classify_postoptics_transfer(
        qualified=qualified,
        actual_live_dominated=actual_live_dominated,
        hybrid_live_dominated=hybrid_live_dominated,
        hybrid_within_material_threshold=hybrid_within,
        expected_particles=EXPECTED_PARTICLES,
    )
    return {
        "schema": "em-k1-postoptics-score-transfer-v1",
        "status": "complete",
        "classification_ready": qualified,
        "classification": classification,
        "metric_policy": (
            "scale-sensitive relative-L2 and centered residual-energy removal; "
            "fixed thresholds; no fitted scale/sign; no correlation"
        ),
        "fixed_gates": {
            "component_dominance_fraction": COMPONENT_DOMINANCE_FRACTION,
            "hybrid_base_material_relative_l2": (
                HYBRID_BASE_MATERIAL_RELATIVE_L2
            ),
            "ctf_zero_threshold": CTF_ZERO_THRESHOLD,
        },
        "fixed_metric": {
            "evaluated_particles": len(particles),
            "expected_particles": EXPECTED_PARTICLES,
            "actual_relion_live_base_dominated": actual_live_dominated,
            "postoptics_recovar_transfer_dominated": hybrid_live_dominated,
            "postoptics_recovar_transfer_within_material_threshold": hybrid_within,
        },
        "full_image_size": full_image_size,
        "voxel_size": voxel_size,
        "preprocess_validation": preprocess_validation,
        "operand_validation_status": operand_validation["status"],
        "component_validation_status": component_validation["status"],
        "prior_preprocess_analysis": {
            "path": str(Path(preprocess_analysis_json).resolve()),
            "sha256": _sha256(preprocess_analysis_json),
            "classification": prior_preprocess["classification"],
        },
        "cohort": {
            "path": str(Path(cohort_json).resolve()),
            "sha256": _sha256(cohort_json),
        },
        "ctf_pickle": {
            "path": str(Path(ctf_pickle).resolve()),
            "sha256": _sha256(ctf_pickle),
        },
        "particles": particles,
        "notes": [
            (
                "The component validator's scientific replay status is reported "
                "but is not an input-integrity gate: the known replay residual is "
                "the target of this intervention."
            ),
            (
                "Component capture structure, identity, topology, and completeness "
                "are still validated and fail closed before analysis."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-json", type=Path, required=True)
    parser.add_argument(
        "--preprocess-capture-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--operand-capture-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--ctf-pickle", type=Path, required=True)
    parser.add_argument("--preprocess-analysis-json", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(
        not args.output_json.exists(),
        f"refusing to overwrite report: {args.output_json}",
    )
    report = build_report(
        cohort_json=args.cohort_json,
        preprocess_capture_directory=args.preprocess_capture_directory,
        operand_capture_directory=args.operand_capture_directory,
        recovar_directory=args.recovar_directory,
        ctf_pickle=args.ctf_pickle,
        preprocess_analysis_json=args.preprocess_analysis_json,
        full_image_size=args.full_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["fixed_metric"], indent=2, sort_keys=True))
    print(report["classification"])


if __name__ == "__main__":
    main()
