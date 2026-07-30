#!/usr/bin/env python3
"""Split the K=1 post-optics score transfer into pixel and corr_img operands."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from scripts.analyze_em_k1_postoptics_score_transfer import (
    CTF_ZERO_THRESHOLD,
    EXPECTED_PARTICLES,
    _load_ctf_half,
    _recover_base_and_phase,
    _relative_l2,
    _require,
    capture_inputs_qualified,
    relion_live_score_base,
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

PARENT_CLASSIFICATION = (
    "raw_coarse_residual_is_postoptics_score_weight_transfer_"
    "dominated_not_preprocessing"
)
CLASSIFICATION = (
    "raw_coarse_residual_is_corr_img_score_weight_dominated_"
    "not_pixel_correction"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_transfer_factorial_bases(
    *,
    relion_postoptics_native: np.ndarray,
    relion_pixel_corrected_native: np.ndarray,
    relion_corr_img: np.ndarray,
    recovar_ctf: np.ndarray,
    recovar_ctf2_data: np.ndarray,
    half_weights: np.ndarray,
    full_image_size: int,
) -> dict[str, np.ndarray]:
    """Construct the fixed 2x2 pixel-correction/corr_img score bases."""

    postoptics = np.asarray(relion_postoptics_native, dtype=np.complex128)
    relion_pixel = np.asarray(
        relion_pixel_corrected_native,
        dtype=np.complex128,
    )
    relion_corr = np.asarray(relion_corr_img, dtype=np.float64)
    recovar_ctf = np.asarray(recovar_ctf, dtype=np.float64)
    recovar_ctf2 = np.asarray(recovar_ctf2_data, dtype=np.float64)
    half_weights = np.asarray(half_weights, dtype=np.float64)
    shape = postoptics.shape
    _require(
        shape
        == relion_pixel.shape
        == relion_corr.shape
        == recovar_ctf.shape
        == recovar_ctf2.shape
        == half_weights.shape,
        "score-transfer operand shapes differ",
    )
    _require(
        full_image_size > 0 and full_image_size % 2 == 0,
        "full image size must be positive and even",
    )
    _require(np.all(half_weights > 0.0), "half weights must be positive")
    _require(
        not np.any(np.abs(recovar_ctf) <= CTF_ZERO_THRESHOLD),
        "fixed cohort contains a zero-CTF score pixel",
    )

    recovar_pixel_correction = -1.0 / recovar_ctf
    recovar_corr_img = (
        float(full_image_size**4) * half_weights * recovar_ctf2
    )

    def base(image: np.ndarray, correction: np.ndarray) -> np.ndarray:
        return (
            -image
            * correction
            / (float(full_image_size**2) * half_weights)
        )

    return {
        "actual_relion": base(relion_pixel, relion_corr),
        "recovar_pixel_correction_only": base(
            postoptics * recovar_pixel_correction,
            relion_corr,
        ),
        "recovar_corr_img_only": base(relion_pixel, recovar_corr_img),
        "recovar_pixel_and_corr_img": base(
            postoptics * recovar_pixel_correction,
            recovar_corr_img,
        ),
    }


def classify_score_transfer_factorial(
    *,
    qualified: bool,
    dominated: dict[str, int],
    expected_particles: int,
) -> str:
    """Classify the predeclared 2x2 intervention at a fixed denominator."""

    if not qualified:
        return "score_transfer_factorial_inputs_not_qualified"
    expected_pattern = {
        "actual_relion": expected_particles,
        "recovar_pixel_correction_only": expected_particles,
        "recovar_corr_img_only": 0,
        "recovar_pixel_and_corr_img": 0,
    }
    if dominated == expected_pattern:
        return CLASSIFICATION
    return "raw_coarse_residual_has_mixed_pixel_correction_corr_img_effect"


def _validate_parent(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    _require(report.get("status") == "complete", "parent report is incomplete")
    _require(
        report.get("classification_ready") is True,
        "parent report is not classification-ready",
    )
    _require(
        report.get("classification") == PARENT_CLASSIFICATION,
        "parent post-optics classification changed",
    )
    fixed = report.get("fixed_metric", {})
    _require(
        fixed.get("evaluated_particles") == EXPECTED_PARTICLES
        and fixed.get("expected_particles") == EXPECTED_PARTICLES,
        "parent report denominator mismatch",
    )
    _require(
        fixed.get("actual_relion_live_base_dominated") == EXPECTED_PARTICLES
        and fixed.get("postoptics_recovar_transfer_dominated") == 0
        and fixed.get(
            "postoptics_recovar_transfer_within_material_threshold"
        )
        == EXPECTED_PARTICLES,
        "parent report fixed intervention did not pass",
    )
    return report


def build_report(
    *,
    cohort_json: Path,
    preprocess_capture_directory: Path,
    operand_capture_directory: Path,
    recovar_directory: Path,
    ctf_pickle: Path,
    parent_analysis_json: Path,
    full_image_size: int,
) -> dict[str, Any]:
    """Build the fixed-cohort pixel-correction/corr_img factorial report."""

    parent = _validate_parent(parent_analysis_json)
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
        current_size = int(recovar["current_size"])
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
        window_indices = recovar["window_indices"]
        postoptics_native = relion_values_on_recovar_window(
            preprocess_item.masked_fourier_post_optics[0].reshape(1, -1),
            window_indices,
            full_image_size=full_image_size,
            current_size=current_size,
        )[0]
        relion_pixel_native = relion_values_on_recovar_window(
            (
                operand.image_real.astype(np.float64)
                + np.complex128(1j) * operand.image_imag.astype(np.float64)
            ).reshape(1, -1),
            window_indices,
            full_image_size=full_image_size,
            current_size=current_size,
        )[0]
        relion_corr = relion_values_on_recovar_window(
            operand.correction.reshape(1, -1),
            window_indices,
            full_image_size=full_image_size,
            current_size=current_size,
        )[0].real
        recovar_ctf = ctf_half[original_index].reshape(-1)[window_indices]
        bases = score_transfer_factorial_bases(
            relion_postoptics_native=postoptics_native,
            relion_pixel_corrected_native=relion_pixel_native,
            relion_corr_img=relion_corr,
            recovar_ctf=recovar_ctf,
            recovar_ctf2_data=recovar["ctf2_data"],
            half_weights=recovar["half_weights"],
            full_image_size=full_image_size,
        )
        _require(
            _relative_l2(bases["actual_relion"], live_base) <= 1.0e-14,
            "factorial actual arm does not replay the live base",
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
        base_relative_l2 = {}
        for label, base in bases.items():
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
            base_relative_l2[label] = _relative_l2(base, recovar_base)

        particles.append(
            {
                "group": cohort_row["group"],
                "stack_index_one_based": stack_index,
                "original_index_zero_based": original_index,
                "base_relative_l2": base_relative_l2,
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

    arm_labels = tuple(particles[0]["counterfactuals"])
    dominated = {
        label: sum(
            row["counterfactuals"][label]["live_reference_dominated"]
            for row in particles
        )
        for label in arm_labels
    }
    capture_qualified = capture_inputs_qualified(
        preprocess_validation=preprocess_validation,
        operand_validation=operand_validation,
    )
    qualified = capture_qualified and parent["classification_ready"]
    classification = classify_score_transfer_factorial(
        qualified=qualified,
        dominated=dominated,
        expected_particles=EXPECTED_PARTICLES,
    )
    return {
        "schema": "em-k1-score-transfer-factorial-v1",
        "status": "complete",
        "classification_ready": qualified,
        "classification": classification,
        "metric_policy": (
            "fixed 2x2 pixel-correction/corr_img intervention; centered "
            "residual-energy removal; scale-sensitive base relative-L2; "
            "no fitted scale/sign; no correlation"
        ),
        "fixed_gates": {
            "expected_particles": EXPECTED_PARTICLES,
            "component_dominance_fraction_strictly_greater_than": 0.5,
            "ctf_zero_threshold": CTF_ZERO_THRESHOLD,
        },
        "fixed_metric": {
            "evaluated_particles": len(particles),
            "expected_particles": EXPECTED_PARTICLES,
            "live_reference_dominated": dominated,
        },
        "full_image_size": full_image_size,
        "voxel_size": voxel_size,
        "preprocess_validation_status": preprocess_validation["status"],
        "operand_validation_status": operand_validation["status"],
        "component_validation_status": component_validation["status"],
        "parent_analysis": {
            "path": str(Path(parent_analysis_json).resolve()),
            "sha256": _sha256(parent_analysis_json),
            "classification": parent["classification"],
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
                "RECOVAR's CTF convention has the fixed opposite sign from "
                "RELION, so the counterfactual pixel correction is -1/CTF."
            ),
            (
                "The fixed cohort has no evaluated CTF magnitude at or below "
                "the 1e-8 zero threshold."
            ),
            (
                "The component validator's known scientific replay rejection "
                "is reported but is not an input-integrity gate."
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
    parser.add_argument("--parent-analysis-json", type=Path, required=True)
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
        parent_analysis_json=args.parent_analysis_json,
        full_image_size=args.full_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["fixed_metric"], indent=2, sort_keys=True))
    print(report["classification"])
    if not report["classification_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
