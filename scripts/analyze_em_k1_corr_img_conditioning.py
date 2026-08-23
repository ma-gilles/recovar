#!/usr/bin/env python3
"""Audit K=1 corr_img attribution away from poorly conditioned CTF pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_shell_indices_half,
)
from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.analyze_em_k1_corr_img_factorial import (
    ACTUAL_ARM_REPLAY_RELATIVE_L2,
    EFFECTIVE_CTF_IMAGINARY_MAX_ABS,
    _sha256,
)
from scripts.analyze_em_k1_corr_img_factorial import (
    CLASSIFICATION as PARENT_CLASSIFICATION,
)
from scripts.analyze_em_k1_live_reference_counterfactual import (
    _load_recovar,
    recovar_score_components,
    reference_swap_counterfactual,
    relion_values_on_recovar_window,
)
from scripts.analyze_em_k1_postoptics_score_transfer import (
    EXPECTED_PARTICLES,
    _load_ctf_half,
    _recover_base_and_phase,
    _relative_l2,
    _require,
    relion_live_score_base,
)
from scripts.validate_relion_coarse_operand_capture import (
    load_artifact as load_operand,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
)
from scripts.validate_relion_coarse_pass1_components import (
    load_artifact as load_component,
)
from scripts.validate_relion_preprocess_capture import (
    load_artifact as load_preprocess,
)

CLASSIFICATION = (
    "inverse_noise_attribution_is_stable_above_fixed_"
    "effective_ctf_thresholds"
)
FIXED_EFFECTIVE_CTF_THRESHOLDS = (0.0, 1.0e-3, 3.0e-3, 1.0e-2)
ARM_LABELS = (
    "actual_relion",
    "recovar_inverse_noise_only",
    "recovar_ctf_scale_squared_only",
    "recovar_inverse_noise_and_ctf_scale_squared",
)
SHELL_PARTITION_CLASSIFICATION = (
    "inverse_noise_residual_is_confined_to_star_fixed_decimal_"
    "shells_1_through_4"
)
SHELL_PARTITION_EFFECTIVE_CTF_THRESHOLD = 1.0e-2
STAR_FIXED_DECIMAL_SHELLS = (1, 2, 3, 4)
SHELL_PARTITION_ARM_LABELS = (
    "actual_relion",
    "recovar_all",
    "relion_inverse_noise_all",
    "relion_inverse_noise_shells_1_through_4",
    "relion_inverse_noise_shells_5_plus",
)


def _threshold_label(threshold: float) -> str:
    return f"{threshold:.6g}"


def _validate_parent(path: Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    _require(report.get("status") == "complete", "parent report is incomplete")
    _require(
        report.get("classification_ready") is True,
        "parent report is not classification-ready",
    )
    _require(
        report.get("classification") == PARENT_CLASSIFICATION,
        "parent corr_img classification changed",
    )
    fixed = report.get("fixed_metric", {})
    expected = {
        "actual_relion": EXPECTED_PARTICLES,
        "recovar_inverse_noise_only": 0,
        "recovar_ctf_scale_squared_only": EXPECTED_PARTICLES,
        "recovar_inverse_noise_and_ctf_scale_squared": 0,
    }
    _require(
        fixed.get("evaluated_particles") == EXPECTED_PARTICLES
        and fixed.get("expected_particles") == EXPECTED_PARTICLES
        and fixed.get("live_reference_dominated") == expected,
        "parent fixed corr_img factorial did not pass",
    )
    return report


def conditioned_corr_img_factorial_values(
    *,
    relion_corr_img: np.ndarray,
    relion_effective_ctf: np.ndarray,
    recovar_corr_img: np.ndarray,
    recovar_effective_ctf: np.ndarray,
    effective_ctf_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Apply the corr_img factorial only to well-conditioned CTF pixels."""

    relion_corr = np.asarray(relion_corr_img, dtype=np.float64)
    relion_ctf = np.asarray(relion_effective_ctf, dtype=np.float64)
    recovar_corr = np.asarray(recovar_corr_img, dtype=np.float64)
    recovar_ctf = np.asarray(recovar_effective_ctf, dtype=np.float64)
    _require(
        relion_corr.shape
        == relion_ctf.shape
        == recovar_corr.shape
        == recovar_ctf.shape,
        "conditioned corr_img operand shapes differ",
    )
    threshold = float(effective_ctf_threshold)
    _require(
        np.isfinite(threshold) and threshold >= 0.0,
        "effective CTF threshold must be finite and nonnegative",
    )
    _require(
        np.all(np.isfinite(relion_corr))
        and np.all(np.isfinite(relion_ctf))
        and np.all(np.isfinite(recovar_corr))
        and np.all(np.isfinite(recovar_ctf)),
        "conditioned corr_img operands must be finite",
    )
    valid = (np.abs(relion_ctf) > threshold) & (
        np.abs(recovar_ctf) > threshold
    )
    _require(np.any(valid), "effective CTF threshold excludes every pixel")

    relion_inverse_noise = relion_corr[valid] / relion_ctf[valid] ** 2
    recovar_inverse_noise = recovar_corr[valid] / recovar_ctf[valid] ** 2
    _require(
        np.all(np.isfinite(relion_inverse_noise))
        and np.all(np.isfinite(recovar_inverse_noise)),
        "conditioned inverse-noise weight is non-finite",
    )

    values = {label: relion_corr.copy() for label in ARM_LABELS}
    values["recovar_inverse_noise_only"][valid] = (
        recovar_inverse_noise * relion_ctf[valid] ** 2
    )
    values["recovar_ctf_scale_squared_only"][valid] = (
        relion_inverse_noise * recovar_ctf[valid] ** 2
    )
    values["recovar_inverse_noise_and_ctf_scale_squared"][valid] = (
        recovar_corr[valid]
    )
    return values, valid


def inverse_noise_shell_partition_values(
    *,
    relion_corr_img: np.ndarray,
    relion_effective_ctf: np.ndarray,
    recovar_corr_img: np.ndarray,
    recovar_effective_ctf: np.ndarray,
    shell_indices: np.ndarray,
    effective_ctf_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Partition the inverse-noise intervention across fixed shell cohorts."""

    relion_corr = np.asarray(relion_corr_img, dtype=np.float64)
    relion_ctf = np.asarray(relion_effective_ctf, dtype=np.float64)
    recovar_corr = np.asarray(recovar_corr_img, dtype=np.float64)
    recovar_ctf = np.asarray(recovar_effective_ctf, dtype=np.float64)
    shells = np.asarray(shell_indices, dtype=np.int64)
    _require(
        relion_corr.shape
        == relion_ctf.shape
        == recovar_corr.shape
        == recovar_ctf.shape
        == shells.shape,
        "inverse-noise shell-partition operand shapes differ",
    )
    _require(
        np.all(shells >= 1),
        "score shell partition unexpectedly contains the excluded origin",
    )
    threshold = float(effective_ctf_threshold)
    _require(
        np.isfinite(threshold) and threshold >= 0.0,
        "effective CTF threshold must be finite and nonnegative",
    )
    _require(
        np.all(np.isfinite(relion_corr))
        and np.all(np.isfinite(relion_ctf))
        and np.all(np.isfinite(recovar_corr))
        and np.all(np.isfinite(recovar_ctf)),
        "inverse-noise shell-partition operands must be finite",
    )
    valid = (np.abs(relion_ctf) > threshold) & (
        np.abs(recovar_ctf) > threshold
    )
    _require(np.any(valid), "effective CTF threshold excludes every pixel")

    relion_inverse_noise = relion_corr[valid] / relion_ctf[valid] ** 2
    recovar_inverse_noise = recovar_corr[valid] / recovar_ctf[valid] ** 2
    _require(
        np.all(np.isfinite(relion_inverse_noise))
        and np.all(np.isfinite(recovar_inverse_noise)),
        "partitioned inverse-noise weight is non-finite",
    )
    valid_shells = shells[valid]
    low_shell = np.isin(valid_shells, STAR_FIXED_DECIMAL_SHELLS)
    high_shell = valid_shells >= 5
    _require(
        np.all(low_shell | high_shell),
        "fixed shell cohorts do not cover every valid score pixel",
    )

    values = {
        label: relion_corr.copy() for label in SHELL_PARTITION_ARM_LABELS
    }
    recovar_ctf_squared = recovar_ctf[valid] ** 2
    values["recovar_all"][valid] = recovar_corr[valid]
    values["relion_inverse_noise_all"][valid] = (
        relion_inverse_noise * recovar_ctf_squared
    )
    values["relion_inverse_noise_shells_1_through_4"][valid] = (
        np.where(low_shell, relion_inverse_noise, recovar_inverse_noise)
        * recovar_ctf_squared
    )
    values["relion_inverse_noise_shells_5_plus"][valid] = (
        np.where(high_shell, relion_inverse_noise, recovar_inverse_noise)
        * recovar_ctf_squared
    )
    return values, valid


def classify_shell_partition(
    *,
    qualified: bool,
    dominated: dict[str, int],
    expected_particles: int,
) -> str:
    """Classify the predeclared shell intervention without fitted parameters."""

    if not qualified:
        return "inverse_noise_shell_partition_inputs_not_qualified"
    expected = {
        "actual_relion": expected_particles,
        "recovar_all": 0,
        "relion_inverse_noise_all": expected_particles,
        "relion_inverse_noise_shells_1_through_4": expected_particles,
        "relion_inverse_noise_shells_5_plus": 0,
    }
    if dominated == expected:
        return SHELL_PARTITION_CLASSIFICATION
    return "inverse_noise_residual_is_not_confined_to_fixed_decimal_shells"


def _sigma2_noise_tokens(model_star: Path) -> dict[int, str]:
    """Read the raw sigma2-noise tokens from the first optics-group block."""

    lines = Path(model_star).read_text().splitlines()
    in_block = False
    found_column = False
    result: dict[int, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped == "data_model_optics_group_1":
            in_block = True
            continue
        if not in_block:
            continue
        if stripped.startswith("data_") and result:
            break
        if stripped.startswith("_rlnSigma2Noise"):
            found_column = True
            continue
        if not found_column or not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 3:
            if result:
                break
            continue
        try:
            shell = int(fields[0])
            float(fields[2])
        except ValueError:
            if result:
                break
            continue
        result[shell] = fields[2]
    _require(result, "RELION sigma2-noise STAR block is missing")
    return result


def _validate_star_precision_partition(model_star: Path) -> dict[str, Any]:
    """Bind shells 1--4 to RELION's fixed-decimal serialization boundary."""

    tokens = _sigma2_noise_tokens(model_star)
    for shell in STAR_FIXED_DECIMAL_SHELLS:
        _require(shell in tokens, f"RELION sigma2-noise shell {shell} missing")
        _require(
            "e" not in tokens[shell].lower()
            and float(tokens[shell]) >= 1.0e-3,
            f"RELION shell {shell} is not fixed-decimal at the 0.001 boundary",
        )
    _require(5 in tokens, "RELION sigma2-noise shell 5 missing")
    _require(
        "e" in tokens[5].lower() and 0.0 < float(tokens[5]) < 1.0e-3,
        "RELION shell 5 does not cross to scientific serialization",
    )
    return {
        "fixed_decimal_shells": list(STAR_FIXED_DECIMAL_SHELLS),
        "first_scientific_shell": 5,
        "serialization_boundary": 1.0e-3,
        "raw_sigma2_noise_tokens": {
            str(shell): tokens[shell] for shell in range(0, 6)
        },
    }


def classify_conditioning_audit(
    *,
    qualified: bool,
    dominated_by_threshold: dict[str, dict[str, int]],
    thresholds: tuple[float, ...],
    expected_particles: int,
) -> str:
    """Classify the fixed threshold audit without fitting any parameters."""

    if not qualified:
        return "corr_img_conditioning_inputs_not_qualified"
    expected = {
        "actual_relion": expected_particles,
        "recovar_inverse_noise_only": 0,
        "recovar_ctf_scale_squared_only": expected_particles,
        "recovar_inverse_noise_and_ctf_scale_squared": 0,
    }
    if all(
        dominated_by_threshold.get(_threshold_label(threshold)) == expected
        for threshold in thresholds
    ):
        return CLASSIFICATION
    return "inverse_noise_attribution_is_not_stable_across_ctf_thresholds"


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def build_report(
    *,
    parent_analysis_json: Path,
    full_image_size: int,
) -> dict[str, Any]:
    """Build the fixed 14-particle effective-CTF conditioning audit."""

    parent = _validate_parent(parent_analysis_json)
    _require(
        int(parent.get("full_image_size", -1)) == full_image_size,
        "parent image size changed",
    )
    for binding_name in ("relion_model_star", "relion_data_star", "ctf_pickle"):
        binding = parent.get(binding_name, {})
        binding_path = Path(binding.get("path", ""))
        _require(binding_path.is_file(), f"{binding_name} path is missing")
        _require(
            _sha256(binding_path) == binding.get("sha256"),
            f"{binding_name} hash changed",
        )
    star_precision_partition = _validate_star_precision_partition(
        Path(parent["relion_model_star"]["path"])
    )

    ctf_path = Path(parent["ctf_pickle"]["path"])
    ctf_half, voxel_size = _load_ctf_half(
        ctf_path,
        full_image_size=full_image_size,
    )
    particles = []
    for parent_row in parent["particles"]:
        paths = {
            key: Path(value)
            for key, value in parent_row["artifact_paths"].items()
        }
        for key, path in paths.items():
            _require(
                _sha256(path) == parent_row["artifact_sha256"][key],
                f"parent artifact hash changed: {key}",
            )
        preprocess = load_preprocess(paths["preprocess"])
        operand = load_operand(paths["operand"])
        component = load_component(paths["component"])
        recovar = _load_recovar(paths["recovar"])
        stack_index = int(parent_row["stack_index_one_based"])
        original_index = int(parent_row["original_index_zero_based"])
        _require(
            preprocess.stack_index
            == operand.stack_index
            == component.stack_index
            == stack_index,
            "cross-capture stack identity mismatch",
        )
        _require(
            preprocess.part_id == operand.part_id == component.part_id,
            "cross-capture particle identity mismatch",
        )
        _require(
            int(recovar["original_index"]) == original_index
            and stack_index == original_index + 1,
            "RECOVAR/RELION particle identity mismatch",
        )

        live_base, live_shifted = relion_live_score_base(
            operand,
            recovar,
            full_image_size=full_image_size,
        )
        _, recovar_phase = _recover_base_and_phase(
            live_base=live_base,
            live_shifted=live_shifted,
            recovar_shifted=recovar["shifted_data"],
        )
        window_indices = recovar["window_indices"]
        current_size = int(recovar["current_size"])
        postoptics_native = relion_values_on_recovar_window(
            preprocess.masked_fourier_post_optics[0].reshape(1, -1),
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
        relion_effective_ctf_complex = postoptics_native / relion_pixel_native
        effective_ctf_imaginary_max_abs = float(
            np.max(np.abs(relion_effective_ctf_complex.imag))
        )
        _require(
            effective_ctf_imaginary_max_abs
            <= EFFECTIVE_CTF_IMAGINARY_MAX_ABS,
            "captured effective CTF has a material imaginary component",
        )
        relion_effective_ctf = relion_effective_ctf_complex.real
        recovar_ctf = ctf_half[original_index].reshape(-1)[window_indices]
        scale_correction = float(parent_row["scale_correction"])
        recovar_effective_ctf = -scale_correction * recovar_ctf
        recovar_corr = (
            float(full_image_size**4)
            * recovar["half_weights"]
            * recovar["ctf2_data"]
        )
        shell_indices = np.asarray(
            make_shell_indices_half((full_image_size, full_image_size)),
            dtype=np.int64,
        )[window_indices]

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

        thresholds = {}
        for threshold in FIXED_EFFECTIVE_CTF_THRESHOLDS:
            corr_values, valid = conditioned_corr_img_factorial_values(
                relion_corr_img=relion_corr,
                relion_effective_ctf=relion_effective_ctf,
                recovar_corr_img=recovar_corr,
                recovar_effective_ctf=recovar_effective_ctf,
                effective_ctf_threshold=threshold,
            )
            bases = {
                label: (
                    -relion_pixel_native
                    * correction
                    / (
                        float(full_image_size**2)
                        * recovar["half_weights"]
                    )
                )
                for label, correction in corr_values.items()
            }
            _require(
                _relative_l2(bases["actual_relion"], live_base)
                <= ACTUAL_ARM_REPLAY_RELATIVE_L2,
                "conditioning actual arm does not replay the live base",
            )
            counterfactuals = {}
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
            thresholds[_threshold_label(threshold)] = {
                "effective_ctf_threshold": threshold,
                "valid_pixel_count": int(np.count_nonzero(valid)),
                "total_pixel_count": int(valid.size),
                "valid_pixel_fraction": float(np.mean(valid)),
                "counterfactuals": counterfactuals,
            }
        particles.append(
            {
                "group": parent_row["group"],
                "stack_index_one_based": stack_index,
                "original_index_zero_based": original_index,
                "scale_correction": scale_correction,
                "effective_ctf_imaginary_max_abs": (
                    effective_ctf_imaginary_max_abs
                ),
                "thresholds": thresholds,
                "shell_partition": {},
                "translation_mapping": translation_mapping,
                "artifact_paths": {
                    key: str(path.resolve()) for key, path in paths.items()
                },
                "artifact_sha256": parent_row["artifact_sha256"],
            }
        )
        shell_values, shell_valid = inverse_noise_shell_partition_values(
            relion_corr_img=relion_corr,
            relion_effective_ctf=relion_effective_ctf,
            recovar_corr_img=recovar_corr,
            recovar_effective_ctf=recovar_effective_ctf,
            shell_indices=shell_indices,
            effective_ctf_threshold=(
                SHELL_PARTITION_EFFECTIVE_CTF_THRESHOLD
            ),
        )
        shell_bases = {
            label: (
                -relion_pixel_native
                * correction
                / (
                    float(full_image_size**2)
                    * recovar["half_weights"]
                )
            )
            for label, correction in shell_values.items()
        }
        _require(
            _relative_l2(shell_bases["actual_relion"], live_base)
            <= ACTUAL_ARM_REPLAY_RELATIVE_L2,
            "shell-partition actual arm does not replay the live base",
        )
        shell_counterfactuals = {}
        for label, base in shell_bases.items():
            shifted = base[np.newaxis, :] * recovar_phase
            swapped_norm, swapped_cross = recovar_score_components(
                recovar["references"],
                shifted,
                recovar["ctf2_data"],
                recovar["half_weights"],
            )
            shell_counterfactuals[label] = reference_swap_counterfactual(
                baseline_residual,
                swapped_norm + swapped_cross + selected_raw,
            )
        particles[-1]["shell_partition"] = {
            "effective_ctf_threshold": (
                SHELL_PARTITION_EFFECTIVE_CTF_THRESHOLD
            ),
            "valid_pixel_count": int(np.count_nonzero(shell_valid)),
            "total_pixel_count": int(shell_valid.size),
            "valid_pixel_fraction": float(np.mean(shell_valid)),
            "counterfactuals": shell_counterfactuals,
        }

    dominated_by_threshold = {}
    valid_fraction_by_threshold = {}
    for threshold in FIXED_EFFECTIVE_CTF_THRESHOLDS:
        label = _threshold_label(threshold)
        dominated_by_threshold[label] = {
            arm: sum(
                row["thresholds"][label]["counterfactuals"][arm][
                    "live_reference_dominated"
                ]
                for row in particles
            )
            for arm in ARM_LABELS
        }
        valid_fraction_by_threshold[label] = _summarize(
            [
                row["thresholds"][label]["valid_pixel_fraction"]
                for row in particles
            ]
        )
    classification = classify_conditioning_audit(
        qualified=True,
        dominated_by_threshold=dominated_by_threshold,
        thresholds=FIXED_EFFECTIVE_CTF_THRESHOLDS,
        expected_particles=EXPECTED_PARTICLES,
    )
    _require(
        len(particles) == EXPECTED_PARTICLES,
        "conditioning audit particle count changed",
    )
    shell_partition_dominated = {
        arm: sum(
            row["shell_partition"]["counterfactuals"][arm][
                "live_reference_dominated"
            ]
            for row in particles
        )
        for arm in SHELL_PARTITION_ARM_LABELS
    }
    shell_partition_classification = classify_shell_partition(
        qualified=True,
        dominated=shell_partition_dominated,
        expected_particles=EXPECTED_PARTICLES,
    )
    return {
        "schema": "em-k1-corr-img-conditioning-v2",
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "metric_policy": (
            "fixed effective-CTF thresholds; apply each corr_img factorial "
            "intervention only where both implementations exceed the "
            "threshold and retain actual RELION correction elsewhere; "
            "centered residual-energy removal; no fitted scale/sign; "
            "no correlation"
        ),
        "fixed_gates": {
            "expected_particles": EXPECTED_PARTICLES,
            "component_dominance_fraction_strictly_greater_than": 0.5,
            "effective_ctf_thresholds": list(FIXED_EFFECTIVE_CTF_THRESHOLDS),
            "effective_ctf_imaginary_max_abs": (
                EFFECTIVE_CTF_IMAGINARY_MAX_ABS
            ),
            "actual_arm_replay_relative_l2": (
                ACTUAL_ARM_REPLAY_RELATIVE_L2
            ),
        },
        "fixed_metric": {
            "evaluated_particles": len(particles),
            "expected_particles": EXPECTED_PARTICLES,
            "live_reference_dominated_by_threshold": (
                dominated_by_threshold
            ),
            "valid_pixel_fraction_by_threshold": (
                valid_fraction_by_threshold
            ),
        },
        "shell_partition_metric": {
            "classification": shell_partition_classification,
            "effective_ctf_threshold": (
                SHELL_PARTITION_EFFECTIVE_CTF_THRESHOLD
            ),
            "fixed_decimal_shells": list(STAR_FIXED_DECIMAL_SHELLS),
            "evaluated_particles": len(particles),
            "expected_particles": EXPECTED_PARTICLES,
            "live_reference_dominated": shell_partition_dominated,
            "valid_pixel_fraction": _summarize(
                [
                    row["shell_partition"]["valid_pixel_fraction"]
                    for row in particles
                ]
            ),
            "star_precision_partition": star_precision_partition,
            "metric_policy": (
                "fixed effective-CTF threshold 0.01; C++ ROUND radial "
                "shells; swap RELION inverse noise on shells 1-4 versus "
                "shells 5+ while using RECOVAR CTF-scale squared; retain "
                "actual RELION correction outside the valid mask; centered "
                "residual-energy removal; no fitted scale/sign; no "
                "correlation"
            ),
        },
        "full_image_size": full_image_size,
        "voxel_size": voxel_size,
        "parent_analysis": {
            "path": str(Path(parent_analysis_json).resolve()),
            "sha256": _sha256(parent_analysis_json),
            "classification": parent["classification"],
        },
        "relion_model_star": parent["relion_model_star"],
        "relion_data_star": parent["relion_data_star"],
        "ctf_pickle": parent["ctf_pickle"],
        "particles": particles,
        "notes": [
            (
                "The actual RELION correction remains in excluded pixels, so "
                "no arm can win by zeroing or discarding low-CTF samples."
            ),
            (
                "The fixed 0.01 threshold excludes the most poorly "
                "conditioned divisions while retaining more than 99% of the "
                "captured score window in the authoritative fixture."
            ),
            (
                "Shells 1-4 are the scored shells serialized with six fixed "
                "decimal places in the bound RELION model STAR; shell 5 is "
                "the first shell serialized in scientific notation."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-analysis-json", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(
        not args.output_json.exists(),
        f"refusing to overwrite report: {args.output_json}",
    )
    report = build_report(
        parent_analysis_json=args.parent_analysis_json,
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
