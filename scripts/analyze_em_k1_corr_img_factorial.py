#!/usr/bin/env python3
"""Split the K=1 corr_img residual into inverse-noise and CTF-scale factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    relion_live_score_base,
)
from scripts.analyze_em_k1_score_transfer_factorial import (
    CLASSIFICATION as PARENT_CLASSIFICATION,
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
    "raw_coarse_residual_is_inverse_noise_weight_dominated_"
    "not_ctf_scale_squared"
)
EFFECTIVE_CTF_IMAGINARY_MAX_ABS = 2.0e-5
ACTUAL_ARM_REPLAY_RELATIVE_L2 = 1.0e-14
STACK_NAME = re.compile(r"(?P<stack>[1-9][0-9]*)@")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "recovar_pixel_correction_only": EXPECTED_PARTICLES,
        "recovar_corr_img_only": 0,
        "recovar_pixel_and_corr_img": 0,
    }
    _require(
        fixed.get("evaluated_particles") == EXPECTED_PARTICLES
        and fixed.get("expected_particles") == EXPECTED_PARTICLES
        and fixed.get("live_reference_dominated") == expected,
        "parent fixed factorial did not pass",
    )
    return report


def _scale_by_stack(
    model: dict[str, Any],
    data: dict[str, Any],
    *,
    full_image_size: int,
) -> dict[int, tuple[float, int]]:
    """Resolve exact one-based stack identities to model group scales."""

    general = model.get("model_general")
    groups = model.get("model_groups")
    particles = data.get("particles")
    _require(isinstance(general, dict), "RELION model general block missing")
    _require(groups is not None, "RELION model group block missing")
    _require(particles is not None, "RELION particle data block missing")
    _require(
        int(general.get("rlnOriginalImageSize", -1)) == full_image_size,
        "RELION model image size mismatch",
    )
    _require(
        int(general.get("rlnNrClasses", -1)) == 1
        and int(general.get("rlnNrBodies", -1)) == 1,
        "RELION model must be K=1 single-body",
    )
    required_group_columns = {
        "rlnGroupNumber",
        "rlnGroupScaleCorrection",
    }
    _require(
        required_group_columns.issubset(groups.columns),
        "RELION model group columns missing",
    )
    group_numbers = np.asarray(groups["rlnGroupNumber"], dtype=np.int64)
    group_scales = np.asarray(
        groups["rlnGroupScaleCorrection"],
        dtype=np.float64,
    )
    _require(
        np.array_equal(group_numbers, np.arange(1, len(groups) + 1)),
        "RELION model groups are not contiguous and ordered",
    )
    _require(
        int(general.get("rlnNrGroups", -1)) == len(groups),
        "RELION model group count mismatch",
    )
    _require(
        np.all(np.isfinite(group_scales)) and np.all(group_scales > 0.0),
        "RELION model group scales must be finite and positive",
    )
    required_particle_columns = {
        "rlnImageName",
        "rlnGroupNumber",
        "rlnRandomSubset",
    }
    _require(
        required_particle_columns.issubset(particles.columns),
        "RELION particle identity columns missing",
    )
    result: dict[int, tuple[float, int]] = {}
    for image_name, group_number, subset in zip(
        particles["rlnImageName"],
        particles["rlnGroupNumber"],
        particles["rlnRandomSubset"],
        strict=True,
    ):
        match = STACK_NAME.match(str(image_name))
        _require(match is not None, f"invalid RELION image name: {image_name}")
        stack_index = int(match["stack"])
        group_index = int(group_number) - 1
        _require(
            0 <= group_index < len(group_scales),
            "RELION particle group is out of range",
        )
        _require(stack_index not in result, "duplicate RELION stack identity")
        result[stack_index] = (
            float(group_scales[group_index]),
            int(subset),
        )
    return result


def load_scale_by_stack(
    model_star: Path,
    data_star: Path,
    *,
    full_image_size: int,
) -> dict[int, tuple[float, int]]:
    """Read and validate the exact RELION model/data STAR pair."""

    import starfile

    model = starfile.read(model_star)
    data = starfile.read(data_star)
    _require(isinstance(model, dict), "RELION model STAR is not multi-block")
    _require(isinstance(data, dict), "RELION data STAR is not multi-block")
    return _scale_by_stack(
        model,
        data,
        full_image_size=full_image_size,
    )


def corr_img_factorial_values(
    *,
    relion_corr_img: np.ndarray,
    relion_effective_ctf: np.ndarray,
    recovar_corr_img: np.ndarray,
    recovar_effective_ctf: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Construct the fixed 2x2 inverse-noise/CTF-scale corr_img values."""

    relion_corr = np.asarray(relion_corr_img, dtype=np.float64)
    relion_ctf = np.asarray(relion_effective_ctf, dtype=np.float64)
    recovar_corr = np.asarray(recovar_corr_img, dtype=np.float64)
    recovar_ctf = np.asarray(recovar_effective_ctf, dtype=np.float64)
    _require(
        relion_corr.shape
        == relion_ctf.shape
        == recovar_corr.shape
        == recovar_ctf.shape,
        "corr_img factorial operand shapes differ",
    )
    _require(
        not np.any(np.abs(relion_ctf) <= CTF_ZERO_THRESHOLD)
        and not np.any(np.abs(recovar_ctf) <= CTF_ZERO_THRESHOLD),
        "fixed cohort contains a zero effective-CTF score pixel",
    )
    relion_inverse_noise = relion_corr / (relion_ctf**2)
    recovar_inverse_noise = recovar_corr / (recovar_ctf**2)
    _require(
        np.all(np.isfinite(relion_inverse_noise))
        and np.all(np.isfinite(recovar_inverse_noise)),
        "inferred inverse-noise weight is non-finite",
    )
    values = {
        "actual_relion": relion_corr,
        "recovar_inverse_noise_only": (
            recovar_inverse_noise * relion_ctf**2
        ),
        "recovar_ctf_scale_squared_only": (
            relion_inverse_noise * recovar_ctf**2
        ),
        "recovar_inverse_noise_and_ctf_scale_squared": recovar_corr,
    }
    return values, relion_inverse_noise, recovar_inverse_noise


def classify_corr_img_factorial(
    *,
    qualified: bool,
    dominated: dict[str, int],
    expected_particles: int,
) -> str:
    """Classify the predeclared nested corr_img intervention."""

    if not qualified:
        return "corr_img_factorial_inputs_not_qualified"
    expected = {
        "actual_relion": expected_particles,
        "recovar_inverse_noise_only": 0,
        "recovar_ctf_scale_squared_only": expected_particles,
        "recovar_inverse_noise_and_ctf_scale_squared": 0,
    }
    if dominated == expected:
        return CLASSIFICATION
    return "raw_coarse_residual_has_mixed_inverse_noise_ctf_scale_effect"


def build_report(
    *,
    parent_analysis_json: Path,
    relion_model_star: Path,
    relion_data_star: Path,
    full_image_size: int,
) -> dict[str, Any]:
    """Build the fixed 14-particle nested corr_img factorial report."""

    parent = _validate_parent(parent_analysis_json)
    scale_by_stack = load_scale_by_stack(
        relion_model_star,
        relion_data_star,
        full_image_size=full_image_size,
    )
    ctf_path = Path(parent["ctf_pickle"]["path"])
    _require(
        _sha256(ctf_path) == parent["ctf_pickle"]["sha256"],
        "parent CTF pickle hash changed",
    )
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
        _require(stack_index in scale_by_stack, "RELION scale identity missing")
        scale_correction, random_subset = scale_by_stack[stack_index]
        _require(random_subset == 1, "fixed cohort particle is not in half 1")

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
        recovar_effective_ctf = -scale_correction * recovar_ctf
        recovar_corr = (
            float(full_image_size**4)
            * recovar["half_weights"]
            * recovar["ctf2_data"]
        )
        corr_values, relion_inverse_noise, recovar_inverse_noise = (
            corr_img_factorial_values(
                relion_corr_img=relion_corr,
                relion_effective_ctf=relion_effective_ctf,
                recovar_corr_img=recovar_corr,
                recovar_effective_ctf=recovar_effective_ctf,
            )
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
            "nested factorial actual arm does not replay the live base",
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
                "group": parent_row["group"],
                "stack_index_one_based": stack_index,
                "original_index_zero_based": original_index,
                "scale_correction": scale_correction,
                "effective_ctf_imaginary_max_abs": (
                    effective_ctf_imaginary_max_abs
                ),
                "effective_ctf_relative_l2": _relative_l2(
                    relion_effective_ctf,
                    recovar_effective_ctf,
                ),
                "inverse_noise_relative_l2": _relative_l2(
                    relion_inverse_noise,
                    recovar_inverse_noise,
                ),
                "base_relative_l2": base_relative_l2,
                "counterfactuals": counterfactuals,
                "translation_mapping": translation_mapping,
                "artifact_paths": {
                    key: str(path.resolve()) for key, path in paths.items()
                },
                "artifact_sha256": parent_row["artifact_sha256"],
            }
        )

    dominated = {
        label: sum(
            row["counterfactuals"][label]["live_reference_dominated"]
            for row in particles
        )
        for label in particles[0]["counterfactuals"]
    }
    classification = classify_corr_img_factorial(
        qualified=True,
        dominated=dominated,
        expected_particles=EXPECTED_PARTICLES,
    )
    return {
        "schema": "em-k1-corr-img-factorial-v1",
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "metric_policy": (
            "fixed 2x2 inverse-noise/CTF-scale-squared intervention; "
            "centered residual-energy removal; scale-sensitive relative-L2; "
            "no fitted scale/sign; no correlation"
        ),
        "fixed_gates": {
            "expected_particles": EXPECTED_PARTICLES,
            "component_dominance_fraction_strictly_greater_than": 0.5,
            "ctf_zero_threshold": CTF_ZERO_THRESHOLD,
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
            "live_reference_dominated": dominated,
        },
        "full_image_size": full_image_size,
        "voxel_size": voxel_size,
        "parent_analysis": {
            "path": str(Path(parent_analysis_json).resolve()),
            "sha256": _sha256(parent_analysis_json),
            "classification": parent["classification"],
        },
        "relion_model_star": {
            "path": str(Path(relion_model_star).resolve()),
            "sha256": _sha256(relion_model_star),
        },
        "relion_data_star": {
            "path": str(Path(relion_data_star).resolve()),
            "sha256": _sha256(relion_data_star),
        },
        "ctf_pickle": parent["ctf_pickle"],
        "particles": particles,
        "notes": [
            (
                "RELION's effective CTF times scale is recovered from the "
                "captured post-optics/pixel-corrected image ratio."
            ),
            (
                "RECOVAR effective CTF uses the exact iteration-1 model group "
                "scale and the fixed opposite CTF sign convention."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-analysis-json", type=Path, required=True)
    parser.add_argument("--relion-model-star", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(
        not args.output_json.exists(),
        f"refusing to overwrite report: {args.output_json}",
    )
    report = build_report(
        parent_analysis_json=args.parent_analysis_json,
        relion_model_star=args.relion_model_star,
        relion_data_star=args.relion_data_star,
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
