#!/usr/bin/env python3
"""Factor live RELION score operands against fixed K=1 coarse residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _relion_parent_to_recovar,
    _translation_permutation,
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

COMPONENT_DOMINANCE_FRACTION = 0.5
RECOVAR_REPLAY_P95_GATE = 5.0e-5
RECOVAR_REPLAY_MAX_GATE = 2.0e-4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values)


def _relative_l2(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source)
    target = np.asarray(target)
    _require(source.shape == target.shape, "relative-L2 shapes differ")
    denominator = float(np.linalg.norm(target))
    _require(denominator > 0.0, "relative-L2 target has zero norm")
    return float(np.linalg.norm(source - target) / denominator)


def relion_reference_on_recovar_window(
    reference: np.ndarray,
    window_indices: np.ndarray,
    *,
    full_image_size: int,
    current_size: int,
) -> np.ndarray:
    """Convert RELION FFTW-row references to RECOVAR centered-DFT units."""

    reference = np.asarray(reference)
    window_indices = np.asarray(window_indices, dtype=np.int64)
    _require(reference.ndim == 2, "RELION reference array must be 2D")
    _require(full_image_size > 0 and current_size > 0, "image sizes must be positive")
    full_half = full_image_size // 2 + 1
    current_half = current_size // 2 + 1
    _require(
        reference.shape[1] == current_size * current_half,
        "RELION reference current-size topology mismatch",
    )
    rows = window_indices // full_half
    columns = window_indices % full_half
    ky = rows - full_image_size // 2
    _require(np.all(np.abs(ky) <= current_size // 2), "window row is outside current size")
    _require(np.all(columns < current_half), "window column is outside current size")
    relion_rows = np.where(ky >= 0, ky, current_size + ky)
    relion_indices = relion_rows * current_half + columns
    return -(full_image_size**2) * reference[:, relion_indices]


def relion_values_on_recovar_window(
    values: np.ndarray,
    window_indices: np.ndarray,
    *,
    full_image_size: int,
    current_size: int,
) -> np.ndarray:
    """Select RELION FFTW rows without applying the projection conversion."""

    return -relion_reference_on_recovar_window(
        values,
        window_indices,
        full_image_size=full_image_size,
        current_size=current_size,
    ) / (full_image_size**2)


def recovar_score_components(
    projected_reference: np.ndarray,
    shifted_data: np.ndarray,
    ctf2_data: np.ndarray,
    half_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute RECOVAR's coarse norm/cross log-score components."""

    projected_reference = np.asarray(projected_reference, dtype=np.complex128)
    shifted_data = np.asarray(shifted_data, dtype=np.complex128)
    ctf2_data = np.asarray(ctf2_data, dtype=np.float64)
    half_weights = np.asarray(half_weights, dtype=np.float64)
    _require(projected_reference.ndim == 2, "projected references must be 2D")
    _require(shifted_data.ndim == 2, "shifted data must be 2D")
    n_pixels = projected_reference.shape[1]
    _require(shifted_data.shape[1] == n_pixels, "shifted-data pixel mismatch")
    _require(ctf2_data.shape == (n_pixels,), "CTF pixel mismatch")
    _require(half_weights.shape == (n_pixels,), "half-weight pixel mismatch")
    norm = -0.5 * np.sum(
        ctf2_data[np.newaxis, :] * np.abs(projected_reference) ** 2 * half_weights[np.newaxis, :],
        axis=1,
    )
    cross = np.real(
        np.einsum(
            "tp,rp,p->rt",
            np.conj(shifted_data),
            projected_reference,
            half_weights,
            optimize=True,
        )
    )
    return np.broadcast_to(norm[:, np.newaxis], cross.shape).copy(), cross


def reference_swap_counterfactual(
    total_residual: np.ndarray,
    counterfactual_residual: np.ndarray,
) -> dict[str, Any]:
    """Measure centered residual-energy removal without correlation."""

    total = np.asarray(total_residual, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_residual, dtype=np.float64)
    _require(total.shape == counterfactual.shape, "counterfactual shape mismatch")
    _require(total.ndim == 2 and min(total.shape) > 1, "score panel is too small")
    baseline = _center(total)
    swapped = _center(counterfactual)
    baseline_energy = float(np.sum(baseline**2))
    _require(baseline_energy > 0.0, "baseline residual has zero energy")
    swapped_energy = float(np.sum(swapped**2))
    removal = 1.0 - swapped_energy / baseline_energy
    return {
        "baseline_centered_energy": baseline_energy,
        "swapped_centered_energy": swapped_energy,
        "counterfactual_energy_removal_fraction": float(removal),
        "live_reference_dominated": bool(removal > COMPONENT_DOMINANCE_FRACTION),
        "swapped_centered_p95_abs": float(np.percentile(np.abs(swapped), 95)),
        "swapped_centered_max_abs": float(np.max(np.abs(swapped))),
    }


def classify_live_reference(*, capture_qualified: bool, dominated: int, expected: int) -> str:
    """Classify the fixed-cohort counterfactual without changing its gate."""

    if not capture_qualified:
        return "operand_capture_not_qualified"
    if dominated == expected:
        return "raw_coarse_residual_is_live_projected_reference_dominated"
    if dominated == 0:
        return "live_projected_reference_rejected_as_raw_coarse_residual_cause"
    return "raw_coarse_residual_has_mixed_live_projected_reference_effect"


def classify_live_operands(
    *,
    capture_qualified: bool,
    dominated: dict[str, int],
    expected: int,
) -> str:
    """Classify the fixed operand factorial at its unchanged majority gate."""

    if not capture_qualified:
        return "operand_capture_not_qualified"
    if (
        dominated["base_corrected_image"] == expected
        and dominated["shifted_image"] == expected
        and dominated["all_live"] == expected
        and dominated["reference"] == 0
        and dominated["correction"] == 0
        and dominated["translation_phase"] == 0
    ):
        return (
            "raw_coarse_residual_is_live_base_corrected_image_dominated_"
            "not_reference_correction_or_translation_phase"
        )
    if dominated["shifted_image"] == expected:
        return "raw_coarse_residual_is_live_shifted_image_dominated"
    return "raw_coarse_residual_has_mixed_live_operand_effect"


def _load_recovar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "original_index",
            "current_size",
            "translations",
            "window_indices",
            "shifted_data",
            "ctf2_data",
            "half_weights",
            "scores_pre_prior_per_class",
            "projected_reference_rotation_ids",
            "projected_reference_per_class",
            "projected_reference_norm_score_per_class",
            "projected_cross_score_per_class",
        }
        _require(required <= set(payload.files), f"missing RECOVAR fields: {path}")
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "original_index": int(payload["original_index"]),
            "current_size": int(payload["current_size"]),
            "translations": np.asarray(payload["translations"], dtype=np.float64),
            "window_indices": np.asarray(payload["window_indices"], dtype=np.int64),
            "shifted_data": np.asarray(payload["shifted_data"], dtype=np.complex128),
            "ctf2_data": np.asarray(payload["ctf2_data"][0], dtype=np.float64),
            "half_weights": np.asarray(payload["half_weights"], dtype=np.float64),
            "scores": np.asarray(
                payload["scores_pre_prior_per_class"][0],
                dtype=np.float64,
            ),
            "rotation_ids": np.asarray(
                payload["projected_reference_rotation_ids"],
                dtype=np.int64,
            ),
            "references": np.asarray(
                payload["projected_reference_per_class"][0],
                dtype=np.complex128,
            ),
            "norms": np.asarray(
                payload["projected_reference_norm_score_per_class"][0],
                dtype=np.float64,
            ),
            "crosses": np.asarray(
                payload["projected_cross_score_per_class"][0],
                dtype=np.float64,
            ),
        }


def _operand_complex(operand: CoarseOperandCapture) -> np.ndarray:
    return operand.reference_real.astype(np.float64) + 1j * operand.reference_imag.astype(np.float64)


def build_report(
    *,
    cohort_json: Path,
    capture_directory: Path,
    recovar_directory: Path,
    full_image_size: int,
) -> dict[str, Any]:
    cohort = json.loads(Path(cohort_json).read_text())
    _require(cohort["selected_particle_count"] == 14, "cohort denominator must be 14")
    expected_stacks = np.asarray(
        cohort["selected_stack_indices_one_based"],
        dtype=np.int64,
    )
    operands, operand_validation = validate_operands(
        capture_directory,
        expected_particles=14,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=1,
    )
    components, _ = validate_components(
        capture_directory,
        expected_particles=14,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=1,
    )
    operands_by_stack = {item.stack_index: item for item in operands}
    components_by_stack = {item.stack_index: item for item in components}
    recovar_items = [_load_recovar(path) for path in sorted(Path(recovar_directory).glob("*.npz"))]
    _require(len(recovar_items) == 14, "RECOVAR denominator must be 14")
    recovar_by_index = {item["original_index"]: item for item in recovar_items}
    _require(len(recovar_by_index) == 14, "duplicate RECOVAR original index")

    particles = []
    for cohort_row in cohort["rows"]:
        stack_index = cohort_row["stack_index_one_based"]
        operand = operands_by_stack[stack_index]
        component = components_by_stack[stack_index]
        recovar = recovar_by_index[cohort_row["original_index_zero_based"]]
        n_directions, n_psi, _ = component.header[10:13]
        mapped_operand_ids = np.asarray(
            [
                _relion_parent_to_recovar(
                    int(key),
                    n_directions=n_directions,
                    n_psi=n_psi,
                )
                for key in operand.rotation_keys
            ],
            dtype=np.int64,
        )
        operand_index = {int(key): index for index, key in enumerate(mapped_operand_ids)}
        _require(
            set(operand_index) == set(recovar["rotation_ids"].tolist()),
            "RELION/RECOVAR captured rotation sets differ",
        )
        order = np.asarray(
            [operand_index[int(key)] for key in recovar["rotation_ids"]],
            dtype=np.int64,
        )
        relion_reference = relion_reference_on_recovar_window(
            _operand_complex(operand)[order],
            recovar["window_indices"],
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )
        replay_norm, replay_cross = recovar_score_components(
            recovar["references"],
            recovar["shifted_data"],
            recovar["ctf2_data"],
            recovar["half_weights"],
        )
        replay_error = replay_norm + replay_cross - recovar["scores"][recovar["rotation_ids"]]
        replay_absolute = np.abs(replay_error)
        recovar_replay = {
            "p95_abs": float(np.percentile(replay_absolute, 95)),
            "max_abs": float(np.max(replay_absolute)),
        }
        translation_permutation, translation_mapping = _translation_permutation(
            component.translations,
            recovar["translations"],
        )
        relion_shifted = relion_values_on_recovar_window(
            operand.shifted_real.astype(np.float64)
            + 1j * operand.shifted_imag.astype(np.float64),
            recovar["window_indices"],
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )
        relion_shifted_ordered = np.empty_like(relion_shifted)
        relion_shifted_ordered[translation_permutation] = relion_shifted
        relion_correction = relion_values_on_recovar_window(
            operand.correction[np.newaxis, :],
            recovar["window_indices"],
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )[0].real
        half_weights = recovar["half_weights"]
        _require(np.all(half_weights > 0.0), "RECOVAR half weights must be positive")
        image_normalization = float(full_image_size**2)
        relion_shifted_recovar = (
            -relion_shifted_ordered
            * relion_correction[np.newaxis, :]
            / (image_normalization * half_weights[np.newaxis, :])
        )
        relion_ctf2_recovar = relion_correction / (
            image_normalization**2 * half_weights
        )
        relion_base = relion_values_on_recovar_window(
            (
                operand.image_real.astype(np.float64)
                + 1j * operand.image_imag.astype(np.float64)
            )[np.newaxis, :],
            recovar["window_indices"],
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )[0]
        relion_base_recovar = (
            -relion_base
            * relion_correction
            / (image_normalization * half_weights)
        )
        base_power = np.abs(relion_base_recovar) ** 2
        nonzero_base = base_power > float(np.max(base_power)) * 1.0e-16
        _require(np.count_nonzero(nonzero_base) > 1, "live base image is empty")
        relion_phase = np.ones_like(relion_shifted_recovar)
        relion_phase[:, nonzero_base] = (
            relion_shifted_recovar[:, nonzero_base]
            * np.conj(relion_base_recovar[np.newaxis, nonzero_base])
            / base_power[np.newaxis, nonzero_base]
        )
        relion_phase[:, nonzero_base] /= np.abs(
            relion_phase[:, nonzero_base]
        )
        recovar_bases = recovar["shifted_data"] / relion_phase
        recovar_base = np.mean(recovar_bases, axis=0)
        recovar_base_power = np.abs(recovar_base) ** 2
        nonzero_recovar_base = (
            recovar_base_power
            > float(np.max(recovar_base_power)) * 1.0e-16
        )
        _require(
            np.count_nonzero(nonzero_recovar_base) > 1,
            "RECOVAR base image is empty",
        )
        recovar_phase = np.ones_like(recovar["shifted_data"])
        recovar_phase[:, nonzero_recovar_base] = (
            recovar["shifted_data"][:, nonzero_recovar_base]
            / recovar_base[np.newaxis, nonzero_recovar_base]
        )
        base_only_shifted = relion_base_recovar[np.newaxis, :] * recovar_phase
        phase_only_shifted = recovar_base[np.newaxis, :] * relion_phase

        mapped_raw = _map_relion_table(
            component.raw_diff2,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=translation_permutation,
        )
        selected_raw = mapped_raw[recovar["rotation_ids"]]
        _require(
            np.all(selected_raw != RELION_INVALID_DIFF2),
            "captured rotation has inactive RELION score",
        )
        recovar_total = recovar["scores"][recovar["rotation_ids"]]
        baseline_residual = recovar_total + selected_raw
        configurations = {
            "reference": (
                relion_reference,
                recovar["shifted_data"],
                recovar["ctf2_data"],
            ),
            "shifted_image": (
                recovar["references"],
                relion_shifted_recovar,
                recovar["ctf2_data"],
            ),
            "correction": (
                recovar["references"],
                recovar["shifted_data"],
                relion_ctf2_recovar,
            ),
            "reference_and_shifted_image": (
                relion_reference,
                relion_shifted_recovar,
                recovar["ctf2_data"],
            ),
            "reference_and_correction": (
                relion_reference,
                recovar["shifted_data"],
                relion_ctf2_recovar,
            ),
            "shifted_image_and_correction": (
                recovar["references"],
                relion_shifted_recovar,
                relion_ctf2_recovar,
            ),
            "all_live": (
                relion_reference,
                relion_shifted_recovar,
                relion_ctf2_recovar,
            ),
            "base_corrected_image": (
                recovar["references"],
                base_only_shifted,
                recovar["ctf2_data"],
            ),
            "translation_phase": (
                recovar["references"],
                phase_only_shifted,
                recovar["ctf2_data"],
            ),
        }
        counterfactuals = {}
        for label, (reference, shifted, ctf2) in configurations.items():
            swapped_norm, swapped_cross = recovar_score_components(
                reference,
                shifted,
                ctf2,
                half_weights,
            )
            counterfactuals[label] = reference_swap_counterfactual(
                baseline_residual,
                swapped_norm + swapped_cross + selected_raw,
            )
        reference_delta = relion_reference - recovar["references"]
        operand_relative_l2 = {
            "reference": _relative_l2(
                relion_reference,
                recovar["references"],
            ),
            "correction": _relative_l2(
                relion_correction,
                recovar["ctf2_data"] * half_weights * image_normalization**2,
            ),
            "shifted_image": _relative_l2(
                relion_shifted_recovar,
                recovar["shifted_data"],
            ),
            "base_corrected_image": _relative_l2(
                relion_base_recovar,
                recovar_base,
            ),
            "translation_dependent_base_variation": float(
                np.linalg.norm(
                    recovar_bases - recovar_base[np.newaxis, :]
                )
                / np.linalg.norm(recovar_bases)
            ),
        }
        particles.append(
            {
                "group": cohort_row["group"],
                "stack_index_one_based": stack_index,
                "original_index_zero_based": cohort_row["original_index_zero_based"],
                "rotation_count": int(recovar["rotation_ids"].size),
                "translation_count": int(recovar["translations"].shape[0]),
                "recovar_replay": recovar_replay,
                "recovar_replay_passed": bool(
                    recovar_replay["p95_abs"] <= RECOVAR_REPLAY_P95_GATE
                    and recovar_replay["max_abs"] <= RECOVAR_REPLAY_MAX_GATE
                ),
                "reference_relative_l2": float(np.linalg.norm(reference_delta) / np.linalg.norm(recovar["references"])),
                "reference_max_abs": float(np.max(np.abs(reference_delta))),
                "operand_relative_l2": operand_relative_l2,
                "counterfactual": counterfactuals["reference"],
                "counterfactuals": counterfactuals,
                "translation_mapping": translation_mapping,
                "artifact_paths": {
                    "operand": str(operand.path.resolve()),
                    "component": str(component.path.resolve()),
                    "recovar": recovar["path"],
                },
                "artifact_sha256": {
                    "operand": operand.sha256,
                    "component": component.sha256,
                    "recovar": recovar["sha256"],
                },
            }
        )

    qualified = operand_validation["status"] == "pass"
    dominated = {
        label: sum(
            row["counterfactuals"][label]["live_reference_dominated"]
            for row in particles
        )
        for label in particles[0]["counterfactuals"]
    }
    fixed_metric = {
        "evaluated_particles": len(particles),
        "expected_particles": 14,
        "operand_capture_qualified": 14 if qualified else 0,
        "recovar_replay_passed": sum(row["recovar_replay_passed"] for row in particles),
        "live_reference_dominated": dominated["reference"],
        "live_operand_dominated": dominated,
    }
    classification = classify_live_operands(
        capture_qualified=qualified,
        dominated=dominated,
        expected=fixed_metric["expected_particles"],
    )
    return {
        "schema": "em-k1-live-operand-factorial-v2",
        "status": "complete",
        "classification_ready": qualified,
        "classification": classification,
        "fixed_metric": fixed_metric,
        "fixed_gates": {
            "component_dominance_fraction_strictly_greater_than": (COMPONENT_DOMINANCE_FRACTION),
            "recovar_replay_p95_abs_max": RECOVAR_REPLAY_P95_GATE,
            "recovar_replay_max_abs_max": RECOVAR_REPLAY_MAX_GATE,
        },
        "fixed_conventions": {
            "relion_reference_to_recovar_scale": -(full_image_size**2),
            "relion_correction_to_recovar_ctf2_scale": (
                f"1 / ({full_image_size}**4 * half_weights)"
            ),
            "relion_shifted_to_recovar_scale": (
                f"-correction / ({full_image_size}**2 * half_weights)"
            ),
            "relion_rows": "FFTW packed",
            "recovar_rows": "centered rFFT",
        },
        "operand_validation": operand_validation,
        "particles": particles,
        "notes": [
            "No correlation is computed.",
            "The fixed factorial replaces reference, shifted image, and correction singly, pairwise, and jointly.",
            "The shifted-image arm is split into base corrected image and translation phase.",
            "The RELION reference uses the fixed RECOVAR sign and 2D DFT normalization.",
            "This diagnostic does not update the immutable parity scorecard.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-json", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        cohort_json=args.cohort_json,
        capture_directory=args.capture_directory,
        recovar_directory=args.recovar_directory,
        full_image_size=args.full_image_size,
    )
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(encoded)
    print(encoded, end="")
    if not report["classification_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
