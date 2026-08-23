#!/usr/bin/env python3
"""Compare passive RELION preprocessing boundaries with RECOVAR's strict CUDA path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.validate_relion_preprocess_capture import (
    RelionPreprocessCapture,
)
from scripts.validate_relion_preprocess_capture import (
    validate_directory as validate_capture,
)

MATERIAL_RELATIVE_L2 = 5.0e-7
STRICT_REPLAY_COUNT = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relion_fourier_to_recovar_centered(
    values: np.ndarray,
    *,
    full_image_size: int,
) -> np.ndarray:
    """Map cropped RELION FFTW rows and 1/N² units to centered-rFFT units."""

    values = np.asarray(values, dtype=np.complex64)
    _require(values.ndim == 3, "RELION Fourier batch must be three-dimensional")
    current_size = int(values.shape[-2])
    _require(
        0 < current_size <= full_image_size and current_size % 2 == 0,
        "RELION Fourier current size must be positive, even, and no larger "
        "than the full image size",
    )
    _require(
        values.shape[-1] == current_size // 2 + 1,
        "RELION Fourier packed width does not match its current size",
    )
    centered = np.fft.fftshift(values, axes=(-2,))
    return (centered * np.float32(full_image_size**2)).astype(
        np.complex64,
        copy=False,
    )


def crop_centered_rfft(values: np.ndarray, *, current_size: int) -> np.ndarray:
    """Window a centered packed rFFT to RELION's current-size row support."""

    values = np.asarray(values, dtype=np.complex64)
    _require(values.ndim == 3, "centered Fourier batch must be three-dimensional")
    full_image_size = int(values.shape[-2])
    _require(
        values.shape[-1] == full_image_size // 2 + 1,
        "centered Fourier packed width does not match its full image size",
    )
    _require(
        0 < current_size <= full_image_size and current_size % 2 == 0,
        "Fourier current size must be positive, even, and no larger than "
        "the full image size",
    )
    current_half_width = current_size // 2 + 1
    natural_rows = np.fft.ifftshift(values, axes=(-2,))
    windowed = np.zeros(
        (values.shape[0], current_size, current_half_width),
        dtype=np.complex64,
    )
    positive_rows = current_size // 2 + 1
    windowed[:, :positive_rows, :] = natural_rows[
        :,
        :positive_rows,
        :current_half_width,
    ]
    negative_rows = current_size - positive_rows
    if negative_rows:
        windowed[:, positive_rows:, :] = natural_rows[
            :,
            full_image_size - negative_rows :,
            :current_half_width,
        ]
    return np.fft.fftshift(windowed, axes=(-2,))


def _per_particle_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Return scale-sensitive error metrics; correlation is intentionally absent."""

    source = np.asarray(source)
    target = np.asarray(target)
    _require(source.shape == target.shape, "stage comparison shapes differ")
    _require(source.ndim >= 2 and source.shape[0] > 0, "stage batch is empty")
    flat_source = source.reshape(source.shape[0], -1).astype(np.complex128)
    flat_target = target.reshape(target.shape[0], -1).astype(np.complex128)
    differences = flat_source - flat_target
    target_norms = np.linalg.norm(flat_target, axis=1)
    _require(np.all(target_norms > 0.0), "stage target contains a zero-norm image")
    relative_l2 = np.linalg.norm(differences, axis=1) / target_norms
    absolute = np.abs(differences)
    bitwise_equal = np.asarray(
        [
            np.array_equal(source[index], target[index])
            for index in range(source.shape[0])
        ],
        dtype=bool,
    )
    return {
        "relative_l2": relative_l2.astype(np.float64).tolist(),
        "relative_l2_min": float(np.min(relative_l2)),
        "relative_l2_median": float(np.median(relative_l2)),
        "relative_l2_max": float(np.max(relative_l2)),
        "p95_abs": np.percentile(absolute, 95, axis=1).astype(np.float64).tolist(),
        "max_abs": np.max(absolute, axis=1).astype(np.float64).tolist(),
        "bitwise_equal": bitwise_equal.tolist(),
        "bitwise_equal_count": int(np.count_nonzero(bitwise_equal)),
        "material_relative_l2_count": int(
            np.count_nonzero(relative_l2 > MATERIAL_RELATIVE_L2)
        ),
        "evaluated_particles": int(source.shape[0]),
    }


def classify_earliest_boundary(
    stage_metrics: dict[str, dict[str, Any]],
    *,
    expected_particles: int,
) -> str:
    """Classify the first cross-engine stage above the fixed material threshold."""

    _require(expected_particles > 0, "expected particle count must be positive")
    for stage, classification in (
        (
            "normalized_shifted_real",
            "material_gap_begins_at_normalization_or_rounded_translation",
        ),
        ("unmasked_fourier_pre_optics", "material_gap_begins_at_unmasked_fft"),
        ("masked_real", "material_gap_begins_at_softmask"),
        ("masked_fourier_pre_optics", "material_gap_begins_at_masked_fft"),
        ("masked_fourier_post_optics", "material_gap_begins_at_optics_correction"),
    ):
        _require(stage in stage_metrics, f"missing stage metric: {stage}")
        evaluated = stage_metrics[stage]["evaluated_particles"]
        _require(evaluated == expected_particles, f"stage denominator mismatch: {stage}")
        if stage_metrics[stage]["material_relative_l2_count"] > 0:
            return classification
    return "all_preprocessing_boundaries_within_fixed_material_threshold"


def build_report_from_arrays(
    *,
    captures: tuple[RelionPreprocessCapture, ...],
    disk_raw: np.ndarray,
    strict_normalized: np.ndarray,
    strict_masked: np.ndarray,
    strict_unmasked_fourier: np.ndarray,
    strict_masked_fourier: np.ndarray,
    repeat_normalized: tuple[np.ndarray, ...],
    repeat_masked: tuple[np.ndarray, ...],
    capture_validation: dict[str, Any],
    particle_stack: Path,
    gpu_uuid: str,
) -> dict[str, Any]:
    """Build the fixed-cohort boundary report from already materialized arrays."""

    expected = len(captures)
    _require(expected > 0, "capture cohort is empty")
    relion_raw = np.stack([artifact.raw_input_real[0] for artifact in captures])
    relion_normalized = np.stack(
        [artifact.normalized_shifted_real[0] for artifact in captures]
    )
    relion_masked = np.stack([artifact.masked_real[0] for artifact in captures])
    full_image_size = relion_raw.shape[-1]
    _require(
        relion_raw.shape == (expected, full_image_size, full_image_size),
        "captured real-space topology mismatch",
    )

    def captured_fourier(name: str) -> np.ndarray:
        values = np.stack([getattr(artifact, name)[0] for artifact in captures])
        return relion_fourier_to_recovar_centered(
            values,
            full_image_size=full_image_size,
        )

    relion_unmasked_pre = captured_fourier("unmasked_fourier_pre_optics")
    relion_unmasked_post = captured_fourier("unmasked_fourier_post_optics")
    relion_masked_pre = captured_fourier("masked_fourier_pre_optics")
    relion_masked_post = captured_fourier("masked_fourier_post_optics")
    current_size = int(relion_unmasked_pre.shape[-2])
    strict_unmasked_fourier = crop_centered_rfft(
        strict_unmasked_fourier,
        current_size=current_size,
    )
    strict_masked_fourier = crop_centered_rfft(
        strict_masked_fourier,
        current_size=current_size,
    )
    stage_metrics = {
        "disk_raw_input": _per_particle_metrics(relion_raw, disk_raw),
        "normalized_shifted_real": _per_particle_metrics(
            relion_normalized,
            strict_normalized,
        ),
        "masked_real": _per_particle_metrics(relion_masked, strict_masked),
        "unmasked_fourier_pre_optics": _per_particle_metrics(
            relion_unmasked_pre,
            strict_unmasked_fourier,
        ),
        "masked_fourier_pre_optics": _per_particle_metrics(
            relion_masked_pre,
            strict_masked_fourier,
        ),
        "masked_fourier_post_optics": _per_particle_metrics(
            relion_masked_post,
            strict_masked_fourier,
        ),
        "relion_unmasked_optics_effect": _per_particle_metrics(
            relion_unmasked_post,
            relion_unmasked_pre,
        ),
        "relion_masked_optics_effect": _per_particle_metrics(
            relion_masked_post,
            relion_masked_pre,
        ),
    }
    _require(
        len(repeat_normalized) == len(repeat_masked) == STRICT_REPLAY_COUNT,
        "strict replay count mismatch",
    )
    repeat_metrics = {
        "normalized_shifted_real": [
            _per_particle_metrics(values, strict_normalized)
            for values in repeat_normalized
        ],
        "masked_real": [
            _per_particle_metrics(values, strict_masked)
            for values in repeat_masked
        ],
    }
    classification = classify_earliest_boundary(
        stage_metrics,
        expected_particles=expected,
    )
    return {
        "schema": "relion-recovar-preprocess-boundary-v1",
        "status": "pass",
        "classification": classification,
        "metric_policy": (
            "scale-sensitive relative-L2/p95/max and bitwise equality; "
            "no correlation; fixed material relative-L2 threshold"
        ),
        "material_relative_l2_threshold": MATERIAL_RELATIVE_L2,
        "particle_count": expected,
        "full_image_size": full_image_size,
        "fourier_current_size": current_size,
        "gpu_uuid": gpu_uuid,
        "particle_stack": str(Path(particle_stack).resolve()),
        "particle_stack_sha256": _sha256(particle_stack),
        "capture_validation": capture_validation,
        "stack_indices_one_based": [artifact.stack_index for artifact in captures],
        "part_ids": [artifact.part_id for artifact in captures],
        "norm_corrections": [artifact.norm_correction for artifact in captures],
        "rounded_old_offsets": [
            artifact.old_offset.astype(np.float64).tolist() for artifact in captures
        ],
        "stage_metrics": stage_metrics,
        "strict_repeat_metrics": repeat_metrics,
        "fixed_metric": {
            "evaluated_particles": expected,
            "expected_particles": expected,
            "disk_raw_bitwise_equal": stage_metrics["disk_raw_input"][
                "bitwise_equal_count"
            ],
            "normalized_material_gap": stage_metrics["normalized_shifted_real"][
                "material_relative_l2_count"
            ],
            "masked_material_gap": stage_metrics["masked_real"][
                "material_relative_l2_count"
            ],
            "unmasked_fft_material_gap": stage_metrics[
                "unmasked_fourier_pre_optics"
            ]["material_relative_l2_count"],
            "masked_fft_material_gap": stage_metrics["masked_fourier_pre_optics"][
                "material_relative_l2_count"
            ],
            "masked_post_optics_material_gap": stage_metrics[
                "masked_fourier_post_optics"
            ]["material_relative_l2_count"],
        },
    }


def _allocated_gpu_uuid(expected_gpu_uuid: str) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            expected_gpu_uuid,
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    _require(actual == expected_gpu_uuid, "allocated GPU UUID mismatch")
    return actual


def run_gpu_analysis(
    *,
    capture_directory: Path,
    particle_stack: Path,
    cohort_json: Path,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    """Replay RECOVAR's current strict preprocessing on the requested GPU."""

    import jax
    import jax.numpy as jnp

    from recovar.cuda_backproject import relion_preprocess_real_f32
    from recovar.data_io.image_backends import _centered_rfft2_jax
    from recovar.data_io.image_loader import ImageLoader

    _require(jax.default_backend() == "gpu", "preprocessing analysis requires JAX GPU")
    gpu_uuid = _allocated_gpu_uuid(expected_gpu_uuid)
    cohort = json.loads(Path(cohort_json).read_text())
    expected = int(cohort["selected_particle_count"])
    _require(expected == 14, "fixed preprocessing denominator must be 14")
    expected_parts = np.asarray(
        [row["relion_part_id"] for row in cohort["rows"]],
        dtype=np.int64,
    )
    expected_stacks = np.asarray(
        cohort["selected_stack_indices_one_based"],
        dtype=np.int64,
    )
    captures, capture_validation = validate_capture(
        capture_directory,
        expected_particles=expected,
        expected_part_ids=expected_parts,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=int(cohort["mpi_rank"]),
        expected_iteration=int(cohort["iteration"]),
    )
    capture_by_stack = {artifact.stack_index: artifact for artifact in captures}
    captures = tuple(capture_by_stack[int(index)] for index in expected_stacks)
    loader = ImageLoader.from_file(str(particle_stack), lazy=True)
    disk_raw = np.asarray(loader[expected_stacks - 1], dtype=np.float32)
    normalization_factors = np.asarray(
        [artifact.norm_correction for artifact in captures],
        dtype=np.float32,
    )
    integer_shifts = np.stack(
        [artifact.old_offset[:2] for artifact in captures]
    ).astype(np.int32)
    mask_parameters = captures[0].mask_parameters
    radius = float(mask_parameters["radius"])
    cosine_width = float(mask_parameters["cosine_width"])
    for artifact in captures[1:]:
        _require(
            artifact.mask_parameters["radius"] == radius
            and artifact.mask_parameters["cosine_width"] == cosine_width,
            "capture cohort mask geometry differs",
        )

    images_jax = jnp.asarray(disk_raw, dtype=jnp.float32)
    factors_jax = jnp.asarray(normalization_factors, dtype=jnp.float32)
    shifts_jax = jnp.asarray(integer_shifts, dtype=jnp.int32)
    normalized_jax, masked_jax = relion_preprocess_real_f32(
        images_jax,
        factors_jax,
        shifts_jax,
        radius,
        cosine_width,
        True,
    )
    strict_unmasked_fourier_jax = _centered_rfft2_jax(normalized_jax)
    strict_masked_fourier_jax = _centered_rfft2_jax(masked_jax)
    strict_masked_fourier_jax.block_until_ready()
    strict_normalized = np.asarray(normalized_jax, dtype=np.float32)
    strict_masked = np.asarray(masked_jax, dtype=np.float32)
    strict_unmasked_fourier = np.asarray(
        strict_unmasked_fourier_jax,
        dtype=np.complex64,
    )
    strict_masked_fourier = np.asarray(
        strict_masked_fourier_jax,
        dtype=np.complex64,
    )
    repeat_normalized = []
    repeat_masked = []
    for _ in range(STRICT_REPLAY_COUNT):
        repeat_normalized_jax, repeat_masked_jax = relion_preprocess_real_f32(
            images_jax,
            factors_jax,
            shifts_jax,
            radius,
            cosine_width,
            True,
        )
        repeat_masked_jax.block_until_ready()
        repeat_normalized.append(
            np.asarray(repeat_normalized_jax, dtype=np.float32)
        )
        repeat_masked.append(np.asarray(repeat_masked_jax, dtype=np.float32))
    return build_report_from_arrays(
        captures=captures,
        disk_raw=disk_raw,
        strict_normalized=strict_normalized,
        strict_masked=strict_masked,
        strict_unmasked_fourier=strict_unmasked_fourier,
        strict_masked_fourier=strict_masked_fourier,
        repeat_normalized=tuple(repeat_normalized),
        repeat_masked=tuple(repeat_masked),
        capture_validation=capture_validation,
        particle_stack=particle_stack,
        gpu_uuid=gpu_uuid,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--particle-stack", type=Path, required=True)
    parser.add_argument("--cohort-json", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = run_gpu_analysis(
        capture_directory=args.capture_directory,
        particle_stack=args.particle_stack,
        cohort_json=args.cohort_json,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["fixed_metric"], indent=2, sort_keys=True))
    print(report["classification"])


if __name__ == "__main__":
    main()
