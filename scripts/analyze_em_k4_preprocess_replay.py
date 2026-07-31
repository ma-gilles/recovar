#!/usr/bin/env python3
"""Replay a sealed K=4 particle through RECOVAR's RELION CUDA preprocessor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

MATERIAL_RELATIVE_L2 = 5.0e-7
REPLAY_COUNT = 4
REPORT_SCHEMA = "recovar.em_k4_preprocess_replay.v1"
REQUIRED_BUNDLE_FIELDS = {
    "class_index",
    "current_size",
    "half",
    "high_precision_operand_bundle",
    "image_corrections",
    "image_mask",
    "image_mask_mode",
    "image_shape",
    "integer_pre_shifts",
    "iteration",
    "original_indices",
    "preprocess_backend",
    "raw_real_images",
    "relion_cuda_preprocess",
    "relion_preprocess_normalization_factors",
    "scale_corrections",
    "schema",
    "score_with_masked_images",
    "voxel_size",
}
STAGE_ORDER = (
    "normalized_shifted_real",
    "masked_real",
    "masked_fourier",
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


def _strict_mismatch_count(reference: np.ndarray, repeat: np.ndarray) -> int:
    reference = np.ascontiguousarray(reference)
    repeat = np.ascontiguousarray(repeat)
    _require(reference.shape == repeat.shape, "replay stage shapes differ")
    _require(reference.dtype == repeat.dtype, "replay stage dtypes differ")
    reference_bytes = (
        reference.reshape(-1)
        .view(np.uint8)
        .reshape(
            -1,
            reference.dtype.itemsize,
        )
    )
    repeat_bytes = (
        repeat.reshape(-1)
        .view(np.uint8)
        .reshape(
            -1,
            repeat.dtype.itemsize,
        )
    )
    return int(np.count_nonzero(np.any(reference_bytes != repeat_bytes, axis=1)))


def _pair_metrics(reference: np.ndarray, repeat: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference)
    repeat = np.asarray(repeat)
    _require(reference.shape == repeat.shape, "replay stage shapes differ")
    _require(reference.dtype == repeat.dtype, "replay stage dtypes differ")
    _require(
        np.all(np.isfinite(reference)) and np.all(np.isfinite(repeat)),
        "replay stage contains non-finite values",
    )
    difference = repeat.astype(np.complex128) - reference.astype(np.complex128)
    absolute = np.abs(difference).reshape(-1)
    reference_norm = float(np.linalg.norm(reference.astype(np.complex128).reshape(-1)))
    _require(reference_norm > 0.0, "replay reference has zero norm")
    mismatch_count = _strict_mismatch_count(reference, repeat)
    relative_l2 = float(np.linalg.norm(difference.reshape(-1)) / reference_norm)
    return {
        "element_count": int(reference.size),
        "mismatch_count": mismatch_count,
        "byte_equal": mismatch_count == 0,
        "relative_l2": relative_l2,
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
        "within_fixed_material_floor": relative_l2 <= MATERIAL_RELATIVE_L2,
    }


def analyze_replay_arrays(
    *,
    normalized_runs: np.ndarray,
    masked_runs: np.ndarray,
    masked_fourier_runs: np.ndarray,
) -> dict[str, Any]:
    """Classify repeated preprocessing arrays against their first execution."""

    runs = {
        "normalized_shifted_real": np.asarray(normalized_runs),
        "masked_real": np.asarray(masked_runs),
        "masked_fourier": np.asarray(masked_fourier_runs),
    }
    for stage, values in runs.items():
        _require(
            values.ndim >= 2 and values.shape[0] == REPLAY_COUNT,
            f"{stage} must contain exactly {REPLAY_COUNT} executions",
        )
    batch_size = int(runs["normalized_shifted_real"].shape[1])
    _require(batch_size > 0, "preprocessing replay batch is empty")
    _require(
        runs["masked_real"].shape[:2] == (REPLAY_COUNT, batch_size)
        and runs["masked_fourier"].shape[:2] == (REPLAY_COUNT, batch_size),
        "preprocessing replay batch shapes differ",
    )

    stages = {}
    for stage in STAGE_ORDER:
        values = runs[stage]
        comparisons = [_pair_metrics(values[0], values[repeat_index]) for repeat_index in range(1, REPLAY_COUNT)]
        stages[stage] = {
            "comparison_count": len(comparisons),
            "bitwise_equal_comparison_count": sum(comparison["byte_equal"] for comparison in comparisons),
            "within_fixed_material_floor_count": sum(
                comparison["within_fixed_material_floor"] for comparison in comparisons
            ),
            "maximum_relative_l2": max(comparison["relative_l2"] for comparison in comparisons),
            "maximum_p95_abs": max(comparison["p95_abs"] for comparison in comparisons),
            "maximum_abs": max(comparison["max_abs"] for comparison in comparisons),
            "comparisons_to_first_execution": comparisons,
        }

    def any_nonexact(stage: str) -> bool:
        return stages[stage]["bitwise_equal_comparison_count"] != REPLAY_COUNT - 1

    def any_material(stage: str) -> bool:
        return stages[stage]["within_fixed_material_floor_count"] != REPLAY_COUNT - 1

    if any_material("normalized_shifted_real"):
        classification = "material_drift_begins_at_normalization_or_translation"
    elif any_nonexact("normalized_shifted_real"):
        classification = "normalization_or_translation_roundoff_within_fixed_material_floor"
    elif any_material("masked_real"):
        classification = "material_drift_begins_at_softmask_background"
    elif any_nonexact("masked_real"):
        classification = "softmask_background_reduction_drift_within_fixed_material_floor"
    elif any_material("masked_fourier"):
        classification = "material_drift_begins_at_masked_fft"
    elif any_nonexact("masked_fourier"):
        classification = "masked_fft_roundoff_within_fixed_material_floor"
    else:
        classification = "preprocessing_replays_bitwise_exact"

    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "classification": classification,
        "metric_policy": (
            "bitwise equality plus scale-sensitive relative-L2/p95/max; "
            "no correlation; fixed material relative-L2 threshold"
        ),
        "material_relative_l2_threshold": MATERIAL_RELATIVE_L2,
        "replay_count": REPLAY_COUNT,
        "comparison_count": REPLAY_COUNT - 1,
        "batch_size": batch_size,
        "stages": stages,
        "fixed_metric": {
            "evaluated_comparisons": (REPLAY_COUNT - 1) * len(STAGE_ORDER),
            "expected_comparisons": (REPLAY_COUNT - 1) * len(STAGE_ORDER),
            "bitwise_equal_comparisons": sum(stages[stage]["bitwise_equal_comparison_count"] for stage in STAGE_ORDER),
            "within_fixed_material_floor_comparisons": sum(
                stages[stage]["within_fixed_material_floor_count"] for stage in STAGE_ORDER
            ),
        },
        "scorecard_change_admissible": False,
        "correlation_used": False,
        "quality_metric_policy": {
            "fsc_auc_evaluated": False,
            "map_gate": "not evaluated by this preprocessing replay diagnostic",
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


def _clean_repo_head(repo: Path) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1"],
        text=True,
    )
    _require(not status, "analyzer repository is dirty")
    return head


def _load_bundle(
    path: Path,
    *,
    expected_original_index: int,
    expected_iteration: int,
    expected_class_one_based: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {field: np.asarray(archive[field]) for field in archive.files}
    missing = sorted(REQUIRED_BUNDLE_FIELDS - values.keys())
    _require(not missing, f"sealed bundle is missing required fields: {missing}")
    _require(str(values["schema"]) == "recovar-bpref-contribution-rows-v3", "bundle schema changed")
    _require(bool(values["high_precision_operand_bundle"]), "bundle is not high precision")
    _require(bool(values["relion_cuda_preprocess"]), "bundle did not use RELION CUDA preprocessing")
    _require(str(values["preprocess_backend"]) == "relion_cuda", "preprocess backend changed")
    _require(bool(values["score_with_masked_images"]), "bundle did not score masked images")
    _require(str(values["image_mask_mode"]) == "relion_background_fill", "mask mode changed")
    _require(int(values["iteration"]) == expected_iteration, "bundle iteration changed")
    _require(
        int(values["class_index"]) + 1 == expected_class_one_based,
        "bundle class changed",
    )
    original_indices = np.asarray(values["original_indices"], dtype=np.int64)
    _require(
        original_indices.shape == (1,) and int(original_indices[0]) == expected_original_index,
        "bundle particle identity changed",
    )
    raw = np.asarray(values["raw_real_images"])
    normalization = np.asarray(
        values["relion_preprocess_normalization_factors"],
    )
    shifts = np.asarray(values["integer_pre_shifts"])
    _require(raw.dtype == np.float32 and raw.shape[0] == 1, "raw image shape/dtype changed")
    _require(
        normalization.dtype == np.float32 and normalization.shape == (1,),
        "normalization shape/dtype changed",
    )
    _require(
        shifts.dtype == np.int32 and shifts.shape == (1, 2),
        "integer shift shape/dtype changed",
    )
    _require(
        np.array_equal(normalization, np.asarray(values["image_corrections"])),
        "normalization and image correction differ",
    )
    _require(
        np.array_equal(np.asarray(values["scale_corrections"]), np.ones(1, np.float32)),
        "target scale correction changed",
    )
    image_shape = tuple(int(value) for value in np.asarray(values["image_shape"]))
    _require(raw.shape[1:] == image_shape, "raw image topology changed")
    _require(image_shape[0] == image_shape[1], "raw image must be square")
    mask = np.asarray(values["image_mask"])
    _require(mask.shape == image_shape and mask.dtype == np.float32, "stored mask changed")
    _require(float(mask.min()) == 0.0 and float(mask.max()) == 1.0, "stored mask range changed")
    return values


def run_gpu_replays(
    *,
    bundle_path: Path,
    expected_gpu_uuid: str,
    expected_original_index: int,
    expected_iteration: int,
    expected_class_one_based: int,
    particle_diameter_angstrom: float,
    mask_edge_width_pixels: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the sealed particle four times through the current CUDA preprocessor."""

    import jax
    import jax.numpy as jnp

    from recovar.cuda_backproject import relion_preprocess_real_f32
    from recovar.data_io.image_backends import _centered_rfft2_jax

    _require(jax.default_backend() == "gpu", "preprocessing replay requires JAX GPU")
    gpu_uuid = _allocated_gpu_uuid(expected_gpu_uuid)
    values = _load_bundle(
        bundle_path,
        expected_original_index=expected_original_index,
        expected_iteration=expected_iteration,
        expected_class_one_based=expected_class_one_based,
    )
    voxel_size = float(values["voxel_size"])
    _require(np.isfinite(voxel_size) and voxel_size > 0.0, "voxel size is invalid")
    _require(
        np.isfinite(particle_diameter_angstrom) and particle_diameter_angstrom > 0.0,
        "particle diameter must be finite and positive",
    )
    _require(
        np.isfinite(mask_edge_width_pixels) and mask_edge_width_pixels > 0.0,
        "mask edge width must be finite and positive",
    )
    mask_radius_pixels = particle_diameter_angstrom / (2.0 * voxel_size)
    images = jnp.asarray(values["raw_real_images"], dtype=jnp.float32)
    normalization = jnp.asarray(
        values["relion_preprocess_normalization_factors"],
        dtype=jnp.float32,
    )
    shifts = jnp.asarray(values["integer_pre_shifts"], dtype=jnp.int32)
    normalized_runs = []
    masked_runs = []
    masked_fourier_runs = []
    for _ in range(REPLAY_COUNT):
        normalized, masked = relion_preprocess_real_f32(
            images,
            normalization,
            shifts,
            mask_radius_pixels,
            mask_edge_width_pixels,
            True,
        )
        masked_fourier = _centered_rfft2_jax(masked)
        masked_fourier.block_until_ready()
        normalized_runs.append(np.asarray(normalized, dtype=np.float32))
        masked_runs.append(np.asarray(masked, dtype=np.float32))
        masked_fourier_runs.append(np.asarray(masked_fourier, dtype=np.complex64))
    arrays = {
        "normalized_shifted_real": np.stack(normalized_runs),
        "masked_real": np.stack(masked_runs),
        "masked_fourier": np.stack(masked_fourier_runs),
    }
    report = analyze_replay_arrays(
        normalized_runs=arrays["normalized_shifted_real"],
        masked_runs=arrays["masked_real"],
        masked_fourier_runs=arrays["masked_fourier"],
    )
    report["scope"] = {
        "original_index_zero_based": expected_original_index,
        "iteration": expected_iteration,
        "class_one_based": expected_class_one_based,
        "current_size": int(values["current_size"]),
        "half": int(values["half"]),
        "image_shape": list(values["raw_real_images"].shape[1:]),
        "voxel_size_angstrom": voxel_size,
        "particle_diameter_angstrom": particle_diameter_angstrom,
        "mask_radius_pixels": mask_radius_pixels,
        "mask_edge_width_pixels": mask_edge_width_pixels,
        "gpu_uuid": gpu_uuid,
    }
    report["inputs"] = {
        "sealed_bundle": {
            "path": str(bundle_path.resolve()),
            "sha256": _sha256(bundle_path),
        }
    }
    return report, arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-original-index", required=True, type=int)
    parser.add_argument("--expected-iteration", required=True, type=int)
    parser.add_argument("--expected-class-one-based", required=True, type=int)
    parser.add_argument("--particle-diameter-angstrom", required=True, type=float)
    parser.add_argument("--mask-edge-width-pixels", required=True, type=float)
    parser.add_argument("--cuda-library", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    _require(not args.output_npz.exists(), f"refusing to overwrite {args.output_npz}")
    configured_cuda = os.environ.get("RECOVAR_CUDA_LIB")
    _require(configured_cuda is not None, "RECOVAR_CUDA_LIB is unset")
    _require(
        Path(configured_cuda).resolve() == args.cuda_library.resolve(),
        "configured CUDA library differs from the declared library",
    )
    report, arrays = run_gpu_replays(
        bundle_path=args.bundle,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_original_index=args.expected_original_index,
        expected_iteration=args.expected_iteration,
        expected_class_one_based=args.expected_class_one_based,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        mask_edge_width_pixels=args.mask_edge_width_pixels,
    )
    report["analyzer_repo_head"] = _clean_repo_head(args.repo)
    report["inputs"]["cuda_library"] = {
        "path": str(args.cuda_library.resolve()),
        "sha256": _sha256(args.cuda_library),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **arrays)
    report["replay_arrays"] = {
        "path": str(args.output_npz.resolve()),
        "sha256": _sha256(args.output_npz),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                **report["fixed_metric"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
