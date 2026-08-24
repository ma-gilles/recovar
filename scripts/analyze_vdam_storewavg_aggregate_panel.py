#!/usr/bin/env python3
"""Replay a complete native VDAM StoreWavg panel against RECOVAR BPref rows."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recovar.em.initial_model.layout import relion_bpref_frame_scales
from scripts.analyze_vdam_bpref_accumulator_boundary import _production_names
from scripts.analyze_vdam_mstep_boundary import _read_relion_array
from scripts.analyze_vdam_storewavg_boundary import (
    _fftw_window_to_native_crop,
    _load_native,
    _load_unmasked_image,
    _match_rotations,
    _metric,
    _native_gradient_rows,
    _posterior_metric,
    _real_2d,
    _require,
    _restore_storewavg_inverse_noise_dc,
    _scalar,
    _scatter_relion_rows,
)
from scripts.analyze_vdam_storewavg_panel import (
    _native_prefixes,
    _part_to_original_indices,
    _pooled_relative_l2,
    _quantiles,
)

SCHEMA = "recovar.vdam_storewavg_aggregate_panel.v1"


@dataclass
class _Capture:
    path: Path
    half: int
    current_size: int
    physical_image_size: int
    padding_factor: int
    window_indices: np.ndarray
    original_indices: np.ndarray
    reconstruction_probs: np.ndarray
    active_particle_rows: np.ndarray
    active_rotation_rows: np.ndarray
    active_summed: np.ndarray
    active_ctf_probs: np.ndarray
    active_rotations: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.active_particle_rows.size)


def _load_capture(path: Path, expected_half: int) -> _Capture:
    required = {
        "half",
        "current_size",
        "image_shape",
        "reconstruction_padding_factor",
        "window_indices",
        "original_indices",
        "reconstruction_probs",
        "active_particle_rows",
        "active_rotation_rows",
        "active_summed",
        "active_ctf_probs",
        "active_rotations",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        _require(not missing, f"RECOVAR contribution capture lacks fields {missing}: {path}")
        values = {name: np.asarray(archive[name]) for name in required}
    half = int(values["half"])
    _require(half == expected_half, f"expected half {expected_half}, got {half}: {path}")
    image_shape = tuple(int(value) for value in values["image_shape"])
    _require(len(image_shape) == 2 and image_shape[0] == image_shape[1], "image must be square")
    original_indices = np.asarray(values["original_indices"], dtype=np.int64)
    active_particle_rows = np.asarray(values["active_particle_rows"], dtype=np.int32)
    active_rotation_rows = np.asarray(values["active_rotation_rows"], dtype=np.int32)
    row_count = active_particle_rows.size
    _require(original_indices.size > 0, f"empty contribution capture: {path}")
    _require(
        np.all((active_particle_rows >= 0) & (active_particle_rows < original_indices.size)),
        f"active particle row lies outside capture: {path}",
    )
    active_summed = np.asarray(values["active_summed"], dtype=np.complex64)
    active_ctf_probs = np.asarray(values["active_ctf_probs"], dtype=np.float32)
    active_rotations = np.asarray(values["active_rotations"], dtype=np.float32)
    window_indices = np.asarray(values["window_indices"], dtype=np.int32)
    _require(
        active_rotation_rows.shape == (row_count,)
        and active_summed.shape == active_ctf_probs.shape == (row_count, window_indices.size)
        and active_rotations.shape == (row_count, 3, 3),
        f"active contribution topology changed: {path}",
    )
    reconstruction_probs = np.asarray(values["reconstruction_probs"])
    _require(
        reconstruction_probs.ndim == 3 and reconstruction_probs.shape[0] == original_indices.size,
        f"posterior topology changed: {path}",
    )
    return _Capture(
        path=path.resolve(),
        half=half,
        current_size=int(values["current_size"]),
        physical_image_size=image_shape[0],
        padding_factor=int(values["reconstruction_padding_factor"]),
        window_indices=window_indices,
        original_indices=original_indices,
        reconstruction_probs=reconstruction_probs,
        active_particle_rows=active_particle_rows,
        active_rotation_rows=active_rotation_rows,
        active_summed=active_summed,
        active_ctf_probs=active_ctf_probs,
        active_rotations=active_rotations,
    )


def _particle_locations(captures: list[_Capture]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for capture_index, capture in enumerate(captures):
        for slot, original_index in enumerate(capture.original_indices):
            identity = int(original_index)
            _require(identity not in result, f"particle {identity} occurs in multiple captures")
            result[identity] = (capture_index, slot)
    return result


def _align_native_rows(native_rows: np.ndarray, rotation_map: np.ndarray) -> np.ndarray:
    native_rows = np.asarray(native_rows)
    rotation_map = np.asarray(rotation_map, dtype=np.int64)
    _require(native_rows.ndim == 2, "native contribution rows must be two-dimensional")
    _require(rotation_map.shape == (native_rows.shape[0],), "rotation map topology changed")
    _require(
        np.array_equal(np.sort(rotation_map), np.arange(rotation_map.size)),
        "rotation map must be a complete permutation",
    )
    aligned = np.empty_like(native_rows)
    aligned[rotation_map] = native_rows
    return aligned


def _residual_geometry(reference_gap: np.ndarray, candidate_gap: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference_gap).astype(np.complex128, copy=False).reshape(-1)
    candidate = np.asarray(candidate_gap).astype(np.complex128, copy=False).reshape(-1)
    _require(reference.shape == candidate.shape and reference.size > 0, "gap topology mismatch")
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    denominator = max(reference_norm * candidate_norm, np.finfo(np.float64).tiny)
    projection_denominator = max(reference_norm * reference_norm, np.finfo(np.float64).tiny)
    projection = float(np.real(np.vdot(reference, candidate)) / projection_denominator)
    orthogonal = candidate - projection * reference
    return {
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "cosine": float(np.real(np.vdot(reference, candidate)) / denominator),
        "candidate_projection_on_reference": projection,
        "candidate_orthogonal_over_reference": float(
            np.linalg.norm(orthogonal) / max(reference_norm, np.finfo(np.float64).tiny)
        ),
    }


def _pooled_metric(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    reference = np.asarray(reference).astype(np.complex128, copy=False)
    candidate = np.asarray(candidate).astype(np.complex128, copy=False)
    _require(reference.shape == candidate.shape, "pooled metric topology mismatch")
    residual = candidate - reference
    return float(np.vdot(reference, reference).real), float(np.vdot(residual, residual).real)


def _scatter_complete_half(
    captures: list[_Capture],
    native_data_rows: list[np.ndarray],
    native_weight_rows: list[np.ndarray],
    controlled_data_rows: list[np.ndarray],
    controlled_weight_rows: list[np.ndarray],
    *,
    get_backprojector_data,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    metadata = captures[0]
    for capture in captures[1:]:
        _require(capture.current_size == metadata.current_size, "capture current sizes differ")
        _require(capture.physical_image_size == metadata.physical_image_size, "capture image sizes differ")
        _require(capture.padding_factor == metadata.padding_factor, "capture padding factors differ")
        _require(np.array_equal(capture.window_indices, metadata.window_indices), "capture windows differ")
    rotations = np.concatenate([capture.active_rotations for capture in captures], axis=0)
    row_sets = {
        "native": (
            np.concatenate(native_data_rows, axis=0),
            np.concatenate(native_weight_rows, axis=0),
        ),
        "candidate": (
            np.concatenate([capture.active_summed for capture in captures], axis=0),
            np.concatenate([capture.active_ctf_probs for capture in captures], axis=0),
        ),
        "same_posterior_control": (
            np.concatenate(controlled_data_rows, axis=0),
            np.concatenate(controlled_weight_rows, axis=0),
        ),
    }
    result = {}
    for name, (data_rows, weight_rows) in row_sets.items():
        result[name] = _scatter_relion_rows(
            data_rows,
            weight_rows,
            rotations,
            metadata.window_indices,
            physical_image_size=metadata.physical_image_size,
            current_size=metadata.current_size,
            padding_factor=metadata.padding_factor,
            get_backprojector_data=get_backprojector_data,
        )
        gc.collect()
    return result


def analyze(
    native_directory: Path,
    relion_data_star: Path,
    half_capture_paths: dict[int, list[Path]],
    *,
    native_production_directory: Path | None = None,
    recovar_production_directory: Path | None = None,
    rotation_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from recovar.relion_bind._relion_bind_core import get_backprojector_data

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    _require(jax.default_backend() == "gpu", "complete StoreWavg panel replay requires a GPU")
    native_by_part, incomplete_native = _native_prefixes(native_directory)
    all_original_by_part = _part_to_original_indices(relion_data_star)
    _require(not incomplete_native, f"incomplete native captures: {incomplete_native}")
    _require(
        set(native_by_part).issubset(all_original_by_part),
        "native part ID lies outside the RELION data STAR",
    )
    original_by_part = {
        part_id: all_original_by_part[part_id] for part_id in sorted(native_by_part)
    }

    captures_by_half = {
        half: [_load_capture(path, half) for path in paths]
        for half, paths in sorted(half_capture_paths.items())
    }
    _require(set(captures_by_half) == {1, 2}, "both RECOVAR halves are required")
    all_captures = [capture for captures in captures_by_half.values() for capture in captures]
    reference_capture = all_captures[0]
    locations = {
        half: _particle_locations(captures) for half, captures in captures_by_half.items()
    }
    all_locations = {
        original_index: (half, capture_index, slot)
        for half, half_locations in locations.items()
        for original_index, (capture_index, slot) in half_locations.items()
    }
    _require(
        set(all_locations) == set(original_by_part.values()),
        "native and RECOVAR particle identities do not close exactly",
    )

    projector = _load_native(
        native_directory,
        native_by_part[min(native_by_part)],
        load_projector=True,
    )
    _require(int(projector["r_max"]) == reference_capture.current_size // 2, "projector size changed")
    crop_indices, centered_indices = _fftw_window_to_native_crop(
        reference_capture.window_indices,
        physical_image_size=reference_capture.physical_image_size,
        current_size=reference_capture.current_size,
    )
    inverse_noise = _real_2d(native_directory / "Minvsigma2.bin").astype(np.float32).reshape(-1)[
        crop_indices
    ]
    inverse_noise = _restore_storewavg_inverse_noise_dc(
        inverse_noise,
        crop_indices,
        _real_2d(native_directory / "sigma2_noise.bin"),
        _scalar(native_directory / "sigma2_fudge.bin"),
    )
    data_scale = np.float32(-float(reference_capture.physical_image_size) ** -2)
    weight_scale = np.float32(float(reference_capture.physical_image_size) ** -4)

    native_data_rows = {
        half: [np.empty_like(capture.active_summed) for capture in captures]
        for half, captures in captures_by_half.items()
    }
    native_weight_rows = {
        half: [np.empty_like(capture.active_ctf_probs) for capture in captures]
        for half, captures in captures_by_half.items()
    }
    controlled_data_rows = {
        half: [np.empty_like(capture.active_summed) for capture in captures]
        for half, captures in captures_by_half.items()
    }
    controlled_weight_rows = {
        half: [np.empty_like(capture.active_ctf_probs) for capture in captures]
        for half, captures in captures_by_half.items()
    }
    filled = {
        half: [np.zeros(capture.row_count, dtype=bool) for capture in captures]
        for half, captures in captures_by_half.items()
    }
    summaries = {
        half: {
            "posterior_relative_l2": [],
            "posterior_l1": [],
            "posterior_support_mismatch": [],
            "posterior_argmax_mismatch": 0,
            "posterior_reference_norm2": 0.0,
            "posterior_residual_norm2": 0.0,
            "native_row_reference_norm2": 0.0,
            "native_row_residual_norm2": 0.0,
            "controlled_row_reference_norm2": 0.0,
            "controlled_row_residual_norm2": 0.0,
            "per_particle": [],
        }
        for half in (1, 2)
    }

    for part_id in sorted(original_by_part):
        original_index = original_by_part[part_id]
        half, capture_index, slot = all_locations[original_index]
        capture = captures_by_half[half][capture_index]
        row_mask = capture.active_particle_rows == slot
        row_indices = np.flatnonzero(row_mask)
        _require(row_indices.size > 0, f"RECOVAR capture has no active rows for particle {original_index}")
        native = _load_native(native_directory, native_by_part[part_id], load_projector=False)
        candidate_rotations = capture.active_rotations[row_mask]
        rotation_map = _match_rotations(
            np.asarray(native["rotations"], dtype=np.float32),
            candidate_rotations,
            rotation_tolerance,
        )
        candidate_posterior = np.asarray(capture.reconstruction_probs[slot], dtype=np.float32)[
            capture.active_rotation_rows[row_mask]
        ][rotation_map]
        native_posterior = np.asarray(native["probabilities"], dtype=np.float32)
        _require(candidate_posterior.shape == native_posterior.shape, "posterior topology differs")

        native_image = _load_unmasked_image(
            native_directory / f"{native_by_part[part_id]}Fimg_unweighted_nomask.bin"
        ).astype(np.complex64).reshape(-1)[crop_indices]
        translated = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(native_image[None, :]),
            jnp.asarray(native["translation_angles"], dtype=jnp.float32),
            jnp.asarray(centered_indices, dtype=jnp.int32),
            (reference_capture.physical_image_size, reference_capture.physical_image_size),
        )
        projections, _ = compute_relion_projector_projections_block(
            jnp.asarray(projector["projector"]),
            jnp.asarray(native["rotations"], dtype=jnp.float32),
            (reference_capture.physical_image_size, reference_capture.physical_image_size),
            r_max=reference_capture.current_size // 2,
            padding_factor=1,
            return_abs2=False,
            centered_rows=True,
            dense_scale=False,
            projector_output_size=reference_capture.current_size,
            pixel_indices=jnp.asarray(centered_indices, dtype=jnp.int32),
            relion_texture_interp=True,
        )
        translated, projections = jax.block_until_ready((translated, projections))
        translated = np.asarray(translated, dtype=np.complex64)
        projections = np.asarray(projections, dtype=np.complex64)
        native_ctf = np.asarray(native["ctf"], dtype=np.float32)[crop_indices]
        native_data, native_weight = _native_gradient_rows(
            native_posterior, translated, projections, native_ctf, inverse_noise
        )
        controlled_data, controlled_weight = _native_gradient_rows(
            candidate_posterior, translated, projections, native_ctf, inverse_noise
        )
        native_data = _align_native_rows(native_data * data_scale, rotation_map)
        native_weight = _align_native_rows(native_weight * weight_scale, rotation_map)
        controlled_data = _align_native_rows(controlled_data * data_scale, rotation_map)
        controlled_weight = _align_native_rows(controlled_weight * weight_scale, rotation_map)
        candidate_data = capture.active_summed[row_mask]
        candidate_weight = capture.active_ctf_probs[row_mask]

        native_data_rows[half][capture_index][row_indices] = native_data
        native_weight_rows[half][capture_index][row_indices] = native_weight
        controlled_data_rows[half][capture_index][row_indices] = controlled_data
        controlled_weight_rows[half][capture_index][row_indices] = controlled_weight
        filled[half][capture_index][row_indices] = True

        posterior = _posterior_metric(native_posterior, candidate_posterior)
        summary = summaries[half]
        summary["posterior_relative_l2"].append(float(posterior["relative_l2"]))
        summary["posterior_l1"].append(float(posterior["l1"]))
        summary["posterior_support_mismatch"].append(float(posterior["support_mismatch_count"]))
        summary["posterior_argmax_mismatch"] += int(
            np.argmax(native_posterior) != np.argmax(candidate_posterior)
        )
        ref2, residual2 = _pooled_metric(native_posterior, candidate_posterior)
        summary["posterior_reference_norm2"] += ref2
        summary["posterior_residual_norm2"] += residual2
        ref2, residual2 = _pooled_metric(native_data, candidate_data)
        summary["native_row_reference_norm2"] += ref2
        summary["native_row_residual_norm2"] += residual2
        ref2, residual2 = _pooled_metric(controlled_data, candidate_data)
        summary["controlled_row_reference_norm2"] += ref2
        summary["controlled_row_residual_norm2"] += residual2
        summary["per_particle"].append(
            {
                "part_id": part_id,
                "original_index": original_index,
                "row_count": int(row_indices.size),
                "posterior": posterior,
                "native_rows_vs_candidate_data": _metric(native_data, candidate_data),
                "native_rows_vs_candidate_weight": _metric(native_weight, candidate_weight),
                "same_posterior_rows_vs_candidate_data": _metric(controlled_data, candidate_data),
                "same_posterior_rows_vs_candidate_weight": _metric(
                    controlled_weight, candidate_weight
                ),
            }
        )
        del translated, projections

    for half in (1, 2):
        for capture, capture_filled in zip(captures_by_half[half], filled[half], strict=True):
            _require(bool(np.all(capture_filled)), f"unfilled RECOVAR rows remain in {capture.path}")

    half_reports = {}
    for half in (1, 2):
        captures = captures_by_half[half]
        scattered = _scatter_complete_half(
            captures,
            native_data_rows[half],
            native_weight_rows[half],
            controlled_data_rows[half],
            controlled_weight_rows[half],
            get_backprojector_data=get_backprojector_data,
        )
        summary = summaries[half]
        native_scatter = scattered["native"]
        candidate_scatter = scattered["candidate"]
        controlled_scatter = scattered["same_posterior_control"]
        report = {
            "particle_count": len(summary["per_particle"]),
            "row_count": sum(capture.row_count for capture in captures),
            "posterior": {
                "relative_l2": _quantiles(summary["posterior_relative_l2"]),
                "l1": _quantiles(summary["posterior_l1"]),
                "support_mismatch_count": _quantiles(summary["posterior_support_mismatch"]),
                "argmax_mismatch_count": int(summary["posterior_argmax_mismatch"]),
                "pooled_relative_l2": _pooled_relative_l2(
                    summary["posterior_reference_norm2"], summary["posterior_residual_norm2"]
                ),
            },
            "pooled_rows": {
                "native_vs_candidate_data_relative_l2": _pooled_relative_l2(
                    summary["native_row_reference_norm2"], summary["native_row_residual_norm2"]
                ),
                "same_posterior_vs_candidate_data_relative_l2": _pooled_relative_l2(
                    summary["controlled_row_reference_norm2"],
                    summary["controlled_row_residual_norm2"],
                ),
                "native_vs_candidate_weight": _metric(
                    np.concatenate(native_weight_rows[half], axis=0),
                    np.concatenate([capture.active_ctf_probs for capture in captures], axis=0),
                ),
                "same_posterior_vs_candidate_weight": _metric(
                    np.concatenate(controlled_weight_rows[half], axis=0),
                    np.concatenate([capture.active_ctf_probs for capture in captures], axis=0),
                ),
            },
            "shared_relion_double_scatter": {
                "native_vs_candidate_data": _metric(native_scatter[0], candidate_scatter[0]),
                "native_vs_candidate_weight": _metric(native_scatter[1], candidate_scatter[1]),
                "same_posterior_vs_candidate_data": _metric(
                    controlled_scatter[0], candidate_scatter[0]
                ),
                "same_posterior_vs_candidate_weight": _metric(
                    controlled_scatter[1], candidate_scatter[1]
                ),
            },
            "per_particle": summary["per_particle"],
        }
        if native_production_directory is not None or recovar_production_directory is not None:
            _require(
                native_production_directory is not None and recovar_production_directory is not None,
                "native and RECOVAR production directories must be supplied together",
            )
            native_data_name, native_weight_name, candidate_data_name, candidate_weight_name = (
                _production_names(half)
            )
            production_native_data = _read_relion_array(
                native_production_directory / native_data_name, complex_values=True
            )
            production_native_weight = _read_relion_array(
                native_production_directory / native_weight_name, complex_values=False
            )
            production_candidate_data = np.load(
                recovar_production_directory / candidate_data_name, allow_pickle=False
            )
            production_candidate_weight = np.load(
                recovar_production_directory / candidate_weight_name, allow_pickle=False
            )
            bpref_data_scale, bpref_weight_scale = relion_bpref_frame_scales(
                reference_capture.physical_image_size
            )
            panel_native_data = native_scatter[0] * bpref_data_scale
            panel_native_weight = native_scatter[1] * bpref_weight_scale
            panel_candidate_data = candidate_scatter[0] * bpref_data_scale
            panel_candidate_weight = candidate_scatter[1] * bpref_weight_scale
            production_gap_data = production_candidate_data - production_native_data
            production_gap_weight = production_candidate_weight - production_native_weight
            panel_gap_data = panel_candidate_data - panel_native_data
            panel_gap_weight = panel_candidate_weight - panel_native_weight
            report["independent_production_accumulator_comparison"] = {
                "caveat": (
                    "The native production accumulator is an independent native repeat; "
                    "use residual geometry diagnostically, not as a bitwise closure gate."
                ),
                "panel_candidate_vs_production_candidate_data": _metric(
                    production_candidate_data, panel_candidate_data
                ),
                "panel_candidate_vs_production_candidate_weight": _metric(
                    production_candidate_weight, panel_candidate_weight
                ),
                "panel_native_vs_production_native_data": _metric(
                    production_native_data, panel_native_data
                ),
                "panel_native_vs_production_native_weight": _metric(
                    production_native_weight, panel_native_weight
                ),
                "production_gap_vs_panel_gap_data": _metric(production_gap_data, panel_gap_data),
                "production_gap_vs_panel_gap_weight": _metric(
                    production_gap_weight, panel_gap_weight
                ),
                "gap_geometry_data": _residual_geometry(production_gap_data, panel_gap_data),
                "gap_geometry_weight": _residual_geometry(
                    production_gap_weight, panel_gap_weight
                ),
            }
        half_reports[str(half)] = report

    return {
        "schema": SCHEMA,
        "status": "complete",
        "coverage": {
            "relion_data_star_particle_count": len(all_original_by_part),
            "native_particle_count": len(native_by_part),
            "recovar_particle_count": len(all_locations),
            "native_incomplete_capture_count": len(incomplete_native),
            "halves": {
                str(half): {
                    "particle_count": len(locations[half]),
                    "capture_count": len(captures_by_half[half]),
                    "row_count": sum(capture.row_count for capture in captures_by_half[half]),
                }
                for half in (1, 2)
            },
        },
        "frame_scales": {
            "native_rows_to_recovar_data": float(data_scale),
            "native_rows_to_recovar_weight": float(weight_scale),
        },
        "halves": half_reports,
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "relion_data_star": str(relion_data_star.resolve()),
            "recovar_half_captures": {
                str(half): [str(path.resolve()) for path in paths]
                for half, paths in half_capture_paths.items()
            },
            **(
                {
                    "native_production_directory": str(native_production_directory.resolve()),
                    "recovar_production_directory": str(recovar_production_directory.resolve()),
                }
                if native_production_directory is not None
                else {}
            ),
        },
        "device": str(jax.devices()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--relion-data-star", required=True, type=Path)
    parser.add_argument("--recovar-half1-capture", required=True, action="append", type=Path)
    parser.add_argument("--recovar-half2-capture", required=True, action="append", type=Path)
    parser.add_argument("--native-production-directory", type=Path)
    parser.add_argument("--recovar-production-directory", type=Path)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.relion_data_star,
        {1: args.recovar_half1_capture, 2: args.recovar_half2_capture},
        native_production_directory=args.native_production_directory,
        recovar_production_directory=args.recovar_production_directory,
        rotation_tolerance=args.rotation_tolerance,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    concise = {
        "schema": report["schema"],
        "status": report["status"],
        "coverage": report["coverage"],
        "halves": {
            half: {
                "posterior": values["posterior"],
                "pooled_rows": values["pooled_rows"],
                "shared_relion_double_scatter": values["shared_relion_double_scatter"],
                **(
                    {
                        "independent_production_accumulator_comparison": values[
                            "independent_production_accumulator_comparison"
                        ]
                    }
                    if "independent_production_accumulator_comparison" in values
                    else {}
                ),
            }
            for half, values in report["halves"].items()
        },
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
