#!/usr/bin/env python3
"""Compare reconstructed-map repeats after the production RELION projector."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    delta = candidate - reference
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    candidate_norm = float(np.linalg.norm(candidate.reshape(-1)))
    denominator = reference_norm * candidate_norm
    return {
        "count": int(reference.size),
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / reference_norm),
        "cosine": (
            float(np.vdot(reference.reshape(-1), candidate.reshape(-1)).real / denominator)
            if denominator > 0.0
            else 1.0
        ),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _read_map(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        volume = np.asarray(handle.data, dtype=np.float64).copy()
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"map must be a cube: {path} {volume.shape}")
    return volume


def _shell_ids(n: int) -> np.ndarray:
    frequency = np.fft.fftfreq(n) * n
    z, y, x = np.meshgrid(
        frequency,
        frequency,
        np.fft.rfftfreq(n) * n,
        indexing="ij",
    )
    return np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32)


def _projector(volume: np.ndarray, current_size: int) -> tuple[np.ndarray, int]:
    from recovar.relion_bind import _relion_bind_core as bind

    projector, *_unused, r_max, _padding, _interpolator = bind.compute_fourier_transform_map(
        np.ascontiguousarray(volume, dtype=np.float64),
        int(volume.shape[0]),
        1,
        1,
        int(current_size),
        True,
        2,
    )
    return np.asarray(projector, dtype=np.complex64), int(r_max)


def analyze(
    maps: dict[str, Path],
    score_dumps: list[Path],
    *,
    reference_label: str,
    native_repeat_label: str,
    decomposition_report: Path | None = None,
) -> dict[str, object]:
    if reference_label not in maps or native_repeat_label not in maps:
        raise ValueError("reference and native-repeat labels must name supplied maps")
    volumes = {label: _read_map(path) for label, path in maps.items()}
    shapes = {volume.shape for volume in volumes.values()}
    if len(shapes) != 1:
        raise ValueError(f"map shapes differ: {sorted(shapes)}")
    n = next(iter(volumes.values())).shape[0]
    shell_ids = _shell_ids(n)
    transforms = {label: np.fft.rfftn(volume) for label, volume in volumes.items()}
    shell_metrics: dict[str, dict[str, object]] = {}
    reference_transform = transforms[reference_label]
    for label, transform in transforms.items():
        if label == reference_label:
            continue
        shell_metrics[label] = {
            str(shell): _metric(
                reference_transform[shell_ids == shell],
                transform[shell_ids == shell],
            )
            for shell in range(n // 2 + 1)
            if np.any(shell_ids == shell)
        }

    projection_records = []
    cutoff_records = []
    decomposition_by_original = {}
    decomposition_physical_size = None
    if decomposition_report is not None:
        decomposition = json.loads(decomposition_report.read_text())
        decomposition_by_original = {
            int(row["original_index"]): row
            for row in decomposition["per_particle"]
        }
        decomposition_physical_size = int(decomposition["identity"]["physical_image_size"])
    projectors_by_size: dict[int, dict[str, tuple[np.ndarray, int]]] = {}
    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    for score_path in score_dumps:
        with np.load(score_path, allow_pickle=False) as archive:
            score = {name: archive[name] for name in archive.files}
        current_size = int(np.asarray(score["current_size"]).reshape(-1)[0])
        if current_size not in projectors_by_size:
            projectors_by_size[current_size] = {
                label: _projector(volume, current_size)
                for label, volume in volumes.items()
            }
        pixel_indices, _ = make_fourier_window_indices_np(
            (n, n),
            current_size,
            include_dc=True,
            exact_radius=False,
        )
        exact_pixel_indices, _ = make_fourier_window_indices_np(
            (n, n),
            current_size,
            include_dc=True,
            exact_radius=True,
        )
        exact_projection_take = np.searchsorted(pixel_indices, exact_pixel_indices)
        if not np.array_equal(pixel_indices[exact_projection_take], exact_pixel_indices):
            raise ValueError("exact projection support is not contained in rounded support")
        rotations = np.asarray(score["local_rotation_matrices"], dtype=np.float32)
        projections = {}
        for label, (projector, r_max) in projectors_by_size[current_size].items():
            projected, _ = compute_relion_projector_projections_block(
                jnp.asarray(projector),
                jnp.asarray(rotations),
                (n, n),
                r_max=r_max,
                padding_factor=1,
                return_abs2=False,
                centered_rows=True,
                # InitialModel's scoring/reconstruction projection frame is
                # the RELION projector value multiplied by ``-ori_size^2``.
                dense_scale=True,
                projector_output_size=current_size,
                pixel_indices=jnp.asarray(pixel_indices, dtype=jnp.int32),
                relion_texture_interp=True,
                mask_current_image_disk=False,
            )
            projections[label] = np.asarray(jax.block_until_ready(projected), dtype=np.complex64)
        reference_projection = projections[reference_label]
        projection_records.append(
            {
                "score_dump": str(score_path),
                "current_size": current_size,
                "rotation_count": int(rotations.shape[0]),
                "pixel_count": int(pixel_indices.size),
                "comparisons": {
                    label: _metric(reference_projection, projection)
                    for label, projection in projections.items()
                    if label != reference_label
                },
                "captured_candidate_validation": (
                    _metric(
                        np.asarray(score["debug_proj_for_recon"], dtype=np.complex64),
                        projections["candidate"][:, exact_projection_take],
                    )
                    if "candidate" in projections and "debug_proj_for_recon" in score
                    else None
                ),
            }
        )
        if decomposition_report is not None:
            match = re.search(r"_image_(\d+)_", score_path.name)
            if match is None:
                raise ValueError(f"cannot recover image id from score dump: {score_path}")
            original_index = int(match.group(1))
            if original_index not in decomposition_by_original:
                raise ValueError(
                    f"score image {original_index} is absent from decomposition report"
                )
            decomposition_row = decomposition_by_original[original_index]
            capture = Path(decomposition_row["artifacts"]["capture_directory"])
            part_id = int(decomposition_row["part_id"])
            prefix = f"img0_part{part_id}_storeWavg_"
            from recovar.em.dense_single_volume.helpers.sparse_pass2_wavg import (
                _make_relion_wavg_rectangle,
            )
            from scripts.analyze_vdam_storewavg_boundary import (
                _load_native,
                _match_rotations,
                _production_score_gradient_rows,
            )
            from scripts.analyze_vdam_storewavg_reference_decomposition import (
                _current_size_from_rectangle_size,
                _cutoff_sums,
                _fine_reference_rectangle,
                _flat_complex,
                _scalar,
                _translate_native_rectangle,
            )

            native = _load_native(capture, prefix, load_projector=False)
            rotation_map = _match_rotations(
                np.asarray(native["rotations"], dtype=np.float32),
                rotations,
                1.0e-5,
            )
            rectangle_size = int(round(_scalar(capture / f"{prefix}image_size.bin")))
            if _current_size_from_rectangle_size(rectangle_size) != current_size:
                raise ValueError("native and candidate current sizes differ")
            recon_indices_for_rectangle, _ = make_fourier_window_indices_np(
                (decomposition_physical_size, decomposition_physical_size),
                current_size,
                include_dc=True,
                exact_radius=True,
            )
            rectangle = _make_relion_wavg_rectangle(
                (decomposition_physical_size, decomposition_physical_size),
                current_size,
                recon_indices_for_rectangle,
            )
            rounded_indices_for_rectangle, _ = make_fourier_window_indices_np(
                (decomposition_physical_size, decomposition_physical_size),
                current_size,
                include_dc=True,
                exact_radius=False,
            )
            rounded_rectangle = _make_relion_wavg_rectangle(
                (decomposition_physical_size, decomposition_physical_size),
                current_size,
                rounded_indices_for_rectangle,
            )
            masked_image = _flat_complex(
                capture,
                "preprocess_img0_masked_fourier_post_optics",
            ).astype(np.complex64)
            translated = _translate_native_rectangle(
                masked_image,
                np.asarray(native["translation_angles"], dtype=np.float32),
                current_size,
            )
            ctf = np.asarray(native["ctf"], dtype=np.float32)
            cutoff_mask = np.asarray(rectangle.shell_indices) == current_size // 2
            cutoff = {}
            native_capture_projection = _fine_reference_rectangle(
                capture,
                int(native["orientation_count"]),
                rectangle_size,
            )
            _score_data, _score_weight, candidate_posterior = (
                _production_score_gradient_rows(score)
            )
            candidate_posterior = np.asarray(
                candidate_posterior[rotation_map], dtype=np.float32
            )
            native_posterior = np.asarray(native["probabilities"], dtype=np.float32)
            if candidate_posterior.shape != native_posterior.shape:
                raise ValueError("candidate and native posterior shapes differ")
            for label, projection in projections.items():
                native_frame_rounded = (
                    projection[rotation_map] * np.float32(-1.0 / n**2)
                ).astype(np.complex64)
                native_frame_exact = native_frame_rounded[:, exact_projection_take]
                exact_only = np.zeros_like(native_capture_projection)
                exact_only[:, rectangle.exact_positions] = native_frame_exact
                hybrid = native_capture_projection.copy()
                hybrid[:, rectangle.exact_positions] = native_frame_exact
                rounded_support = np.zeros_like(native_capture_projection)
                rounded_support[:, rounded_rectangle.exact_positions] = native_frame_rounded
                native_with_candidate_posterior = _cutoff_sums(
                    native_capture_projection,
                    translated,
                    ctf,
                    candidate_posterior,
                    cutoff_mask,
                )
                exact_only_values = _cutoff_sums(
                    exact_only, translated, ctf, candidate_posterior, cutoff_mask
                )
                hybrid_values = _cutoff_sums(
                    hybrid, translated, ctf, candidate_posterior, cutoff_mask
                )
                rounded_values = _cutoff_sums(
                    rounded_support, translated, ctf, candidate_posterior, cutoff_mask
                )
                cutoff[label] = {
                    "native_reference_candidate_posterior": native_with_candidate_posterior,
                    "exact_only_candidate_posterior": exact_only_values,
                    "hybrid_candidate_posterior": hybrid_values,
                    "rounded_support_candidate_posterior": rounded_values,
                }
            captured_native_cutoff = _cutoff_sums(
                native_capture_projection,
                translated,
                ctf,
                native_posterior,
                cutoff_mask,
            )
            component_effects = {}
            for label, values in cutoff.items():
                component_effects[label] = {}
                for name in ("xa", "aa"):
                    native_candidate = values["native_reference_candidate_posterior"][name]
                    exact_only = values["exact_only_candidate_posterior"][name]
                    hybrid = values["hybrid_candidate_posterior"][name]
                    rounded = values["rounded_support_candidate_posterior"][name]
                    component_effects[label][name] = {
                        "posterior": float(native_candidate - captured_native_cutoff[name]),
                        "inside_exact_reference": float(hybrid - native_candidate),
                        "missing_rounded_rim": float(exact_only - hybrid),
                        "total_exact_only": float(exact_only - captured_native_cutoff[name]),
                        "restored_candidate_rim": float(rounded - exact_only),
                        "total_rounded_support": float(
                            rounded - captured_native_cutoff[name]
                        ),
                    }
            cutoff_records.append(
                {
                    "part_id": part_id,
                    "original_index": original_index,
                    "values": cutoff,
                    "captured_native_values": captured_native_cutoff,
                    "captured_native_projection_validation": _metric(
                        native_capture_projection[:, rectangle.exact_positions],
                        (
                            projections[reference_label][rotation_map][:, exact_projection_take]
                            * np.float32(-1.0 / n**2)
                        ).astype(np.complex64),
                    ),
                    "captured_native_rounded_projection_validation": _metric(
                        native_capture_projection[:, rounded_rectangle.exact_positions],
                        (
                            projections[reference_label][rotation_map]
                            * np.float32(-1.0 / n**2)
                        ).astype(np.complex64),
                    ),
                    "component_effects": component_effects,
                }
            )

    def _pooled(label: str) -> dict[str, float]:
        candidate_sq = 0.0
        reference_sq = 0.0
        for row in projection_records:
            metric = row["comparisons"][label]
            reference_sq += float(metric["reference_norm"]) ** 2
            candidate_sq += (
                float(metric["relative_l2"]) * float(metric["reference_norm"])
            ) ** 2
        return {
            "relative_l2": float(np.sqrt(candidate_sq / reference_sq)),
            "record_count": len(projection_records),
        }

    pooled = {
        label: _pooled(label)
        for label in maps
        if label != reference_label
    }
    native_floor = pooled[native_repeat_label]["relative_l2"]
    for label, values in pooled.items():
        values["native_repeat_floor_ratio"] = (
            float(values["relative_l2"] / native_floor)
            if native_floor > 0.0
            else float("inf")
        )
    cutoff_summary = None
    if cutoff_records:
        cutoff_summary = {}
        for label in maps:
            cutoff_summary[label] = {}
            for name in ("xa", "aa"):
                cutoff_summary[label][name] = {}
                for component in (
                    "posterior",
                    "inside_exact_reference",
                    "missing_rounded_rim",
                    "total_exact_only",
                    "restored_candidate_rim",
                    "total_rounded_support",
                ):
                    effects = np.asarray(
                        [
                            row["component_effects"][label][name][component]
                            for row in cutoff_records
                        ],
                        dtype=np.float64,
                    )
                    cutoff_summary[label][name][component] = {
                        "signed_sum": float(np.sum(effects)),
                        "mean_abs": float(np.mean(np.abs(effects))),
                        "max_abs": float(np.max(np.abs(effects))),
                    }
    return {
        "schema": "recovar.vdam_reference_repeat_projection.v1",
        "identity": {
            "maps": {label: str(path) for label, path in maps.items()},
            "reference_label": reference_label,
            "native_repeat_label": native_repeat_label,
            "score_dump_count": len(score_dumps),
        },
        "volume": {
            label: _metric(volumes[reference_label], volume)
            for label, volume in volumes.items()
            if label != reference_label
        },
        "fourier_shells": shell_metrics,
        "projection_pooled": pooled,
        "per_score_dump": projection_records,
        "cutoff_component_summary": cutoff_summary,
        "per_particle_cutoff_components": cutoff_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--score-dump", action="append", required=True, type=Path)
    parser.add_argument("--reference-label", default="native_a")
    parser.add_argument("--native-repeat-label", default="native_b")
    parser.add_argument("--decomposition-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    maps = {}
    for token in args.map:
        label, separator, path = token.partition("=")
        if not separator or not label or label in maps:
            raise ValueError(f"invalid or duplicate --map value: {token}")
        maps[label] = Path(path).resolve()
    report = analyze(
        maps,
        [path.resolve() for path in args.score_dump],
        reference_label=args.reference_label,
        native_repeat_label=args.native_repeat_label,
        decomposition_report=(
            None
            if args.decomposition_report is None
            else args.decomposition_report.resolve()
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "projection_pooled": report["projection_pooled"],
                "cutoff_component_summary": report["cutoff_component_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
