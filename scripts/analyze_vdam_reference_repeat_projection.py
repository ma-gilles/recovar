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
            exact_radius=True,
        )
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
                mask_current_image_disk=True,
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
                        projections["candidate"],
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
            from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
                _make_relion_wavg_rectangle,
            )
            from scripts.analyze_vdam_storewavg_boundary import (
                _load_native,
                _match_rotations,
            )
            from scripts.analyze_vdam_storewavg_reference_decomposition import (
                _current_size_from_rectangle_size,
                _cutoff_sums,
                _flat_complex,
                _fine_reference_rectangle,
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
            masked_image = _flat_complex(
                capture,
                "preprocess_img0_masked_fourier_post_optics",
            ).astype(np.complex64)
            translated = _translate_native_rectangle(
                masked_image,
                np.asarray(native["translation_angles"], dtype=np.float32),
                current_size,
            )[:, rectangle.exact_positions]
            ctf = np.asarray(native["ctf"], dtype=np.float32)[rectangle.exact_positions]
            cutoff_mask = (
                np.asarray(rectangle.shell_indices)[rectangle.exact_positions]
                == current_size // 2
            )
            cutoff = {}
            native_capture_projection = _fine_reference_rectangle(
                capture,
                int(native["orientation_count"]),
                rectangle_size,
            )[:, rectangle.exact_positions]
            for label, projection in projections.items():
                native_frame_projection = (
                    projection[rotation_map] * np.float32(-1.0 / n**2)
                ).astype(np.complex64)
                cutoff[label] = _cutoff_sums(
                    native_frame_projection,
                    translated,
                    ctf,
                    np.asarray(native["probabilities"], dtype=np.float32),
                    cutoff_mask,
                )
            captured_native_cutoff = _cutoff_sums(
                native_capture_projection,
                translated,
                ctf,
                np.asarray(native["probabilities"], dtype=np.float32),
                cutoff_mask,
            )
            cutoff_records.append(
                {
                    "part_id": part_id,
                    "original_index": original_index,
                    "values": cutoff,
                    "captured_native_values": captured_native_cutoff,
                    "captured_native_projection_validation": _metric(
                        native_capture_projection,
                        (
                            projections[reference_label][rotation_map]
                            * np.float32(-1.0 / n**2)
                        ).astype(np.complex64),
                    ),
                    "effects": {
                        label: {
                            name: float(values[name] - cutoff[reference_label][name])
                            for name in ("xa", "aa")
                        }
                        for label, values in cutoff.items()
                        if label != reference_label
                    },
                    "effects_vs_captured_native": {
                        label: {
                            name: float(values[name] - captured_native_cutoff[name])
                            for name in ("xa", "aa")
                        }
                        for label, values in cutoff.items()
                    },
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
    cutoff_vs_captured_summary = None
    if cutoff_records:
        cutoff_summary = {}
        for label in maps:
            if label == reference_label:
                continue
            cutoff_summary[label] = {}
            for name in ("xa", "aa"):
                effects = np.asarray(
                    [row["effects"][label][name] for row in cutoff_records],
                    dtype=np.float64,
                )
                cutoff_summary[label][name] = {
                    "signed_sum": float(np.sum(effects)),
                    "mean_abs": float(np.mean(np.abs(effects))),
                    "max_abs": float(np.max(np.abs(effects))),
                }
        for name in ("xa", "aa"):
            native_floor = abs(cutoff_summary[native_repeat_label][name]["signed_sum"])
            for label in cutoff_summary:
                cutoff_summary[label][name]["native_signed_sum_floor_ratio"] = (
                    float(abs(cutoff_summary[label][name]["signed_sum"]) / native_floor)
                    if native_floor > 0.0
                    else float("inf")
                )
        cutoff_vs_captured_summary = {}
        for label in maps:
            cutoff_vs_captured_summary[label] = {}
            for name in ("xa", "aa"):
                effects = np.asarray(
                    [
                        row["effects_vs_captured_native"][label][name]
                        for row in cutoff_records
                    ],
                    dtype=np.float64,
                )
                cutoff_vs_captured_summary[label][name] = {
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
        "cutoff_vs_captured_native_summary": cutoff_vs_captured_summary,
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
                "cutoff_vs_captured_native_summary": report[
                    "cutoff_vs_captured_native_summary"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
