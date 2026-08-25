#!/usr/bin/env python3
"""Compare reconstructed-map repeats after the production RELION projector."""

from __future__ import annotations

import argparse
import json
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
                dense_scale=False,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--score-dump", action="append", required=True, type=Path)
    parser.add_argument("--reference-label", default="native_a")
    parser.add_argument("--native-repeat-label", default="native_b")
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["projection_pooled"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
