#!/usr/bin/env python3
"""Prove which stable-shape construction preserves VDAM's RELION projector.

The InitialModel path currently rebuilds RELION ``Projector::data`` at every
logical ``current_size``.  Those arrays (and their static ``r_max`` values)
become operands of the large local-EM JIT, so GF46's resolution schedule
creates many otherwise redundant executables.

This diagnostic keeps production code unchanged.  It builds the projector
maps with the RELION C++ binding, then uses RECOVAR's production JAX
interpolator on a CPU device to compare every logical size observed in the
frozen GF46 0--200 trajectory with its stable physical-size bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.em.dense_single_volume.helpers.fourier_window import (  # noqa: E402
    stable_fourier_window_current_size,
)
from recovar.em.dense_single_volume.helpers.projection import (  # noqa: E402
    project_relion_projector_half_spectrum,
)

SCHEMA = "recovar.vdam_projector_capacity_proof.v1"
GF46_IMAGE_SIZE = 128

# Distinct rlnCurrentImageSize values in the frozen v3 GF46 0--200 trajectory.
# Keeping the observed values explicit makes schedule drift visible instead of
# silently weakening this proof to whichever sizes a new run happens to visit.
GF46_LOGICAL_CURRENT_SIZES = (
    30,
    32,
    34,
    38,
    44,
    46,
    48,
    50,
    56,
    60,
    62,
    66,
    68,
    70,
    72,
    76,
    78,
    82,
    84,
    86,
    88,
    90,
    98,
    100,
    104,
    106,
    110,
    112,
    114,
    118,
    126,
    128,
)


def gf46_logical_physical_pairs() -> tuple[tuple[int, int], ...]:
    """Return the complete frozen GF46 logical-to-capacity mapping."""

    return tuple(
        (
            int(logical_size),
            int(
                stable_fourier_window_current_size(
                    logical_size,
                    GF46_IMAGE_SIZE,
                )
            ),
        )
        for logical_size in GF46_LOGICAL_CURRENT_SIZES
    )


def center_pad_relion_projector(
    logical_projector: np.ndarray,
    physical_shape: tuple[int, int, int],
) -> np.ndarray:
    """Embed a cropped RELION PPref without changing any logical texel.

    RELION stores signed z/y frequencies around the center of the first two
    axes and non-negative x frequencies from index zero.  Stable storage must
    therefore pad z/y symmetrically and append capacity only at the x end.
    """

    logical = np.asarray(logical_projector)
    target = tuple(int(value) for value in physical_shape)
    if logical.ndim != 3 or len(target) != 3:
        raise ValueError("logical projector and physical shape must both be three-dimensional")
    differences = tuple(dst - src for src, dst in zip(logical.shape, target))
    if any(value < 0 for value in differences):
        raise ValueError(f"physical shape {target} cannot contain logical shape {logical.shape}")
    if differences[0] % 2 or differences[1] % 2:
        raise ValueError("RELION projector z/y capacity differences must be even for center alignment")

    z_offset = differences[0] // 2
    y_offset = differences[1] // 2
    padded = np.zeros(target, dtype=logical.dtype)
    padded[
        z_offset : z_offset + logical.shape[0],
        y_offset : y_offset + logical.shape[1],
        : logical.shape[2],
    ] = logical
    return padded


def crop_relion_projector_capacity(
    physical_projector: np.ndarray,
    logical_shape: tuple[int, int, int],
) -> np.ndarray:
    """Return the coordinate-aligned logical box from physical PPref storage."""

    physical = np.asarray(physical_projector)
    target = tuple(int(value) for value in logical_shape)
    differences = tuple(src - dst for src, dst in zip(physical.shape, target))
    if physical.ndim != 3 or len(target) != 3 or any(value < 0 for value in differences):
        raise ValueError(f"logical shape {target} is not contained in physical shape {physical.shape}")
    if differences[0] % 2 or differences[1] % 2:
        raise ValueError("RELION projector z/y capacity differences must be even for center alignment")
    z_offset = differences[0] // 2
    y_offset = differences[1] // 2
    return physical[
        z_offset : z_offset + target[0],
        y_offset : y_offset + target[1],
        : target[2],
    ]


def crop_relion_projection_capacity(
    physical_projection: np.ndarray,
    *,
    physical_size: int,
    logical_size: int,
) -> np.ndarray:
    """Gather a logical FFTW half-image from a larger output-size capacity."""

    physical = np.asarray(physical_projection)
    physical_size = int(physical_size)
    logical_size = int(logical_size)
    if logical_size > physical_size or logical_size < 1:
        raise ValueError(f"invalid projection capacity {logical_size=} {physical_size=}")
    expected_pixels = physical_size * (physical_size // 2 + 1)
    if physical.ndim != 2 or physical.shape[1] != expected_pixels:
        raise ValueError(
            f"physical projection must have shape (R, physical_size * (physical_size // 2 + 1)), got {physical.shape}"
        )

    logical_rows = np.arange(logical_size, dtype=np.int64)
    signed_y = np.where(
        logical_rows <= logical_size // 2,
        logical_rows,
        logical_rows - logical_size,
    )
    physical_rows = np.where(signed_y >= 0, signed_y, signed_y + physical_size)
    return physical.reshape(
        physical.shape[0],
        physical_size,
        physical_size // 2 + 1,
    )[:, physical_rows, : logical_size // 2 + 1].reshape(physical.shape[0], -1)


def _logical_sphere_mask(shape: tuple[int, int, int], r_max: int) -> np.ndarray:
    z_size, y_size, x_size = (int(value) for value in shape)
    z = np.arange(z_size, dtype=np.int64) - z_size // 2
    y = np.arange(y_size, dtype=np.int64) - y_size // 2
    x = np.arange(x_size, dtype=np.int64)
    return z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2 <= int(r_max) ** 2


def _sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _complex_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.ascontiguousarray(reference)
    candidate = np.ascontiguousarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    if reference.dtype != candidate.dtype:
        raise ValueError(f"dtype mismatch: {reference.dtype} != {candidate.dtype}")
    residual = candidate.astype(np.complex128) - reference.astype(np.complex128)
    reference_norm = float(np.linalg.norm(reference.astype(np.complex128).reshape(-1)))
    residual_norm = float(np.linalg.norm(residual.reshape(-1)))
    return {
        "count": int(reference.size),
        "different_values": int(np.count_nonzero(reference != candidate)),
        "bitwise_equal": bool(reference.tobytes(order="C") == candidate.tobytes(order="C")),
        "reference_sha256": _sha256(reference),
        "candidate_sha256": _sha256(candidate),
        "max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "relative_l2": residual_norm / reference_norm if reference_norm > 0.0 else residual_norm,
    }


def _make_reference_volume(image_size: int, seed: int) -> np.ndarray:
    """Create a deterministic non-symmetric, band-limited cryo-EM-like map."""

    rng = np.random.default_rng(int(seed))
    volume = rng.standard_normal((image_size, image_size, image_size))
    transform = np.fft.rfftn(volume, axes=(0, 1, 2))
    z = np.fft.fftfreq(image_size) * image_size
    y = np.fft.fftfreq(image_size) * image_size
    x = np.fft.rfftfreq(image_size) * image_size
    radius_squared = z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2
    transform *= np.exp(-radius_squared / (2.0 * (image_size / 5.0) ** 2))
    return np.asarray(
        np.fft.irfftn(
            transform,
            s=(image_size, image_size, image_size),
            axes=(0, 1, 2),
        ),
        dtype=np.float64,
    )


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.eye(3, dtype=np.float64)
    first = (axis + 1) % 3
    second = (axis + 2) % 3
    rotation[first, first] = cosine
    rotation[first, second] = -sine
    rotation[second, first] = sine
    rotation[second, second] = cosine
    return rotation


def _fixture_rotations() -> np.ndarray:
    """Use fixed general rotations, including exact and interpolating views."""

    angle_triples = (
        (0.0, 0.0, 0.0),
        (0.17, -0.31, 0.43),
        (-0.52, 0.29, 0.71),
        (0.83, -0.47, -0.19),
        (-0.38, -0.67, 0.23),
        (0.61, 0.37, -0.79),
    )
    rotations = []
    for x_angle, y_angle, z_angle in angle_triples:
        rotations.append(_axis_rotation(2, z_angle) @ _axis_rotation(1, y_angle) @ _axis_rotation(0, x_angle))
    return np.asarray(rotations, dtype=np.float32)


def _build_projector(
    reference: np.ndarray,
    current_size: int,
) -> tuple[np.ndarray, int]:
    try:
        from recovar.relion_bind import _relion_bind_core as binding
    except ImportError as error:  # pragma: no cover - depends on optional build
        raise RuntimeError(
            "the RELION C++ binding is required; build it or set RECOVAR_RELION_BIND_BUILD_DIR"
        ) from error

    projector, *_unused, r_max, _padding_factor, _interpolator = binding.compute_fourier_transform_map(
        np.ascontiguousarray(reference, dtype=np.float64),
        int(reference.shape[0]),
        1,
        1,
        int(current_size),
        True,
        2,
    )
    return np.asarray(projector, dtype=np.complex128), int(r_max)


def _project_on_cpu(
    projector: np.ndarray,
    rotations: np.ndarray,
    *,
    output_size: int,
    r_max: int,
) -> np.ndarray:
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("the projector-capacity proof requires a JAX CPU device")
    with jax.default_device(cpu_devices[0]):
        projected = project_relion_projector_half_spectrum(
            jnp.asarray(projector, dtype=jnp.complex64),
            jnp.asarray(rotations, dtype=jnp.float32),
            (int(output_size), int(output_size)),
            int(r_max),
            1,
        )
        return np.asarray(jax.block_until_ready(projected))


def analyze(
    *,
    logical_sizes: Iterable[int] = GF46_LOGICAL_CURRENT_SIZES,
    volume_seed: int = 20260903,
) -> dict[str, object]:
    """Run the portable projector-capacity proof and return a JSON report."""

    logical_sizes = tuple(int(value) for value in logical_sizes)
    expected_sizes = set(GF46_LOGICAL_CURRENT_SIZES)
    if not logical_sizes or any(value not in expected_sizes for value in logical_sizes):
        raise ValueError("logical_sizes must be a non-empty subset of the frozen GF46 schedule")

    pairs = tuple(
        (
            logical_size,
            int(
                stable_fourier_window_current_size(
                    logical_size,
                    GF46_IMAGE_SIZE,
                )
            ),
        )
        for logical_size in logical_sizes
    )
    reference = _make_reference_volume(GF46_IMAGE_SIZE, volume_seed)
    rotations = _fixture_rotations()
    maps: dict[int, tuple[np.ndarray, int]] = {}
    for current_size in sorted({value for pair in pairs for value in pair}):
        maps[current_size] = _build_projector(reference, current_size)

    rows = []
    for logical_size, physical_size in pairs:
        logical_raw, logical_r_max = maps[logical_size]
        physical_raw, physical_r_max = maps[physical_size]
        physical_overlap = crop_relion_projector_capacity(
            physical_raw,
            logical_raw.shape,
        )
        logical_mask = _logical_sphere_mask(logical_raw.shape, logical_r_max)
        padded_raw = center_pad_relion_projector(logical_raw, physical_raw.shape)

        logical_projector = logical_raw.astype(np.complex64)
        physical_projector = physical_raw.astype(np.complex64)
        padded_projector = padded_raw.astype(np.complex64)
        logical_projection = _project_on_cpu(
            logical_projector,
            rotations,
            output_size=logical_size,
            r_max=logical_r_max,
        )
        projection_candidates_physical = {
            "physical_rebuild_logical_cutoff": _project_on_cpu(
                physical_projector,
                rotations,
                output_size=physical_size,
                r_max=logical_r_max,
            ),
            "physical_rebuild_physical_cutoff": _project_on_cpu(
                physical_projector,
                rotations,
                output_size=physical_size,
                r_max=physical_r_max,
            ),
            "center_padded_logical_cutoff": _project_on_cpu(
                padded_projector,
                rotations,
                output_size=physical_size,
                r_max=logical_r_max,
            ),
            "center_padded_physical_cutoff": _project_on_cpu(
                padded_projector,
                rotations,
                output_size=physical_size,
                r_max=physical_r_max,
            ),
        }
        projection_candidates = {
            name: crop_relion_projection_capacity(
                candidate,
                physical_size=physical_size,
                logical_size=logical_size,
            )
            for name, candidate in projection_candidates_physical.items()
        }

        rows.append(
            {
                "logical_size": logical_size,
                "physical_size": physical_size,
                "logical_shape": list(logical_raw.shape),
                "physical_shape": list(physical_raw.shape),
                "logical_r_max": logical_r_max,
                "physical_r_max": physical_r_max,
                "texels": {
                    "physical_rebuild_full_logical_box": _complex_metric(
                        logical_raw,
                        physical_overlap,
                    ),
                    "physical_rebuild_logical_sphere": _complex_metric(
                        logical_raw[logical_mask],
                        physical_overlap[logical_mask],
                    ),
                    "center_padded_logical_box": _complex_metric(
                        logical_raw,
                        crop_relion_projector_capacity(
                            padded_raw,
                            logical_raw.shape,
                        ),
                    ),
                    "physical_only_nonzero_texels": int(np.count_nonzero(physical_raw != padded_raw)),
                },
                "projections": {
                    name: _complex_metric(logical_projection, candidate)
                    for name, candidate in projection_candidates.items()
                },
            }
        )

    projection_names = tuple(rows[0]["projections"])
    nonidentity_rows = [row for row in rows if row["logical_size"] != row["physical_size"]]
    summary = {
        "pair_count": len(rows),
        "nonidentity_pair_count": len(nonidentity_rows),
        "physical_class_count": len({row["physical_size"] for row in rows}),
        "active_sphere_texels_all_bitwise": all(
            row["texels"]["physical_rebuild_logical_sphere"]["bitwise_equal"] for row in rows
        ),
        "center_padded_logical_box_all_bitwise": all(
            row["texels"]["center_padded_logical_box"]["bitwise_equal"] for row in rows
        ),
        "projection_constructions": {
            name: {
                "all_bitwise": all(row["projections"][name]["bitwise_equal"] for row in rows),
                "nonidentity_bitwise_count": sum(
                    bool(row["projections"][name]["bitwise_equal"]) for row in nonidentity_rows
                ),
                "max_relative_l2": max(float(row["projections"][name]["relative_l2"]) for row in rows),
                "max_abs": max(float(row["projections"][name]["max_abs"]) for row in rows),
            }
            for name in projection_names
        },
    }
    report = {
        "schema": SCHEMA,
        "fixture": {
            "image_size": GF46_IMAGE_SIZE,
            "logical_sizes": list(logical_sizes),
            "logical_physical_pairs": [list(pair) for pair in pairs],
            "volume_seed": int(volume_seed),
            "rotation_count": int(rotations.shape[0]),
            "projector_dtype_from_cpp": "complex128",
            "projector_dtype_used_by_vdam": "complex64",
            "jax_projection_device": str(jax.devices("cpu")[0]),
        },
        "summary": summary,
        "pairs": rows,
    }
    report["markdown"] = render_markdown(report)
    return report


def render_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# VDAM RELION projector stable-capacity proof",
        "",
        (
            f"Pairs: `{summary['pair_count']}`; physical classes: "
            f"`{summary['physical_class_count']}`; CPU rotations per pair: "
            f"`{report['fixture']['rotation_count']}`."
        ),
        "",
        "| Construction | All pairs bitwise | Max relative L2 | Max abs |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "physical_rebuild_logical_cutoff": "Physical rebuild + logical r_max",
        "physical_rebuild_physical_cutoff": "Physical rebuild + physical r_max",
        "center_padded_logical_cutoff": "Center-padded logical + logical r_max",
        "center_padded_physical_cutoff": "Center-padded logical + physical r_max",
    }
    for name, values in summary["projection_constructions"].items():
        lines.append(
            f"| {labels[name]} | {values['all_bitwise']} | {values['max_relative_l2']:.6e} | {values['max_abs']:.6e} |"
        )
    lines.extend(
        [
            "",
            "| L→P | Active texels exact | Physical/logical rel-L2 | "
            "Physical/physical rel-L2 | Padded/logical exact | Padded/physical rel-L2 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["pairs"]:
        projections = row["projections"]
        lines.append(
            f"| {row['logical_size']}→{row['physical_size']} | "
            f"{row['texels']['physical_rebuild_logical_sphere']['bitwise_equal']} | "
            f"{projections['physical_rebuild_logical_cutoff']['relative_l2']:.3e} | "
            f"{projections['physical_rebuild_physical_cutoff']['relative_l2']:.3e} | "
            f"{projections['center_padded_logical_cutoff']['bitwise_equal']} | "
            f"{projections['center_padded_physical_cutoff']['relative_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            "The proof changes no production behavior. A safe production seam must retain "
            "the logical `r_max` while stabilizing only projector storage capacity.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write(path: str | None, payload: str) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    parser.add_argument("--volume-seed", type=int, default=20260903)
    arguments = parser.parse_args()

    report = analyze(volume_seed=arguments.volume_seed)
    _write(
        arguments.output_json,
        json.dumps({key: value for key, value in report.items() if key != "markdown"}, indent=2) + "\n",
    )
    _write(arguments.output_markdown, report["markdown"])
    print(report["markdown"], end="")


if __name__ == "__main__":
    main()
