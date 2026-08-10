#!/usr/bin/env python3
"""Compare RELION and RECOVAR K=1 BPref scatter geometry on a fixed panel.

The input RELION factor captures contain the exact post-translation-reduction
source value, source weight, Euler matrix, and current-size Fourier coordinate.
This diagnostic places those native values into RECOVAR's production fused
x-half scatter and captures its exact device coordinates, Hermitian fold,
neighbor destinations, and trilinear coefficients.  It then evaluates the
corresponding RELION CUDA equations independently in NumPy float32.

No map correlation is computed.  Exact identity and relative L2 are the
appropriate metrics at this operand/geometry boundary; signed FSC/FSC-AUC
remain the map-level acceptance metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.analyze_k1_bpref_factor_boundary import _rotation_map
from scripts.validate_relion_bpref_factor_capture import (
    FOOTER_STRUCT,
    HEADER_MAGIC,
    HEADER_STRUCT,
    HYPOTHESIS_DTYPE,
    PIXEL_DTYPE,
    ROTATION_DTYPE,
    TERM_DTYPE,
    TRANSLATION_DTYPE,
    load_factor_capture,
)

SCHEMA = "recovar.em.k1_bpref_scatter_geometry.v1"
EXTENDED_SUMMARY_DTYPE = np.dtype(
    {
        "names": (
            "state",
            "orientation_local",
            "pixel",
            "flags",
            "x",
            "y",
            "z",
            "source_re",
            "source_im",
            "source_weight",
            "rotated_x",
            "rotated_y",
            "rotated_z",
            "neighbor_indices",
            "neighbor_coefficients",
        ),
        "formats": (
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<i4",
            "<i4",
            "<i4",
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            ("<i4", 8),
            ("<f4", 8),
        ),
        "offsets": (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 84),
        "itemsize": 116,
    }
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


def _load_factor(path: Path):
    """Load the qualified v2 capture or its diagnostic geometry extension."""

    payload = path.read_bytes()
    _require(len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size, f"truncated factor: {path}")
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload, 0)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"factor header magic mismatch: {path}")
    if header[6] != EXTENDED_SUMMARY_DTYPE.itemsize:
        return load_factor_capture(path)
    _require(header[0] == 2 and header[1] == HEADER_STRUCT.size, "extended factor schema changed")
    _require(
        header[2:9]
        == (
            ROTATION_DTYPE.itemsize,
            TRANSLATION_DTYPE.itemsize,
            HYPOTHESIS_DTYPE.itemsize,
            PIXEL_DTYPE.itemsize,
            EXTENDED_SUMMARY_DTYPE.itemsize,
            TERM_DTYPE.itemsize,
            FOOTER_STRUCT.size,
        ),
        "extended factor record sizes changed",
    )
    counts = header[46:52]
    offset = HEADER_STRUCT.size

    def read(dtype, count):
        nonlocal offset
        value = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
        offset += int(count) * dtype.itemsize
        return value

    rotations = read(ROTATION_DTYPE, counts[0])
    translations = read(TRANSLATION_DTYPE, counts[1])
    hypotheses = read(HYPOTHESIS_DTYPE, counts[2])
    pixels = read(PIXEL_DTYPE, counts[3])
    summaries = read(EXTENDED_SUMMARY_DTYPE, counts[4])
    terms = read(TERM_DTYPE, counts[5])
    _require(offset + FOOTER_STRUCT.size == len(payload), "extended factor byte count changed")
    footer_magic, *footer_counts = FOOTER_STRUCT.unpack_from(payload, offset)
    _require(footer_magic == b"RLNBPRF2FOOTER\0\0", "extended factor footer magic changed")
    _require(tuple(int(value) for value in footer_counts) == counts, "extended factor footer count changed")
    _require(np.all(summaries["state"] == 1), "extended factor contains inactive summaries")
    _require(np.all((summaries["flags"] & np.uint32(3)) == np.uint32(3)), "extended summary support changed")
    return SimpleNamespace(
        path=path,
        sha256=_sha256(path),
        header=header,
        rotations=rotations,
        translations=translations,
        hypotheses=hypotheses,
        pixels=pixels,
        summaries=summaries,
        terms=terms,
        stack_index=header[12],
        geometry_only=bool(header[53]),
    )


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "metric arrays differ or are empty")
    delta = right.astype(np.float64) - left.astype(np.float64)
    denominator = max(float(np.linalg.norm(left.astype(np.float64))), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_reference": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _factor_native_rows(factor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense native numerator/weight and the serialized support mask."""

    rotations = int(factor.rotations.size)
    pixels = int(factor.pixels.size)
    data = np.zeros((rotations, pixels), dtype=np.complex64)
    weight = np.zeros((rotations, pixels), dtype=np.float32)
    support = np.zeros((rotations, pixels), dtype=bool)
    rows = factor.summaries
    rr = np.asarray(rows["orientation_local"], dtype=np.int64)
    pp = np.asarray(rows["pixel"], dtype=np.int64)
    _require(
        np.unique(rr * pixels + pp).size == rows.size,
        "RELION summary rows are not unique",
    )
    n = np.float32(int(factor.header[17]))
    data[rr, pp] = -(
        np.asarray(rows["source_re"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(rows["source_im"], dtype=np.float32)
    ) / np.float32(n * n)
    weight[rr, pp] = np.asarray(rows["source_weight"], dtype=np.float32) / np.float32(
        n * n * n * n
    )
    support[rr, pp] = True
    return data, weight, support


def _native_geometry(
    factor,
    *,
    accumulator_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    """Evaluate RELION cuda_kernel_backproject3D geometry in float32."""

    rotations = np.asarray(factor.rotations["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    pixels = factor.pixels
    x = np.asarray(pixels["x"], dtype=np.float32)
    y = np.asarray(pixels["y"], dtype=np.float32)
    padding = np.float32(2.0)

    # Preserve the source expression and addend order from BP.cuh.
    xp = (rotations[:, 0, 0, None] * x + rotations[:, 0, 1, None] * y) * padding
    yp = (rotations[:, 1, 0, None] * x + rotations[:, 1, 1, None] * y) * padding
    zp = (rotations[:, 2, 0, None] * x + rotations[:, 2, 1, None] * y) * padding
    coordinates_zyx = np.stack((zp, yp, xp), axis=-1).astype(np.float32, copy=False)
    folded = xp < np.float32(0.0)
    scatter_coordinates = np.where(folded[..., None], -coordinates_zyx, coordinates_zyx)

    floor_zyx = np.floor(scatter_coordinates).astype(np.int32)
    fractions = scatter_coordinates - floor_zyx.astype(np.float32)
    complements = np.float32(1.0) - fractions
    center = np.asarray(
        [accumulator_shape[0] // 2, accumulator_shape[1] // 2, 0],
        dtype=np.int32,
    )
    strides = np.asarray(
        [accumulator_shape[1] * (accumulator_shape[2] // 2 + 1), accumulator_shape[2] // 2 + 1, 1],
        dtype=np.int32,
    )
    neighbor_indices = np.empty((*xp.shape, 8), dtype=np.int32)
    coefficients = np.empty((*xp.shape, 8), dtype=np.float32)
    slot = 0
    for dz in range(2):
        for dy in range(2):
            pair = np.float32(
                (fractions[..., 0] if dz else complements[..., 0])
                * (fractions[..., 1] if dy else complements[..., 1])
            )
            for dx in range(2):
                coefficients[..., slot] = np.float32(
                    pair * (fractions[..., 2] if dx else complements[..., 2])
                )
                offset = floor_zyx + center + np.asarray([dz, dy, dx], dtype=np.int32)
                neighbor_indices[..., slot] = np.sum(offset * strides, axis=-1, dtype=np.int32)
                slot += 1
    return {
        "coordinates_zyx": coordinates_zyx,
        "folded": folded,
        "neighbor_indices": neighbor_indices,
        "neighbor_coefficients": coefficients,
    }


def _captured_native_geometry(factor) -> dict[str, np.ndarray] | None:
    """Expand passive native-device geometry to dense orientation/pixel rows."""

    if "rotated_x" not in factor.summaries.dtype.names:
        return None
    shape = (int(factor.rotations.size), int(factor.pixels.size))
    coordinates = np.full((*shape, 3), np.nan, dtype=np.float32)
    neighbor_indices = np.full((*shape, 8), -1, dtype=np.int32)
    coefficients = np.full((*shape, 8), np.nan, dtype=np.float32)
    folded = np.zeros(shape, dtype=bool)
    rows = factor.summaries
    rr = np.asarray(rows["orientation_local"], dtype=np.int64)
    pp = np.asarray(rows["pixel"], dtype=np.int64)
    coordinates[rr, pp] = np.stack(
        (rows["rotated_z"], rows["rotated_y"], rows["rotated_x"]), axis=-1
    )
    neighbor_indices[rr, pp] = rows["neighbor_indices"]
    coefficients[rr, pp] = rows["neighbor_coefficients"]
    folded[rr, pp] = (rows["flags"] & np.uint32(4)) != 0
    return {
        "coordinates_zyx": coordinates,
        "folded": folded,
        "neighbor_indices": neighbor_indices,
        "neighbor_coefficients": coefficients,
    }


def _analyze_particle(factor_path: Path, pass2_path: Path) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject

    factor = _load_factor(factor_path)
    with np.load(pass2_path, allow_pickle=False) as archive:
        recovar_rotations = np.asarray(archive["rotations"], dtype=np.float32)
        recon_indices = np.asarray(archive["recon_window_indices"], dtype=np.int32)
        current_size = int(np.asarray(archive["current_size"]).item())
    _require(current_size == int(factor.header[17]), "current-size mismatch")
    _require(tuple(int(v) for v in factor.header[16:19]) == (31, 60, 1), "unexpected factor layout")
    rotation_map, rotation_error = _rotation_map(factor.rotations, recovar_rotations)
    native_rotations = np.asarray(factor.rotations["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    _require(
        np.array_equal(recovar_rotations[rotation_map], native_rotations.transpose(0, 2, 1)),
        "RELION/RECOVAR rotation mapping is not bit exact",
    )

    native_data, native_weight, native_support = _factor_native_rows(factor)
    # RECOVAR accepts physical-box packed indices and expands them to the
    # native current-size square internally.  Map factor current-size pixels
    # to the corresponding physical 128-pixel packed positions.
    factor_x = np.asarray(factor.pixels["x"], dtype=np.int32)
    factor_y = np.asarray(factor.pixels["y"], dtype=np.int32)
    physical_size = 128
    physical_indices = (factor_y % physical_size) * (physical_size // 2 + 1) + factor_x
    recovar_centered_rows = recon_indices // (physical_size // 2 + 1)
    recovar_x = recon_indices % (physical_size // 2 + 1)
    recovar_y = recovar_centered_rows - physical_size // 2
    recovar_fftw_indices = (
        (recovar_y % physical_size) * (physical_size // 2 + 1) + recovar_x
    )
    _require(
        np.array_equal(
            np.sort(recovar_fftw_indices),
            np.sort(physical_indices[native_support.any(axis=0)]),
        ),
        "factor/RECOVAR reconstruction pixel support changed",
    )

    # Reorder native rows into RECOVAR's rotation order.  Every dense pixel is
    # retained so the signature uses RELION's exact current-size block topology.
    data_rows = np.zeros_like(native_data)
    weight_rows = np.zeros_like(native_weight)
    data_rows[rotation_map] = native_data
    weight_rows[rotation_map] = native_weight
    contributor_rows = np.flatnonzero(np.any(weight_rows > 0, axis=1)).astype(np.int32)
    accumulator_shape = (123, 123, 123)
    accumulator_size = accumulator_shape[0] * accumulator_shape[1] * (accumulator_shape[2] // 2 + 1)
    with jax.default_device(jax.devices("gpu")[0]):
        outputs = cuda_backproject.relion_fused_x_half_backproject_signature_indexed(
            jnp.zeros(accumulator_size, dtype=jnp.complex64),
            jnp.zeros(accumulator_size, dtype=jnp.float32),
            jnp.asarray(data_rows),
            jnp.asarray(weight_rows),
            jnp.asarray(physical_indices, dtype=jnp.int32),
            jnp.asarray(recovar_rotations),
            jnp.arange(recovar_rotations.shape[0], dtype=jnp.int32),
            jnp.asarray(contributor_rows, dtype=jnp.int32),
            (physical_size, physical_size),
            accumulator_shape,
            30.0,
        )
        device = [np.asarray(value) for value in outputs[:9]]

    rotation_keys, pixel_ids, row_flags, source_values, neighbor_indices, coefficients, neighbor_flags = device[2:]
    _require(np.array_equal(rotation_keys[:, 0], contributor_rows), "signature rotation identity changed")
    _require(np.array_equal(pixel_ids, np.broadcast_to(np.arange(current_size * (current_size // 2 + 1)), pixel_ids.shape)), "signature pixel order changed")
    recovar_support = (row_flags & np.int32(64)) != 0
    factor_rows = np.asarray([np.flatnonzero(rotation_map == row)[0] for row in contributor_rows], dtype=np.int64)
    selected_native_support = native_support[factor_rows]
    _require(np.array_equal(recovar_support, selected_native_support), "RELION/RECOVAR scatter support differs")

    captured_geometry = _captured_native_geometry(factor)
    native_geometry = (
        captured_geometry
        if captured_geometry is not None
        else _native_geometry(factor, accumulator_shape=accumulator_shape)
    )
    expected_coordinates = native_geometry["coordinates_zyx"][factor_rows]
    expected_fold = native_geometry["folded"][factor_rows]
    expected_indices = native_geometry["neighbor_indices"][factor_rows]
    expected_coefficients = native_geometry["neighbor_coefficients"][factor_rows]
    active = selected_native_support
    live_neighbors = active[..., None] & ((neighbor_flags & np.int32(1)) != 0)
    _require(np.all(live_neighbors[active]), "active scatter row has an invalid neighbor")

    return {
        "stack_index_one_based": int(factor.stack_index),
        "factor_path": str(factor_path.resolve()),
        "factor_sha256": _sha256(factor_path),
        "pass2_path": str(pass2_path.resolve()),
        "pass2_sha256": _sha256(pass2_path),
        "rotation_map_max_abs": rotation_error,
        "rotation_map_bit_exact": True,
        "contributor_rotation_count": int(contributor_rows.size),
        "active_scatter_row_count": int(np.count_nonzero(active)),
        "support_exact": True,
        "native_geometry_oracle": (
            "passive_relion_device_capture"
            if captured_geometry is not None
            else "independent_host_float32_equations"
        ),
        "fold_exact": bool(np.array_equal(expected_fold[active], (row_flags[active] & 16) != 0)),
        "source_data": _metric(
            np.stack(
                (
                    data_rows[contributor_rows].real,
                    data_rows[contributor_rows].imag,
                ),
                axis=-1,
            )[active],
            source_values[..., :2][active],
        ),
        "source_weight": _metric(weight_rows[contributor_rows][active], source_values[..., 2][active]),
        "coordinates_zyx": _metric(expected_coordinates[active], source_values[..., 3:6][active]),
        "neighbor_indices_exact": bool(np.array_equal(expected_indices[active], neighbor_indices[active])),
        "neighbor_index_mismatch_count": int(np.count_nonzero(expected_indices[active] != neighbor_indices[active])),
        "neighbor_coefficients": _metric(expected_coefficients[active], coefficients[active]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-directory", type=Path, required=True)
    parser.add_argument("--pass2-directory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")

    particles = []
    for factor_path in sorted(args.factor_directory.glob("*.bpre-v2.bin")):
        factor = _load_factor(factor_path)
        # The frozen panel records zero-based original index in the pass-2 file;
        # identify it through the immutable one-based stack index used here.
        candidates = []
        for pass2_path in sorted(args.pass2_directory.glob("pass2_orig*_cs060.npz")):
            with np.load(pass2_path, allow_pickle=False) as archive:
                original = int(np.asarray(archive["original_index"]).item())
            if original + 1 == factor.stack_index:
                candidates.append(pass2_path)
        _require(len(candidates) == 1, f"stack {factor.stack_index}: pass-2 capture identity is ambiguous")
        particles.append(_analyze_particle(factor_path, candidates[0]))

    geometry_closes = all(
        particle["support_exact"]
        and particle["fold_exact"]
        and particle["neighbor_indices_exact"]
        and particle["coordinates_zyx"]["relative_l2_over_reference"] <= 1.0e-6
        and particle["neighbor_coefficients"]["relative_l2_over_reference"] <= 1.0e-6
        for particle in particles
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact/relative-L2 intermediates; no correlation; FSC/FSC-AUC remain map acceptance",
        "particle_count": len(particles),
        "classification": (
            "fixed_panel_scatter_geometry_closes"
            if geometry_closes
            else "fixed_panel_scatter_geometry_mismatch"
        ),
        "next_boundary": (
            "all_particle_posterior_operand_distribution_or_inter_particle_reduction"
            if geometry_closes
            else "first_reported_scatter_geometry_field"
        ),
        "particles": particles,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": report["classification"], "output": str(args.output_json.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
