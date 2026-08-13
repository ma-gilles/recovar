#!/usr/bin/env python3
"""Localize a K=1 norm-residual mismatch at the BPref operand boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .validate_relion_bpref_prescatter import (
        FOOTER_MAGIC,
        FOOTER_STRUCT,
        HEADER_MAGIC,
        HEADER_STRUCT,
        ROTATION_DTYPE,
        ROW_DTYPE,
        load_artifact,
    )
else:
    from validate_relion_bpref_prescatter import (  # type: ignore[no-redef]
        FOOTER_MAGIC,
        FOOTER_STRUCT,
        HEADER_MAGIC,
        HEADER_STRUCT,
        ROTATION_DTYPE,
        ROW_DTYPE,
        load_artifact,
    )


SCHEMA = "recovar.em.k1_norm_residual_bpref_boundary.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_native_memmap(path: Path) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Open a sealed native artifact without copying its multi-GB row table."""

    path = Path(path)
    with path.open("rb") as stream:
        magic, *values = HEADER_STRUCT.unpack(stream.read(HEADER_STRUCT.size))
        _require(magic == HEADER_MAGIC, "native BPref header magic changed")
        header = tuple(int(value) for value in values)
        _require(header[0] == 1, "native BPref schema changed")
        _require(header[1] == HEADER_STRUCT.size, "native BPref header size changed")
        _require(header[2] == ROW_DTYPE.itemsize, "native BPref row size changed")
        _require(header[3] == ROTATION_DTYPE.itemsize, "native BPref rotation size changed")
        rotation_count = header[16]
        row_count = header[17]
        expected_size = (
            HEADER_STRUCT.size
            + rotation_count * ROTATION_DTYPE.itemsize
            + row_count * ROW_DTYPE.itemsize
            + FOOTER_STRUCT.size
        )
        _require(path.stat().st_size == expected_size, "native BPref byte count changed")
        stream.seek(expected_size - FOOTER_STRUCT.size)
        footer_magic, footer_rows, footer_rotations = FOOTER_STRUCT.unpack(stream.read(FOOTER_STRUCT.size))
    _require(footer_magic == FOOTER_MAGIC, "native BPref footer magic changed")
    _require(footer_rows == row_count, "native BPref footer row count changed")
    _require(footer_rotations == rotation_count, "native BPref footer rotation count changed")
    rotation_offset = HEADER_STRUCT.size
    row_offset = rotation_offset + rotation_count * ROTATION_DTYPE.itemsize
    rotations = np.memmap(
        path,
        dtype=ROTATION_DTYPE,
        mode="r",
        offset=rotation_offset,
        shape=(rotation_count,),
    )
    rows = np.memmap(
        path,
        dtype=ROW_DTYPE,
        mode="r",
        offset=row_offset,
        shape=(row_count,),
    )
    _require(
        np.array_equal(rotations["orientation_local"], np.arange(rotation_count, dtype=np.uint32)),
        "native BPref rotation order changed",
    )
    _require(np.all(np.isfinite(rotations["matrix"])), "native BPref rotations are non-finite")
    return header, rotations, rows


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape, f"shape mismatch: {left.shape} != {right.shape}")
    promoted_left = left.astype(np.complex128, copy=False).reshape(-1)
    promoted_right = right.astype(np.complex128, copy=False).reshape(-1)
    delta = promoted_right - promoted_left
    denominator = max(float(np.linalg.norm(promoted_left)), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_reference": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _rotation_map(native: np.ndarray, recovar: np.ndarray) -> tuple[np.ndarray, float]:
    native_matrices = np.asarray(native["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    recovar_matrices = np.asarray(recovar, dtype=np.float32).reshape(-1, 3, 3)
    distances = np.max(
        np.abs(native_matrices.transpose(0, 2, 1)[:, None] - recovar_matrices[None]),
        axis=(2, 3),
    )
    nearest = np.argmin(distances, axis=1)
    error = distances[np.arange(nearest.size), nearest]
    _require(np.all(error <= 1.0e-6), "native/RECOVAR rotations do not match within 1e-6")
    _require(np.unique(nearest).size == nearest.size, "native rotation mapping is not one-to-one")
    return nearest.astype(np.int64), float(np.max(error, initial=0.0))


def _centered_coordinates(indices: np.ndarray, physical_image_size: int) -> list[tuple[int, int]]:
    packed = np.asarray(indices, dtype=np.int64).reshape(-1)
    half_width = physical_image_size // 2 + 1
    return [(int(index % half_width), int(index // half_width - physical_image_size // 2)) for index in packed]


def _dense_native_operands(
    *,
    rows: np.ndarray,
    native_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
    recon_window_indices: np.ndarray,
    physical_image_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rotation_rows, rotation_error = _rotation_map(native_rotations, recovar_rotations)
    coordinates = _centered_coordinates(recon_window_indices, physical_image_size)
    coordinate_to_pixel = {coordinate: index for index, coordinate in enumerate(coordinates)}
    _require(len(coordinate_to_pixel) == len(coordinates), "RECOVAR reconstruction pixels are duplicated")
    native_coordinates = list(zip(rows["x"].astype(int), rows["y"].astype(int)))
    _require(
        all(coordinate in coordinate_to_pixel for coordinate in native_coordinates),
        "native BPref emitted a pixel outside the RECOVAR reconstruction window",
    )
    shape = (np.asarray(recovar_rotations).shape[0], len(coordinates))
    data = np.zeros(shape, dtype=np.complex64)
    weight = np.zeros(shape, dtype=np.float32)
    seen: set[tuple[int, int]] = set()
    data_scale = np.float32(-1.0 / physical_image_size**2)
    weight_scale = np.float32(1.0 / physical_image_size**4)
    for row in rows:
        rotation = int(rotation_rows[int(row["orientation_local"])])
        pixel = coordinate_to_pixel[(int(row["x"]), int(row["y"]))]
        key = (rotation, pixel)
        _require(key not in seen, "native BPref rotation/pixel row is duplicated")
        seen.add(key)
        data[rotation, pixel] = np.complex64(row["source_re"] + 1j * row["source_im"]) * data_scale
        weight[rotation, pixel] = np.float32(row["source_weight"]) * weight_scale
    return (
        data,
        weight,
        {
            "rotation_max_abs": rotation_error,
            "native_supported_rows": len(seen),
            "recovar_rotation_count": int(shape[0]),
            "reconstruction_pixel_count": int(shape[1]),
            "data_scale": float(data_scale),
            "weight_scale": float(weight_scale),
        },
    )


def _exact_rotation_map(native: np.ndarray, recovar: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Map native transposed matrices to exact RECOVAR float32 rotation rows."""

    native_matrices = np.asarray(native["matrix"], dtype=np.float32).reshape(-1, 3, 3).transpose(0, 2, 1)
    recovar_matrices = np.asarray(recovar, dtype=np.float32).reshape(-1, 3, 3)
    if native_matrices.shape[0] <= recovar_matrices.shape[0] and np.array_equal(
        native_matrices, recovar_matrices[: native_matrices.shape[0]]
    ):
        mapping = np.arange(native_matrices.shape[0], dtype=np.int64)
        mode = "exact_prefix"
    else:
        recovar_keys: dict[bytes, int] = {}
        for index, matrix in enumerate(recovar_matrices):
            key = matrix.tobytes()
            if key not in recovar_keys:
                recovar_keys[key] = index
        mapping = np.asarray(
            [recovar_keys.get(matrix.tobytes(), -1) for matrix in native_matrices],
            dtype=np.int64,
        )
        _require(np.all(mapping >= 0), "native BPref rotation is absent from RECOVAR candidates")
        _require(np.unique(mapping).size == mapping.size, "native BPref rotation mapping is not one-to-one")
        mode = "exact_key"
    return mapping, {
        "mapping_mode": mode,
        "native_rotation_count": int(native_matrices.shape[0]),
        "recovar_rotation_count": int(recovar_matrices.shape[0]),
        "mapped_rotation_count": int(mapping.size),
        "unused_recovar_rotation_count": int(recovar_matrices.shape[0] - mapping.size),
    }


def _rectangle_to_reconstruction_pixels(
    *,
    rectangle_xdim: int,
    rectangle_ydim: int,
    reconstruction_indices: np.ndarray,
    physical_image_size: int,
) -> np.ndarray:
    coordinates = _centered_coordinates(reconstruction_indices, physical_image_size)
    coordinate_to_pixel = {coordinate: index for index, coordinate in enumerate(coordinates)}
    _require(len(coordinate_to_pixel) == len(coordinates), "RECOVAR reconstruction pixels are duplicated")
    mapping = np.full(rectangle_xdim * rectangle_ydim, -1, dtype=np.int32)
    for rectangle_pixel in range(mapping.size):
        x = rectangle_pixel % rectangle_xdim
        raw_y = rectangle_pixel // rectangle_xdim
        y = raw_y - rectangle_ydim if raw_y > rectangle_ydim // 2 else raw_y
        mapping[rectangle_pixel] = coordinate_to_pixel.get((x, y), -1)
    return mapping


def _native_norm_block_terms(
    projection: np.ndarray,
    projection_abs2: np.ndarray,
    rows: np.ndarray,
    *,
    orientation_start: int,
    rectangle_to_reconstruction: np.ndarray,
    noise_variance: np.ndarray,
    physical_image_size: int,
) -> dict[str, object]:
    """Form high-precision A2/XA totals from one canonical native row block."""

    proj = np.asarray(projection, dtype=np.complex64)
    proj_abs2 = np.asarray(projection_abs2, dtype=np.float32)
    _require(proj.shape == proj_abs2.shape, "native block projection/abs2 topology changed")
    block_rows = np.asarray(rows)
    _require(np.all(block_rows["state"] == 1), "native BPref block contains an inactive row")
    _require(np.all((block_rows["flags"] & np.uint32(3)) == np.uint32(3)), "native BPref row lacks support flags")
    _require(np.all(block_rows["source_weight"] > 0.0), "native BPref row has non-positive weight")
    _require(
        np.all(np.isfinite(block_rows["source_re"]))
        and np.all(np.isfinite(block_rows["source_im"]))
        and np.all(np.isfinite(block_rows["source_weight"])),
        "native BPref block contains non-finite operands",
    )
    local_rotation = block_rows["orientation_local"].astype(np.int64) - int(orientation_start)
    pixels = rectangle_to_reconstruction[block_rows["pixel"].astype(np.int64)]
    _require(np.all((local_rotation >= 0) & (local_rotation < proj.shape[0])), "native row left rotation block")
    _require(np.all(pixels >= 0), "native BPref row left exact-radius support")
    keys = local_rotation * proj.shape[1] + pixels
    _require(np.unique(keys).size == keys.size, "native BPref block contains duplicate remapped rows")
    source = (
        block_rows["source_re"].astype(np.float32) + np.complex64(1j) * block_rows["source_im"].astype(np.float32)
    ).astype(np.complex64)
    source *= np.float32(-1.0 / physical_image_size**2)
    weight = block_rows["source_weight"].astype(np.float32) * np.float32(1.0 / physical_image_size**4)
    noise = np.asarray(noise_variance, dtype=np.float32).reshape(-1)
    _require(proj.shape[1] == noise.size, "native block noise/pixel topology changed")
    selected_projection = proj[local_rotation, pixels]
    selected_noise = noise[pixels]
    raw_weight = (weight * selected_noise).astype(np.float32)
    a2_rows = (proj_abs2[local_rotation, pixels] * raw_weight).astype(np.float32)
    xa_rows = (selected_noise * (selected_projection * np.conj(source)).real.astype(np.float32)).astype(np.float32)
    return {
        "a2": float(np.sum(a2_rows, dtype=np.float64)),
        "xa": float(np.sum(xa_rows, dtype=np.float64)),
        "row_count": int(block_rows.size),
        "active_rotation_count": int(np.unique(local_rotation).size),
    }


def _norm_terms(
    projection: np.ndarray,
    projection_abs2: np.ndarray,
    summed: np.ndarray,
    ctf_prob: np.ndarray,
    noise_variance: np.ndarray,
) -> dict[str, np.ndarray]:
    proj = np.asarray(projection, dtype=np.complex64)
    proj_abs2 = np.asarray(projection_abs2, dtype=np.float32)
    data = np.asarray(summed, dtype=np.complex64)
    weight = np.asarray(ctf_prob, dtype=np.float32)
    noise = np.asarray(noise_variance, dtype=np.float32).reshape(-1)
    _require(proj.shape == proj_abs2.shape == data.shape == weight.shape, "norm operand shapes differ")
    _require(proj.shape[1] == noise.size, "noise/pixel dimensions differ")
    has_mass = weight != 0.0
    raw_weight = np.where(has_mass, weight * noise[None, :], np.float32(0.0)).astype(np.float32)
    a2 = np.where(has_mass, proj_abs2 * raw_weight, np.float32(0.0)).astype(np.float32)
    cross = np.where(data != 0.0, proj * np.conj(data), np.complex64(0.0)).astype(np.complex64)
    xa = (noise[None, :] * cross.real).astype(np.float32)
    return {
        "ctf_has_mass": has_mass,
        "ctf_probs_raw": raw_weight,
        "a2": a2,
        "cross": cross,
        "xa": xa,
    }


def _float64_scalar_summary(terms: dict[str, np.ndarray]) -> dict[str, float]:
    a2 = float(np.sum(terms["a2"], dtype=np.float64))
    xa = float(np.sum(terms["xa"], dtype=np.float64))
    return {"a2": a2, "xa": xa, "residual_a2_minus_2xa": a2 - 2.0 * xa}


def _load_native_norm_panel(path: Path, original_index: int) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as panel:
        _require(panel["schema"].item() == "relion-k1-wavg-direct-norm-v1", "native norm schema changed")
        rows = np.flatnonzero(np.asarray(panel["input_row"], dtype=np.int64) == original_index)
        _require(rows.size == 1, "native norm panel does not contain exactly one target row")
        row = int(rows[0])
        return {
            "direct_current_size": float(panel["direct_current_size"][row]),
            "powerclass_high_shell": float(panel["powerclass_high_shell"][row]),
            "total": float(panel["total"][row]),
        }


def _counterfactual_totals(
    *,
    weighted_image: float,
    recovar_a2: float,
    recovar_xa: float,
    native_a2: float,
    native_xa: float,
) -> dict[str, float]:
    return {
        "recovar": weighted_image + recovar_a2 - 2.0 * recovar_xa,
        "native_a2_only": weighted_image + native_a2 - 2.0 * recovar_xa,
        "native_xa_only": weighted_image + recovar_a2 - 2.0 * native_xa,
        "native_a2_and_xa": weighted_image + native_a2 - 2.0 * native_xa,
    }


def _analyze_chunked(
    native_path: Path,
    recovar: dict[str, np.ndarray],
    *,
    recovar_path: Path,
    recovar_projector: Path,
    physical_image_size: int,
    native_norm_panel: Path | None,
    rotation_block_size: int,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    _require(jax.default_backend() == "gpu", "chunked native BPref substitution requires a GPU")
    _require(rotation_block_size > 0, "rotation block size must be positive")
    header, native_rotations, native_rows = _open_native_memmap(native_path)
    original_index = int(recovar["original_index"])
    current_size = int(recovar["current_size"])
    _require(header[8] == original_index + 1, "stack identity changed")
    _require(header[5] == int(recovar["iteration"]), "iteration identity changed")
    _require(
        header[12] == current_size // 2 + 1 and header[13] == current_size and header[14] == 1,
        "native/RECOVAR current-size layouts differ",
    )
    recovar_rotations = np.asarray(recovar["candidate_rotation_matrices"], dtype=np.float32)
    rotation_mapping, rotation_identity = _exact_rotation_map(native_rotations, recovar_rotations)
    reconstruction_indices, _ = make_fourier_window_indices_np(
        (physical_image_size, physical_image_size),
        current_size,
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    noise = np.asarray(recovar["noise_variance_for_noise"], dtype=np.float32).reshape(-1)
    _require(reconstruction_indices.size == noise.size, "RECOVAR exact-radius/noise topology changed")
    rectangle_to_reconstruction = _rectangle_to_reconstruction_pixels(
        rectangle_xdim=header[12],
        rectangle_ydim=header[13],
        reconstruction_indices=reconstruction_indices,
        physical_image_size=physical_image_size,
    )
    with np.load(recovar_projector, allow_pickle=False) as projector:
        ppref = np.asarray(projector["projector_half"], dtype=np.complex64)
        if ppref.ndim == 4:
            ppref = ppref[0]
        _require(int(projector["projector_r_max"]) == current_size // 2, "RECOVAR projector radius changed")

    native_a2 = 0.0
    native_xa = 0.0
    processed_rows = 0
    active_rotations = 0
    row_orientations = native_rows["orientation_local"]
    for start in range(0, rotation_mapping.size, rotation_block_size):
        end = min(start + rotation_block_size, rotation_mapping.size)
        row_start = int(np.searchsorted(row_orientations, start, side="left"))
        row_end = int(np.searchsorted(row_orientations, end, side="left"))
        projection, projection_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(ppref),
            jnp.asarray(recovar_rotations[rotation_mapping[start:end]], dtype=jnp.float32),
            (physical_image_size, physical_image_size),
            r_max=current_size // 2,
            padding_factor=2,
            return_abs2=True,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=current_size,
            pixel_indices=jnp.asarray(reconstruction_indices, dtype=jnp.int32),
            relion_texture_interp=True,
        )
        projection_np = np.asarray(jax.block_until_ready(projection), dtype=np.complex64)
        projection_abs2_np = np.asarray(jax.block_until_ready(projection_abs2), dtype=np.float32)
        block = _native_norm_block_terms(
            projection_np,
            projection_abs2_np,
            native_rows[row_start:row_end],
            orientation_start=start,
            rectangle_to_reconstruction=rectangle_to_reconstruction,
            noise_variance=noise,
            physical_image_size=physical_image_size,
        )
        native_a2 += float(block["a2"])
        native_xa += float(block["xa"])
        processed_rows += int(block["row_count"])
        active_rotations += int(block["active_rotation_count"])
    _require(processed_rows == header[17], "native BPref streaming row count changed")

    recovar_a2_chunks = np.asarray(recovar["norm_a2_per_image_by_chunk"], dtype=np.float64)
    recovar_xa_chunks = np.asarray(recovar["norm_xa_per_image_by_chunk"], dtype=np.float64)
    recovar_a2 = float(recovar["norm_a2_per_image"])
    recovar_xa = float(recovar["norm_xa_per_image"])
    _require(
        np.isclose(np.sum(recovar_a2_chunks), recovar_a2, rtol=0.0, atol=1.0e-10),
        "captured RECOVAR A2 chunks do not close",
    )
    _require(
        np.isclose(np.sum(recovar_xa_chunks), recovar_xa, rtol=0.0, atol=1.0e-10),
        "captured RECOVAR XA chunks do not close",
    )
    weighted_image = float(recovar["weighted_img_per_image"])
    totals = _counterfactual_totals(
        weighted_image=weighted_image,
        recovar_a2=recovar_a2,
        recovar_xa=recovar_xa,
        native_a2=native_a2,
        native_xa=native_xa,
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "scope": {
            "native_capture": str(Path(native_path).resolve()),
            "recovar_capture": str(Path(recovar_path).resolve()),
            "recovar_projector": str(Path(recovar_projector).resolve()),
            "original_index_zero_based": original_index,
            "stack_index_one_based": header[8],
            "iteration": int(recovar["iteration"]),
            "half": int(recovar["half"]),
            "current_size": current_size,
            "physical_image_size": physical_image_size,
            "rotation_block_size": rotation_block_size,
        },
        "identity": {
            **rotation_identity,
            "native_supported_row_count": processed_rows,
            "native_active_rotation_count_sum": active_rotations,
            "reconstruction_pixel_count": int(reconstruction_indices.size),
            "native_rectangle_pixel_count": int(header[15]),
            "native_data_scale": float(np.float32(-1.0 / physical_image_size**2)),
            "native_weight_scale": float(np.float32(1.0 / physical_image_size**4)),
        },
        "recovar_replay_closure": {
            "captured_a2": recovar_a2,
            "sum_chunk_a2": float(np.sum(recovar_a2_chunks)),
            "captured_xa": recovar_xa,
            "sum_chunk_xa": float(np.sum(recovar_xa_chunks)),
            "captured_residual": float(recovar["norm_residual_per_image"]),
            "recomputed_residual": recovar_a2 - 2.0 * recovar_xa,
        },
        "native_bpref_substitution": {
            "recovar_a2": recovar_a2,
            "recovar_xa": recovar_xa,
            "native_a2": native_a2,
            "native_xa": native_xa,
            "a2_delta_native_minus_recovar": native_a2 - recovar_a2,
            "xa_delta_native_minus_recovar": native_xa - recovar_xa,
            "recovar_weighted_image_held_fixed": weighted_image,
            "counterfactual_totals": totals,
        },
        "artifacts": {
            "recovar_projector_sha256": _sha256(recovar_projector),
        },
    }
    if native_norm_panel is not None:
        native_norm = _load_native_norm_panel(native_norm_panel, original_index)
        target = native_norm["total"]
        errors = {name: abs(value - target) for name, value in totals.items()}
        baseline_error = errors["recovar"]
        report["native_norm_target"] = {
            **native_norm,
            "absolute_errors": errors,
            "absolute_gap_closure_fraction": {
                name: (baseline_error - error) / baseline_error if baseline_error else 0.0
                for name, error in errors.items()
            },
        }
    return report


def analyze(
    native_path: Path,
    recovar_path: Path,
    *,
    physical_image_size: int,
    native_norm_panel: Path | None = None,
    recovar_projector: Path | None = None,
    rotation_block_size: int = 512,
) -> dict[str, object]:
    with np.load(recovar_path, allow_pickle=False) as capture:
        schema = capture["schema"].item()
        recovar = {name: np.asarray(capture[name]) for name in capture.files}
    if schema == "recovar-k1-scale-xa-aa-chunked-v3":
        _require(recovar_projector is not None, "chunked RECOVAR capture requires --recovar-projector")
        return _analyze_chunked(
            native_path,
            recovar,
            recovar_path=recovar_path,
            recovar_projector=recovar_projector,
            physical_image_size=physical_image_size,
            native_norm_panel=native_norm_panel,
            rotation_block_size=rotation_block_size,
        )
    _require(schema == "recovar-k1-norm-residual-inputs-v3", "RECOVAR schema changed")
    native = load_artifact(native_path)
    original_index = int(recovar["original_index"])
    _require(native.stack_index == original_index + 1, "stack identity changed")
    _require(int(native.header[5]) == int(recovar["iteration"]), "iteration identity changed")
    current_size = int(recovar["current_size"])
    _require(
        int(native.header[12]) == current_size // 2 + 1
        and int(native.header[13]) == current_size
        and int(native.header[14]) == 1,
        "native/RECOVAR current-size layouts differ",
    )
    native_data, native_weight, identity = _dense_native_operands(
        rows=native.rows,
        native_rotations=native.rotations,
        recovar_rotations=recovar["rotations_for_noise"],
        recon_window_indices=recovar["recon_window_indices"],
        physical_image_size=physical_image_size,
    )
    recovar_data = np.asarray(recovar["summed_masked_noise"], dtype=np.complex64)
    recovar_weight = np.asarray(recovar["ctf_probs"], dtype=np.float32)
    _require(native_data.shape == recovar_data.shape, "aligned native/RECOVAR data shapes differ")
    recovar_terms = _norm_terms(
        recovar["proj_for_noise"],
        recovar["proj_abs2_for_noise"],
        recovar_data,
        recovar_weight,
        recovar["noise_variance_for_noise"],
    )
    native_operand_terms = _norm_terms(
        recovar["proj_for_noise"],
        recovar["proj_abs2_for_noise"],
        native_data,
        native_weight,
        recovar["noise_variance_for_noise"],
    )
    recovar_scalar = _float64_scalar_summary(recovar_terms)
    native_operand_scalar = _float64_scalar_summary(native_operand_terms)
    captured_residual = float(recovar["block_norm_residual"])
    weighted_image = float(recovar["weighted_img_per_image"])
    report: dict[str, object] = {
        "schema": SCHEMA,
        "scope": {
            "native_capture": str(Path(native_path).resolve()),
            "recovar_capture": str(Path(recovar_path).resolve()),
            "original_index_zero_based": original_index,
            "stack_index_one_based": native.stack_index,
            "iteration": int(recovar["iteration"]),
            "half": int(recovar["half"]),
            "current_size": current_size,
            "physical_image_size": physical_image_size,
        },
        "identity": identity,
        "support": {
            "native_positive_count": int(np.count_nonzero(native_weight)),
            "recovar_positive_count": int(np.count_nonzero(recovar_weight)),
            "mask_exact": bool(np.array_equal(native_weight != 0.0, recovar_weight != 0.0)),
        },
        "bpref_operands": {
            "summed_data_recovar_vs_native": _metric(recovar_data, native_data),
            "ctf_probability_recovar_vs_native": _metric(recovar_weight, native_weight),
        },
        "recovar_replay_closure": {
            "ctf_has_mass": _metric(recovar["norm_ctf_has_mass"], recovar_terms["ctf_has_mass"]),
            "ctf_probs_raw": _metric(recovar["norm_ctf_probs_raw"], recovar_terms["ctf_probs_raw"]),
            "a2_terms": _metric(recovar["norm_a2_terms"], recovar_terms["a2"]),
            "cross_terms": _metric(recovar["norm_cross_terms"], recovar_terms["cross"]),
            "xa_terms": _metric(recovar["norm_xa_terms"], recovar_terms["xa"]),
            "captured_a2_per_image": float(recovar["norm_a2_per_image"]),
            "captured_xa_per_image": float(recovar["norm_xa_per_image"]),
            "captured_residual": captured_residual,
            "host_float64": recovar_scalar,
        },
        "native_bpref_substitution": {
            "host_float64": native_operand_scalar,
            "recovar_weighted_image_held_fixed": weighted_image,
            "counterfactual_total": weighted_image + native_operand_scalar["residual_a2_minus_2xa"],
            "recovar_total": weighted_image + captured_residual,
        },
    }
    if native_norm_panel is not None:
        native_norm = _load_native_norm_panel(native_norm_panel, original_index)
        recovar_total = weighted_image + captured_residual
        counterfactual_total = weighted_image + native_operand_scalar["residual_a2_minus_2xa"]
        original_gap = abs(recovar_total - native_norm["total"])
        counterfactual_gap = abs(counterfactual_total - native_norm["total"])
        report["native_norm_target"] = {
            **native_norm,
            "recovar_total_abs_error": original_gap,
            "native_bpref_substitution_abs_error": counterfactual_gap,
            "absolute_gap_closure_fraction": (
                (original_gap - counterfactual_gap) / original_gap if original_gap else 0.0
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bpref", type=Path, required=True)
    parser.add_argument("--recovar-norm", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--native-norm-panel", type=Path)
    parser.add_argument("--recovar-projector", type=Path)
    parser.add_argument("--rotation-block-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.native_bpref,
        args.recovar_norm,
        physical_image_size=args.physical_image_size,
        native_norm_panel=args.native_norm_panel,
        recovar_projector=args.recovar_projector,
        rotation_block_size=args.rotation_block_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
