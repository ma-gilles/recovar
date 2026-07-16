#!/usr/bin/env python3
"""Recompute frozen BPref scatter operands in float64/complex128.

This diagnostic consumes complete RECOVAR device-signature shards and their
version-3 companion contribution bundles.  It starts from the captured raw
float32 particle images (the native stack precision), then recomputes integer
pre-shifting, FFTs, CTFs, translation phases, posterior-weighted source rows,
and trilinear coefficients in float64/complex128.  Captured complex64 source
values are never promoted and reused as recomputed operands.

The output is the hash-bound schema accepted by
``validate_bpref_device_signature.py --recomputed-high-precision``.  The
recomputation fails closed when its float64 geometry changes a captured target
index; such a boundary is a geometry/precision result and cannot be represented
by the current same-target replay schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import validate_bpref_device_signature as validator


def _shift_real_image(image: np.ndarray, shift: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    dx, dy = np.asarray(shift, dtype=np.int64).tolist()
    height, width = image.shape
    shifted = np.zeros_like(image)
    src_x0, src_x1 = max(0, -dx), width - max(0, dx)
    src_y0, src_y1 = max(0, -dy), height - max(0, dy)
    dst_x0, dst_y0 = max(0, dx), max(0, dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        shifted[
            dst_y0 : dst_y0 + src_y1 - src_y0,
            dst_x0 : dst_x0 + src_x1 - src_x0,
        ] = image[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _centered_rfft2(image: np.ndarray) -> np.ndarray:
    transformed = np.fft.rfft2(np.fft.fftshift(image))
    return np.fft.fftshift(transformed, axes=-2).reshape(-1).astype(
        np.complex128, copy=False
    )


def _half_lattice(image_shape: tuple[int, int], voxel_size: float) -> np.ndarray:
    height, width = image_shape
    if height != width:
        raise ValueError("high-precision BPref replay currently requires square images")
    y = np.arange(-height // 2, height - height // 2, dtype=np.float64)
    x = np.arange(0, width // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x / (width * voxel_size), y / (height * voxel_size))
    return np.stack((xx.ravel(), yy.ravel()), axis=-1)


def _ctf_float64(
    params: np.ndarray,
    image_shape: tuple[int, int],
    voxel_size: float,
) -> np.ndarray:
    values = np.asarray(params, dtype=np.float64)
    if values.size < 9:
        raise ValueError("CTF parameter vector must contain at least nine values")
    dfu, dfv, dfang, volt, cs, amplitude, phase_shift, bfactor, contrast = values[:9]
    frequency = _half_lattice(image_shape, voxel_size)
    x, y = frequency[:, 0], frequency[:, 1]
    angle = np.arctan2(y, x)
    s2 = x * x + y * y
    dfang = np.deg2rad(dfang)
    phase_shift = np.deg2rad(phase_shift)
    volt *= 1000.0
    cs *= 1.0e7
    wavelength = 12.2643247 / np.sqrt(volt * (1.0 + volt * 0.978466e-6))
    defocus = 0.5 * (
        dfu + dfv + (dfu - dfv) * np.cos(2.0 * (angle - dfang))
    )
    gamma = 2.0 * np.pi * (
        -0.5 * defocus * wavelength * s2
        + 0.25 * cs * wavelength**3 * s2**2
    ) - phase_shift
    ctf = np.sqrt(1.0 - amplitude * amplitude) * np.sin(gamma)
    ctf -= amplitude * np.cos(gamma)
    return ctf * np.exp(-bfactor * s2 / 4.0) * contrast


def _dense_to_compact(contribution: dict[str, np.ndarray], dense_height: int) -> np.ndarray:
    image_shape = tuple(int(value) for value in contribution["image_shape"])
    half_width = image_shape[1] // 2 + 1
    compact = np.asarray(contribution["window_indices"], dtype=np.int64)
    full_rows = compact // half_width
    columns = compact % half_width
    signed_rows = np.where(
        full_rows <= image_shape[0] // 2,
        full_rows,
        full_rows - image_shape[0],
    )
    dense_indices = np.mod(signed_rows, dense_height) * (dense_height // 2 + 1)
    dense_indices += columns
    if np.unique(dense_indices).size != dense_indices.size:
        raise ValueError("compact reconstruction window is not unique in dense layout")
    lookup = np.full(dense_height * (dense_height // 2 + 1), -1, dtype=np.int64)
    lookup[dense_indices] = np.arange(compact.size, dtype=np.int64)
    return lookup


def _particle_processed_fft(
    contribution: dict[str, np.ndarray], particle: int
) -> np.ndarray:
    if not bool(np.asarray(contribution["relion_cuda_preprocess"]).item()):
        raise NotImplementedError(
            "genuine recomputation currently supports the frozen relion_cuda raw-image boundary"
        )
    raw = np.asarray(contribution["raw_real_images"][particle], dtype=np.float64)
    factor = float(contribution["relion_preprocess_normalization_factors"][particle])
    shifted = _shift_real_image(
        raw * np.float64(factor), contribution["integer_pre_shifts"][particle]
    )
    return _centered_rfft2(shifted)


def _source_rows_float64(
    contribution: dict[str, np.ndarray],
    signature: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    image_shape = tuple(int(value) for value in contribution["image_shape"])
    fftw_indices = np.asarray(contribution["window_indices"], dtype=np.int64)
    half_width = image_shape[1] // 2 + 1
    fftw_rows = fftw_indices // half_width
    compact_indices = (
        ((fftw_rows - image_shape[0] // 2) % image_shape[0]) * half_width
        + fftw_indices % half_width
    ).astype(np.int64, copy=False)
    noise = np.asarray(contribution["noise_variance_half"], dtype=np.float64)
    if noise.shape != (image_shape[0] * (image_shape[1] // 2 + 1),):
        raise ValueError("noise_variance_half does not match packed image shape")
    ctf_mode = str(np.asarray(contribution["ctf_mode"]).item())
    if ctf_mode not in {"legacy", "SPA"}:
        raise NotImplementedError(
            f"high-precision recomputation does not support CTF mode {ctf_mode!r}"
        )
    if float(np.asarray(contribution["ctf_dose_per_tilt"]).item()) != 0.0:
        raise NotImplementedError("dose-weighted CTF recomputation is not implemented")

    translations = np.asarray(contribution["fine_translations"], dtype=np.float64)
    phase_lattice = _half_lattice(image_shape, 1.0)
    phases = np.exp(-2j * np.pi * (translations @ phase_lattice.T))[:, compact_indices]
    probs = np.asarray(contribution["reconstruction_probs"], dtype=np.float64)
    active_particle = np.asarray(contribution["active_particle_rows"], dtype=np.int32)
    active_row = np.asarray(contribution["active_rotation_rows"], dtype=np.int32)
    particle_launches = np.asarray(signature["particle_launch_ordinals"], dtype=np.int64)
    row_launches = np.asarray(signature["launch_ordinal"], dtype=np.int64)
    row_local = np.asarray(signature["particle_local_row"], dtype=np.int32)
    launch_to_particle = {int(value): index for index, value in enumerate(particle_launches)}

    q = row_launches.size
    source_data = np.empty((q, compact_indices.size), dtype=np.complex128)
    source_weight = np.empty((q, compact_indices.size), dtype=np.float64)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for q_row, (launch, local_row) in enumerate(zip(row_launches, row_local)):
        particle = launch_to_particle[int(launch)]
        if particle not in cache:
            fft = _particle_processed_fft(contribution, particle)
            ctf = _ctf_float64(
                contribution["ctf_params"][particle],
                image_shape,
                float(np.asarray(contribution["voxel_size"]).item()),
            )
            applied_scale = float(contribution["scale_corrections"][particle])
            weighted = fft[compact_indices] * ctf[compact_indices]
            weighted /= noise[compact_indices]
            weighted *= np.float64(applied_scale)
            ctf_weight = ctf[compact_indices] ** 2 / noise[compact_indices]
            ctf_weight *= np.float64(applied_scale) ** 2
            cache[particle] = (weighted, ctf_weight)
        weighted, ctf_weight = cache[particle]
        active_match = np.flatnonzero(
            (active_particle == particle) & (active_row == int(local_row))
        )
        if active_match.size != 1:
            raise ValueError("signature row does not have one exact companion active row")
        row_probs = probs[particle, int(local_row)]
        source_data[q_row] = np.sum(
            row_probs[:, None] * weighted[None, :] * phases,
            axis=0,
            dtype=np.complex128,
        )
        source_weight[q_row] = np.sum(row_probs, dtype=np.float64) * ctf_weight
    return source_data, source_weight


def _geometry_float64(
    contribution: dict[str, np.ndarray],
    signature: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    dense_height = 2 * int(round(float(np.asarray(signature["max_r"]).item())))
    dense_width = dense_height // 2 + 1
    dense_pixels = dense_height * dense_width
    rows = np.arange(dense_pixels, dtype=np.int64) // dense_width
    x = np.arange(dense_pixels, dtype=np.int64) % dense_width
    y = np.where(rows <= dense_height // 2, rows, rows - dense_height)
    source_xyz = np.stack((x, y, np.zeros_like(x)), axis=-1).astype(np.float64)

    active_particle = np.asarray(contribution["active_particle_rows"], dtype=np.int32)
    active_row = np.asarray(contribution["active_rotation_rows"], dtype=np.int32)
    active_rotations = np.asarray(contribution["active_rotations"], dtype=np.float64)
    particle_launches = np.asarray(signature["particle_launch_ordinals"], dtype=np.int64)
    launch_to_particle = {int(value): index for index, value in enumerate(particle_launches)}
    row_launches = np.asarray(signature["launch_ordinal"], dtype=np.int64)
    row_local = np.asarray(signature["particle_local_row"], dtype=np.int32)
    padding = float(np.asarray(signature["reconstruction_padding_factor"]).item())
    rotations = []
    for launch, local_row in zip(row_launches, row_local):
        particle = launch_to_particle[int(launch)]
        match = np.flatnonzero(
            (active_particle == particle) & (active_row == int(local_row))
        )
        if match.size != 1:
            raise ValueError("cannot align signature geometry to one active rotation")
        rotations.append(active_rotations[int(match[0])])
    rotations = np.asarray(rotations, dtype=np.float64)
    source_x = source_xyz[:, 0]
    source_y = source_xyz[:, 1]

    # The CUDA binding stores RECOVAR's matrix with reversed xyz columns in
    # RELION's six-value 2-D backprojection layout.  Validate that convention
    # by emulating the exact float32 multiply/add/scale order against the
    # device-produced coordinates before evaluating the same formula in f64.
    rotations_f32 = rotations.astype(np.float32)
    x_f32 = source_x.astype(np.float32)
    y_f32 = source_y.astype(np.float32)
    padding_f32 = np.float32(padding)
    promoted_control = []
    recomputed = []
    for column in (2, 1, 0):
        x_term = rotations_f32[:, None, 0, column] * x_f32[None, :]
        y_term = rotations_f32[:, None, 1, column] * y_f32[None, :]
        promoted_control.append((x_term + y_term) * padding_f32)
        recomputed.append(
            (
                rotations[:, None, 0, column] * source_x[None, :]
                + rotations[:, None, 1, column] * source_y[None, :]
            ) * padding
        )
    promoted_control = np.stack(promoted_control, axis=-1).astype(np.float32, copy=False)
    xyz = np.stack(recomputed, axis=-1)
    loaded = (np.asarray(signature["row_flags"], dtype=np.int32) & (8 | 32 | 64)) != 0
    captured_xyz = np.asarray(signature["source_values"], dtype=np.float32)[..., 3:6]
    coordinate_control_equal = np.array_equal(promoted_control[loaded], captured_xyz[loaded])
    coordinate_delta = np.abs(
        promoted_control[loaded].astype(np.float64)
        - captured_xyz[loaded].astype(np.float64)
    )
    coordinate_control_max_abs = float(np.max(coordinate_delta))
    coordinate_ulp = np.abs(np.spacing(captured_xyz[loaded])).astype(np.float64)
    coordinate_control_max_ulp = float(
        np.max(np.divide(
            coordinate_delta,
            coordinate_ulp,
            out=np.zeros_like(coordinate_delta),
            where=coordinate_ulp > 0,
        ))
    )
    coordinate_control_normalized_max = coordinate_control_max_abs / max(
        1.0, float(np.max(np.abs(captured_xyz[loaded])))
    )
    coordinate_control_bound = 2.0 * float(np.finfo(np.float32).eps)
    # CUDA contracts these multiply/add pairs to FMA and its six matrix values
    # are prepared on device, whereas NumPy evaluates separate operations from
    # the captured host matrix.  A scale-normalized two-epsilon envelope
    # validates sign, lattice, matrix orientation, axis order, and padding
    # without pretending the host expression reconstructs every low-magnitude
    # device ULP.
    if coordinate_control_normalized_max > coordinate_control_bound:
        raise ValueError(
            "promoted-f32 rotation/axis control does not reproduce captured device coordinates: "
            f"normalized_max={coordinate_control_normalized_max:.9g} "
            f"bound={coordinate_control_bound:.9g}"
        )

    rk0, rk1, rk2 = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    zp, yp, xp = rk0, rk1, rk2
    folded = xp < 0.0
    xp = np.where(folded, -xp, xp)
    yp = np.where(folded, -yp, yp)
    zp = np.where(folded, -zp, zp)
    x_floor, y_floor, z_floor = np.floor(xp), np.floor(yp), np.floor(zp)
    fx, fy, fz = xp - x_floor, yp - y_floor, zp - z_floor
    wx = np.stack((1.0 - fx, fx), axis=-1)
    wy = np.stack((1.0 - fy, fy), axis=-1)
    wz = np.stack((1.0 - fz, fz), axis=-1)
    coefficients = np.empty((len(rotations), dense_pixels, 8), dtype=np.float64)
    for neighbor in range(8):
        dz, rem = divmod(neighbor, 4)
        dy, dx = divmod(rem, 2)
        coefficients[..., neighbor] = wz[..., dz] * wy[..., dy] * wx[..., dx]

    volume_shape = tuple(int(value) for value in signature["volume_shape"])
    n0, n1, n2 = volume_shape
    n2_half = n2 // 2 + 1
    z0 = z_floor.astype(np.int64) + n0 // 2
    y0 = y_floor.astype(np.int64) + n1 // 2
    x0 = x_floor.astype(np.int64)
    targets = np.empty_like(coefficients, dtype=np.int64)
    for neighbor in range(8):
        dz, rem = divmod(neighbor, 4)
        dy, dx = divmod(rem, 2)
        targets[..., neighbor] = (
            (z0 + dz) * n1 * n2_half + (y0 + dy) * n2_half + x0 + dx
        )
    max_r = float(np.asarray(signature["max_r"]).item())
    max_radius_padded = max_r * padding
    redundant = (x[None, :] == 0) & (rows[None, :] >= dense_width)
    radius_2d = (source_x[None, :] * padding) ** 2 + (source_y[None, :] * padding) ** 2
    outside_2d = radius_2d > max_radius_padded**2
    positive_weight = np.asarray(signature["source_values"], dtype=np.float32)[..., 2] > 0
    outside_3d = rk0 * rk0 + rk1 * rk1 + rk2 * rk2 > max_radius_padded**2
    max_radius_int = int(np.floor(max_radius_padded + 0.5))
    x_compact = np.floor(xp).astype(np.int64)
    y_compact = np.floor(yp).astype(np.int64) + max_radius_int + 1
    z_compact = np.floor(zp).astype(np.int64) + max_radius_int + 1
    compact_xdim = max_radius_int + 2
    compact_ydim = 2 * max_radius_int + 3
    compact_oob = (
        (x_compact < 0) | (x_compact + 1 >= compact_xdim)
        | (y_compact < 0) | (y_compact + 1 >= compact_ydim)
        | (z_compact < 0) | (z_compact + 1 >= compact_ydim)
    )
    reached_f64 = (
        ~redundant & ~outside_2d & positive_weight & ~outside_3d & ~compact_oob
    )
    reached_device = (np.asarray(signature["row_flags"], dtype=np.int32) & 64) != 0
    support_mismatch = reached_f64 != reached_device
    control = {
        "rotation_formula": (
            "rk0=(R[0,2]*x+R[1,2]*y)*padding; "
            "rk1=(R[0,1]*x+R[1,1]*y)*padding; "
            "rk2=(R[0,0]*x+R[1,0]*y)*padding"
        ),
        "axis_order": "captured rk0,rk1,rk2; compact scatter z,y,x = rk0,rk1,abs(rk2)",
        "promoted_f32_coordinate_control_equal": coordinate_control_equal,
        "promoted_f32_coordinate_control_max_abs": coordinate_control_max_abs,
        "promoted_f32_coordinate_control_max_ulp": coordinate_control_max_ulp,
        "promoted_f32_coordinate_control_normalized_max": coordinate_control_normalized_max,
        "promoted_f32_coordinate_control_normalized_bound": coordinate_control_bound,
        "promoted_f32_coordinate_control_policy": (
            "two float32 eps relative to coordinate scale; host matrix/add contraction differs from device"
        ),
        "float64_reached_support_mismatch_count": int(np.count_nonzero(support_mismatch)),
        "float64_reached_support_mismatch_dense_rows": np.argwhere(support_mismatch).tolist(),
    }
    return coefficients, targets, folded, control


def _recompute_shard(result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    contribution = result["contribution"]
    signature = result["signature"]
    records = result["contribution_records"]
    dense_height = 2 * int(round(float(np.asarray(signature["max_r"]).item())))
    dense_lookup = _dense_to_compact(contribution, dense_height)
    row_data, row_weight = _source_rows_float64(contribution, signature)
    coefficients, targets, folded, geometry_control = _geometry_float64(
        contribution, signature
    )

    row_keys = {
        (int(launch), int(local_row)): index
        for index, (launch, local_row) in enumerate(
            zip(signature["launch_ordinal"], signature["particle_local_row"])
        )
    }
    q_index = np.asarray(
        [row_keys[(int(launch), int(local_row))] for launch, local_row in zip(
            records.launch_ordinal, records.particle_local_row
        )],
        dtype=np.int64,
    )
    compact_position = dense_lookup[records.dense_pixel]
    if np.any(compact_position < 0):
        raise ValueError("valid device contribution refers to a pixel outside compact window")
    recomputed_data = row_data[q_index, compact_position]
    recomputed_weight = row_weight[q_index, compact_position]
    recomputed_coefficients = coefficients[
        q_index, records.dense_pixel, records.neighbor
    ]
    recomputed_targets = targets[q_index, records.dense_pixel, records.neighbor]
    target_mismatch = recomputed_targets != records.target_indices
    fold_mismatch = folded[q_index, records.dense_pixel] != records.row_conjugated
    geometry = {
        "record_count": int(records.size),
        "target_mismatch_count": int(np.count_nonzero(target_mismatch)),
        "row_fold_mismatch_count": int(np.count_nonzero(fold_mismatch)),
        "coefficient_max_abs_vs_captured_f32": float(
            np.max(np.abs(recomputed_coefficients - records.coefficients.astype(np.float64)))
        ),
        "source_data_float64_vs_captured_complex64": validator.exact_array_metrics(
            recomputed_data, records.source_data
        ),
        "source_weight_float64_vs_captured_float32": validator.exact_array_metrics(
            recomputed_weight, records.source_weight
        ),
        "geometry_formula_control": geometry_control,
    }
    geometry["same_target_replay_compatible"] = not (
        geometry["target_mismatch_count"]
        or geometry["row_fold_mismatch_count"]
        or geometry_control["float64_reached_support_mismatch_count"]
    )
    return recomputed_coefficients, recomputed_data, recomputed_weight, geometry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signatures", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    results = sorted(
        (validator._validate_signature(path) for path in args.signatures),
        key=lambda result: (result["launch_min"], result["path"]),
    )
    reference = results[0]["signature"]
    boundary = tuple(
        int(validator._scalar(reference, key))
        for key in ("iteration", "half", "class_index", "rank")
    )
    for result in results:
        current = tuple(
            int(validator._scalar(result["signature"], key))
            for key in ("iteration", "half", "class_index", "rank")
        )
        if current != boundary:
            raise ValueError("recomputation shards mix iteration/half/class/rank boundaries")

    coefficients, source_data, source_weight, geometry = [], [], [], []
    for result in results:
        shard_coefficients, shard_data, shard_weight, shard_geometry = _recompute_shard(result)
        coefficients.append(shard_coefficients)
        source_data.append(shard_data)
        source_weight.append(shard_weight)
        shard_geometry["signature"] = result["path"]
        geometry.append(shard_geometry)

    incompatible_geometry = [
        item for item in geometry if not item["same_target_replay_compatible"]
    ]
    if incompatible_geometry:
        summary = {
            "schema": "recovar-bpref-high-precision-recomputation-report-v1",
            "status": "GEOMETRY_PRECISION_BOUNDARY",
            "classification": "discrete_geometry_difference_under_float64_recomputation",
            "iteration": boundary[0],
            "half": boundary[1],
            "class_index_zero_based": boundary[2],
            "rank": boundary[3],
            "signature_count": len(results),
            "geometry_by_shard": geometry,
            "artifact_written": False,
            "reason": (
                "float64 recomputation changes support, fold, or target geometry; "
                "the same-target verified replay schema must not hide that difference"
            ),
        }
        text = json.dumps(summary, indent=2) + "\n"
        if args.summary_out is not None:
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(text)
        print(text, end="")
        raise SystemExit(2)

    records = validator.concatenate_contribution_records(
        [result["contribution_records"] for result in results]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    companions = [
        Path(str(validator._scalar(result["signature"], "companion_contribution_path")))
        for result in results
    ]
    companion_values = [result["contribution"] for result in results]

    def companion_digest(fields: tuple[str, ...]) -> str:
        return validator._sha256_named_arrays(
            (f"shard{shard}:{field}", values[field])
            for shard, values in enumerate(companion_values)
            for field in fields
        )

    np.savez(
        args.output,
        magic=np.asarray(validator.RECOMPUTATION_MAGIC),
        schema=np.asarray(validator.RECOMPUTATION_SCHEMA),
        schema_version=np.int32(2),
        parent_signature_sha256=np.asarray(
            [validator._sha256_file(Path(result["path"])) for result in results]
        ),
        companion_contribution_sha256=np.asarray(
            [validator._sha256_file(path) for path in companions]
        ),
        semantic_identity_sha256=np.asarray(validator._semantic_identity_digest(records)),
        formula_name=np.asarray(validator.RECOMPUTATION_FORMULA_NAME),
        formula_version=np.asarray(validator.RECOMPUTATION_FORMULA_VERSION),
        numeric_policy=np.asarray(validator.RECOMPUTATION_NUMERIC_POLICY),
        source_dtype=np.asarray(validator.RECOMPUTATION_SOURCE_POLICY),
        source_boundary=np.asarray(
            "native float32 stack pixels; downstream operands recomputed without captured-complex64 promotion"
        ),
        fft_layout=np.asarray("centered-y packed-x rfft; flattened C order"),
        fft_normalization=np.asarray("unnormalized forward numpy.fft.rfft2"),
        posterior_weight_policy=np.asarray(
            "captured reconstruction_probs frozen at M-step boundary"
        ),
        canonical_sort_key_legend=np.asarray(
            "original_index,canonical_rotation_key,dense_pixel,neighbor"
        ),
        iteration=np.int32(boundary[0]),
        half=np.int32(boundary[1]),
        class_index=np.int32(boundary[2]),
        rank=np.int32(boundary[3]),
        raw_image_identity_sha256=np.asarray(companion_digest(("image_identities",))),
        raw_image_input_sha256=np.asarray(companion_digest((
            "raw_real_images", "integer_pre_shifts",
            "relion_preprocess_normalization_factors",
        ))),
        ctf_noise_input_sha256=np.asarray(companion_digest((
            "ctf_params", "noise_variance_half", "scale_corrections",
            "voxel_size", "ctf_mode", "ctf_dose_per_tilt", "ctf_angle_per_tilt",
        ))),
        posterior_weight_sha256=np.asarray(companion_digest((
            "reconstruction_probs", "reconstruction_mask",
            "reconstruction_sum_weight", "reconstruction_threshold",
        ))),
        hypothesis_geometry_input_sha256=np.asarray(companion_digest((
            "active_particle_rows", "active_rotation_rows", "active_rotations",
            "oversampled_rotation_indices", "fine_translations", "window_indices",
        ))),
        canonical_original_index=records.original_index,
        canonical_rotation_key=records.canonical_rotation_key,
        canonical_dense_pixel=records.dense_pixel,
        canonical_neighbor=records.neighbor,
        captured_target_indices=records.target_indices,
        captured_row_conjugated=records.row_conjugated,
        captured_neighbor_conjugated=records.neighbor_conjugated,
        recomputed_coefficients=np.concatenate(coefficients).astype(np.float64, copy=False),
        recomputed_source_data=np.concatenate(source_data).astype(np.complex128, copy=False),
        recomputed_source_weight=np.concatenate(source_weight).astype(np.float64, copy=False),
    )
    # Round-trip through the fail-closed loader before reporting success.
    validator.load_verified_recomputation(
        args.output,
        records,
        parent_signature_paths=tuple(Path(result["path"]) for result in results),
        companion_contribution_paths=tuple(companions),
    )
    summary = {
        "schema": "recovar-bpref-high-precision-recomputation-report-v1",
        "status": "PASS",
        "iteration": boundary[0],
        "half": boundary[1],
        "class_index_zero_based": boundary[2],
        "rank": boundary[3],
        "signature_count": len(results),
        "record_count": records.size,
        "source_boundary": (
            "native float32 stack pixels; normalization, integer shift, FFT, CTF, "
            "translation phase, posterior-weighted source, and geometry recomputed in float64/complex128"
        ),
        "geometry_by_shard": geometry,
        "artifact": str(args.output.resolve()),
        "artifact_sha256": validator._sha256_file(args.output),
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
