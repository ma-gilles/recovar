#!/usr/bin/env python3
"""Compare one native RELION VDAM StoreWavg call with RECOVAR pre-scatter rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

SCHEMA = "recovar.vdam_storewavg_boundary.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat(path: Path, dtype: np.dtype) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 4, f"truncated flat array: {path}")
    count = struct.unpack_from("<i", payload)[0]
    values = np.frombuffer(payload, dtype=dtype, offset=4).copy()
    _require(values.size == count, f"flat-array size mismatch: {path}")
    return values


def _scalar(path: Path) -> float:
    payload = path.read_bytes()
    _require(len(payload) == 8, f"scalar size mismatch: {path}")
    return float(struct.unpack("<d", payload)[0])


def _real_2d(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 8, f"truncated real 2-D array: {path}")
    rows, columns = struct.unpack_from("<ii", payload)
    values = np.frombuffer(payload, dtype="<f8", offset=8).copy()
    _require(values.size == rows * columns, f"real 2-D size mismatch: {path}")
    return values.reshape(rows, columns)


def _complex_2d(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 8, f"truncated complex 2-D array: {path}")
    rows, columns = struct.unpack_from("<ii", payload)
    values = np.frombuffer(payload, dtype="<c16", offset=8).copy()
    _require(values.size == rows * columns, f"complex 2-D size mismatch: {path}")
    return values.reshape(rows, columns)


def _complex_long_3d(path: Path) -> np.ndarray:
    """Read a RELION MultidimArray dump with three native ``long`` dimensions."""

    payload = path.read_bytes()
    header_bytes = 3 * np.dtype(np.int_).itemsize
    _require(len(payload) >= header_bytes, f"truncated complex 3-D array: {path}")
    dimensions = np.frombuffer(payload, dtype=np.int_, count=3).astype(np.int64)
    _require(np.all(dimensions > 0), f"invalid complex 3-D dimensions: {path}")
    values = np.frombuffer(payload, dtype=np.complex128, offset=header_bytes).copy()
    _require(values.size == int(np.prod(dimensions)), f"complex 3-D size mismatch: {path}")
    return values.reshape(tuple(int(value) for value in dimensions))


def _load_unmasked_image(path: Path) -> np.ndarray:
    """Load only a dump whose name explicitly identifies the unmasked operand."""

    _require(path.is_file(), f"unmasked StoreWavg image is unavailable: {path}")
    _require("nomask" in path.name, f"refusing masked StoreWavg image: {path}")
    if path.name.endswith("Fimg_unweighted_nomask.bin"):
        return _complex_2d(path)
    if path.name.endswith("Fimg_nomask.bin"):
        return _complex_long_3d(path)
    raise ValueError(f"unrecognized unmasked StoreWavg image dump: {path}")


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    _require(reference.shape == candidate.shape and reference.size > 0, "metric topology mismatch")
    ref = reference.astype(np.complex128, copy=False).reshape(-1)
    cand = candidate.astype(np.complex128, copy=False).reshape(-1)
    residual = cand - ref
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    denominator = max(ref_norm, np.finfo(np.float64).tiny)
    cosine_denominator = max(ref_norm * cand_norm, np.finfo(np.float64).tiny)
    return {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "cosine": float(np.real(np.vdot(ref, cand)) / cosine_denominator),
        "max_abs": float(np.max(np.abs(residual), initial=0.0)),
        "reference_norm": ref_norm,
        "candidate_norm": cand_norm,
    }


def _match_rotations(native: np.ndarray, recovar: np.ndarray, tolerance: float) -> np.ndarray:
    native = np.asarray(native, dtype=np.float32).reshape(-1, 3, 3)
    recovar = np.asarray(recovar, dtype=np.float32).reshape(-1, 3, 3)
    distances = np.max(np.abs(native[:, None] - recovar[None, :]), axis=(2, 3))
    mapping = np.argmin(distances, axis=1).astype(np.int64)
    nearest = distances[np.arange(native.shape[0]), mapping]
    _require(float(np.max(nearest, initial=0.0)) <= tolerance, "native rotation is absent from RECOVAR")
    _require(np.unique(mapping).size == mapping.size, "native-to-RECOVAR rotation mapping is not one-to-one")
    return mapping


def _select_recovar_particle_rows(
    recovar: dict[str, np.ndarray], original_index: int | None
) -> tuple[int, np.ndarray]:
    original_indices = np.asarray(recovar["original_indices"], dtype=np.int64)
    _require(original_indices.size > 0, "RECOVAR capture contains no particles")
    if original_index is None:
        _require(original_indices.size == 1, "--recovar-original-index is required for a panel capture")
        particle_slot = 0
    else:
        matches = np.flatnonzero(original_indices == int(original_index))
        _require(matches.size == 1, f"RECOVAR capture does not uniquely contain original index {original_index}")
        particle_slot = int(matches[0])
    active_particle_rows = np.asarray(recovar["active_particle_rows"], dtype=np.int64)
    row_mask = active_particle_rows == particle_slot
    _require(np.any(row_mask), f"RECOVAR capture has no active rows for original index {original_indices[particle_slot]}")
    return particle_slot, row_mask


def _fftw_window_to_native_crop(
    window_indices: np.ndarray,
    *,
    physical_image_size: int,
    current_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map full-box FFTW-half rows to native current-size and centered indices."""

    indices = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    full_half_width = physical_image_size // 2 + 1
    rows = indices // full_half_width
    x = indices - rows * full_half_width
    ky = np.where(rows <= physical_image_size // 2, rows, rows - physical_image_size)
    _require(np.all(x <= current_size // 2), "window x coordinate exceeds native crop")
    _require(np.all(np.abs(ky) <= current_size // 2), "window y coordinate exceeds native crop")
    crop_rows = np.where(ky >= 0, ky, ky + current_size)
    crop_indices = crop_rows * (current_size // 2 + 1) + x
    centered_indices = (ky + physical_image_size // 2) * full_half_width + x
    return crop_indices.astype(np.int32), centered_indices.astype(np.int32)


def _native_gradient_rows(
    probabilities: np.ndarray,
    translated_images: np.ndarray,
    projections: np.ndarray,
    ctf: np.ndarray,
    inverse_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay RELION ``cuda_kernel_backproject3D_SGD`` before scatter."""

    probabilities = np.asarray(probabilities, dtype=np.float32)
    translated_images = np.asarray(translated_images, dtype=np.complex64)
    projections = np.asarray(projections, dtype=np.complex64)
    ctf = np.asarray(ctf, dtype=np.float32).reshape(-1)
    inverse_noise = np.asarray(inverse_noise, dtype=np.float32).reshape(-1)
    _require(probabilities.ndim == 2, "probabilities must have shape (rotation, translation)")
    _require(
        translated_images.shape == (probabilities.shape[1], ctf.size),
        "translated-image topology changed",
    )
    _require(projections.shape == (probabilities.shape[0], ctf.size), "projection topology changed")
    rotation_mass = np.sum(probabilities, axis=1, dtype=np.float32)
    ctf_inverse_noise = (ctf * inverse_noise).astype(np.float32)
    ctf2_inverse_noise = (ctf * ctf * inverse_noise).astype(np.float32)
    weighted_images = (probabilities @ translated_images).astype(np.complex64)
    data = (
        weighted_images * ctf_inverse_noise[None, :]
        - projections * (rotation_mass[:, None] * ctf2_inverse_noise[None, :])
    ).astype(np.complex64)
    weight = (rotation_mass[:, None] * ctf2_inverse_noise[None, :]).astype(np.float32)
    return data, weight


def _load_native(native_directory: Path, prefix: str) -> dict[str, np.ndarray | float | int]:
    root = native_directory / prefix
    orientation_count = int(round(_scalar(Path(f"{root}orientation_num.bin"))))
    translation_count = int(round(_scalar(Path(f"{root}translation_num.bin"))))
    raw = _flat(Path(f"{root}sorted_weights.bin"), np.dtype("<f8")).astype(np.float32)
    raw = raw.reshape(orientation_count, translation_count)
    sum_weight = np.float32(_scalar(Path(f"{root}sum_weight.bin")))
    threshold = np.float32(_scalar(Path(f"{root}significant_weight.bin")))
    probabilities = np.where(raw >= threshold, raw / sum_weight, np.float32(0.0)).astype(np.float32)
    rotations = _flat(Path(f"{root}eulers.bin"), np.dtype("<f8")).astype(np.float32)
    rotations = rotations.reshape(orientation_count, 3, 3).transpose(0, 2, 1)
    phases = _flat(Path(f"{root}trans_xyz.bin"), np.dtype("<f8")).astype(np.float32)
    _require(phases.size == 3 * translation_count, "native translation phase topology changed")
    translation_angles = np.stack(
        (phases[:translation_count], phases[translation_count : 2 * translation_count]),
        axis=1,
    )
    ctf = _flat(Path(f"{root}ctfs.bin"), np.dtype("<f8")).astype(np.float32)
    dims = _flat(Path(f"{root}wavg_ppref_dims.bin"), np.dtype("<i4")).astype(np.int64)
    _require(dims.size == 7, "native Projector dimensions changed")
    xdim, ydim, zdim, _xinit, _yinit, _zinit, r_max = (int(value) for value in dims)
    real = _flat(Path(f"{root}wavg_ppref_real.bin"), np.dtype("<f8"))
    imag = _flat(Path(f"{root}wavg_ppref_imag.bin"), np.dtype("<f8"))
    _require(real.size == imag.size == xdim * ydim * zdim, "native Projector payload changed")
    projector = (real + 1j * imag).astype(np.complex64).reshape(zdim, ydim, xdim)
    return {
        "orientation_count": orientation_count,
        "translation_count": translation_count,
        "probabilities": probabilities,
        "rotations": rotations,
        "translation_angles": translation_angles,
        "ctf": ctf,
        "projector": projector,
        "r_max": r_max,
        "retained_mass": float(np.sum(probabilities, dtype=np.float64)),
    }


def analyze(
    native_directory: Path,
    recovar_capture: Path,
    *,
    native_image_path: Path,
    native_inverse_noise_path: Path,
    recovar_original_index: int | None,
    native_prefix: str,
    physical_image_size: int,
    current_size: int,
    rotation_tolerance: float,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    _require(jax.default_backend() == "gpu", "VDAM StoreWavg replay requires a GPU")
    native = _load_native(native_directory, native_prefix)
    with np.load(recovar_capture, allow_pickle=False) as archive:
        recovar = {name: archive[name] for name in archive.files}
    _require(int(recovar["current_size"]) == current_size, "RECOVAR current size changed")
    _require(int(native["r_max"]) == current_size // 2, "native current size changed")
    particle_slot, recovar_row_mask = _select_recovar_particle_rows(recovar, recovar_original_index)

    crop_indices, centered_indices = _fftw_window_to_native_crop(
        recovar["window_indices"],
        physical_image_size=physical_image_size,
        current_size=current_size,
    )
    native_ctf = np.asarray(native["ctf"], dtype=np.float32)[crop_indices]
    inverse_noise = _real_2d(native_inverse_noise_path).astype(np.float32).reshape(-1)[crop_indices]
    native_image = _load_unmasked_image(native_image_path).astype(np.complex64).reshape(-1)[crop_indices]
    rotations = np.asarray(native["rotations"], dtype=np.float32)
    recovar_rotations = np.asarray(recovar["active_rotations"])[recovar_row_mask]
    rotation_map = _match_rotations(rotations, recovar_rotations, rotation_tolerance)

    translated = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(native_image[None, :]),
        jnp.asarray(native["translation_angles"], dtype=jnp.float32),
        jnp.asarray(centered_indices, dtype=jnp.int32),
        (physical_image_size, physical_image_size),
    )
    translated = np.asarray(jax.block_until_ready(translated), dtype=np.complex64)
    projections, _ = compute_relion_projector_projections_block(
        jnp.asarray(native["projector"]),
        jnp.asarray(rotations),
        (physical_image_size, physical_image_size),
        r_max=current_size // 2,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        dense_scale=False,
        projector_output_size=current_size,
        pixel_indices=jnp.asarray(centered_indices, dtype=jnp.int32),
        relion_texture_interp=True,
    )
    projections = np.asarray(jax.block_until_ready(projections), dtype=np.complex64)
    native_data, native_weight = _native_gradient_rows(
        native["probabilities"],
        translated,
        projections,
        native_ctf,
        inverse_noise,
    )
    recovar_data = np.asarray(recovar["active_summed"], dtype=np.complex64)[recovar_row_mask][rotation_map]
    recovar_weight = np.asarray(recovar["active_ctf_probs"], dtype=np.float32)[recovar_row_mask][rotation_map]
    data_scale = -float(physical_image_size) ** -2
    weight_scale = float(physical_image_size) ** -4
    return {
        "schema": SCHEMA,
        "identity": {
            "original_index": int(np.asarray(recovar["original_indices"])[particle_slot]),
            "image_identity": str(np.asarray(recovar["image_identities"])[particle_slot]),
            "iteration": int(recovar["iteration"]),
            "half": int(recovar["half"]),
            "physical_image_size": physical_image_size,
            "current_size": current_size,
            "orientation_count": int(native["orientation_count"]),
            "translation_count": int(native["translation_count"]),
            "retained_mass": float(native["retained_mass"]),
        },
        "frame_scales": {
            "native_data_to_recovar": data_scale,
            "native_weight_to_recovar": weight_scale,
        },
        "comparisons": {
            "gradient_numerator": _metric(native_data * data_scale, recovar_data),
            "gradient_denominator": _metric(native_weight * weight_scale, recovar_weight),
        },
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "native_unmasked_image": str(native_image_path.resolve()),
            "native_unmasked_image_sha256": _sha256(native_image_path),
            "native_inverse_noise": str(native_inverse_noise_path.resolve()),
            "native_inverse_noise_sha256": _sha256(native_inverse_noise_path),
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
        },
        "device": str(jax.devices()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-image", type=Path, required=True)
    parser.add_argument("--native-inverse-noise", type=Path)
    parser.add_argument("--recovar-original-index", type=int)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--native-prefix", default="img0_part0_storeWavg_")
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--current-size", type=int, default=38)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.recovar_capture,
        native_image_path=args.native_image,
        native_inverse_noise_path=(
            args.native_inverse_noise
            if args.native_inverse_noise is not None
            else args.native_directory / "Minvsigma2.bin"
        ),
        recovar_original_index=args.recovar_original_index,
        native_prefix=args.native_prefix,
        physical_image_size=args.physical_image_size,
        current_size=args.current_size,
        rotation_tolerance=args.rotation_tolerance,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
