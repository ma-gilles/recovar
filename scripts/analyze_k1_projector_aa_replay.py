#!/usr/bin/env python3
"""Replay scale-AA shells with RECOVAR's projector and native/RECOVAR PPref inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_indices_np
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
)
from scripts.analyze_k1_scale_aa_boundary import _native_aa_shells


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
    count = struct.unpack_from("<i", payload)[0]
    values = np.frombuffer(payload, dtype=dtype, offset=4).copy()
    _require(values.size == count, f"flat-array size mismatch: {path}")
    return values


def _load_native_ppref(native_directory: Path, prefix: str) -> tuple[np.ndarray, dict[str, object]]:
    root = native_directory / prefix
    dims = _flat(Path(f"{root}dims.bin"), np.dtype("<i4"))
    _require(dims.size == 7, "native PPref dimensions changed")
    xdim, ydim, zdim, xinit, yinit, zinit, r_max = (int(value) for value in dims)
    real = _flat(Path(f"{root}real.bin"), np.dtype("<f8"))
    imag = _flat(Path(f"{root}imag.bin"), np.dtype("<f8"))
    ppref = (real + 1j * imag).astype(np.complex64).reshape(zdim, ydim, xdim)
    return ppref, {
        "shape_zyx": list(ppref.shape),
        "origin_xyz": [xinit, yinit, zinit],
        "r_max": r_max,
    }


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    residual = candidate - reference
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "median_abs": float(np.median(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _replay_shells(
    ppref: np.ndarray,
    rotations: np.ndarray,
    rotation_mass: np.ndarray,
    pixel_indices: np.ndarray,
    ctf2: np.ndarray,
    pixel_shells: np.ndarray,
    shell_ids: np.ndarray,
    *,
    image_size: int,
    current_size: int,
    chunk_size: int,
) -> np.ndarray:
    shell_total = np.zeros(shell_ids.size, dtype=np.float64)
    active_rows = np.flatnonzero(rotation_mass > 0.0)
    for start in range(0, active_rows.size, chunk_size):
        rows = active_rows[start : start + chunk_size]
        _, abs2 = compute_relion_projector_projections_block(
            jnp.asarray(ppref),
            jnp.asarray(rotations[rows], dtype=jnp.float32),
            (image_size, image_size),
            r_max=current_size // 2,
            padding_factor=2,
            return_abs2=True,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=current_size,
            pixel_indices=jnp.asarray(pixel_indices, dtype=jnp.int32),
            relion_texture_interp=True,
        )
        abs2_np = np.asarray(jax.block_until_ready(abs2), dtype=np.float32)
        weighted = (
            abs2_np.astype(np.float64)
            * ctf2[None, :]
            * rotation_mass[rows, None]
        )
        for shell_row, shell in enumerate(shell_ids.tolist()):
            shell_total[shell_row] += np.sum(
                weighted[:, pixel_shells == shell],
                dtype=np.float64,
            )
    return shell_total


def analyze(
    recovar_candidates: Path,
    recovar_projector: Path,
    native_directory: Path,
    native_components: Path,
    *,
    native_prefix: str,
    image_size: int,
    recovar_term_divisor: float,
    chunk_size: int,
) -> dict[str, object]:
    _require(jax.default_backend() == "gpu", "projector AA replay requires a GPU")
    with np.load(recovar_candidates, allow_pickle=False) as payload:
        rotations = np.asarray(payload["candidate_rotation_matrices"], dtype=np.float32)
        probabilities = np.asarray(payload["candidate_posterior_probs"], dtype=np.float32)
        rotation_mass = np.sum(probabilities, axis=1, dtype=np.float64)
        scale_mask = np.asarray(payload["scale_correction_pixel_mask"], dtype=bool)
        shell_indices = np.asarray(payload["scale_shell_indices"], dtype=np.int32)
        ctf_raw_sum = np.asarray(payload["ctf_probs_raw_sum_per_pixel"], dtype=np.float64)
        captured_shells = np.asarray(payload["scale_aa_per_shell"], dtype=np.float64)
        current_size = int(payload["current_size"])
        iteration = int(payload["iteration"])
        half = int(payload["half"])
        part_id = int(payload["group_id"])
    retained_mass = float(np.sum(rotation_mass, dtype=np.float64))
    active_pixel_rows = np.flatnonzero(scale_mask)
    ctf2 = ctf_raw_sum[active_pixel_rows] / retained_mass
    pixel_shells = shell_indices[active_pixel_rows]
    shell_ids = np.unique(pixel_shells)
    window_indices, _ = make_fourier_window_indices_np(
        (image_size, image_size),
        current_size,
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    pixel_indices = window_indices[active_pixel_rows]

    with np.load(recovar_projector, allow_pickle=False) as payload:
        recovar_ppref = np.asarray(payload["projector_half"], dtype=np.complex64)
        if recovar_ppref.ndim == 4:
            recovar_ppref = recovar_ppref[0]
        _require(int(payload["projector_r_max"]) == current_size // 2, "RECOVAR PPref r_max changed")
    native_ppref, native_metadata = _load_native_ppref(native_directory, native_prefix)
    _require(native_ppref.shape == recovar_ppref.shape, "native and RECOVAR PPref topology differs")

    recovar_replay = _replay_shells(
        recovar_ppref,
        rotations,
        rotation_mass,
        pixel_indices,
        ctf2,
        pixel_shells,
        shell_ids,
        image_size=image_size,
        current_size=current_size,
        chunk_size=chunk_size,
    )
    native_ppref_replay = _replay_shells(
        native_ppref,
        rotations,
        rotation_mass,
        pixel_indices,
        ctf2,
        pixel_shells,
        shell_ids,
        image_size=image_size,
        current_size=current_size,
        chunk_size=chunk_size,
    )
    native_shells = _native_aa_shells(
        native_components,
        iteration=iteration,
        half=half,
        part_id=part_id,
    )[shell_ids]
    captured_native_units = captured_shells[shell_ids] / recovar_term_divisor
    recovar_replay_native_units = recovar_replay / recovar_term_divisor
    native_ppref_replay_native_units = native_ppref_replay / recovar_term_divisor
    baseline_norm = float(np.linalg.norm(captured_native_units - native_shells))
    native_ppref_norm = float(np.linalg.norm(native_ppref_replay_native_units - native_shells))
    closure = 1.0 - native_ppref_norm / baseline_norm

    return {
        "schema": "recovar.em.k1_projector_aa_replay.v1",
        "identity": {
            "iteration": iteration,
            "half": half,
            "part_id": part_id,
            "current_size": current_size,
            "active_rotation_count": int(np.count_nonzero(rotation_mass)),
            "active_pixel_count": int(active_pixel_rows.size),
            "shell_ids": shell_ids.tolist(),
            "chunk_size": chunk_size,
            "native_ppref": native_metadata,
        },
        "replay": {
            "recovar_ppref_vs_captured_recovar": _metric(
                recovar_replay_native_units,
                captured_native_units,
            ),
            "captured_recovar_vs_native": _metric(captured_native_units, native_shells),
            "native_ppref_recovar_projector_vs_native": _metric(
                native_ppref_replay_native_units,
                native_shells,
            ),
            "native_ppref_residual_closure_fraction": closure,
        },
        "artifacts": {
            "recovar_candidates": str(recovar_candidates.resolve()),
            "recovar_candidates_sha256": _sha256(recovar_candidates),
            "recovar_projector": str(recovar_projector.resolve()),
            "recovar_projector_sha256": _sha256(recovar_projector),
            "native_components": str(native_components.resolve()),
            "native_components_sha256": _sha256(native_components),
        },
        "classification": (
            "incoming PPref is causally sufficient for the AA residual"
            if closure >= 0.8
            else "incoming PPref is not the dominant AA residual; projector interpolation or AA arithmetic differs"
            if closure <= 0.2
            else "incoming PPref partially contributes to the AA residual"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-candidates", type=Path, required=True)
    parser.add_argument("--recovar-projector", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-components", type=Path, required=True)
    parser.add_argument("--native-prefix", default="img0_part109_storeWavg_wavg_ppref_")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--recovar-term-divisor", type=float, default=float(128**4))
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.recovar_candidates,
        args.recovar_projector,
        args.native_directory,
        args.native_components,
        native_prefix=args.native_prefix,
        image_size=args.image_size,
        recovar_term_divisor=args.recovar_term_divisor,
        chunk_size=args.chunk_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
