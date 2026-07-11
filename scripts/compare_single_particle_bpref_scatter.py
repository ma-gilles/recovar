#!/usr/bin/env python
"""Compare one dumped RECOVAR pass-2 M-step contribution in RELION C++ and RECOVAR CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pass2-dump", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--ori-size", type=int, default=128)
    parser.add_argument("--padding-factor", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-30)


def _complex_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs).reshape(-1).astype(np.complex128, copy=False)
    b = np.asarray(rhs).reshape(-1).astype(np.complex128, copy=False)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.real(np.vdot(a, b)) / denom)


def _real_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=np.float64).reshape(-1)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _summary(actual: np.ndarray, expected: np.ndarray, *, complex_values: bool) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    rel = _relative_error(actual, expected)
    nz = np.abs(expected) > 1e-20
    rel_nz = rel[nz]
    return {
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "norm_relative_error": float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "expected_norm": float(np.linalg.norm(expected)),
        "actual_norm": float(np.linalg.norm(actual)),
        "corr": _complex_corr(actual, expected) if complex_values else _real_corr(actual, expected),
        "nonzero_expected": int(np.count_nonzero(nz)),
        "median_relative_error_nonzero": float(np.median(rel_nz)) if rel_nz.size else float("nan"),
        "p95_relative_error_nonzero": float(np.percentile(rel_nz, 95)) if rel_nz.size else float("nan"),
        "p99_relative_error_nonzero": float(np.percentile(rel_nz, 99)) if rel_nz.size else float("nan"),
        "max_relative_error_nonzero": float(np.max(rel_nz)) if rel_nz.size else float("nan"),
        "sum_actual": [float(np.sum(actual).real), float(np.sum(actual).imag)]
        if complex_values
        else float(np.sum(actual)),
        "sum_expected": [float(np.sum(expected).real), float(np.sum(expected).imag)]
        if complex_values
        else float(np.sum(expected)),
    }


def _active_mstep_rows(dump: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(
        dump["reconstruction_probs"] if "reconstruction_probs" in dump else dump["probs"],
        dtype=np.float64,
    )
    shifted = np.asarray(dump["shifted_recon"])
    ctf2 = np.asarray(dump["ctf2_over_nv_recon"], dtype=np.float64)
    summed = probs @ shifted
    ctf_probs = probs.sum(axis=1)[:, None] * ctf2[None, :]
    active = (np.sum(np.abs(summed), axis=1) > 0.0) | (np.sum(np.abs(ctf_probs), axis=1) > 0.0)
    rotations = np.asarray(dump["rotations"], dtype=np.float64)[active]
    return summed[active], ctf_probs[active], rotations


def _dense_half_images(
    rows: np.ndarray,
    fftw_indices: np.ndarray,
    *,
    ori_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    out = np.zeros((rows.shape[0], ori_size, ori_size // 2 + 1), dtype=dtype)
    out.reshape(rows.shape[0], -1)[:, fftw_indices] = rows.astype(dtype, copy=False)
    return out


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.helpers.fourier_window import centered_half_indices_to_fftw_half_indices
    from recovar.em.dense_single_volume.helpers.half_volume_mstep import relion_backprojector_volume_shape
    from recovar.relion_bind._relion_bind_core import TRILINEAR, get_backprojector_data

    with np.load(args.pass2_dump, allow_pickle=True) as dump:
        current_size = int(np.asarray(dump["current_size"]).item())
        centered_indices = np.asarray(dump["recon_window_indices"], dtype=np.int32)
        summed, ctf_probs, rotations = _active_mstep_rows(dump)

    fftw_indices = np.asarray(
        centered_half_indices_to_fftw_half_indices((args.ori_size, args.ori_size), centered_indices),
        dtype=np.int32,
    )
    real_dtype = np.float64 if args.dtype == "float64" else np.float32
    complex_dtype = np.complex128 if args.dtype == "float64" else np.complex64
    summed = summed.astype(complex_dtype, copy=False)
    ctf_probs = ctf_probs.astype(real_dtype, copy=False)

    relion_images = _dense_half_images(summed, fftw_indices, ori_size=args.ori_size, dtype=complex_dtype)
    relion_weights = _dense_half_images(ctf_probs, fftw_indices, ori_size=args.ori_size, dtype=real_dtype)
    relion_data, relion_weight = get_backprojector_data(
        relion_images,
        rotations,
        relion_weights,
        ori_size=args.ori_size,
        padding_factor=args.padding_factor,
        interpolator=TRILINEAR,
        current_size=current_size,
    )

    accum_shape = relion_backprojector_volume_shape(
        (args.ori_size, args.ori_size, args.ori_size),
        args.padding_factor,
        current_size=current_size,
    )
    half_shape = ftu.volume_shape_to_half_volume_shape(accum_shape)
    data0 = jnp.zeros((int(np.prod(half_shape)),), dtype=jnp.asarray(summed).dtype)
    weight0 = jnp.zeros((int(np.prod(half_shape)),), dtype=jnp.asarray(ctf_probs).dtype)
    recovar_data = cuda_backproject.backproject_indexed(
        data0,
        jnp.asarray(summed),
        jnp.asarray(fftw_indices, dtype=jnp.int32),
        jnp.asarray(rotations),
        image_shape=(args.ori_size, args.ori_size),
        volume_shape=accum_shape,
        order=1,
        half_volume=True,
        half_image=True,
        max_r=float(current_size // 2),
        relion_x_half=True,
    )
    recovar_weight = cuda_backproject.backproject_indexed(
        weight0,
        jnp.asarray(ctf_probs),
        jnp.asarray(fftw_indices, dtype=jnp.int32),
        jnp.asarray(rotations),
        image_shape=(args.ori_size, args.ori_size),
        volume_shape=accum_shape,
        order=1,
        half_volume=True,
        half_image=True,
        max_r=float(current_size // 2),
        relion_x_half=True,
    )

    recovar_data_np = np.asarray(jax.device_get(recovar_data)).reshape(half_shape)
    recovar_weight_np = np.asarray(jax.device_get(recovar_weight)).reshape(half_shape)
    relion_data_np = np.asarray(relion_data).reshape(half_shape)
    relion_weight_np = np.asarray(relion_weight).reshape(half_shape)

    result = {
        "pass2_dump": str(args.pass2_dump),
        "ori_size": int(args.ori_size),
        "padding_factor": int(args.padding_factor),
        "current_size": int(current_size),
        "accumulator_shape": list(accum_shape),
        "half_shape": list(half_shape),
        "n_active_rows": int(summed.shape[0]),
        "n_window_pixels": int(centered_indices.shape[0]),
        "devices": [str(d) for d in jax.devices()],
        "data": _summary(recovar_data_np, relion_data_np, complex_values=True),
        "weight": _summary(recovar_weight_np, relion_weight_np, complex_values=False),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
