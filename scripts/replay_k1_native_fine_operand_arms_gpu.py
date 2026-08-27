#!/usr/bin/env python3
"""Replay aligned native/RECOVAR K=1 fine operands through the exact CUDA tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
)
from scripts.analyze_em_k1_native_fine_operands import _flat_memmap, _full_to_compact
from scripts.analyze_k1_native_fine_operand_boundary import _complex_metric, _metric
from scripts.compare_relion_recovar_estep_dump import _nearest_rotation_rows_by_matrix

SCHEMA = "recovar.em.k1_native_fine_operand_arms_gpu.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dense_compact_rows(
    compact_rows: np.ndarray,
    native_full_to_compact: np.ndarray,
) -> np.ndarray:
    """Scatter compact rows into RELION's current-size packed-rFFT layout."""

    compact_rows = np.asarray(compact_rows)
    lookup = np.asarray(native_full_to_compact, dtype=np.int64)
    if compact_rows.ndim < 1:
        raise ValueError("compact_rows must have at least one dimension")
    if np.any(lookup >= compact_rows.shape[-1]):
        raise ValueError("native-to-compact lookup exceeds compact row width")
    dense = np.zeros(compact_rows.shape[:-1] + (lookup.size,), dtype=compact_rows.dtype)
    valid = lookup >= 0
    dense[..., valid] = compact_rows[..., lookup[valid]]
    return dense


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.subtract(values, np.max(values), dtype=np.float32)


def _native_packed_physical_indices(
    *,
    current_size: int,
    physical_image_size: int,
) -> np.ndarray:
    """Map RELION's packed current-size rows into centered physical indices."""

    if current_size <= 0 or current_size > physical_image_size:
        raise ValueError("current_size must fit inside physical_image_size")
    current_half_width = current_size // 2 + 1
    physical_half_width = physical_image_size // 2 + 1
    packed = np.arange(current_size * current_half_width, dtype=np.int64)
    x = packed % current_half_width
    row = packed // current_half_width
    y = np.where(row > current_size // 2, row - current_size, row)
    return ((y + physical_image_size // 2) * physical_half_width + x).astype(
        np.int32
    )


def replay(
    native_dir: Path,
    recovar_capture: Path,
    *,
    physical_image_size: int,
    alternate_reference_capture: Path | None = None,
) -> dict[str, object]:
    native_dir = Path(native_dir)
    recovar_capture = Path(recovar_capture)
    with np.load(recovar_capture, allow_pickle=False) as archive:
        rec = {name: np.asarray(archive[name]) for name in archive.files}

    native_raw = np.asarray(
        _flat_memmap(native_dir / "pass1_exp_Mweight_raw_preprior.bin"),
        dtype=np.float32,
    )
    native_corr = np.asarray(
        _flat_memmap(native_dir / "pass1_img0_corr_img.bin"),
        dtype=np.float32,
    )
    candidate_count = int(native_raw.size)
    native_pixel_count = int(native_corr.size)

    def native_complex(stem: str) -> np.ndarray:
        real = np.asarray(
            _flat_memmap(native_dir / f"pass1_class0_{stem}_real.bin"),
            dtype=np.float32,
        )
        imag = np.asarray(
            _flat_memmap(native_dir / f"pass1_class0_{stem}_imag.bin"),
            dtype=np.float32,
        )
        if real.size != candidate_count * native_pixel_count or imag.size != real.size:
            raise ValueError(f"native {stem} tensor has incompatible shape")
        # RECOVAR and RELION use opposite Fourier signs at this capture boundary.
        return np.negative(
            (real + np.complex64(1j) * imag).astype(np.complex64),
            dtype=np.complex64,
        ).reshape(candidate_count, native_pixel_count)

    native_reference = native_complex("fine_ref")
    native_shifted = native_complex("fine_shifted")

    native_eulers = _flat_memmap(
        native_dir / "pass1_class0_fine_eulers.bin"
    ).reshape(-1, 3, 3)
    nearest, rotation_distance, rotation_orientation = _nearest_rotation_rows_by_matrix(
        native_eulers,
        rec["rotations"],
    )
    native_rotation = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_rot_idx.bin", np.int32),
        dtype=np.int64,
    )
    native_translation = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_trans_idx.bin", np.int32),
        dtype=np.int64,
    )
    recovar_rotation = nearest[native_rotation]
    if native_rotation.shape != (candidate_count,) or native_translation.shape != (
        candidate_count,
    ):
        raise ValueError("native candidate-key arrays have incompatible shape")

    lookup = _full_to_compact(
        rec["window_indices"],
        full_size=physical_image_size,
        current_size=int(rec["current_size"]),
    )
    if lookup.size != native_pixel_count:
        raise ValueError("native and RECOVAR current-size layouts differ")
    recovar_reference = _dense_compact_rows(
        rec["raw_operand_proj_half"][recovar_rotation],
        lookup,
    ).astype(np.complex64)
    recovar_shifted = _dense_compact_rows(
        rec["raw_operand_shifted_corrected"][native_translation],
        lookup,
    ).astype(np.complex64)
    recovar_corr = _dense_compact_rows(
        rec["raw_operand_corr_img_score"],
        lookup,
    ).astype(np.float32)

    ppref_dims_path = native_dir / "pass1_class0_ppref_dims.bin"
    ppref_dims_count = int(np.fromfile(ppref_dims_path, dtype=np.int32, count=1)[0])
    ppref_dims = np.fromfile(
        ppref_dims_path,
        dtype=np.int32,
        count=ppref_dims_count,
        offset=4,
    )
    if ppref_dims.shape != (7,):
        raise ValueError("native PPref dimensions must contain seven values")
    ppref_shape = (int(ppref_dims[2]), int(ppref_dims[1]), int(ppref_dims[0]))
    ppref_real = np.asarray(
        _flat_memmap(native_dir / "pass1_class0_ppref_real.bin"),
        dtype=np.float32,
    )
    ppref_imag = np.asarray(
        _flat_memmap(native_dir / "pass1_class0_ppref_imag.bin"),
        dtype=np.float32,
    )
    if ppref_real.size != int(np.prod(ppref_shape)) or ppref_imag.size != ppref_real.size:
        raise ValueError("native PPref payload has incompatible shape")
    native_ppref = (ppref_real + np.complex64(1j) * ppref_imag).astype(
        np.complex64
    ).reshape(ppref_shape)
    padding_factor = int(
        round(
            float(
                np.fromfile(
                    native_dir / "pass1_class0_ppref_padding_factor.bin",
                    dtype=np.float64,
                    count=1,
                )[0]
            )
        )
    )
    current_size = int(rec["current_size"])
    physical_indices = _native_packed_physical_indices(
        current_size=current_size,
        physical_image_size=physical_image_size,
    )
    projected, _ = compute_relion_projector_projections_block(
        jnp.asarray(native_ppref),
        jnp.asarray(rec["rotations"][nearest], dtype=jnp.float32),
        (physical_image_size, physical_image_size),
        r_max=int(ppref_dims[6]),
        padding_factor=padding_factor,
        return_abs2=False,
        centered_rows=True,
        dense_scale=False,
        projector_output_size=current_size,
        pixel_indices=jnp.asarray(physical_indices),
        relion_texture_interp=True,
    )
    native_ppref_projection = np.asarray(
        jax.block_until_ready(projected),
        dtype=np.complex64,
    )
    if native_ppref_projection.shape != (native_eulers.shape[0], native_pixel_count):
        raise ValueError("native PPref projection has incompatible shape")
    native_ppref_reference = np.negative(
        native_ppref_projection[native_rotation],
        dtype=np.complex64,
    )

    alternate_reference = None
    if alternate_reference_capture is not None:
        with np.load(alternate_reference_capture, allow_pickle=False) as archive:
            alternate = {name: np.asarray(archive[name]) for name in archive.files}
        for name in ("rotations", "window_indices"):
            if not np.array_equal(rec[name], alternate[name]):
                raise ValueError(f"alternate reference has different {name}")
        alternate_reference = _dense_compact_rows(
            alternate["raw_operand_proj_half"][recovar_rotation],
            lookup,
        ).astype(np.complex64)

    full_to_compact = np.arange(native_pixel_count, dtype=np.int32)

    def device_reduce(
        reference: np.ndarray,
        shifted: np.ndarray,
        corr: np.ndarray,
    ) -> np.ndarray:
        values = cuda_backproject.relion_fine_diff2_pairs_f32(
            jnp.asarray(reference[None, :, :]),
            jnp.asarray(shifted[None, :, :]),
            jnp.asarray(corr[None, :]),
            jnp.asarray(full_to_compact),
        )
        return np.asarray(jax.block_until_ready(values), dtype=np.float32).reshape(-1)

    arms = {
        "native_all": (native_reference, native_shifted, native_corr),
        "recovar_all": (recovar_reference, recovar_shifted, recovar_corr),
        "native_reference_only": (
            native_reference,
            recovar_shifted,
            recovar_corr,
        ),
        "native_shifted_only": (
            recovar_reference,
            native_shifted,
            recovar_corr,
        ),
        "native_reference_and_shifted": (
            native_reference,
            native_shifted,
            recovar_corr,
        ),
        "native_corr_only": (
            recovar_reference,
            recovar_shifted,
            native_corr,
        ),
        "native_ppref_reference_only": (
            native_ppref_reference,
            recovar_shifted,
            recovar_corr,
        ),
        "native_ppref_reference_and_native_shifted": (
            native_ppref_reference,
            native_shifted,
            recovar_corr,
        ),
    }
    if alternate_reference is not None:
        arms["alternate_reference_only"] = (
            alternate_reference,
            recovar_shifted,
            recovar_corr,
        )
    reduced = {
        name: device_reduce(reference, shifted, corr)
        for name, (reference, shifted, corr) in arms.items()
    }
    inferred_highres = np.subtract(
        native_raw,
        reduced["native_all"],
        dtype=np.float32,
    )
    if not np.all(inferred_highres == inferred_highres[0]):
        raise ValueError("native operands do not replay with one exact highres addend")
    highres = np.float32(inferred_highres[0])
    reduced = {
        name: np.add(score, highres, dtype=np.float32)
        for name, score in reduced.items()
    }

    recovar_raw = np.asarray(rec["raw_operand_raw_diff2"], dtype=np.float32)[
        recovar_rotation,
        native_translation,
    ]
    current_half_width = current_size // 2 + 1
    row = np.arange(native_pixel_count, dtype=np.int64) // current_half_width
    even_nyquist = row == current_size // 2
    contributing = native_corr != 0.0
    ordinary_contributing = contributing & ~even_nyquist

    comparisons = {}
    for name, score in reduced.items():
        comparisons[name] = {
            "versus_native_raw_centered": _metric(
                _center(native_raw),
                _center(score),
            ),
            "versus_recovar_raw_centered": _metric(
                _center(recovar_raw),
                _center(score),
            ),
        }

    return {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact bytes and relative L2; no correlation",
        "candidate_count": candidate_count,
        "native_pixel_count": native_pixel_count,
        "current_size": current_size,
        "devices": [str(device) for device in jax.devices()],
        "cuda_library": os.environ.get("RECOVAR_CUDA_LIB"),
        "candidate_keys_exact": True,
        "rotation_matrix_orientation": rotation_orientation,
        "rotation_matrix_median_frobenius": float(np.median(rotation_distance)),
        "rotation_matrix_max_frobenius": float(np.max(rotation_distance)),
        "inferred_highres_xi2_half": float(highres),
        "native_exact_replay": _metric(native_raw, reduced["native_all"]),
        "recovar_production_replay": _metric(recovar_raw, reduced["recovar_all"]),
        "operands": {
            "corr_img": _metric(native_corr, recovar_corr),
            "reference_all_contributing": _complex_metric(
                native_reference[:, contributing],
                recovar_reference[:, contributing],
            ),
            "reference_ordinary_contributing": _complex_metric(
                native_reference[:, ordinary_contributing],
                recovar_reference[:, ordinary_contributing],
            ),
            "native_ppref_reference_all_contributing": _complex_metric(
                native_reference[:, contributing],
                native_ppref_reference[:, contributing],
            ),
            "shifted_all_contributing": _complex_metric(
                native_shifted[:, contributing],
                recovar_shifted[:, contributing],
            ),
            "shifted_ordinary_contributing": _complex_metric(
                native_shifted[:, ordinary_contributing],
                recovar_shifted[:, ordinary_contributing],
            ),
            "even_nyquist_contributing_pixel_count": int(
                np.count_nonzero(contributing & even_nyquist)
            ),
            "ordinary_contributing_pixel_count": int(
                np.count_nonzero(ordinary_contributing)
            ),
        },
        "native_ppref": {
            "shape": list(native_ppref.shape),
            "dims_xyz_origins_rmax": ppref_dims.tolist(),
            "padding_factor": padding_factor,
            "physical_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(physical_indices).tobytes()
            ).hexdigest(),
        },
        "counterfactual_scores": comparisons,
        "artifacts": {
            "native_dir": str(native_dir.resolve()),
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            "alternate_reference_capture": (
                None
                if alternate_reference_capture is None
                else str(Path(alternate_reference_capture).resolve())
            ),
            "alternate_reference_capture_sha256": (
                None
                if alternate_reference_capture is None
                else _sha256(Path(alternate_reference_capture))
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--recovar-capture", required=True, type=Path)
    parser.add_argument("--physical-image-size", required=True, type=int)
    parser.add_argument("--alternate-reference-capture", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = replay(
        args.native_dir,
        args.recovar_capture,
        physical_image_size=args.physical_image_size,
        alternate_reference_capture=args.alternate_reference_capture,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
