#!/usr/bin/env python3
"""Audit the live VDAM fine-score image/translation boundary against RELION."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar import cuda_backproject
from recovar.em.dense_single_volume.local_big_jit import _centered_rfft2_per_image
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_cuda_pixel_correction_from_rfloat_ctf,
    _relion_cuda_powerclass_highres_xi2_half,
    _relion_translation_angles_f32,
)
from scripts.analyze_em_k1_native_fine_operands import (
    _center,
    _flat_memmap,
    _stats,
)
from scripts.analyze_vdam_storewavg_boundary import _complex_2d, _real_2d
from scripts.compare_relion_recovar_estep_dump import (
    _nearest_rotation_rows_by_matrix,
)


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | list[int]]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError(f"metric shape mismatch: {reference.shape} != {candidate.shape}")
    residual = candidate.astype(np.complex128) - reference.astype(np.complex128)
    denominator = float(np.linalg.norm(reference.astype(np.complex128).reshape(-1)))
    return {
        "shape": list(reference.shape),
        "exact_count": int(np.count_nonzero(candidate == reference)),
        "value_count": int(reference.size),
        "relative_l2": float(np.linalg.norm(residual.reshape(-1)) / denominator),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _native_crop_rows(score_indices: np.ndarray, full_size: int, current_size: int) -> np.ndarray:
    score_indices = np.asarray(score_indices, dtype=np.int64)
    half_width = full_size // 2 + 1
    centered_row = score_indices // half_width
    x = score_indices - centered_row * half_width
    ky = centered_row - full_size // 2
    native_row = np.where(ky >= 0, ky, ky + current_size)
    crop = native_row * (current_size // 2 + 1) + x
    if np.unique(crop).size != crop.size:
        raise ValueError("score window does not map bijectively into the native crop")
    return crop.astype(np.int32)


def _current_crop_to_compact(crop: np.ndarray, current_size: int) -> np.ndarray:
    """Map RELION's full current-size FFTW grid to the radial compact rows."""

    crop = np.asarray(crop, dtype=np.int64)
    current_pixel_count = current_size * (current_size // 2 + 1)
    if np.any(crop < 0) or np.any(crop >= current_pixel_count):
        raise ValueError("native crop index lies outside the current-size Fourier grid")
    lookup = np.full(current_pixel_count, -1, dtype=np.int32)
    lookup[crop] = np.arange(crop.size, dtype=np.int32)
    return lookup


def analyze(
    native_dir: Path,
    live_score_path: Path,
    *,
    full_size: int,
    particle_stack: Path | None = None,
    particle_index: int | None = None,
    normalization_factor: float = 1.0,
    integer_pre_shift: tuple[int, int] = (0, 0),
    mask_radius: float | None = None,
    mask_cosine_width: float = 5.0,
) -> dict[str, object]:
    native_dir = Path(native_dir)
    with np.load(live_score_path, allow_pickle=False) as payload:
        live = {name: np.array(payload[name]) for name in payload.files}

    current_size = int(np.asarray(live["current_size"]).reshape(-1)[0])
    score_indices, _ = make_fourier_window_indices_np(
        (full_size, full_size),
        current_size,
        square=False,
        include_dc=False,
    )
    if score_indices.size != live["debug_ctf2_over_nv"].shape[-1]:
        raise ValueError("live score dump has a different radial window")
    crop = _native_crop_rows(score_indices, full_size, current_size)
    native_pixel_count = current_size * (current_size // 2 + 1)

    native_rotation = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_rot_idx.bin", np.int32),
        dtype=np.int32,
    )
    translation = np.asarray(
        _flat_memmap(native_dir / "pass1_acc_trans_idx.bin", np.int32),
        dtype=np.int32,
    )
    native_eulers = np.asarray(
        _flat_memmap(native_dir / "pass1_class0_fine_eulers.bin")
    ).reshape(-1, 3, 3)
    nearest, rotation_distance, orientation = _nearest_rotation_rows_by_matrix(
        native_eulers,
        live["local_rotation_matrices"],
    )
    rotation = nearest[native_rotation]
    native_raw = np.asarray(
        _flat_memmap(native_dir / "pass1_exp_Mweight_raw_preprior.bin"),
        dtype=np.float64,
    )
    live_raw = np.asarray(live["pass2_scores_raw"])[0, rotation, translation]

    scale = np.float32(full_size**2)
    native_weight_full = np.asarray(
        _flat_memmap(native_dir / "pass1_img0_corr_img.bin"),
        dtype=np.float32,
    ) / np.float32(full_size**4)
    native_weight = native_weight_full[crop]
    live_weight = np.asarray(live["debug_ctf2_over_nv"], dtype=np.float32)

    candidate_count = native_raw.size
    shape = (candidate_count, native_pixel_count)
    native_reference = -scale * (
        np.asarray(
            _flat_memmap(native_dir / "pass1_class0_fine_ref_real.bin"),
            dtype=np.float32,
        ).reshape(shape)
        + np.complex64(1j)
        * np.asarray(
            _flat_memmap(native_dir / "pass1_class0_fine_ref_imag.bin"),
            dtype=np.float32,
        ).reshape(shape)
    )
    native_shifted = -scale * (
        np.asarray(
            _flat_memmap(native_dir / "pass1_class0_fine_shifted_real.bin"),
            dtype=np.float32,
        ).reshape(shape)
        + np.complex64(1j)
        * np.asarray(
            _flat_memmap(native_dir / "pass1_class0_fine_shifted_imag.bin"),
            dtype=np.float32,
        ).reshape(shape)
    )
    native_reference = native_reference[:, crop]
    native_shifted = native_shifted[:, crop]
    live_reference = np.asarray(live["debug_proj_weighted"], dtype=np.complex64)[rotation]
    live_shifted_weighted = np.asarray(live["debug_shifted_score"], dtype=np.complex64)[translation]

    base_corrected = -scale * (
        np.asarray(
            _flat_memmap(native_dir / "pass1_img0_Fimg_corrected_real.bin"),
            dtype=np.float32,
        )[crop]
        + np.complex64(1j)
        * np.asarray(
            _flat_memmap(native_dir / "pass1_img0_Fimg_corrected_imag.bin"),
            dtype=np.float32,
        )[crop]
    )
    translation_angles = jnp.asarray(
        _relion_translation_angles_f32(live["translations"], (full_size, full_size)),
        dtype=jnp.float32,
    )
    translated_unweighted = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(base_corrected[None], dtype=jnp.complex64),
        translation_angles,
        jnp.asarray(score_indices, dtype=jnp.int32),
        (full_size, full_size),
    )
    translated_preweighted = cuda_backproject.relion_translate_score_f32(
        jnp.asarray((base_corrected * native_weight)[None], dtype=jnp.complex64),
        translation_angles,
        jnp.asarray(score_indices, dtype=jnp.int32),
        (full_size, full_size),
    )
    native_processed = (
        -scale
        * np.asarray(_complex_2d(native_dir / "Fimg_unweighted.bin"), dtype=np.complex64)
        .reshape(-1)[crop]
    ).astype(np.complex64)
    native_ctf = np.asarray(
        _real_2d(native_dir / "Fctf.bin"), dtype=np.float64
    ).astype(np.float32).reshape(-1)[crop]
    native_inverse_noise_engine = (
        np.asarray(_real_2d(native_dir / "Minvsigma2.bin"), dtype=np.float64).reshape(-1)[crop]
        / float(full_size**4)
    ).astype(np.float32)
    native_production_weighted = (
        native_processed * (native_ctf * native_inverse_noise_engine).astype(np.float32)
    ).astype(np.complex64)
    translated_native_production = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(native_production_weighted[None], dtype=jnp.complex64),
        translation_angles,
        jnp.asarray(score_indices, dtype=jnp.int32),
        (full_size, full_size),
    )
    translated_unweighted, translated_preweighted, translated_native_production = (
        np.asarray(value, dtype=np.complex64)
        for value in jax.block_until_ready(
            (translated_unweighted, translated_preweighted, translated_native_production)
        )
    )

    expected_weighted = (native_shifted * native_weight[None]).astype(np.complex64)
    comparisons = {
        "live_centered_raw_score_residual": _stats(_center(live_raw + native_raw)),
        "live_projected_reference": _metric(native_reference, live_reference),
        "live_score_weight": _metric(native_weight, live_weight),
        "live_weighted_shifted_image": _metric(expected_weighted, live_shifted_weighted),
        "native_base_then_translate": _metric(
            native_shifted,
            translated_unweighted[translation],
        ),
        "native_base_preweighted_then_translate": _metric(
            expected_weighted,
            translated_preweighted[translation],
        ),
        "native_translate_then_weight_vs_live": _metric(
            expected_weighted,
            live_shifted_weighted,
        ),
        "native_preweight_then_translate_vs_live": _metric(
            translated_preweighted[translation],
            live_shifted_weighted,
        ),
        "native_production_product_then_translate_vs_live": _metric(
            translated_native_production[translation],
            live_shifted_weighted,
        ),
        "native_corrected_weight_product_vs_native_production_product": _metric(
            (base_corrected * native_weight).astype(np.complex64),
            native_production_weighted,
        ),
    }
    if particle_stack is not None:
        if particle_index is None or mask_radius is None:
            raise ValueError("particle preprocessing replay requires particle_index and mask_radius")
        with mrcfile.open(particle_stack, permissive=False) as stack_file:
            raw_image = np.asarray(stack_file.data[int(particle_index)], dtype=np.float32)
        for mode, native_lane_reduction, native_atomic_reduction in (
            ("block_first", False, False),
            ("native_lane", True, False),
            ("native_atomic", False, True),
        ):
            _, masked_real = cuda_backproject.relion_preprocess_real_f32(
                jnp.asarray(raw_image[None], dtype=jnp.float32),
                jnp.asarray([normalization_factor], dtype=jnp.float32),
                jnp.asarray([integer_pre_shift], dtype=jnp.int32),
                float(mask_radius),
                float(mask_cosine_width),
                apply_mask=True,
                native_lane_reduction=native_lane_reduction,
                native_atomic_reduction=native_atomic_reduction,
            )
            replay_half = _centered_rfft2_per_image(masked_real)
            replay_half = np.asarray(jax.block_until_ready(replay_half), dtype=np.complex64)[0]
            comparisons[f"native_masked_fourier_vs_{mode}_preprocess_replay"] = _metric(
                -native_processed,
                replay_half[score_indices],
            )
            pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
                jnp.asarray([1.0], dtype=jnp.float32),
                jnp.asarray(
                    -np.asarray(_real_2d(native_dir / "Fctf.bin"), dtype=np.float64)
                    .reshape(-1)[crop][None],
                    dtype=jnp.float64,
                ),
            )
            corrected = (
                jnp.asarray(replay_half[score_indices][None], dtype=jnp.complex64)
                * pixel_correction
            )
            translated_corrected = cuda_backproject.relion_translate_score_f32(
                corrected,
                translation_angles,
                jnp.asarray(score_indices, dtype=jnp.int32),
                (full_size, full_size),
            )
            lookup = _relion_cuda_fine_full_to_compact_lookup(
                (full_size, full_size),
                current_size,
                score_indices,
            )
            highres_xi2_half = _relion_cuda_powerclass_highres_xi2_half(
                jnp.asarray(replay_half[None], dtype=jnp.complex64),
                image_shape=(full_size, full_size),
                current_size=current_size,
            )[0]
            lookup_variants = {
                "physical_grid": lookup,
                "current_grid": _current_crop_to_compact(crop, current_size),
                "radial_compact": np.arange(score_indices.size, dtype=np.int32),
            }
            for lookup_name, lookup_variant in lookup_variants.items():
                direct_diff2 = cuda_backproject.relion_fine_diff2_rectangular_f32(
                    jnp.asarray(live["debug_proj_weighted"][None], dtype=jnp.complex64),
                    translated_corrected[None],
                    jnp.asarray(live_weight[None], dtype=jnp.float32),
                    jnp.asarray(lookup_variant, dtype=jnp.int32),
                )[0]
                direct_lowres = np.asarray(
                    jax.block_until_ready(direct_diff2), dtype=np.float32
                )[rotation, translation]
                inferred_highres = np.subtract(
                    native_raw.astype(np.float32), direct_lowres, dtype=np.float32
                )
                inferred_bits, inferred_counts = np.unique(
                    inferred_highres.view(np.uint32), return_counts=True
                )
                inferred_mode_index = int(np.argmax(inferred_counts))
                inferred_mode = inferred_bits[inferred_mode_index].view(np.float32)
                direct_cost = np.add(
                    direct_lowres, np.float32(highres_xi2_half), dtype=np.float32
                )
                centered_direct_residual = _stats(_center(direct_cost - native_raw))
                centered_direct_residual.update(
                    {
                        "raw_exact_count": int(
                            np.count_nonzero(direct_cost == native_raw.astype(np.float32))
                        ),
                        "highres_xi2_half": float(highres_xi2_half),
                        "inferred_highres_mode": float(inferred_mode),
                        "inferred_highres_mode_count": int(
                            inferred_counts[inferred_mode_index]
                        ),
                        "inferred_highres_unique_count": int(inferred_bits.size),
                    }
                )
                comparisons[
                    f"native_raw_diff2_vs_{mode}_{lookup_name}_direct_replay"
                ] = centered_direct_residual

    return {
        "schema": "recovar.vdam_native_translation_boundary.v1",
        "status": "complete",
        "device": str(jax.devices()[0]),
        "identity": {
            "current_size": current_size,
            "candidate_count": candidate_count,
            "score_pixel_count": int(score_indices.size),
            "rotation_matrix_orientation": orientation,
            "rotation_matrix_max_frobenius": float(np.max(rotation_distance)),
        },
        "comparisons": comparisons,
        "artifacts": {
            "native_directory": str(native_dir.resolve()),
            "live_score_dump": str(Path(live_score_path).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--live-score", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--particle-stack", type=Path)
    parser.add_argument("--particle-index", type=int)
    parser.add_argument("--normalization-factor", type=float, default=1.0)
    parser.add_argument("--integer-pre-shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--mask-radius", type=float)
    parser.add_argument("--mask-cosine-width", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.live_score,
        full_size=args.full_image_size,
        particle_stack=args.particle_stack,
        particle_index=args.particle_index,
        normalization_factor=args.normalization_factor,
        integer_pre_shift=tuple(args.integer_pre_shift),
        mask_radius=args.mask_radius,
        mask_cosine_width=args.mask_cosine_width,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
