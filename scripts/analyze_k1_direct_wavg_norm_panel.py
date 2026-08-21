#!/usr/bin/env python3
"""Evaluate source-faithful direct Wavg norm totals on a native K=1 panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

if __package__:
    from scripts.compare_k1_relion_recovar_bpref_primitives import (
        _load_relion_ctf_inputs,
        _native_ctf_image,
        _standard_half_indices,
    )
    from scripts.validate_relion_preprocess_capture import load_artifact
else:
    from compare_k1_relion_recovar_bpref_primitives import (  # type: ignore[no-redef]
        _load_relion_ctf_inputs,
        _native_ctf_image,
        _standard_half_indices,
    )
    from validate_relion_preprocess_capture import load_artifact  # type: ignore[no-redef]


def _direct_wavg_pixel_sum(
    projection: np.ndarray,
    translated_image: np.ndarray,
    ctf: np.ndarray,
) -> float:
    """Mirror the deployed CUDA Wavg float32 operation order, then host-sum."""

    proj = np.asarray(projection, dtype=np.complex64)
    image = np.asarray(translated_image, dtype=np.complex64)
    ctf_f32 = np.asarray(ctf, dtype=np.float32)
    ref_real = np.asarray(proj.real * ctf_f32, dtype=np.float32)
    ref_imag = np.asarray(proj.imag * ctf_f32, dtype=np.float32)
    diff_real = np.asarray(ref_real - image.real, dtype=np.float32)
    diff_imag = np.asarray(ref_imag - image.imag, dtype=np.float32)
    pixel_diff2 = np.asarray(
        np.asarray(diff_real * diff_real, dtype=np.float32)
        + np.asarray(diff_imag * diff_imag, dtype=np.float32),
        dtype=np.float32,
    )
    return float(np.sum(pixel_diff2, dtype=np.float64))


def _direct_wavg_posterior_pixel_sum(
    projections: np.ndarray,
    translated_images: np.ndarray,
    ctf: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Replay CUDA Wavg's translation then orientation accumulation in float32."""

    proj = np.asarray(projections, dtype=np.complex64)
    images = np.asarray(translated_images, dtype=np.complex64)
    ctf_f32 = np.asarray(ctf, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape != (proj.shape[0], images.shape[0]):
        raise ValueError(
            "probabilities must have shape (n_rotations, n_translations), got "
            f"{probs.shape} for {proj.shape[0]} rotations and {images.shape[0]} translations"
        )
    pixel_totals = np.zeros(proj.shape[1], dtype=np.float32)
    for rotation_row in range(proj.shape[0]):
        ref_real = np.asarray(proj[rotation_row].real * ctf_f32, dtype=np.float32)
        ref_imag = np.asarray(proj[rotation_row].imag * ctf_f32, dtype=np.float32)
        rotation_totals = np.zeros(proj.shape[1], dtype=np.float32)
        for translation_row in range(images.shape[0]):
            weight = probs[rotation_row, translation_row]
            if weight == 0:
                continue
            diff_real = np.asarray(ref_real - images[translation_row].real, dtype=np.float32)
            diff_imag = np.asarray(ref_imag - images[translation_row].imag, dtype=np.float32)
            diff2 = np.asarray(
                np.asarray(diff_real * diff_real, dtype=np.float32)
                + np.asarray(diff_imag * diff_imag, dtype=np.float32),
                dtype=np.float32,
            )
            rotation_totals = np.asarray(
                rotation_totals + np.asarray(weight * diff2, dtype=np.float32),
                dtype=np.float32,
            )
        pixel_totals = np.asarray(pixel_totals + rotation_totals, dtype=np.float32)
    return float(np.sum(pixel_totals, dtype=np.float64))


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": float(np.median(array)),
        "spread": float(np.ptp(array)),
        "relative_spread_over_median": float(np.ptp(array) / np.median(array)),
        "std": float(np.std(array)),
    }


def _inferred_average_norm(native_factor: float, particle_norm_squared: float) -> float:
    """Invert RELION's image scale ``average_norm / particle_norm``."""

    return float(native_factor * np.sqrt(2.0 * particle_norm_squared))


def _predicted_factor(average_norm: float, particle_norm_squared: float) -> np.float32:
    return np.float32(average_norm / np.sqrt(2.0 * np.float64(particle_norm_squared)))


def _factor_rounding_interval(
    native_factor: float,
    particle_norm_squared: float,
) -> tuple[float, float]:
    factor = np.float32(native_factor)
    previous = np.nextafter(factor, np.float32(-np.inf), dtype=np.float32)
    following = np.nextafter(factor, np.float32(np.inf), dtype=np.float32)
    particle_norm = np.sqrt(2.0 * np.float64(particle_norm_squared))
    return (
        float((np.float64(previous) + np.float64(factor)) * 0.5 * particle_norm),
        float((np.float64(factor) + np.float64(following)) * 0.5 * particle_norm),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass2-dir", type=Path, required=True)
    parser.add_argument("--native-capture-dir", type=Path, required=True)
    parser.add_argument("--source-star", type=Path, required=True)
    parser.add_argument("--relion-bind-dir", type=Path, required=True)
    parser.add_argument("--recovar-iteration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["RECOVAR_RELION_BIND_BUILD_DIR"] = str(args.relion_bind_dir.resolve())
    particles, optics_by_id, relion_bind = _load_relion_ctf_inputs(
        args.source_star,
        args.relion_bind_dir,
    )
    native_by_source = {
        int(capture.stack_index) - 1: capture
        for capture in (
            load_artifact(path)
            for path in sorted(args.native_capture_dir.glob("*.preprocess-v1.bin"))
        )
    }
    with np.load(args.recovar_iteration, allow_pickle=False) as iteration:
        recovar_average_norm = {
            half: float(iteration[f"half{half}_avg_norm_correction"])
            for half in (1, 2)
        }
    records: list[dict[str, object]] = []
    for norm_path in sorted(args.pass2_dir.glob("norm_residual_orig*_cs056.npz")):
        with np.load(norm_path, allow_pickle=False) as norm:
            source_index = int(norm["original_index"])
            half = int(norm["half"])
            raw_translated = np.asarray(norm["raw_translated_recon"], dtype=np.complex64)
            raw_translated_wavg = np.asarray(norm["raw_translated_wavg"], dtype=np.complex64)
            captured_wavg_indices = np.asarray(norm["wavg_window_indices"], dtype=np.int32)
            ctf_probs = np.asarray(norm["ctf_probs"])
            noise_variance = np.asarray(norm["noise_variance_for_noise"], dtype=np.float64)
            shell_indices_half = np.asarray(norm["shell_indices_half"], dtype=np.int32)
            high_shell = float(norm["relion_norm_high_shell"])
            current_total = float(norm["weighted_img_per_image"] + norm["block_norm_residual"])
            weighted_image = float(norm["weighted_img_per_image"])
            block_residual = float(norm["block_norm_residual"])
        pass2_path = args.pass2_dir / f"pass2_orig{source_index:06d}_cs056.npz"
        with np.load(pass2_path, allow_pickle=False) as pass2:
            probs = np.asarray(pass2["probs"], dtype=np.float32)
            reconstruction_probs = np.asarray(pass2["reconstruction_probs"], dtype=np.float32)
            winner = int(np.argmax(probs.reshape(-1)))
            rotation_row, translation_row = np.unravel_index(winner, probs.shape)
            window_indices = np.asarray(pass2["window_indices"], dtype=np.int32)
            recon_indices = np.asarray(pass2["recon_window_indices"], dtype=np.int32)
            projection_rows = np.asarray(pass2["proj_half"], dtype=np.complex64)
            current_size = int(pass2["current_size"])
        if not np.array_equal(captured_wavg_indices, window_indices):
            raise ValueError(
                f"Wavg translated-image indices differ from pass-2 score indices for {source_index}"
            )
        native = native_by_source[source_index]
        native_factor = float(np.float32(native.norm_correction))
        particle = particles.iloc[source_index]
        optics_group = int(
            particle["rlnOpticsGroup"]
            if "rlnOpticsGroup" in particle
            else particle["_rlnOpticsGroup"]
        )
        native_ctf = _native_ctf_image(
            particle_row=particle,
            optics_row=optics_by_id[optics_group],
            relion_bind=relion_bind,
            image_size=128,
        )
        native_ctf_flat = native_ctf.reshape(-1)
        recon_standard_indices = _standard_half_indices(recon_indices, 128)
        ctf_recon = native_ctf_flat[recon_standard_indices].astype(np.float32)
        standard_indices = _standard_half_indices(window_indices, 128)
        valid_wavg = shell_indices_half[window_indices] <= current_size // 2
        ctf = native_ctf_flat[standard_indices].astype(np.float32)[valid_wavg]
        ctf_magnitude_from_recovar = np.sqrt(
            np.asarray(ctf_probs[rotation_row], dtype=np.float64) * noise_variance
        )
        ctf_magnitude_max_abs = float(
            np.max(np.abs(np.abs(ctf_recon.astype(np.float64)) - ctf_magnitude_from_recovar))
        )
        direct_plus_low = _direct_wavg_pixel_sum(
            projection_rows[rotation_row, valid_wavg],
            raw_translated_wavg[translation_row, valid_wavg],
            ctf,
        )
        direct_minus_low = _direct_wavg_pixel_sum(
            projection_rows[rotation_row, valid_wavg],
            raw_translated_wavg[translation_row, valid_wavg],
            -ctf,
        )
        posterior_native_low = _direct_wavg_posterior_pixel_sum(
            projection_rows[: reconstruction_probs.shape[0], valid_wavg],
            raw_translated_wavg[:, valid_wavg],
            ctf,
            reconstruction_probs,
        )
        posterior_flipped_low = _direct_wavg_posterior_pixel_sum(
            projection_rows[: reconstruction_probs.shape[0], valid_wavg],
            raw_translated_wavg[:, valid_wavg],
            -ctf,
            reconstruction_probs,
        )
        totals = {
            "current_separated": current_total,
            "hard_winner_native_ctf_sign": direct_plus_low + high_shell,
            "hard_winner_flipped_ctf_sign": direct_minus_low + high_shell,
            "posterior_native_ctf_sign": posterior_native_low + high_shell,
            "posterior_flipped_ctf_sign": posterior_flipped_low + high_shell,
        }
        inferred_average = {
            name: _inferred_average_norm(native_factor, total)
            for name, total in totals.items()
        }
        records.append(
            {
                "source_index": source_index,
                "half": half,
                "native_factor": native_factor,
                "winner_rotation_row": int(rotation_row),
                "winner_translation_row": int(translation_row),
                "wavg_valid_pixel_count": int(np.count_nonzero(valid_wavg)),
                "bpref_reconstruction_pixel_count": int(raw_translated.shape[-1]),
                "ctf_magnitude_max_abs": ctf_magnitude_max_abs,
                "weighted_image_power": weighted_image,
                "block_norm_residual": block_residual,
                "high_shell_power": high_shell,
                "totals": totals,
                "inferred_average_norm": inferred_average,
                "norm_capture": str(norm_path.resolve()),
                "pass2_capture": str(pass2_path.resolve()),
                "native_capture": str(native.path.resolve()),
            }
        )

    if set(native_by_source) != {int(record["source_index"]) for record in records}:
        raise ValueError("direct Wavg panel does not match the native preprocess panel identities")
    methods = tuple(records[0]["totals"])
    for record in records:
        half = int(record["half"])
        native_factor = np.float32(record["native_factor"])
        record["predicted_factor"] = {
            method: {
                "float32": float(
                    _predicted_factor(
                        recovar_average_norm[half],
                        float(record["totals"][method]),
                    )
                ),
                "bit_exact_native": bool(
                    _predicted_factor(
                        recovar_average_norm[half],
                        float(record["totals"][method]),
                    ).view(np.uint32)
                    == native_factor.view(np.uint32)
                ),
            }
            for method in methods
        }
    summaries = {
        f"half{half}": {
            method: {
                **_summary(
                    [
                        float(record["inferred_average_norm"][method])
                        for record in records
                        if int(record["half"]) == half
                    ]
                ),
                "recovar_average_norm": recovar_average_norm[half],
                "median_relative_error_vs_recovar_average": float(
                    abs(
                        np.median(
                            [
                                float(record["inferred_average_norm"][method])
                                for record in records
                                if int(record["half"]) == half
                            ]
                        )
                        - recovar_average_norm[half]
                    )
                    / recovar_average_norm[half]
                ),
                "native_factor_bit_exact_count": sum(
                    bool(record["predicted_factor"][method]["bit_exact_native"])
                    for record in records
                    if int(record["half"]) == half
                ),
                "native_factor_count": sum(
                    int(record["half"]) == half for record in records
                ),
                "factor_rounding_interval_signed_margin": float(
                    min(
                        _factor_rounding_interval(
                            float(record["native_factor"]),
                            float(record["totals"][method]),
                        )[1]
                        for record in records
                        if int(record["half"]) == half
                    )
                    - max(
                        _factor_rounding_interval(
                            float(record["native_factor"]),
                            float(record["totals"][method]),
                        )[0]
                        for record in records
                        if int(record["half"]) == half
                    )
                ),
            }
            for method in methods
        }
        for half in sorted({int(record["half"]) for record in records})
    }
    report = {
        "schema": "recovar.em.k1_direct_wavg_norm_panel.v1",
        "status": "complete",
        "interpretation": (
            "A source-faithful per-particle total should infer one common average norm "
            "within each half when multiplied by the native normalization factors."
        ),
        "particle_count": len(records),
        "half_summaries": summaries,
        "particles": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
