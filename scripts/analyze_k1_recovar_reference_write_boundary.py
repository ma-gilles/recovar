#!/usr/bin/env python3
"""Replay one K=1 RECOVAR reference from its accumulator through map output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils, mask, padding
from recovar.em.dense_single_volume.mean_helpers import (
    _apply_relion_initial_lowpass_filter,
)
from recovar.reconstruction import regularization, relion_functions
from recovar.utils import helpers
from scripts.analyze_k1_reconstruction_stage_boundary import (
    _load,
    _metrics,
    _stage_path,
)
from scripts.summarize_em_completion_bench import normalized_fsc_auc, shell_fsc


def _fsc_auc(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        normalized_fsc_auc(
            np.asarray(shell_fsc(np.asarray(actual), np.asarray(expected)), dtype=np.float64)
        )
    )


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    return {
        "array": _metrics(actual, expected),
        "signed_fsc_auc_non_dc": _fsc_auc(actual, expected),
    }


def _choose_sign(actual: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, int]:
    positive = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))
    negative = float(np.linalg.norm(-np.asarray(actual) - np.asarray(expected)))
    if negative < positive:
        return -np.asarray(actual), -1
    return np.asarray(actual), 1


def _firstiter_postprocess(
    volume: np.ndarray,
    *,
    volume_shape: tuple[int, int, int],
    voxel_size: float,
    ini_high_angstrom: float,
    fourier_mask_edge: float,
    solvent_mask: np.ndarray,
) -> np.ndarray:
    volume_ft = fourier_transform_utils.get_dft3(jnp.asarray(volume)).reshape(-1)
    lowpass_ft = _apply_relion_initial_lowpass_filter(
        volume_ft,
        volume_shape,
        voxel_size,
        ini_high_angstrom,
        filter_edgewidth=fourier_mask_edge,
    )
    lowpass_real = np.asarray(
        fourier_transform_utils.get_idft3(jnp.asarray(lowpass_ft).reshape(volume_shape))
    ).real
    return lowpass_real * solvent_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-accumulator", type=Path, required=True)
    parser.add_argument("--recovar-fsc", type=Path, required=True)
    parser.add_argument("--recovar-map-dir", type=Path, required=True)
    parser.add_argument("--native-stage-dir", type=Path, required=True)
    parser.add_argument("--native-relion-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--ini-high-angstrom", type=float, required=True)
    parser.add_argument("--fourier-mask-edge", type=float, default=2.0)
    parser.add_argument("--particle-diameter-angstrom", type=float, required=True)
    parser.add_argument("--solvent-mask-edge-pixels", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.recovar_accumulator, allow_pickle=False) as archive:
        grid_size = int(archive["grid_size"])
        current_size = int(archive["current_size"])
        padding_factor = int(archive["padding_factor"])
        volume_shape = tuple(int(value) for value in archive["volume_shape"])
        accumulator_shape = tuple(
            int(value) for value in archive["mstep_accumulator_shape"]
        )
        numerators = [np.asarray(archive[f"Ft_y_{index}"]) for index in (0, 1)]
        weights = [np.asarray(archive[f"Ft_ctf_{index}"]) for index in (0, 1)]
    if volume_shape != (grid_size,) * 3:
        raise ValueError("volume shape and grid size disagree")
    if accumulator_shape != (accumulator_shape[0],) * 3:
        raise ValueError("accumulator is not cubic")

    fsc = jnp.asarray(np.load(args.recovar_fsc, allow_pickle=False))
    wiener_radius = padding_factor * (current_size // 2)
    valid = relion_functions._relion_current_size_decenter_mask(
        accumulator_shape,
        wiener_radius,
        half_volume=False,
    ).reshape(-1)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        volume_shape, padding_factor
    )
    flatten_radius = args.particle_diameter_angstrom / (2.0 * args.voxel_size)
    solvent_mask = np.asarray(
        mask.raised_cosine_mask(
            volume_shape,
            radius=flatten_radius,
            radius_p=flatten_radius + args.solvent_mask_edge_pixels,
            offset=jnp.zeros(3),
        ),
        dtype=np.float64,
    )

    halves: dict[str, object] = {}
    for half in (1, 2):
        weight = jnp.asarray(weights[half - 1])
        numerator = jnp.asarray(numerators[half - 1])
        live_tau, _, live_tau_details = regularization.compute_relion_tau2_from_weights(
            weight,
            weight,
            fsc,
            volume_shape,
            tau2_fudge=1.0,
            padding_factor=padding_factor,
            r_max=current_size // 2,
            return_details=True,
            full_half_axis=0,
            accumulator_volume_shape=accumulator_shape,
        )
        denominator = relion_functions.adjust_regularization_relion_style(
            weight.real,
            accumulator_shape,
            tau=jnp.asarray(live_tau, dtype=jnp.float64),
            padding_factor=padding_factor,
            max_res_shell=current_size // 2,
            half_volume=False,
            tau2_fudge=1.0,
            minres_map=5,
            relion_native_shell_floor=True,
            native_volume_shape=volume_shape,
            tau_is_1d=False,
            relion_filter_scale=float(grid_size**4),
        )
        divided = numerator * valid.astype(numerator.real.dtype) / denominator
        divided_half = fourier_transform_utils.full_volume_to_half_volume(
            divided.reshape(accumulator_shape), accumulator_shape
        )
        divided_half = relion_functions._relion_window_centered_half_fourier(
            divided_half, accumulator_shape, reconstruction_shape
        )
        replay_after = fourier_transform_utils.get_idft3_real(
            divided_half,
            volume_shape=reconstruction_shape,
        )
        replay_after = padding.unpad_volume_spatial_domain(
            replay_after, reconstruction_shape[0] - volume_shape[0]
        )
        replay_after, _ = mask.soft_mask_outside_map(replay_after, cosine_width=3)
        replay_after, _ = relion_functions.griddingCorrect(
            replay_after.reshape(volume_shape),
            volume_shape[0],
            padding_factor,
            order=1,
        )
        replay_after = np.asarray(replay_after)

        native_after = helpers.relion_volume_to_recovar(
            _load(
                _stage_path(args.native_stage_dir, half, "volume_after_gridding", 0),
                np.dtype("<f8"),
            )
        )
        replay_after, selected_sign = _choose_sign(replay_after, native_after)
        replay_written = _firstiter_postprocess(
            replay_after,
            volume_shape=volume_shape,
            voxel_size=args.voxel_size,
            ini_high_angstrom=args.ini_high_angstrom,
            fourier_mask_edge=args.fourier_mask_edge,
            solvent_mask=solvent_mask,
        )
        recovar_map_path = args.recovar_map_dir / f"it000_half{half}_reg.mrc"
        native_map_path = args.native_relion_dir / f"run_it001_half{half}_class001.mrc"
        recovar_written = np.asarray(helpers.load_mrc(recovar_map_path), dtype=np.float64)
        native_written = np.asarray(
            helpers.load_relion_volume(native_map_path), dtype=np.float64
        )
        halves[str(half)] = {
            "selected_reconstruction_sign": selected_sign,
            "live_tau_prior_shells_head": np.asarray(
                live_tau_details["prior_shells"][:8], dtype=np.float64
            ).tolist(),
            "replay_after_gridding_vs_native": _comparison(replay_after, native_after),
            "replay_postprocess_vs_recovar_written": _comparison(
                replay_written, recovar_written
            ),
            "replay_postprocess_vs_native_written": _comparison(
                replay_written, native_written
            ),
            "recovar_written_vs_native_written": _comparison(
                recovar_written, native_written
            ),
            "recovar_map": str(recovar_map_path.resolve()),
            "native_map": str(native_map_path.resolve()),
        }

    report = {
        "schema": "recovar.em.k1_recovar_reference_write_boundary.v1",
        "metric_policy": "scale-sensitive relative-L2 and signed non-DC FSC-AUC; no fitted rescaling",
        "recovar_accumulator": str(args.recovar_accumulator.resolve()),
        "recovar_fsc": str(args.recovar_fsc.resolve()),
        "parameters": {
            "grid_size": grid_size,
            "current_size": current_size,
            "padding_factor": padding_factor,
            "voxel_size": args.voxel_size,
            "ini_high_angstrom": args.ini_high_angstrom,
            "fourier_mask_edge": args.fourier_mask_edge,
            "particle_diameter_angstrom": args.particle_diameter_angstrom,
            "solvent_mask_edge_pixels": args.solvent_mask_edge_pixels,
        },
        "halves": halves,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
