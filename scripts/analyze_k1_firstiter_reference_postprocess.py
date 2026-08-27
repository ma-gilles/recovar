#!/usr/bin/env python3
"""Compare RECOVAR's first-iteration reference postprocess with native RELION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils, mask
from recovar.em.dense_single_volume.mean_helpers import (
    _apply_relion_initial_lowpass_filter,
)
from recovar.utils import helpers
from scripts.analyze_k1_reconstruction_stage_boundary import (
    _load,
    _metrics,
    _stage_path,
)
from scripts.summarize_em_completion_bench import normalized_fsc_auc, shell_fsc


def _region_metrics(actual: np.ndarray, expected: np.ndarray, region: np.ndarray) -> dict[str, object]:
    selected_actual = np.asarray(actual)[region]
    selected_expected = np.asarray(expected)[region]
    if selected_actual.size == 0:
        raise ValueError("comparison region is empty")
    return _metrics(selected_actual, selected_expected)


def _fsc_auc(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        normalized_fsc_auc(
            np.asarray(shell_fsc(np.asarray(actual), np.asarray(expected)), dtype=np.float64)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-stage-dir", type=Path, required=True)
    parser.add_argument("--native-relion-dir", type=Path, required=True)
    parser.add_argument("--native-call-index", type=int, default=0)
    parser.add_argument("--volume-size", type=int, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--ini-high-angstrom", type=float, required=True)
    parser.add_argument("--fourier-mask-edge", type=float, default=2.0)
    parser.add_argument("--particle-diameter-angstrom", type=float, required=True)
    parser.add_argument("--solvent-mask-edge-pixels", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.native_call_index < 0:
        raise ValueError("native call index must be non-negative")
    if args.volume_size <= 0 or args.voxel_size <= 0.0:
        raise ValueError("volume and voxel sizes must be positive")

    volume_shape = (args.volume_size,) * 3
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
    inside = solvent_mask == 1.0
    transition = (solvent_mask > 0.0) & (solvent_mask < 1.0)
    outside = solvent_mask == 0.0

    halves: dict[str, object] = {}
    for half in (1, 2):
        native_before_postprocess = helpers.relion_volume_to_recovar(
            _load(
                _stage_path(
                    args.native_stage_dir,
                    half,
                    "volume_after_gridding",
                    args.native_call_index,
                ),
                np.dtype("<f8"),
            )
        )
        before_ft = fourier_transform_utils.get_dft3(
            jnp.asarray(native_before_postprocess)
        ).reshape(-1)
        lowpass_ft = _apply_relion_initial_lowpass_filter(
            before_ft,
            volume_shape,
            args.voxel_size,
            args.ini_high_angstrom,
            filter_edgewidth=args.fourier_mask_edge,
        )
        after_lowpass = np.asarray(
            fourier_transform_utils.get_idft3(
                jnp.asarray(lowpass_ft).reshape(volume_shape)
            )
        ).real
        after_solvent = after_lowpass * solvent_mask

        native_map_path = args.native_relion_dir / f"run_it001_half{half}_class001.mrc"
        native_written_map = np.asarray(
            helpers.load_relion_volume(native_map_path), dtype=np.float64
        )
        halves[str(half)] = {
            "native_reconstruction_dump": str(
                _stage_path(
                    args.native_stage_dir,
                    half,
                    "volume_after_gridding",
                    args.native_call_index,
                ).resolve()
            ),
            "native_written_map": str(native_map_path.resolve()),
            "postprocess_vs_native_written_map": {
                "whole_volume": _metrics(after_solvent, native_written_map),
                "inside_flat_mask": _region_metrics(after_solvent, native_written_map, inside),
                "solvent_transition": _region_metrics(
                    after_solvent, native_written_map, transition
                ),
                "outside_mask": _region_metrics(after_solvent, native_written_map, outside),
                "signed_fsc_auc_non_dc": _fsc_auc(after_solvent, native_written_map),
            },
            "lowpass_change_from_reconstruction": _metrics(
                after_lowpass, native_before_postprocess
            ),
        }

    report = {
        "schema": "recovar.em.k1_firstiter_reference_postprocess.v1",
        "metric_policy": "scale-sensitive relative-L2 and signed non-DC FSC-AUC; no fitted rescaling",
        "parameters": {
            "volume_size": args.volume_size,
            "voxel_size": args.voxel_size,
            "ini_high_angstrom": args.ini_high_angstrom,
            "fourier_mask_edge": args.fourier_mask_edge,
            "particle_diameter_angstrom": args.particle_diameter_angstrom,
            "solvent_mask_edge_pixels": args.solvent_mask_edge_pixels,
            "flatten_radius_pixels": flatten_radius,
            "inside_voxels": int(np.count_nonzero(inside)),
            "transition_voxels": int(np.count_nonzero(transition)),
            "outside_voxels": int(np.count_nonzero(outside)),
        },
        "halves": halves,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
