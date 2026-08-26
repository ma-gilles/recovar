#!/usr/bin/env python3
"""Evaluate fused RELION coarse translation/scoring on captured rotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.oversampling import (
    relion_cuda_f32_coarse_log_weights,
    relion_cuda_f32_coarse_posterior,
)
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.significance import (
    _compact_projection_window_positions,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_translation_angles_f32,
)
from scripts.analyze_em_k1_coarse_pass1_boundary import _map_relion_table
from scripts.analyze_k1_native_coarse_boundary import (
    _float32_from_bits,
    load_native_coarse_capture,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(
    capture_path: Path,
    native_path: Path,
    *,
    physical_image_size: int,
    native_directions: int,
    native_psi: int,
) -> dict[str, object]:
    required = {
        "current_size",
        "adaptive_fraction",
        "max_significants",
        "n_rot",
        "n_trans",
        "scores_pre_prior_per_class",
        "rotation_log_prior",
        "translation_log_prior",
        "translations",
        "translation_phase_source",
        "coarse_gaussian_score_indices",
        "coarse_gaussian_unshifted_corrected",
        "coarse_gaussian_pixel_weight",
        "coarse_gaussian_initial_diff2",
        "projected_reference_rotation_ids",
        "projected_reference_per_class",
    }
    with np.load(capture_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"coarse capture misses fields: {sorted(missing)}")
        payload = {name: np.asarray(archive[name]) for name in required}

    current_size = int(payload["current_size"])
    n_rot = int(payload["n_rot"])
    n_trans = int(payload["n_trans"])
    rotation_ids = np.asarray(
        payload["projected_reference_rotation_ids"], dtype=np.int64
    ).reshape(-1)
    references = np.asarray(
        payload["projected_reference_per_class"], dtype=np.complex64
    )[0]
    if references.shape[0] != rotation_ids.size:
        raise ValueError("captured projection ids and reference rows differ")

    image_shape = (int(physical_image_size), int(physical_image_size))
    score_indices = np.asarray(
        payload["coarse_gaussian_score_indices"], dtype=np.int32
    )
    lookup = _relion_cuda_fine_full_to_compact_lookup(
        image_shape,
        current_size,
        score_indices,
    ).astype(np.int32, copy=False)
    if references.shape[-1] != score_indices.size:
        active_indices, _ = make_fourier_window_indices_np(
            image_shape,
            current_size,
            square=False,
            include_dc=False,
        )
        active_positions = _compact_projection_window_positions(
            score_indices,
            active_indices,
        )
        if references.shape[-1] != active_positions.size:
            raise ValueError(
                "captured projection cannot be restored to the square score layout: "
                f"projection={references.shape}, active={active_positions.shape}, "
                f"score_indices={score_indices.shape}"
            )
        square_references = np.zeros(
            references.shape[:-1] + (score_indices.size,),
            dtype=np.complex64,
        )
        square_references[..., active_positions] = references
        references = square_references
    angles = _relion_translation_angles_f32(
        np.asarray(payload["translation_phase_source"], dtype=np.float64),
        image_shape,
    ).astype(np.float32, copy=False)
    image = np.asarray(
        payload["coarse_gaussian_unshifted_corrected"], dtype=np.complex64
    )
    if image.ndim == 1:
        image = image[None, :]
    weight = np.asarray(
        payload["coarse_gaussian_pixel_weight"], dtype=np.float32
    )
    if weight.ndim == 1:
        weight = weight[None, :]
    compact_pixels = references.shape[-1]
    if image.shape[-1] != compact_pixels or weight.shape[-1] != compact_pixels:
        raise ValueError(
            "captured coarse operands do not share the restored square layout: "
            f"projection={references.shape}, image={image.shape}, weight={weight.shape}, "
            f"score_indices={score_indices.shape}, lookup={lookup.shape}"
        )

    initial_diff2 = np.asarray(
        payload["coarse_gaussian_initial_diff2"], dtype=np.float32
    ).reshape(-1)
    fused_lane = cuda_backproject.relion_coarse_diff2_fused_translate_rectangular_f32(
        jnp.asarray(references),
        jnp.asarray(image),
        jnp.asarray(angles),
        jnp.asarray(weight),
        jnp.asarray(initial_diff2),
        jnp.asarray(lookup),
        current_size=current_size,
    )
    fused_diff2 = np.asarray(jax.block_until_ready(fused_lane), dtype=np.float32)[0]

    raw = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float32).copy()
    raw_fused = raw.copy()
    raw_fused[0, rotation_ids, :] = -fused_diff2
    rotation_prior = np.asarray(payload["rotation_log_prior"], dtype=np.float32).reshape(-1)
    translation_prior = np.asarray(payload["translation_log_prior"], dtype=np.float32).reshape(-1)
    exact_log = relion_cuda_f32_coarse_log_weights(
        jnp.asarray(raw_fused),
        jnp.asarray(rotation_prior),
        jnp.asarray(translation_prior[None, :]),
    )
    posterior = relion_cuda_f32_coarse_posterior(
        exact_log.reshape(1, -1),
        adaptive_fraction=float(payload["adaptive_fraction"]),
        max_significants=int(payload["max_significants"]),
    )
    weights, selected, n_selected, cutoff_count, sum_weight, threshold = (
        np.asarray(jax.block_until_ready(value)) for value in posterior
    )
    selected = selected.reshape(raw.shape[1:])

    native = load_native_coarse_capture(native_path)
    native_raw = _map_relion_table(
        native.raw_diff2.reshape(int(native.header[16]), int(native.header[17])),
        n_directions=int(native_directions),
        n_psi=int(native_psi),
        relion_to_recovar_translation=np.arange(int(native.header[17]), dtype=np.int64),
    )
    native_threshold = _float32_from_bits(int(native.header[13]))
    native_selected = _map_relion_table(
        (native.postexponent >= native_threshold).reshape(
            int(native.header[16]), int(native.header[17])
        ),
        n_directions=int(native_directions),
        n_psi=int(native_psi),
        relion_to_recovar_translation=np.arange(int(native.header[17]), dtype=np.int64),
    ).astype(bool)

    records = []
    for row, rotation in enumerate(rotation_ids):
        for translation in range(n_trans):
            records.append(
                {
                    "rotation": int(rotation),
                    "translation": int(translation),
                    "production_raw_score": float(raw[0, rotation, translation]),
                    "fused_raw_score": float(raw_fused[0, rotation, translation]),
                    "native_raw_score": float(-native_raw[rotation, translation]),
                    "fused_raw_exact_native": bool(
                        raw_fused[0, rotation, translation]
                        == np.float32(-native_raw[rotation, translation])
                    ),
                    "counterfactual_selected": bool(selected[rotation, translation]),
                    "native_selected": bool(native_selected[rotation, translation]),
                }
            )

    return {
        "schema": "recovar.em.k1_coarse_fused_translate_counterfactual.v2",
        "status": "complete",
        "device": str(jax.devices()[0]),
        "fused_kernel": "relion_coarse_diff2_fused_translate_rectangular_f32",
        "capture": str(capture_path.resolve()),
        "capture_sha256": _sha256(capture_path),
        "native": str(native_path.resolve()),
        "native_sha256": _sha256(native_path),
        "captured_rotation_ids": rotation_ids.tolist(),
        "captured_candidate_count": int(len(records)),
        "fused_raw_exact_native_count": int(
            sum(record["fused_raw_exact_native"] for record in records)
        ),
        "counterfactual_selected_count": int(n_selected[0]),
        "native_selected_count": int(np.count_nonzero(native_selected)),
        "support_mismatch_count": int(np.count_nonzero(selected != native_selected)),
        "cutoff_count": int(cutoff_count[0]),
        "sum_weight": float(sum_weight[0]),
        "significant_weight": float(threshold[0]),
        "candidates": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--native-directions", type=int, required=True)
    parser.add_argument("--native-psi", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.capture,
        args.native,
        physical_image_size=args.physical_image_size,
        native_directions=args.native_directions,
        native_psi=args.native_psi,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
