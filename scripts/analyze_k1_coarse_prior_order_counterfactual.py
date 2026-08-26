#!/usr/bin/env python3
"""Test RELION's coarse prior/min-diff operation order on one saved particle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.oversampling import (
    relion_cuda_f32_coarse_posterior,
)
from scripts.analyze_em_k1_coarse_pass1_boundary import _map_relion_table
from scripts.analyze_k1_native_coarse_boundary import (
    _float32_from_bits,
    load_native_coarse_capture,
)


def relion_ordered_log_weights(raw_scores, rotation_prior, translation_prior):
    """Apply ``orientation + offset + min_diff2 - diff2`` in float32."""

    raw = jnp.asarray(raw_scores, dtype=jnp.float32)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"expected one K=1 raw-score table, got {raw.shape}")
    rotation = jnp.asarray(rotation_prior, dtype=jnp.float32).reshape(-1)
    translation = jnp.asarray(translation_prior, dtype=jnp.float32).reshape(-1)
    if raw.shape[1:] != (rotation.size, translation.size):
        raise ValueError("raw-score and prior topology mismatch")
    finite = jnp.isfinite(raw)
    raw_best = jnp.max(jnp.where(finite, raw, -jnp.inf), axis=(1, 2))
    min_diff2 = -raw_best
    diff2 = -raw
    prior_sum = rotation[:, None] + translation[None, :]
    ordered = (prior_sum[None, :, :] + min_diff2[:, None, None]) - diff2
    return jnp.where(finite, ordered, -jnp.inf)


def analyze(recovar_path: Path, native_path: Path, targets: list[tuple[int, int]], *, n_directions: int, n_psi: int):
    with np.load(recovar_path, allow_pickle=False) as payload:
        raw = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float32)
        rotation_prior = np.asarray(payload["rotation_log_prior"], dtype=np.float32)
        translation_prior = np.asarray(payload["translation_log_prior"], dtype=np.float32)
        recorded_mask = np.asarray(payload["significant_mask"], dtype=bool).reshape(raw.shape[1:])
        adaptive_fraction = float(np.asarray(payload["adaptive_fraction"]).item())
        max_significants = int(np.asarray(payload["max_significants"]).item())

    ordered = relion_ordered_log_weights(raw, rotation_prior, translation_prior)
    posterior = relion_cuda_f32_coarse_posterior(
        ordered.reshape(1, -1),
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
    )
    weights, mask, n_significant, cutoff_count, sum_weight, threshold = (
        np.asarray(jax.block_until_ready(value)) for value in posterior
    )
    ordered_np = np.asarray(ordered[0], dtype=np.float32)
    weights = weights.reshape(raw.shape[1:])
    mask = mask.reshape(raw.shape[1:])

    native = load_native_coarse_capture(native_path)
    native_threshold = _float32_from_bits(int(native.header[13]))
    native_selected = (native.postexponent >= native_threshold).reshape(
        int(native.header[16]), int(native.header[17])
    )
    native_mask = _map_relion_table(
        native_selected,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=np.arange(int(native.header[17]), dtype=np.int64),
    ).astype(bool)

    target_records = []
    for rotation, translation in targets:
        target_records.append(
            {
                "rotation": rotation,
                "translation": translation,
                "ordered_log_weight": float(ordered_np[rotation, translation]),
                "posterior_weight": float(weights[rotation, translation]),
                "recorded_selected": bool(recorded_mask[rotation, translation]),
                "counterfactual_selected": bool(mask[rotation, translation]),
                "native_selected": bool(native_mask[rotation, translation]),
            }
        )
    return {
        "schema": "recovar.em.k1_coarse_prior_order_counterfactual.v1",
        "status": "complete",
        "jax_backend": jax.default_backend(),
        "recovar_capture": str(recovar_path.resolve()),
        "native_capture": str(native_path.resolve()),
        "recorded_selected_count": int(np.count_nonzero(recorded_mask)),
        "counterfactual_selected_count": int(n_significant[0]),
        "native_selected_count": int(np.count_nonzero(native_mask)),
        "recorded_mismatch_vs_native": int(np.count_nonzero(recorded_mask != native_mask)),
        "counterfactual_mismatch_vs_native": int(np.count_nonzero(mask != native_mask)),
        "cutoff_count": int(cutoff_count[0]),
        "sum_weight": float(sum_weight[0]),
        "significant_weight": float(threshold[0]),
        "targets": target_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--native-directions", type=int, required=True)
    parser.add_argument("--native-psi", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    targets = [tuple(map(int, value.split(":"))) for value in args.target]
    report = analyze(
        args.recovar,
        args.native,
        targets,
        n_directions=args.native_directions,
        n_psi=args.native_psi,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
