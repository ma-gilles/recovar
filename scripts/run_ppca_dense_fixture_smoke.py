#!/usr/bin/env python
"""Run one small dense PPCA EM smoke iteration from a K-class fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import starfile

from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.ppca_refinement.dense_dataset import run_dense_ppca_fused_em_iteration
from recovar.em.ppca_refinement.fixture_validation import validate_kclass_to_ppca_initialization
from recovar.em.sampling import (
    apply_relion_rotation_perturbation_to_eulers,
    apply_relion_translation_perturbation,
    get_relion_rotation_grid_eulers,
    get_rotation_grid_at_order,
    get_translation_grid,
    read_relion_sampling_metadata,
    relion_angular_sampling_deg,
)
from recovar.reconstruction import noise as recon_noise


def _scalar(table_or_dict, name: str, default=None):
    if table_or_dict is None:
        if default is None:
            raise KeyError(name)
        return default
    if isinstance(table_or_dict, dict):
        if name not in table_or_dict:
            if default is None:
                raise KeyError(name)
            return default
        return table_or_dict[name]
    if name not in table_or_dict.columns:
        if default is None:
            raise KeyError(name)
        return default
    return table_or_dict[name].iloc[0]


def _make_rotations(sampling, *, use_relion_bind_grid: bool) -> tuple[np.ndarray, str]:
    healpix_order = int(sampling["healpix_order"])
    if use_relion_bind_grid:
        eulers = get_relion_rotation_grid_eulers(healpix_order)
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            eulers,
            float(sampling["random_perturbation"]),
            relion_angular_sampling_deg(healpix_order, adaptive_oversampling=0),
        )
        return np.asarray(rotations, dtype=np.float32), "relion_bind"
    return np.asarray(get_rotation_grid_at_order(healpix_order), dtype=np.float32), "generic_healpix"


def _noise_from_relion_model(model, image_shape) -> np.ndarray:
    grid_size = int(image_shape[0])
    noise_spectrum = np.asarray(model["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64)
    return np.asarray(recon_noise.make_radial_noise(noise_spectrum * (grid_size**4), tuple(image_shape))).reshape(-1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, help="K-class parity summary.json")
    parser.add_argument("--output-json", required=True, help="Dense PPCA smoke summary path")
    parser.add_argument("--output-npz", help="Optional output arrays path")
    parser.add_argument("--q", type=int, default=None, help="PPCA components; default K-1")
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--n-rotations", type=int, default=12)
    parser.add_argument("--n-translations", type=int, default=None)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--rotation-block-size", type=int, default=4)
    parser.add_argument("--current-size", type=int, default=None, help="Default min(summary current_size, 16)")
    parser.add_argument("--mstep-chunk-size", type=int, default=65536)
    parser.add_argument("--weight-source", choices=("recovar", "relion", "uniform"), default="recovar")
    parser.add_argument("--kclass-frame", choices=("relion", "recovar"), default="relion")
    parser.add_argument("--mean-prior-variance", type=float, default=1.0)
    parser.add_argument("--W-prior-variance", type=float, default=1.0)
    parser.add_argument(
        "--use-relion-bind-grid",
        action="store_true",
        help="Use RELION bind coarse orientations if recovar.relion_bind is built.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text())
    relion_dir = Path(summary["relion_dir"])
    data_star = Path(summary["data_star"])
    target_iter = int(summary.get("target_iter", 1))
    model = starfile.read(str(relion_dir / f"run_it{target_iter:03d}_model.star"))
    sampling = read_relion_sampling_metadata(relion_dir / f"run_it{target_iter:03d}_sampling.star")

    validation, init, _gt_init = validate_kclass_to_ppca_initialization(
        summary_path,
        q=args.q,
        map_key="output_maps",
        kclass_frame=args.kclass_frame,
        weight_source=args.weight_source,
    )
    if not validation.passed:
        raise SystemExit("K-class-to-PPCA initialization validation failed: " + "; ".join(validation.failures))

    dataset = load_dataset(str(data_star))
    rotations, rotation_source = _make_rotations(sampling, use_relion_bind_grid=args.use_relion_bind_grid)
    voxel_size = float(getattr(dataset, "voxel_size", _scalar(model["model_general"], "rlnPixelSize")))
    offset_range_px = float(sampling["offset_range"]) / voxel_size
    offset_step_px = float(sampling["offset_step"]) / voxel_size
    translations = apply_relion_translation_perturbation(
        get_translation_grid(offset_range_px, offset_step_px).astype(np.float32),
        float(sampling["random_perturbation"]),
        offset_step_px,
    ).astype(np.float32)

    n_images = min(int(args.n_images), int(dataset.n_images))
    n_rotations = min(int(args.n_rotations), int(rotations.shape[0]))
    n_translations = int(translations.shape[0]) if args.n_translations is None else min(
        int(args.n_translations),
        int(translations.shape[0]),
    )
    image_indices = np.arange(n_images, dtype=np.int64)
    rotations = rotations[:n_rotations]
    translations = translations[:n_translations]
    current_size = (
        min(int(summary.get("current_size", dataset.image_shape[0])), 16)
        if args.current_size is None
        else int(args.current_size)
    )

    q = int(init.W.shape[0])
    half_size = int(np.prod((dataset.volume_shape[0], dataset.volume_shape[1], dataset.volume_shape[2] // 2 + 1)))
    mean_prior = jnp.full((half_size,), float(args.mean_prior_variance), dtype=jnp.float32)
    W_prior = jnp.full((half_size, q), float(args.W_prior_variance), dtype=jnp.float32)
    noise_variance = _noise_from_relion_model(model, dataset.image_shape)

    t0 = time.time()
    result = run_dense_ppca_fused_em_iteration(
        dataset,
        init.mu,
        init.W,
        mean_prior=mean_prior,
        W_prior=W_prior,
        noise_variance=noise_variance,
        rotations=rotations,
        translations=translations,
        image_batch_size=int(args.image_batch_size),
        rotation_block_size=int(args.rotation_block_size),
        current_size=current_size,
        q=q,
        volume_domain="real",
        image_indices=image_indices,
        mstep_chunk_size=int(args.mstep_chunk_size),
        half_spectrum_scoring=False,
        square_window=False,
    )
    jax.block_until_ready(result.mu_half)
    jax.block_until_ready(result.W_half)
    elapsed_s = float(time.time() - t0)

    delta_mu = float(jnp.sqrt(jnp.mean(jnp.abs(result.mu_half - result.mu_half.mean()) ** 2)))
    output = {
        "passed": bool(np.isfinite(result.diagnostics["log_likelihood"])),
        "summary_json": str(summary_path),
        "relion_dir": str(relion_dir),
        "data_star": str(data_star),
        "n_images": n_images,
        "n_rotations": n_rotations,
        "n_translations": n_translations,
        "rotation_source": rotation_source,
        "current_size": current_size,
        "q": q,
        "elapsed_s": elapsed_s,
        "init_validation": validation.summary,
        "diagnostics": {
            "log_likelihood": float(result.diagnostics["log_likelihood"]),
            "logZ_mean": float(result.diagnostics["logZ_mean"]),
            "pmax_mean": float(result.diagnostics["pmax_mean"]),
            "nsig_mean": float(result.diagnostics["nsig_mean"]),
            "n_images_accumulated": int(result.stats.n_images),
            "best_rotation_shape": list(np.asarray(result.diagnostics["best_rotation_idx"]).shape),
            "best_translation_shape": list(np.asarray(result.diagnostics["best_translation_idx"]).shape),
        },
        "output_stats": {
            "mu_half_shape": list(result.mu_half.shape),
            "W_half_shape": list(result.W_half.shape),
            "mu_half_finite": bool(np.all(np.isfinite(np.asarray(result.mu_half)))),
            "W_half_finite": bool(np.all(np.isfinite(np.asarray(result.W_half)))),
            "mu_half_rms": float(jnp.sqrt(jnp.mean(jnp.abs(result.mu_half) ** 2))),
            "W_half_rms": float(jnp.sqrt(jnp.mean(jnp.abs(result.W_half) ** 2))) if q else 0.0,
            "mu_half_centered_rms": delta_mu,
        },
    }
    output["passed"] = bool(
        output["passed"]
        and output["output_stats"]["mu_half_finite"]
        and output["output_stats"]["W_half_finite"]
        and result.stats.n_images == n_images
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))

    if args.output_npz:
        npz_path = Path(args.output_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            mu_half=np.asarray(result.mu_half),
            W_half=np.asarray(result.W_half),
            best_rotation_idx=np.asarray(result.diagnostics["best_rotation_idx"]),
            best_translation_idx=np.asarray(result.diagnostics["best_translation_idx"]),
        )
        print(f"saved_output_npz={npz_path}")

    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
