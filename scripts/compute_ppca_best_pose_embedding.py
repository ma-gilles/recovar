#!/usr/bin/env python
"""Compute a best-pose PPCA latent embedding from a dense PPCA result."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from recovar.core.configs import ForwardModelConfig
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.dense_single_volume.helpers.preprocessing import preprocess_batch
from recovar.em.ppca_refinement.dense_dataset import (
    _project_augmented_half_volumes,
    prepare_dense_ppca_dataset_inputs,
)
from recovar.em.sampling import get_rotation_grid_at_order, get_translation_grid
from recovar.reconstruction import noise as noise_utils


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_simulation_info(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _load_noise_variance(simulation_info: dict[str, Any] | None, image_shape) -> np.ndarray:
    if simulation_info is None:
        return np.ones(int(np.prod(image_shape)), dtype=np.float32)
    radial = np.asarray(simulation_info["noise_variance"], dtype=np.float32).reshape(-1)
    return np.asarray(noise_utils.make_radial_noise(radial, tuple(image_shape)), dtype=np.float32).reshape(-1)


def _candidate_rotations_from_source(
    *,
    rotation_source: str,
    simulation_info: dict[str, Any] | None,
    healpix_order: int,
    n_total: int,
) -> np.ndarray:
    if rotation_source == "simulation-info":
        if simulation_info is None:
            raise ValueError("--rotation-source=simulation-info requires --simulation-info")
        return np.asarray(simulation_info["rots"], dtype=np.float32)[:n_total]
    if rotation_source == "simulation-info-plus-healpix":
        if simulation_info is None:
            raise ValueError("--rotation-source=simulation-info-plus-healpix requires --simulation-info")
        return np.concatenate(
            [
                np.asarray(simulation_info["rots"], dtype=np.float32)[:n_total],
                np.asarray(get_rotation_grid_at_order(int(healpix_order)), dtype=np.float32),
            ],
            axis=0,
        )
    if rotation_source == "healpix":
        return np.asarray(get_rotation_grid_at_order(int(healpix_order)), dtype=np.float32)
    raise ValueError("rotation_source must be 'healpix', 'simulation-info', or 'simulation-info-plus-healpix'")


def _best_pose_z_mean(Y_best, ctf2_over_noise, proj_aug):
    """Return ``E[z | image, best pose]`` for paired image/projection rows."""

    q = int(proj_aug.shape[1]) - 1
    if q == 0:
        return jnp.zeros((Y_best.shape[0], 0), dtype=jnp.float32)
    proj_mu = proj_aug[:, 0, :]
    proj_W = proj_aug[:, 1:, :]
    g_zx = jnp.einsum("bf,bqf->bq", Y_best, jnp.conj(proj_W)).real
    h_zm = jnp.einsum("bf,bqf,bf->bq", ctf2_over_noise, jnp.conj(proj_W), proj_mu).real
    Hzz = jnp.einsum("bf,bqf,bpf->bqp", ctf2_over_noise, jnp.conj(proj_W), proj_W).real
    Hzz = 0.5 * (Hzz + jnp.swapaxes(Hzz, -1, -2))
    M = Hzz + jnp.eye(q, dtype=Hzz.dtype)
    b = g_zx - h_zm
    return jnp.linalg.solve(M, b[..., None])[..., 0]


def compute_best_pose_embedding(
    *,
    data_star: str | Path,
    simulation_info_path: str | Path | None,
    ppca_result_npz: str | Path,
    output_dir: str | Path,
    rotation_source: str,
    translation_source: str,
    healpix_order: int,
    offset_range_px: float,
    offset_step_px: float,
    current_size: int,
    image_batch_size: int,
    n_images: int | None,
    relion_texture_interp: bool,
    image_scale_source: str = "none",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = np.load(ppca_result_npz)
    mu_half = np.asarray(result["mu_half"])
    W_half = np.asarray(result["W_half"])
    best_rotation_idx = np.asarray(result["best_rotation_idx"], dtype=np.int64)
    best_translation_idx = np.asarray(result["best_translation_idx"], dtype=np.int64)
    if rotation_source == "result-matrices":
        if "best_rotation_matrix" not in result:
            raise ValueError("--rotation-source=result-matrices requires best_rotation_matrix in result npz")
        result_rotations = np.asarray(result["best_rotation_matrix"], dtype=np.float32)
        best_rotation_idx = np.arange(result_rotations.shape[0], dtype=np.int64)
    else:
        result_rotations = None
    if translation_source == "result-vectors":
        if "best_translation" not in result:
            raise ValueError("--translation-source=result-vectors requires best_translation in result npz")
        result_translations = np.asarray(result["best_translation"], dtype=np.float32)
        best_translation_idx = np.arange(result_translations.shape[0], dtype=np.int64)
    else:
        result_translations = None

    dataset = load_dataset(str(data_star))
    n_total = int(dataset.n_images if n_images is None else min(int(n_images), int(dataset.n_images)))
    n_total = min(n_total, int(best_rotation_idx.shape[0]), int(best_translation_idx.shape[0]))
    image_indices = np.arange(n_total, dtype=np.int64)
    simulation_info = _load_simulation_info(simulation_info_path)
    if result_rotations is None:
        rotations = _candidate_rotations_from_source(
            rotation_source=rotation_source,
            simulation_info=simulation_info,
            healpix_order=int(healpix_order),
            n_total=int(n_total),
        )
    else:
        rotations = result_rotations[:n_total]

    if result_translations is not None:
        translations = result_translations[:n_total]
    elif translation_source == "simulation-info-unique":
        if simulation_info is None:
            raise ValueError("--translation-source=simulation-info-unique requires --simulation-info")
        translations = np.unique(np.asarray(simulation_info["trans"], dtype=np.float32)[:n_total], axis=0)
    elif translation_source == "grid":
        translations = np.asarray(get_translation_grid(float(offset_range_px), float(offset_step_px)), dtype=np.float32)
    else:
        raise ValueError("translation_source must be 'grid', 'simulation-info-unique', or 'result-vectors'")
    if int(best_rotation_idx[:n_total].max(initial=0)) >= int(rotations.shape[0]):
        raise ValueError("best_rotation_idx exceeds rotation candidate count for the requested rotation source")
    if int(best_translation_idx[:n_total].max(initial=0)) >= int(translations.shape[0]):
        raise ValueError("best_translation_idx exceeds translation candidate count for the requested translation source")

    noise_variance = _load_noise_variance(simulation_info, dataset.image_shape)
    image_scale_corrections = None
    if image_scale_source == "simulation-info-contrast":
        if simulation_info is None or "per_image_contrast" not in simulation_info:
            raise ValueError("--image-scale-source=simulation-info-contrast requires simulation_info['per_image_contrast']")
        image_scale_corrections = np.asarray(simulation_info["per_image_contrast"], dtype=np.float32)
    elif image_scale_source != "none":
        raise ValueError("image_scale_source must be 'none' or 'simulation-info-contrast'")
    resolved = prepare_dense_ppca_dataset_inputs(
        dataset,
        mu_half,
        W_half,
        q=int(W_half.shape[1]),
        volume_domain="fourier_half",
        current_size=int(current_size),
        half_spectrum_scoring=False,
        square_window=False,
    )
    config = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp", process_fn=dataset.process_images)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, resolved.image_shape).squeeze()

    z_parts = []
    row_indices = []
    batch_iter = dataset.iter_batches(int(image_batch_size), indices=image_indices, by_image=False)
    for batch_data, _rots, _trans, ctf_params, _noise, _particle_indices, indices in batch_iter:
        indices_np = np.asarray(indices, dtype=np.int64)
        batch_count = int(indices_np.shape[0])
        shifted_half, _batch_norm, ctf2_over_nv_half = preprocess_batch(
            dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            translations,
            config,
            score_with_masked_images=False,
        )
        F = int(shifted_half.shape[-1])
        if image_scale_corrections is None:
            batch_scale = jnp.ones((batch_count,), dtype=shifted_half.real.dtype)
        else:
            batch_scale = jnp.asarray(
                image_scale_corrections[indices_np],
                dtype=shifted_half.real.dtype,
            )
        Y1 = (
            shifted_half.reshape(batch_count, int(translations.shape[0]), F)
            * batch_scale[:, None, None]
            * resolved.score_mask[None, None, :]
        )
        ctf2_score = ctf2_over_nv_half * (batch_scale[:, None] ** 2) * resolved.score_mask[None, :]
        best_t = jnp.asarray(best_translation_idx[indices_np], dtype=jnp.int32)
        Y_best = Y1[jnp.arange(batch_count), best_t, :]

        best_rotations = rotations[best_rotation_idx[indices_np]]
        proj_aug = _project_augmented_half_volumes(
            resolved.augmented_half_volumes,
            best_rotations,
            resolved.image_shape,
            resolved.volume_shape,
            "linear_interp",
            max_r=resolved.projection_max_r,
            relion_texture_interp=bool(relion_texture_interp),
        )
        z_batch = _best_pose_z_mean(Y_best, ctf2_score, proj_aug)
        z_parts.append(np.asarray(jax.block_until_ready(z_batch)))
        row_indices.append(indices_np)

    z_complex = np.concatenate(z_parts, axis=0) if z_parts else np.zeros((0, int(W_half.shape[1])), dtype=np.complex64)
    rows = np.concatenate(row_indices, axis=0) if row_indices else np.zeros((0,), dtype=np.int64)
    order = np.argsort(rows)
    z_complex = z_complex[order]
    rows = rows[order]
    z_real = z_complex.real.astype(np.float32)

    embedding_npy = output_dir / "embedding_z.npy"
    embedding_complex_npy = output_dir / "embedding_z_complex.npy"
    embedding_csv = output_dir / "embedding_z.csv"
    np.save(embedding_npy, z_real)
    np.save(embedding_complex_npy, z_complex)
    with embedding_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index"] + [f"z{i + 1}" for i in range(z_real.shape[1])])
        writer.writerows([[int(idx), *map(float, row)] for idx, row in zip(rows, z_real)])

    plot_path = output_dir / "embedding_z1_z2.png"
    if z_real.shape[1] >= 2 and z_real.shape[0]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        color = None
        color_label = None
        if simulation_info is not None and "image_assignment" in simulation_info:
            color = np.asarray(simulation_info["image_assignment"])[rows]
            color_label = "GT volume index"
        fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
        scatter = ax.scatter(z_real[:, 0], z_real[:, 1], c=color, s=5, alpha=0.65, cmap="viridis", linewidths=0)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title("Best-pose PPCA embedding")
        ax.grid(alpha=0.2)
        if color is not None:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(color_label)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    summary = {
        "passed": bool(np.all(np.isfinite(z_real))),
        "data_star": Path(data_star),
        "simulation_info": None if simulation_info_path is None else Path(simulation_info_path),
        "ppca_result_npz": Path(ppca_result_npz),
        "output_dir": output_dir,
        "embedding_npy": embedding_npy,
        "embedding_complex_npy": embedding_complex_npy,
        "embedding_csv": embedding_csv,
        "plot_path": plot_path if plot_path.exists() else None,
        "n_images": int(z_real.shape[0]),
        "q": int(z_real.shape[1]) if z_real.ndim == 2 else 0,
        "rotation_source": rotation_source,
        "translation_source": translation_source,
        "healpix_order": int(healpix_order),
        "n_rotations": int(rotations.shape[0]),
        "n_translations": int(translations.shape[0]),
        "current_size": int(current_size),
        "relion_texture_interp": bool(relion_texture_interp),
        "image_scale_source": str(image_scale_source),
        "uses_image_scale_corrections": bool(image_scale_corrections is not None),
        "z_mean": np.mean(z_real, axis=0) if z_real.size else [],
        "z_std": np.std(z_real, axis=0) if z_real.size else [],
        "z_abs_max": float(np.max(np.abs(z_real))) if z_real.size else 0.0,
        "z_imag_abs_max": float(np.max(np.abs(z_complex.imag))) if z_complex.size else 0.0,
        "best_rotation_minmax": [int(best_rotation_idx[:n_total].min(initial=0)), int(best_rotation_idx[:n_total].max(initial=0))],
        "best_translation_minmax": [
            int(best_translation_idx[:n_total].min(initial=0)),
            int(best_translation_idx[:n_total].max(initial=0)),
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return _jsonable(summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-star", required=True)
    parser.add_argument("--simulation-info")
    parser.add_argument("--ppca-result-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--rotation-source",
        choices=("healpix", "simulation-info", "simulation-info-plus-healpix", "result-matrices"),
        default="healpix",
    )
    parser.add_argument("--translation-source", choices=("grid", "simulation-info-unique", "result-vectors"), default="grid")
    parser.add_argument("--healpix-order", type=int, default=3)
    parser.add_argument("--offset-range-px", type=float, default=6.0)
    parser.add_argument("--offset-step-px", type=float, default=2.0)
    parser.add_argument("--current-size", type=int, required=True)
    parser.add_argument("--image-batch-size", type=int, default=100)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--relion-texture-interp", action="store_true")
    parser.add_argument(
        "--image-scale-source",
        choices=("none", "simulation-info-contrast"),
        default="none",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = compute_best_pose_embedding(
        data_star=args.data_star,
        simulation_info_path=args.simulation_info,
        ppca_result_npz=args.ppca_result_npz,
        output_dir=args.output_dir,
        rotation_source=args.rotation_source,
        translation_source=args.translation_source,
        healpix_order=int(args.healpix_order),
        offset_range_px=float(args.offset_range_px),
        offset_step_px=float(args.offset_step_px),
        current_size=int(args.current_size),
        image_batch_size=int(args.image_batch_size),
        n_images=args.n_images,
        relion_texture_interp=bool(args.relion_texture_interp),
        image_scale_source=str(args.image_scale_source),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
