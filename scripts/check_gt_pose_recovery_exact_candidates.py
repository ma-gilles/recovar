#!/usr/bin/env python
"""Check dense PPCA pose scoring against exact simulator GT pose candidates.

This is a synthetic-data diagnostic, not a production refinement path. It uses
the first N simulator rotations as the candidate rotation set, so image i should
recover candidate i when the model and particle amplitudes are consistent.
Translations are checked against the unique GT translations present in the same
image subset.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.ppca_refinement.dense_dataset import iter_dense_ppca_dataset_blocks
from recovar.em.ppca_refinement.engine import dense_pose_ppca_E_step_blocked
from recovar.reconstruction import noise as recon_noise


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


def _load_simulation_info(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _load_noise_variance(simulation_info, image_shape):
    radial = np.asarray(simulation_info["noise_variance"], dtype=np.float32).reshape(-1)
    return np.asarray(recon_noise.make_radial_noise(radial, tuple(image_shape)), dtype=np.float32).reshape(-1)


def _rotation_error_deg(lhs, rhs):
    traces = np.einsum("nij,nij->n", lhs, rhs)
    cosang = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def run_check(
    *,
    data_star: Path,
    simulation_info_path: Path,
    init_npz: Path,
    output_json: Path,
    n_images: int,
    q: int | None,
    current_size: int | None,
    disc_type: str,
    relion_texture_interp: bool,
):
    dataset = load_dataset(str(data_star))
    simulation_info = _load_simulation_info(simulation_info_path)
    init = np.load(init_npz, allow_pickle=True)
    mu = np.asarray(init["mu"], dtype=np.float32)
    W = np.asarray(init["W"], dtype=np.float32)
    if q is not None:
        W = W[: int(q)]
    q_resolved = int(W.shape[0])

    n = min(int(n_images), int(dataset.n_images))
    image_indices = np.arange(n, dtype=np.int64)
    rotations = np.asarray(simulation_info["rots"], dtype=np.float32)[:n]
    gt_trans = np.asarray(simulation_info["trans"], dtype=np.float32)[:n]
    translations, inverse = np.unique(gt_trans, axis=0, return_inverse=True)
    if translations.shape[0] > n:
        raise AssertionError("internal unique translation count cannot exceed n")

    block_iter = iter_dense_ppca_dataset_blocks(
        dataset,
        mu,
        W,
        noise_variance=_load_noise_variance(simulation_info, dataset.image_shape),
        rotations=rotations,
        translations=translations,
        image_batch_size=n,
        rotation_block_size=n,
        current_size=current_size,
        q=q_resolved,
        volume_domain="real",
        disc_type=disc_type,
        image_indices=image_indices,
        relion_texture_interp=relion_texture_interp,
    )
    blocks = list(block_iter)
    if len(blocks) != 1:
        raise RuntimeError(f"expected exactly one block, got {len(blocks)}")
    block = blocks[0]
    _stats, diagnostics = dense_pose_ppca_E_step_blocked(
        block.Y1,
        block.proj_aug,
        block.ctf2_over_noise,
        block.y_norm,
        block.pose_log_prior,
    )
    jax.block_until_ready(diagnostics.logZ)

    best_rot = np.asarray(diagnostics.best_rotation_idx, dtype=np.int64)
    best_trans = np.asarray(diagnostics.best_translation_idx, dtype=np.int64)
    expected_rot = np.arange(n, dtype=np.int64)
    expected_trans = np.asarray(inverse, dtype=np.int64)
    rot_exact = best_rot == expected_rot
    trans_exact = best_trans == expected_trans
    rot_err = _rotation_error_deg(rotations[best_rot], rotations[expected_rot])
    trans_err = np.linalg.norm(translations[best_trans] - gt_trans, axis=1)

    result = {
        "passed": bool(np.all(rot_exact) and np.all(trans_exact)),
        "data_star": data_star,
        "simulation_info": simulation_info_path,
        "init_npz": init_npz,
        "n_images_checked": int(n),
        "q": q_resolved,
        "current_size": None if current_size is None else int(current_size),
        "disc_type": disc_type,
        "relion_texture_interp": bool(relion_texture_interp),
        "n_candidate_rotations": int(rotations.shape[0]),
        "n_candidate_translations": int(translations.shape[0]),
        "rotation_exact_fraction": float(np.mean(rot_exact)),
        "translation_exact_fraction": float(np.mean(trans_exact)),
        "rotation_error_deg_median": float(np.median(rot_err)),
        "rotation_error_deg_p90": float(np.percentile(rot_err, 90)),
        "rotation_error_deg_max": float(np.max(rot_err)),
        "translation_error_px_median": float(np.median(trans_err)),
        "translation_error_px_p90": float(np.percentile(trans_err, 90)),
        "translation_error_px_max": float(np.max(trans_err)),
        "pmax_min": float(np.min(np.asarray(diagnostics.pmax))),
        "pmax_mean": float(np.mean(np.asarray(diagnostics.pmax))),
        "nsig_mean": float(np.mean(np.asarray(diagnostics.n_significant_per_image))),
        "logZ_mean": float(np.mean(np.asarray(diagnostics.logZ))),
        "first_failures": [
            {
                "image": int(i),
                "best_rotation_idx": int(best_rot[i]),
                "expected_rotation_idx": int(expected_rot[i]),
                "best_translation_idx": int(best_trans[i]),
                "expected_translation_idx": int(expected_trans[i]),
                "rotation_error_deg": float(rot_err[i]),
                "translation_error_px": float(trans_err[i]),
                "pmax": float(np.asarray(diagnostics.pmax)[i]),
            }
            for i in np.flatnonzero(~(rot_exact & trans_exact))[:10]
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-star", required=True)
    parser.add_argument("--simulation-info", required=True)
    parser.add_argument("--init-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--n-images", type=int, default=128)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--current-size", type=int, default=None)
    parser.add_argument("--disc-type", default="linear_interp")
    parser.add_argument("--relion-texture-interp", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    result = run_check(
        data_star=Path(args.data_star),
        simulation_info_path=Path(args.simulation_info),
        init_npz=Path(args.init_npz),
        output_json=Path(args.output_json),
        n_images=int(args.n_images),
        q=args.q,
        current_size=args.current_size,
        disc_type=str(args.disc_type),
        relion_texture_interp=bool(args.relion_texture_interp),
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
