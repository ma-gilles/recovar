#!/usr/bin/env python
"""Check a synthetic PPCA refinement run against simulator ground truth."""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import jax.numpy as jnp

from recovar.core import fourier_transform_utils as ftu
from recovar.em.sampling import get_rotation_grid_at_order, get_translation_grid
from recovar.simulation import synthetic_dataset
from recovar.utils import helpers


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


def _load_mrc(path: Path) -> np.ndarray:
    data, _voxel_size = helpers.load_mrc(path, return_voxel_size=True)
    return np.asarray(data, dtype=np.float64)


def _half_volume_to_real(half_volume, volume_shape) -> np.ndarray:
    full = ftu.half_volume_to_full_volume(jnp.asarray(half_volume), tuple(volume_shape))
    return np.asarray(ftu.get_idft3(full.reshape(tuple(volume_shape))).real, dtype=np.float64)


def _load_estimated_maps(run_dir: Path, result_npz, q: int) -> tuple[np.ndarray, np.ndarray, str]:
    mu_mrc = run_dir / "final_mu.mrc"
    if mu_mrc.exists():
        return (
            _load_mrc(mu_mrc),
            np.stack([_load_mrc(run_dir / f"final_W{i + 1:02d}.mrc") for i in range(int(q))], axis=0),
            "mrc",
        )

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{mu_mrc} is missing and {summary_path} is unavailable; cannot infer volume_shape for final npz maps"
        )
    summary = json.loads(summary_path.read_text())
    volume_shape = tuple(int(x) for x in summary["volume_shape"])
    mu = _half_volume_to_real(result_npz["mu_half"], volume_shape)
    W_half = np.asarray(result_npz["W_half"])
    W = np.stack([_half_volume_to_real(W_half[:, i], volume_shape) for i in range(int(q))], axis=0)
    return mu, W, "final_ppca_dense_npz"


def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _orthonormal_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    keep = np.linalg.norm(matrix, axis=1) > 0
    if not np.any(keep):
        return np.zeros((0, matrix.shape[1]), dtype=np.float64)
    q, _r = np.linalg.qr(matrix[keep].T, mode="reduced")
    return q.T


def _gt_pca_from_volume_stack(volumes: np.ndarray, assignments: np.ndarray, q: int):
    volumes = np.asarray(volumes, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=np.int64)
    flat = volumes.reshape(volumes.shape[0], -1)
    counts = np.bincount(assignments, minlength=volumes.shape[0]).astype(np.float64)
    weights = counts / max(float(np.sum(counts)), 1.0)
    mean = np.sum(flat * weights[:, None], axis=0)
    centered = flat - mean
    weighted_centered = centered * np.sqrt(weights[:, None])
    _u, _s, vt = np.linalg.svd(weighted_centered, full_matrices=False)
    pcs = vt[:q]
    scores_by_volume = centered @ pcs.T
    scores = scores_by_volume[assignments]
    return mean.reshape(volumes.shape[1:]), pcs, scores, volumes


def _gt_pca_from_assignments(volume_paths: list[Path], assignments: np.ndarray, q: int):
    volumes = np.stack([_load_mrc(path) for path in volume_paths], axis=0)
    return _gt_pca_from_volume_stack(volumes, assignments, q)


def _gt_pca_from_simulation_info(simulation_info: dict[str, Any], assignments: np.ndarray, q: int):
    heterogeneous = synthetic_dataset.load_heterogeneous_reconstruction(simulation_info)
    volume_shape = tuple(int(x) for x in heterogeneous.volume_shape)
    volumes_fourier = np.asarray(heterogeneous.volumes).reshape((-1,) + volume_shape)
    volumes = np.stack(
        [np.asarray(ftu.get_idft3(volume).real, dtype=np.float64) for volume in volumes_fourier],
        axis=0,
    )
    return _gt_pca_from_volume_stack(volumes, assignments, q)


def _pose_checks(
    simulation_info: dict[str, Any],
    result_npz,
    rotation_source: str,
    healpix_order: int,
    offset_range_px: float,
    offset_step_px: float,
    translation_source: str,
):
    if rotation_source == "result-matrices":
        if "best_rotation_matrix" not in result_npz:
            raise ValueError("rotation_source=result-matrices requires best_rotation_matrix in result npz")
        best_rot_idx = np.arange(np.asarray(result_npz["best_rotation_matrix"]).shape[0], dtype=np.int64)
    else:
        best_rot_idx = np.asarray(result_npz["best_rotation_idx"], dtype=np.int64)
    best_trans_idx = np.asarray(result_npz["best_translation_idx"], dtype=np.int64)
    gt_rots = np.asarray(simulation_info["rots"], dtype=np.float64)[: best_rot_idx.shape[0]]
    gt_trans = np.asarray(simulation_info["trans"], dtype=np.float64)[: best_trans_idx.shape[0]]
    if rotation_source == "result-matrices":
        rotations = np.asarray(result_npz["best_rotation_matrix"], dtype=np.float64)
    elif rotation_source == "healpix":
        rotations = np.asarray(get_rotation_grid_at_order(int(healpix_order)), dtype=np.float64)
    elif rotation_source == "simulation-info":
        rotations = gt_rots
    elif rotation_source == "simulation-info-plus-healpix":
        rotations = np.concatenate(
            [gt_rots, np.asarray(get_rotation_grid_at_order(int(healpix_order)), dtype=np.float64)],
            axis=0,
        )
    else:
        raise ValueError(
            "rotation_source must be 'healpix', 'simulation-info', "
            "'simulation-info-plus-healpix', or 'result-matrices'"
        )
    if translation_source == "result-vectors":
        if "best_translation" not in result_npz:
            raise ValueError("translation_source=result-vectors requires best_translation in result npz")
        translations = np.asarray(result_npz["best_translation"], dtype=np.float64)
        best_trans_idx = np.arange(translations.shape[0], dtype=np.int64)
    elif translation_source == "grid":
        translations = np.asarray(get_translation_grid(float(offset_range_px), float(offset_step_px)), dtype=np.float64)
    elif translation_source == "simulation-info-unique":
        translations = np.unique(gt_trans, axis=0)
    else:
        raise ValueError("translation_source must be 'grid' or 'simulation-info-unique'")
    est_rots = rotations[best_rot_idx]
    est_trans = translations[best_trans_idx]

    def angular_error_deg(lhs, rhs):
        traces = np.einsum("nij,nij->n", lhs, rhs)
        cosang = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    candidates = {
        "est_vs_gt": angular_error_deg(est_rots, gt_rots),
        "estT_vs_gt": angular_error_deg(np.swapaxes(est_rots, -1, -2), gt_rots),
        "est_vs_gtT": angular_error_deg(est_rots, np.swapaxes(gt_rots, -1, -2)),
    }
    best_name = min(candidates, key=lambda key: float(np.median(candidates[key])))
    rot_err = candidates[best_name]
    trans_err = np.linalg.norm(est_trans - gt_trans, axis=1)
    zero_trans_idx = int(np.argmin(np.linalg.norm(translations, axis=1)))
    return {
        "rotation_convention": best_name,
        "rotation_source": rotation_source,
        "rotation_error_deg_median": float(np.median(rot_err)),
        "rotation_error_deg_p90": float(np.percentile(rot_err, 90)),
        "rotation_error_deg_mean": float(np.mean(rot_err)),
        "translation_error_px_median": float(np.median(trans_err)),
        "translation_error_px_p90": float(np.percentile(trans_err, 90)),
        "translation_source": translation_source,
        "zero_translation_grid_index": zero_trans_idx,
        "zero_translation_fraction": float(np.mean(best_trans_idx == zero_trans_idx)),
    }


def _embedding_checks(embedding_z: np.ndarray, gt_scores: np.ndarray):
    z = np.asarray(embedding_z, dtype=np.float64)
    y = np.asarray(gt_scores[:, : z.shape[1]], dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    y_hat = z @ coef
    ss_res = np.sum((y - y_hat) ** 2, axis=0)
    ss_tot = np.sum(y**2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, np.finfo(np.float64).eps)
    zq = _orthonormal_rows(z.T)
    yq = _orthonormal_rows(y.T)
    singular_values = np.linalg.svd(zq @ yq.T, compute_uv=False) if zq.size and yq.size else np.zeros((0,))
    return {
        "linear_r2_estimated_z_to_gt_pcs": r2,
        "embedding_gt_pc_subspace_cosines": singular_values,
        "embedding_gt_pc_subspace_mean_cosine": float(np.mean(singular_values)) if singular_values.size else float("nan"),
    }


def _linear_r2(features: np.ndarray, targets: np.ndarray):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}")
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError(f"targets must be 1D or 2D, got shape {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"feature/target row mismatch: {x.shape[0]} vs {y.shape[0]}")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ coef
    ss_res = np.sum((y - y_hat) ** 2, axis=0)
    ss_tot = np.sum(y**2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, np.finfo(np.float64).eps)


def _assignment_and_contrast_checks(embedding_z: np.ndarray, simulation_info: dict[str, Any]):
    z = np.asarray(embedding_z, dtype=np.float64)
    assignments = np.asarray(simulation_info["image_assignment"], dtype=np.int64)[: z.shape[0]]
    n_classes = int(np.max(assignments)) + 1 if assignments.size else 0
    onehot = np.eye(n_classes, dtype=np.float64)[assignments] if n_classes else np.zeros((z.shape[0], 0))
    assignment_r2 = _linear_r2(z, onehot) if onehot.shape[1] else np.zeros((0,), dtype=np.float64)
    checks = {
        "assignment_onehot_r2": assignment_r2,
        "assignment_onehot_mean_r2": float(np.mean(assignment_r2)) if assignment_r2.size else float("nan"),
        "assignment_counts": np.bincount(assignments, minlength=n_classes),
    }
    if "per_image_contrast" in simulation_info and simulation_info["per_image_contrast"] is not None:
        contrast = np.asarray(simulation_info["per_image_contrast"], dtype=np.float64)[: z.shape[0]]
        checks["contrast_r2"] = float(_linear_r2(z, contrast)[0])
    return checks


def run_checks(
    *,
    run_dir: Path,
    simulation_info_path: Path,
    volume_glob: str,
    healpix_order: int,
    rotation_source: str,
    offset_range_px: float,
    offset_step_px: float,
    q: int,
    gt_volume_source: str,
    translation_source: str,
):
    with simulation_info_path.open("rb") as f:
        simulation_info = pickle.load(f)
    result_path = run_dir / "final_ppca_local.npz"
    if not result_path.exists():
        result_path = run_dir / "final_ppca_dense.npz"
    result = np.load(result_path)
    embedding_z = np.load(run_dir / "embedding_best_pose" / "embedding_z.npy")
    assignments = np.asarray(simulation_info["image_assignment"], dtype=np.int64)[: embedding_z.shape[0]]
    if gt_volume_source == "simulation-info":
        volume_paths = []
        gt_mean, gt_pcs, gt_scores, _volumes = _gt_pca_from_simulation_info(simulation_info, assignments, int(q))
    elif gt_volume_source == "mrc-glob":
        volume_paths = [Path(path) for path in sorted(glob.glob(volume_glob))]
        if not volume_paths:
            raise ValueError(f"no volumes matched {volume_glob!r}")
        gt_mean, gt_pcs, gt_scores, _volumes = _gt_pca_from_assignments(volume_paths, assignments, int(q))
    else:
        raise ValueError("gt_volume_source must be 'simulation-info' or 'mrc-glob'")

    mu, W, estimated_map_source = _load_estimated_maps(run_dir, result, int(q))
    if tuple(mu.shape) != tuple(gt_mean.shape):
        raise ValueError(
            f"estimated mean shape {mu.shape} does not match GT mean shape {gt_mean.shape}; "
            "use --gt-volume-source simulation-info for simulator-scaled/downsampled fixtures"
        )
    W_flat = W.reshape(int(q), -1)
    gt_pc_flat = gt_pcs[: int(q)]
    Wq = _orthonormal_rows(W_flat)
    GTq = _orthonormal_rows(gt_pc_flat)
    subspace_cosines = np.linalg.svd(Wq @ GTq.T, compute_uv=False) if Wq.size and GTq.size else np.zeros((0,))

    checks = {
        "run_dir": run_dir,
        "result_npz": result_path,
        "simulation_info": simulation_info_path,
        "gt_volume_source": gt_volume_source,
        "volume_glob": volume_glob,
        "estimated_map_source": estimated_map_source,
        "noise_level": float(simulation_info.get("noise_level", float("nan"))),
        "n_images": int(embedding_z.shape[0]),
        "q": int(q),
        "pose": _pose_checks(
            simulation_info,
            result,
            rotation_source,
            healpix_order,
            offset_range_px,
            offset_step_px,
            translation_source,
        ),
        "maps": {
            "mu_vs_empirical_gt_mean_corr": _corr(mu, gt_mean),
            "W_vs_gt_pc_subspace_cosines": subspace_cosines,
            "W_vs_gt_pc_subspace_mean_cosine": float(np.mean(subspace_cosines)) if subspace_cosines.size else float("nan"),
            "mu_rms": float(np.sqrt(np.mean(mu**2))),
            "W_rms": float(np.sqrt(np.mean(W**2))),
            "gt_mean_rms": float(np.sqrt(np.mean(gt_mean**2))),
        },
        "embedding": _embedding_checks(embedding_z, gt_scores),
        "embedding_dataset_metadata": _assignment_and_contrast_checks(embedding_z, simulation_info),
    }
    output_path = run_dir / "synthetic_recovery_checks.json"
    output_path.write_text(json.dumps(_jsonable(checks), indent=2, sort_keys=True) + "\n")
    return _jsonable(checks)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--simulation-info", required=True)
    parser.add_argument("--volume-glob")
    parser.add_argument(
        "--gt-volume-source",
        choices=("simulation-info", "mrc-glob"),
        default="simulation-info",
        help="Use simulator-scaled GT volumes from simulation_info, or load explicit MRCs from --volume-glob.",
    )
    parser.add_argument("--healpix-order", type=int, default=3)
    parser.add_argument(
        "--rotation-source",
        choices=("healpix", "simulation-info", "simulation-info-plus-healpix", "result-matrices"),
        default="healpix",
    )
    parser.add_argument("--offset-range-px", type=float, default=6.0)
    parser.add_argument("--offset-step-px", type=float, default=2.0)
    parser.add_argument("--translation-source", choices=("grid", "simulation-info-unique", "result-vectors"), default="grid")
    parser.add_argument("--q", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checks = run_checks(
        run_dir=Path(args.run_dir),
        simulation_info_path=Path(args.simulation_info),
        volume_glob=args.volume_glob,
        healpix_order=int(args.healpix_order),
        rotation_source=str(args.rotation_source),
        offset_range_px=float(args.offset_range_px),
        offset_step_px=float(args.offset_step_px),
        q=int(args.q),
        gt_volume_source=str(args.gt_volume_source),
        translation_source=str(args.translation_source),
    )
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
