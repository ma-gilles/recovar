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

from recovar.em.sampling import get_rotation_grid_at_order, get_translation_grid
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


def _gt_pca_from_assignments(volume_paths: list[Path], assignments: np.ndarray, q: int):
    volumes = np.stack([_load_mrc(path) for path in volume_paths], axis=0)
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


def _pose_checks(simulation_info: dict[str, Any], result_npz, healpix_order: int, offset_range_px: float, offset_step_px: float):
    best_rot_idx = np.asarray(result_npz["best_rotation_idx"], dtype=np.int64)
    best_trans_idx = np.asarray(result_npz["best_translation_idx"], dtype=np.int64)
    gt_rots = np.asarray(simulation_info["rots"], dtype=np.float64)[: best_rot_idx.shape[0]]
    gt_trans = np.asarray(simulation_info["trans"], dtype=np.float64)[: best_trans_idx.shape[0]]
    rotations = np.asarray(get_rotation_grid_at_order(int(healpix_order)), dtype=np.float64)
    translations = np.asarray(get_translation_grid(float(offset_range_px), float(offset_step_px)), dtype=np.float64)
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
        "rotation_error_deg_median": float(np.median(rot_err)),
        "rotation_error_deg_p90": float(np.percentile(rot_err, 90)),
        "rotation_error_deg_mean": float(np.mean(rot_err)),
        "translation_error_px_median": float(np.median(trans_err)),
        "translation_error_px_p90": float(np.percentile(trans_err, 90)),
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


def run_checks(
    *,
    run_dir: Path,
    simulation_info_path: Path,
    volume_glob: str,
    healpix_order: int,
    offset_range_px: float,
    offset_step_px: float,
    q: int,
):
    with simulation_info_path.open("rb") as f:
        simulation_info = pickle.load(f)
    result = np.load(run_dir / "final_ppca_dense.npz")
    embedding_z = np.load(run_dir / "embedding_best_pose" / "embedding_z.npy")
    assignments = np.asarray(simulation_info["image_assignment"], dtype=np.int64)[: embedding_z.shape[0]]
    volume_paths = [Path(path) for path in sorted(glob.glob(volume_glob))]
    if not volume_paths:
        raise ValueError(f"no volumes matched {volume_glob!r}")
    gt_mean, gt_pcs, gt_scores, _volumes = _gt_pca_from_assignments(volume_paths, assignments, int(q))

    mu = _load_mrc(run_dir / "final_mu.mrc")
    W = np.stack([_load_mrc(run_dir / f"final_W{i + 1:02d}.mrc") for i in range(int(q))], axis=0)
    W_flat = W.reshape(int(q), -1)
    gt_pc_flat = gt_pcs[: int(q)]
    Wq = _orthonormal_rows(W_flat)
    GTq = _orthonormal_rows(gt_pc_flat)
    subspace_cosines = np.linalg.svd(Wq @ GTq.T, compute_uv=False) if Wq.size and GTq.size else np.zeros((0,))

    checks = {
        "run_dir": run_dir,
        "simulation_info": simulation_info_path,
        "volume_glob": volume_glob,
        "noise_level": float(simulation_info.get("noise_level", float("nan"))),
        "n_images": int(embedding_z.shape[0]),
        "q": int(q),
        "pose": _pose_checks(simulation_info, result, healpix_order, offset_range_px, offset_step_px),
        "maps": {
            "mu_vs_empirical_gt_mean_corr": _corr(mu, gt_mean),
            "W_vs_gt_pc_subspace_cosines": subspace_cosines,
            "W_vs_gt_pc_subspace_mean_cosine": float(np.mean(subspace_cosines)) if subspace_cosines.size else float("nan"),
            "mu_rms": float(np.sqrt(np.mean(mu**2))),
            "W_rms": float(np.sqrt(np.mean(W**2))),
            "gt_mean_rms": float(np.sqrt(np.mean(gt_mean**2))),
        },
        "embedding": _embedding_checks(embedding_z, gt_scores),
    }
    output_path = run_dir / "synthetic_recovery_checks.json"
    output_path.write_text(json.dumps(_jsonable(checks), indent=2, sort_keys=True) + "\n")
    return _jsonable(checks)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--simulation-info", required=True)
    parser.add_argument("--volume-glob", required=True)
    parser.add_argument("--healpix-order", type=int, default=3)
    parser.add_argument("--offset-range-px", type=float, default=6.0)
    parser.add_argument("--offset-step-px", type=float, default=2.0)
    parser.add_argument("--q", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checks = run_checks(
        run_dir=Path(args.run_dir),
        simulation_info_path=Path(args.simulation_info),
        volume_glob=args.volume_glob,
        healpix_order=int(args.healpix_order),
        offset_range_px=float(args.offset_range_px),
        offset_step_px=float(args.offset_step_px),
        q=int(args.q),
    )
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
