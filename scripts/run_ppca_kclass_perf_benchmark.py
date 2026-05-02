#!/usr/bin/env python
"""Quality + perf benchmark: PPCA pose-marginal refinement vs k-class
refinement on a big Ribosembly dataset (256³ × 100k images, 16 classes).

Produces:

  ``benchmark_summary.json``
      runtime, FSC area mean (Hungarian), FSC@0.5, fair-mode subspace
      score for PPCA, per-iter wall time breakdown.

  ``benchmark_runs/<method>/iter_NNN/``
      per-iter μ/W (PPCA) or class-{NN} (k-class) MRCs.

  ``benchmark_summary_table.txt``
      rendered side-by-side comparison.

Usage::

    python run_ppca_kclass_perf_benchmark.py \\
        --data-dir /scratch/.../ribosembly_allk_g256_n100000_snr1 \\
        --output-dir /scratch/.../_agent_scratch/ppca_kclass_perf_2026_05_01 \\
        --n-images 100000 --grid-size 256 --n-pcs 10 --em-iters 15 \\
        --healpix-order 2 --image-batch-size 16 --rotation-block-size 64

The dataset is the same one used by the k-class ↔ RELION parity benchmark
(``scripts/run_cryobench_ribosembly_parity_slurm.sh``); reuse it so we
compare both methods on identical inputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class BenchSpec:
    data_dir: Path
    output_dir: Path
    n_images: int = 100_000
    grid_size: int = 256
    n_pcs: int = 10
    n_classes_kclass: int = 10
    em_iters: int = 15
    healpix_order: int = 2
    image_batch_size: int = 16
    rotation_block_size: int = 64
    seed: int = 42


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------


def load_cryoem_dataset(spec: BenchSpec):
    """Load the prepared parity dataset as a CryoEMDataset.

    The benchmark always uses the on-disk grid (256³); ``spec.grid_size``
    is read from the dataset rather than passed in. ``downsample_D`` is
    only used if the caller requests a smaller grid than what's on disk.
    """
    from recovar.data_io.cryoem_dataset import load_dataset

    star = spec.data_dir / "particles.star"
    if not star.exists():
        raise FileNotFoundError(f"Dataset star not found: {star}")
    return load_dataset(
        particles_file=str(star),
        n_images=spec.n_images,
        ind=None,
        lazy=True,
        padding=0,
    )


def load_gt_volumes(spec: BenchSpec) -> np.ndarray:
    """Read all reference_gt_class*.mrc volumes; returns ``[K, D, D, D]``
    real32 at the on-disk grid (caller downsamples if needed)."""
    import mrcfile

    gt_paths = sorted(spec.data_dir.glob("reference_gt_class*.mrc"))
    vols = []
    for p in gt_paths:
        with mrcfile.open(str(p), permissive=True) as mrc:
            vols.append(mrc.data.copy().astype(np.float32))
    return np.stack(vols, axis=0)


# ---------------------------------------------------------------------------
# K-class refinement run
# ---------------------------------------------------------------------------


def run_kclass(spec: BenchSpec, cryo, gt_vols: np.ndarray) -> dict:
    """Multi-iter dense k-class EM, initialised from a downsampled subset
    of the GT volumes (first ``n_classes_kclass``). Mirrors the iter
    pattern from ``scripts/ppca_refine_eval.py::_run_kclass_refinement``.
    """
    import jax.numpy as jnp
    import mrcfile

    import recovar.core.fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.k_class import run_dense_k_class_em
    from recovar.em.sampling import get_rotation_grid

    out_dir = spec.output_dir / "benchmark_runs" / "kclass"
    out_dir.mkdir(parents=True, exist_ok=True)

    K = spec.n_classes_kclass
    vol_shape = (cryo.grid_size,) * 3
    vol_size = int(np.prod(vol_shape))
    image_size = int(np.prod(cryo.image_shape))

    # Initialise: first K GT volumes (already at on-disk resolution).
    vols_init = gt_vols[:K]
    means = jnp.stack(
        [ftu.get_dft3(jnp.asarray(v, dtype=jnp.float32)).reshape(vol_size) for v in vols_init],
        axis=0,
    ).astype(jnp.complex64)

    rotations = jnp.asarray(get_rotation_grid(spec.healpix_order, matrices=True), dtype=jnp.float32)
    translations = jnp.zeros((1, 2), dtype=jnp.float32)
    noise_variance = jnp.ones((image_size,), dtype=jnp.float32)
    mean_variance = jnp.ones((vol_size,), dtype=jnp.float32)

    per_iter_walls = []
    log_evidences = []
    t_total0 = time.time()
    for it in range(spec.em_iters):
        t0 = time.time()
        result = run_dense_k_class_em(
            cryo,
            means,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            disc_type="linear_interp",
        )
        means = jnp.asarray(result.new_means)
        per_iter_walls.append(time.time() - t0)
        try:
            log_evidences.append(float(jnp.sum(result.stats.log_evidence_per_image)))
        except Exception:
            log_evidences.append(float("nan"))

        iter_dir = out_dir / f"iter_{it:03d}"
        iter_dir.mkdir(exist_ok=True)
        for k in range(K):
            real = np.real(np.asarray(ftu.get_idft3(means[k].reshape(vol_shape)))).astype(np.float32)
            with mrcfile.new(str(iter_dir / f"class_{k:02d}.mrc"), overwrite=True) as mrc:
                mrc.set_data(real)
                mrc.voxel_size = cryo.voxel_size
    runtime_s = time.time() - t_total0

    final_real = np.stack(
        [np.real(np.asarray(ftu.get_idft3(means[k].reshape(vol_shape)))).astype(np.float32) for k in range(K)]
    )
    np.save(out_dir / "final_classes_real.npy", final_real)
    for k in range(K):
        with mrcfile.new(str(out_dir / f"class_{k:02d}.mrc"), overwrite=True) as mrc:
            mrc.set_data(final_real[k])
            mrc.voxel_size = cryo.voxel_size

    return {
        "method": "kclass",
        "n_classes": K,
        "runtime_s": runtime_s,
        "per_iter_walls_s": per_iter_walls,
        "log_evidence_history": log_evidences,
        "n_iters": spec.em_iters,
        **score_fsc_hungarian(final_real, gt_vols, cryo.grid_size),
    }


# ---------------------------------------------------------------------------
# PPCA mature pipeline run
# ---------------------------------------------------------------------------


def init_ppca_state_from_kclass(
    spec: BenchSpec,
    kclass_result_path: Path,
):
    """Initialize PPCA state by:

      μ      = mean of k-class final volumes
      W_k    = top-q PCA components of (class_vols - μ) / ||·||
      noise  = ones (per-half-volume voxel)
      mean_prior = ones

    This matches the user's earlier instruction: 'mean of input vols +
    PCA the init vols and use that as init'.
    """
    import jax.numpy as jnp
    import numpy as np

    import recovar.core.fourier_transform_utils as ftu
    from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState

    final_real = np.load(str(kclass_result_path / "final_classes_real.npy"))  # (K, D, D, D)
    K, D, _, _ = final_real.shape
    mu = final_real.mean(axis=0).astype(np.float32)
    centered = final_real - mu  # (K, D, D, D)
    flat = centered.reshape(K, -1)
    # Top-q principal directions of class-mean variation.
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    q = spec.n_pcs
    W_init = (Vt[:q].reshape(q, D, D, D) * S[:q, None, None, None]).astype(np.float32)

    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape((D, D, D))))
    return PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu), jnp.asarray(mu)),
        W_half=(jnp.asarray(W_init), jnp.asarray(W_init)),
        mu_score=jnp.asarray(mu),
        W_score=jnp.asarray(W_init),
        W_prior=jnp.full((half_vol, q), 1.0, dtype=jnp.float32),
        mean_prior=jnp.full((half_vol,), 1.0, dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((q,), dtype=jnp.float32),
        noise_variance=jnp.ones((half_vol,), dtype=jnp.float32),
        contrast_params=None,
        masks=None,
        pose_estimates={},
        pose_priors=None,
        refinement_schedule_state=None,
        hyperparams=None,
    )


def run_ppca(spec: BenchSpec, cryo, gt_vols: np.ndarray, kclass_result_path: Path) -> dict:
    """Run the mature PPCA pipeline on the same dataset."""
    import jax.numpy as jnp
    import mrcfile
    import numpy as np

    from recovar.em.ppca_refinement.iterations import IterationOpts
    from recovar.em.ppca_refinement.refinement_loop import (
        PPCAScheduleOpts,
        run_pose_marginal_refinement,
    )

    out_dir = spec.output_dir / "benchmark_runs" / "ppca"
    out_dir.mkdir(parents=True, exist_ok=True)

    state = init_ppca_state_from_kclass(spec, kclass_result_path)

    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )

    schedule = PPCAScheduleOpts(
        healpix_order_init=spec.healpix_order,
        healpix_order_max=max(spec.healpix_order, 4),
        max_iters=spec.em_iters,
        min_iters=min(spec.em_iters, 5),
        enable_low_resol_join=True,
        enable_per_iter_prior=True,
        enable_per_iter_noise=True,
        enable_x0_hermitian=True,
    )

    iter_log = []
    per_iter_walls = []
    last_t = time.time()

    def cb(it, st, info):
        nonlocal last_t
        now = time.time()
        per_iter_walls.append(now - last_t)
        last_t = now
        iter_log.append(info)
        iter_dir = out_dir / f"iter_{it:03d}"
        iter_dir.mkdir(exist_ok=True)
        with mrcfile.new(str(iter_dir / "mu_score.mrc"), overwrite=True) as mrc:
            mrc.set_data(np.asarray(st.mu_score, dtype=np.float32))
            mrc.voxel_size = cryo.voxel_size
        for k in range(spec.n_pcs):
            with mrcfile.new(str(iter_dir / f"W_{k:02d}_score.mrc"), overwrite=True) as mrc:
                mrc.set_data(np.asarray(st.W_score[k], dtype=np.float32))
                mrc.voxel_size = cryo.voxel_size
        with (out_dir / "iter_log.pkl").open("wb") as fh:
            pickle.dump(iter_log, fh)

    t0 = time.time()
    final_state, _ = run_pose_marginal_refinement(
        state,
        cryo,
        halfset_indices=halfset_indices,
        mask=jnp.ones((cryo.grid_size,) * 3, dtype=jnp.float32),
        image_batch_size=spec.image_batch_size,
        rotation_block_size=spec.rotation_block_size,
        schedule_opts=schedule,
        iteration_opts=IterationOpts(EM_iter=spec.em_iters, pcg_maxiter=20),
        iteration_callback=cb,
    )
    runtime_s = time.time() - t0

    # Build trial volumes for FSC: standard (μ ± W_k) and fair (LSQ projection).
    mu = np.asarray(final_state.mu_score)
    W = np.asarray(final_state.W_score)
    fsc_standard = score_ppca_standard(mu, W, gt_vols, cryo.grid_size)
    fsc_fair = score_ppca_fair(mu, W, gt_vols, cryo.grid_size)

    return {
        "method": "ppca",
        "n_pcs": spec.n_pcs,
        "runtime_s": runtime_s,
        "per_iter_walls_s": per_iter_walls,
        "n_iters": spec.em_iters,
        **{f"std_{k}": v for k, v in fsc_standard.items()},
        **{f"fair_{k}": v for k, v in fsc_fair.items()},
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_fsc_hungarian(pred_vols: np.ndarray, gt_vols: np.ndarray, grid_size: int) -> dict:
    from scipy.optimize import linear_sum_assignment

    from recovar.reconstruction.regularization import get_fsc

    vol_shape = (grid_size,) * 3
    K_pred, K_gt = pred_vols.shape[0], gt_vols.shape[0]
    fsc_curves = np.zeros((K_pred, K_gt), dtype=object)
    fsc_areas = np.zeros((K_pred, K_gt), dtype=np.float32)
    for i in range(K_pred):
        for j in range(K_gt):
            fsc = np.asarray(get_fsc(pred_vols[i].reshape(-1), gt_vols[j].reshape(-1), vol_shape))
            fsc_curves[i, j] = fsc
            fsc_areas[i, j] = float(np.mean(fsc))
    row, col = linear_sum_assignment(-fsc_areas[: min(K_pred, K_gt), : min(K_pred, K_gt)])
    matched = list(zip(map(int, row), map(int, col)))
    return {
        "fsc_assignment": matched,
        "fsc_area_per_class": [float(fsc_areas[i, j]) for i, j in matched],
        "fsc_area_mean": float(np.mean([fsc_areas[i, j] for i, j in matched])),
    }


def score_ppca_standard(mu, W, gt_vols, grid_size) -> dict:
    """μ + 2q ± perturbations; Hungarian-match against GT (biased upward
    when 2q+1 > K_gt)."""
    rms_mu = float(np.sqrt(np.mean(mu**2)) + 1e-12)
    trials = [mu]
    for k in range(W.shape[0]):
        rms_W = float(np.sqrt(np.mean(W[k] ** 2)) + 1e-12)
        s = rms_mu / rms_W if rms_W > 0 else 0.0
        trials.extend([mu + s * W[k], mu - s * W[k]])
    return score_fsc_hungarian(np.stack(trials), gt_vols, grid_size)


def score_ppca_fair(mu, W, gt_vols, grid_size) -> dict:
    """For each GT class, find the best μ + W·z_k via least-squares in
    voxel space → exactly K_gt trial volumes (no combinatorial bias)."""
    mu_flat = mu.reshape(-1)
    W_flat = W.reshape(W.shape[0], -1)
    K_gt = gt_vols.shape[0]
    trials = []
    for k in range(K_gt):
        b = gt_vols[k].reshape(-1).astype(np.float32) - mu_flat
        z_k, *_ = np.linalg.lstsq(W_flat.T, b, rcond=None)
        trials.append((mu_flat + W_flat.T @ z_k).reshape(mu.shape))
    return score_fsc_hungarian(np.stack(trials), gt_vols, grid_size)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=100_000)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--n-classes-kclass", type=int, default=10)
    parser.add_argument("--em-iters", type=int, default=15)
    parser.add_argument("--healpix-order", type=int, default=2)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--rotation-block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-kclass", action="store_true", help="reuse existing benchmark_runs/kclass output")
    parser.add_argument("--skip-ppca", action="store_true", help="run only k-class")
    args = parser.parse_args()

    spec = BenchSpec(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_images=args.n_images,
        grid_size=args.grid_size,
        n_pcs=args.n_pcs,
        n_classes_kclass=args.n_classes_kclass,
        em_iters=args.em_iters,
        healpix_order=args.healpix_order,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        seed=args.seed,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    (spec.output_dir / "SAFE_TO_DELETE").touch()

    cryo = load_cryoem_dataset(spec)
    gt_vols = load_gt_volumes(spec)
    print(f"Loaded {gt_vols.shape[0]} GT classes at grid {spec.grid_size}³")

    summary = {
        "spec": {
            **asdict(spec),
            "data_dir": str(spec.data_dir),
            "output_dir": str(spec.output_dir),
        }
    }
    if not args.skip_kclass:
        summary["kclass"] = run_kclass(spec, cryo, gt_vols)
    if not args.skip_ppca:
        kclass_path = spec.output_dir / "benchmark_runs" / "kclass"
        summary["ppca"] = run_ppca(spec, cryo, gt_vols, kclass_path)

    out_json = spec.output_dir / "benchmark_summary.json"
    with out_json.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"Wrote {out_json}")

    if "kclass" in summary and "ppca" in summary:
        print()
        print("=== Quality + perf comparison ===")
        kc = summary["kclass"]
        pp = summary["ppca"]
        print(
            f"  k-class  ({kc['n_classes']} classes): runtime={kc['runtime_s']:.0f}s  "
            f"fsc_area_mean={kc['fsc_area_mean']:.4f}"
        )
        print(
            f"  PPCA std ({pp['n_pcs']} PCs):     runtime={pp['runtime_s']:.0f}s  "
            f"fsc_area_mean={pp['std_fsc_area_mean']:.4f}"
        )
        print(f"  PPCA fair (LSQ projection):     fair_fsc_area_mean={pp['fair_fsc_area_mean']:.4f}")
        if pp.get("per_iter_walls_s"):
            walls = pp["per_iter_walls_s"]
            print(
                f"  PPCA per-iter walls (s): "
                f"min={min(walls):.1f} median={sorted(walls)[len(walls) // 2]:.1f} max={max(walls):.1f}"
            )


if __name__ == "__main__":
    main()
