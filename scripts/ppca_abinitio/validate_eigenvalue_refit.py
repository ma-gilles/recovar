"""Phase 2 validation: does post-EM ProjCov refit improve eigenvalue
calibration vs the flat-s default?

Runs `run_two_stage` once per dataset on the same synthetic harness
as `run_cryobench.py`, applies the refit, and compares the calibration
ratio  s_refit[k] / s_true[k]  vs the baseline s_em / s_true.

Calibration metric (lower is better, 0 = perfect calibration):
    err = mean_k |log(s_est[k] / s_true[k])|

Phase 2 acceptance: err(s_refit) reduces by >2× vs err(s_em=1) on
Ribosembly + IgG-1D + IgG-RL at vol=32, n=1024, default sigmas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recovar.em.ppca_abinitio.eigenvalue_refit import refit_eigenvalues_post_em
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from scripts.ppca_abinitio.run_cryobench import (  # noqa: I001
    _Cfg,
    load_cryobench_gt_volumes,
    run_two_stage,
)


def calibration_error(s_est: np.ndarray, s_true: np.ndarray) -> float:
    """mean_k |log(s_est_k / s_true_k)|. 0 = perfect."""
    s_est = np.asarray(s_est, dtype=np.float64)
    s_true = np.asarray(s_true, dtype=np.float64)
    s_est = np.maximum(s_est, 1e-30)
    s_true = np.maximum(s_true, 1e-30)
    return float(np.mean(np.abs(np.log(s_est / s_true))))


def run_one(dataset_name: str, q: int, sigma: float, n_joint: int, seed: int):
    volume_shape = (32, 32, 32)
    image_shape = (32, 32)
    grid = build_fixed_grid(healpix_order=1, max_shift=1)

    gt_root = Path("/home/mg6942/mytigress/cryobench2") / dataset_name
    gt_vols = load_cryobench_gt_volumes(gt_root, target_D=32)

    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=volume_shape,
        image_shape=image_shape,
        grid=grid,
        q=q,
        n_images_train=1024,
        n_images_val=0,
        sigma_real=sigma,
        seed=seed,
        external_volumes_real=gt_vols,
        external_sampling_mode="discrete_volumes",
    )
    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)

    final, *_ = run_two_stage(
        cfg,
        ds,
        q=q,
        n_burnin=0,
        n_joint=n_joint,
        mu_init_kind="perturbed",
        u_init_kind="svd",
        weighted_svd=True,
        anneal_schedule="none",
        seed=seed,
        s_init_kind="flat",
    )
    refit_state, info = refit_eigenvalues_post_em(final, cfg, ds)

    s_true = np.asarray(ds.s_true, dtype=np.float64)
    s_em = np.asarray(info.s_em, dtype=np.float64)  # ones
    s_refit = np.asarray(info.s_refit, dtype=np.float64)

    # Top-q true variances, descending. ds.s_true is a (q,) array of
    # per-PC empirical std's of the GT ensemble in the U_true gauge.
    s_true_sorted = np.sort(s_true)[::-1]
    s_refit_sorted = np.sort(s_refit)[::-1]

    return {
        "dataset": dataset_name,
        "q": q,
        "sigma": sigma,
        "seed": seed,
        "s_true": s_true_sorted.tolist(),
        "s_em": s_em.tolist(),
        "s_refit": s_refit_sorted.tolist(),
        "err_em": calibration_error(s_em, s_true_sorted),
        "err_refit": calibration_error(s_refit_sorted, s_true_sorted),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="JSON output path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = [
        ("Ribosembly", 4, 0.01, 30),
        ("IgG-1D", 2, 0.1, 30),
        ("IgG-RL", 2, 0.1, 30),
    ]
    results = []
    for dataset, q, sigma, n_joint in cases:
        print(f"\n=== {dataset} q={q} sigma={sigma} ===", flush=True)
        r = run_one(dataset, q, sigma, n_joint, args.seed)
        results.append(r)
        print(f"  s_true:  {[f'{v:.4g}' for v in r['s_true']]}")
        print(f"  s_em:    {[f'{v:.4g}' for v in r['s_em']]}")
        print(f"  s_refit: {[f'{v:.4g}' for v in r['s_refit']]}")
        print(f"  err_em (flat)     = {r['err_em']:.4f}")
        print(f"  err_refit (proj)  = {r['err_refit']:.4f}")
        ratio = r["err_em"] / max(r["err_refit"], 1e-12)
        print(f"  improvement       = {ratio:.2f}×")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")

    # Phase 2 acceptance check
    print("\n=== Phase 2 acceptance: err(refit) <= err(em) / 2 on all 3 datasets ===")
    pass_count = 0
    for r in results:
        passed = r["err_refit"] <= r["err_em"] / 2.0
        symbol = "PASS" if passed else "FAIL"
        print(f"  {symbol} {r['dataset']} q={r['q']}: err_em={r['err_em']:.4f}, err_refit={r['err_refit']:.4f}")
        pass_count += int(passed)
    print(f"\n{pass_count}/3 datasets passed.")


if __name__ == "__main__":
    main()
