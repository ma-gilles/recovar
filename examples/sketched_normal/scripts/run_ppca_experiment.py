"""Run PPCA EM on one dataset and output JSON comparable to run_experiment.py.

Outputs {rel_var, rel_var_per_pc, rv@1..10 (real-space-only via metrics), iteration_data, ...}.
"""

import argparse
import json
import os
import sys
import time
import warnings

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
warnings.filterwarnings("ignore", module="finufft")
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from recovar import utils  # noqa: E402
from recovar.output import metrics  # noqa: E402
from recovar.ppca import ppca as ppca_mod  # noqa: E402
from recovar.ppca.ppca_scale_sweep import (  # noqa: E402
    _load_simulated_dataset,
    _with_trailing_separator,
    warmstart_from_pca,
)
from recovar.reconstruction import homogeneous  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--grid-size", type=int, required=True)
    ap.add_argument("--n-images", type=int, required=True)
    ap.add_argument("--basis-size", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--prior-scale", type=float, default=1.0, help="multiplier on W_prior_base")
    ap.add_argument("--use-whitening", action="store_true")
    ap.add_argument("--whitening-mode", choices=["cz", "proj_ls"], default="cz")
    ap.add_argument("--mean-batch-size", type=int, default=500)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cryos, sim_info, gt, noise_variance = _load_simulated_dataset(
        _with_trailing_separator(args.dataset_dir), args.grid_size, args.n_images, lazy=False
    )
    vs = cryos.volume_shape
    print(f"[ds] grid={args.grid_size} n={int(cryos.n_images)} V={int(np.prod(vs))}", flush=True)

    t_mean = time.time()
    noise_var_image = utils.make_radial_image(noise_variance, cryos.image_shape)
    means, _mean_prior, _fsc = homogeneous.get_mean_conformation_relion(
        cryos, args.mean_batch_size, noise_variance=noise_var_image, use_regularization=False
    )
    mean_estimate = means.combined.flatten()
    gt_mean = gt.get_mean()
    mean_err = float(np.linalg.norm(np.asarray(mean_estimate) - gt_mean.flatten()) / np.linalg.norm(gt_mean))
    print(f"[mean] err={mean_err:.4f}  [{time.time() - t_mean:.1f}s]", flush=True)

    t_ws = time.time()
    W_init, W_prior_base, U_gt, s_gt, pca_results = warmstart_from_pca(
        cryos, means, gt, args.basis_size, batch_size=100, gpu_memory=40
    )
    print(f"[warmstart] PCA rv={pca_results['rel_var']:.4f}  [{time.time() - t_ws:.1f}s]", flush=True)

    W_prior = args.prior_scale * W_prior_base

    t_em = time.time()
    em_output = ppca_mod.EM(
        cryos,
        mean_estimate,
        W_init.copy(),
        W_prior,
        U_gt=U_gt,
        S_gt=s_gt**2,
        EM_iter=args.n_iter,
        use_whitening=args.use_whitening,
        whitening_mode=args.whitening_mode,
        sparse_PCA=False,
        disc_type_mean="cubic",
        disc_type="linear_interp",
        return_iteration_data=True,
    )
    u_fin, _s_fin, W_fin, ez, sm, iteration_data = em_output
    print(f"[em] done in {time.time() - t_em:.1f}s", flush=True)

    _, rel_var_per_pc, _ = metrics.get_all_variance_scores(u_fin, U_gt, s_gt**2)
    rel_var_per_pc = np.asarray(rel_var_per_pc)
    final = {
        "rel_var_mean": float(np.mean(rel_var_per_pc)),
        "rel_var_per_pc": [float(x) for x in rel_var_per_pc],
    }
    for k in (1, 2, 5, 10):
        kk = min(k, len(rel_var_per_pc))
        final[f"rv@{k}"] = float(rel_var_per_pc[kk - 1])

    # Full-GT s² metric for apples-to-apples comparison with sketched runner.
    # The returned U_gt/s_gt from warmstart_from_pca are truncated to basis_size; re-fetch the full spectrum.
    U_gt_full, s_gt_full, _ = gt.get_vol_svd()
    U_gt_full = np.asarray(U_gt_full).astype(np.complex64)
    s_gt_full = np.asarray(s_gt_full).astype(np.float64)
    s2_total = float((s_gt_full**2).sum())
    u_fin_np = np.asarray(u_fin).astype(np.complex64)
    # u_fin is Fourier-domain; project each GT dir onto first-k learned PCs.
    n_learned = u_fin_np.shape[1]
    for k in (1, 2, 5, 10):
        kk = min(k, n_learned)
        M = np.conj(u_fin_np[:, :kk].T) @ U_gt_full  # (kk, K)
        coeff_sq = (np.abs(M) ** 2).sum(axis=0)
        num = float((coeff_sq.astype(np.float64) * (s_gt_full**2)).sum())
        final[f"rv_s2@{k}"] = num / max(s2_total, 1e-30)

    history = []
    for i, d in enumerate(iteration_data or []):
        row = {"it": i + 1}
        if "rel_var_per_pc" in d:
            rv = np.asarray(d["rel_var_per_pc"])
            for k in (1, 2, 5, 10):
                kk = min(k, len(rv))
                row[f"rv@{k}"] = float(rv[kk - 1])
            row["rel_var_mean"] = float(np.mean(rv))
        history.append(row)

    result = {
        "config": vars(args),
        "method": "ppca_em",
        "final": final,
        "pca_warmstart_rv": float(pca_results["rel_var"]),
        "mean_estimation_error": mean_err,
        "history": history,
        "s_gt_top10": [float(x) for x in np.asarray(s_gt)[:10]],
        "elapsed_total_s": time.time() - t0,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[done] wrote {args.output}  total {time.time() - t0:.1f}s  final rv@10={final.get('rv@10'):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
