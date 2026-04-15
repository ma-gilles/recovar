#!/usr/bin/env python
"""Stage cryoSPARC 3DVA outputs into a recovar-style result/ dir so
ppca_refit_subspace_em.score_one_result can score it.

Conventions (verified against cov baseline, Ribosembly SNR=1):
  - eigen_pos*.mrc are real volumes normalized such that their
    Fourier-Hermitian norm is 1 (==> real L2 norm is 1/sqrt(N)).
  - latent_coords are in the Fourier-Hermitian projection units, i.e.
    z_pipeline[i,k] = <U_pipe_k, v_i - mean>_fourier
                   = sqrt(N) * <Q_k_real_unit, v_i - mean>_real.
    For 3DVA where (v_i - mean) = M @ z_old[i] and M = Q @ R (real QR
    of flattened components stacked as columns), the new coord in Q is
    R @ z_old, and in pipeline convention z = sqrt(N) * R @ z_old.
  - s = diag of sample covariance of z_pipeline.
  - mean.mrc is taken from cov baseline (cryoSPARC 3DVA re-reconstructs its
    own map at a different absolute scale; using the consensus mean that
    3DVA was seeded with puts the subspace in GT scale).
  - Optional: fit a single global scalar c by OLS so z_scaled = c * z_pipe
    matches alpha_gt (GT-leaky-by-one-scalar; user-approved for embed_metric).
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ZDIM = 10


def adapt(
    staged_dir: str,
    cov_result_dir: str,
    out_result_dir: str,
    var_uid: str,
    gt_pack: dict | None = None,
) -> Path:
    staged = Path(staged_dir)
    cov_result = Path(cov_result_dir)
    out_result = Path(out_result_dir)

    map_mrc = staged / f"{var_uid}_map.mrc"
    comps = [staged / f"{var_uid}_component_{k}.mrc" for k in range(ZDIM)]
    particles_cs = staged / f"{var_uid}_particles.cs"
    for p in [map_mrc, particles_cs] + comps:
        assert p.exists(), f"missing {p}"

    from recovar import utils

    # Inherit volume_shape / voxel_size from cov baseline
    with open(cov_result / "model" / "params.pkl", "rb") as f:
        cov_params = pickle.load(f)
    volume_shape = tuple(cov_params["volume_shape"])
    voxel_size = float(cov_params["voxel_size"])
    N = int(np.prod(volume_shape))
    sqrtN = float(np.sqrt(N))

    mean_real = np.asarray(utils.load_mrc(str(map_mrc))).astype(np.float32)
    comp_real = np.stack([np.asarray(utils.load_mrc(str(c))).astype(np.float32) for c in comps], axis=0)
    assert comp_real.shape == (ZDIM, *volume_shape), f"shape mismatch {comp_real.shape} vs (ZDIM,{volume_shape})"

    # M[:,k] = component k flat. Real QR ⇒ Q real orthonormal (real L2), R upper-tri.
    M = comp_real.reshape(ZDIM, -1).T.astype(np.float64)  # (N, q)
    Q, R = np.linalg.qr(M)  # Q: (N, q), R: (q, q)

    # Eigenvolumes in pipeline convention: Fourier-Hermitian unit norm
    # ⇒ real L2 norm = 1/sqrt(N) ⇒ scale Q columns by 1/sqrt(N).
    eigen_pipe = (Q / sqrtN).T.reshape(ZDIM, *volume_shape).astype(np.float32)

    # z in pipeline convention: sqrt(N) * R @ z_old
    recs = np.load(str(particles_cs))
    z_old = np.stack([recs[f"components_mode_{k}/value"] for k in range(ZDIM)], axis=1).astype(
        np.float64
    )  # (n_images, q)
    z_pipe = (z_old @ R.T) * sqrtN  # (n, q)

    c_scale = 1.0
    if gt_pack is not None:
        from recovar.core import fourier_transform_utils as ftu

        cov_mean_real = np.asarray(utils.load_mrc(str(cov_result / "output" / "volumes" / "mean.mrc")))
        mean_f = np.asarray(ftu.get_dft3(cov_mean_real)).reshape(-1).astype(np.complex64)
        U_f = np.asarray(ftu.get_dft3(eigen_pipe)).reshape(ZDIM, -1).astype(np.complex64)
        v_f = gt_pack["v_fourier"]
        centered = v_f - mean_f[None, :]
        alpha_gt = (centered @ np.conj(U_f).T).real.astype(np.float64)
        labels = gt_pack["image_assignments"].astype(np.int64)
        n = min(z_pipe.shape[0], labels.size)
        valid = labels[:n] >= 0
        z_fit = z_pipe[:n][valid]
        target = alpha_gt[labels[:n][valid]]
        num = float((z_fit * target).sum())
        den = float((z_fit * z_fit).sum())
        c_scale = num / den if den > 0 else 1.0
        print(f"fitted c_scale = {c_scale:.6f}")
        z_pipe = z_pipe * c_scale

    s = (z_pipe**2).mean(axis=0)

    # ----- write out -----
    model_dir = out_result / "model"
    vols_dir = out_result / "output" / "volumes"
    zdim_dir = model_dir / f"zdim_{ZDIM}"
    for d in [model_dir, vols_dir, zdim_dir]:
        d.mkdir(parents=True, exist_ok=True)

    params_out = dict(cov_params)
    params_out["s"] = s.astype(np.float64)
    params_out.setdefault("volume_shape", np.asarray(volume_shape))
    params_out.setdefault("voxel_size", voxel_size)
    params_out["zdims_computed"] = [ZDIM]
    params_out.pop("ppca_refit_info", None)
    with open(model_dir / "params.pkl", "wb") as f:
        pickle.dump(params_out, f)

    for fn in ["halfsets.pkl", "particles_halfsets.pkl"]:
        src = cov_result / "model" / fn
        if src.exists():
            shutil.copy(src, model_dir / fn)

    np.save(zdim_dir / "latent_coords.npy", z_pipe.astype(np.float32))
    np.save(zdim_dir / "contrasts.npy", np.ones(z_pipe.shape[0], dtype=np.float32))
    prec = np.tile(np.eye(ZDIM, dtype=np.float32)[None], (z_pipe.shape[0], 1, 1))
    np.save(zdim_dir / "latent_precision.npy", prec)

    # Use consensus (= cov) mean — 3DVA re-reconstructed map is at different scale
    shutil.copy(cov_result / "output" / "volumes" / "mean.mrc", vols_dir / "mean.mrc")
    for k in range(ZDIM):
        utils.write_mrc(
            str(vols_dir / f"eigen_pos{k:04d}.mrc"),
            eigen_pipe[k],
            voxel_size=voxel_size,
        )

    for fn in ["mask.mrc", "dilated_mask.mrc"]:
        src = cov_result / "output" / "volumes" / fn
        if src.exists():
            shutil.copy(src, vols_dir / fn)

    metadata = {
        "volume_shape": list(volume_shape),
        "voxel_size": voxel_size,
        "zdims_computed": [ZDIM],
        "files": {
            "params": "model/params.pkl",
            "halfsets": "model/halfsets.pkl",
            "particles_halfsets": "model/particles_halfsets.pkl",
            "mean_volume": "output/volumes/mean.mrc",
            "mask": "output/volumes/mask.mrc",
        },
        "source": f"cryoSPARC 3DVA {var_uid} var_K={ZDIM}",
        "c_scale": float(c_scale),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("WROTE", out_result)
    return out_result


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: adapt_3dva_to_result.py <staged_dir> <cov_result_dir> <out_result_dir> <var_uid>")
        sys.exit(2)
    adapt(*sys.argv[1:])
