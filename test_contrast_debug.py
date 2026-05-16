"""Isolated contrast debug: GT W/mean → E-step → check mean_c."""
import os; os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
import jax.numpy as jnp

from recovar.ppca.ppca_scale_sweep import _load_simulated_dataset, _with_trailing_separator
from recovar.ppca.ppca import (
    _e_step_half_inner, _iter_processed_batches_half,
    _normalize_experiment_datasets, E_M_step_batch_half,
)
from recovar.ppca import contrast_posterior
import recovar.core.fourier_transform_utils as ftu

# Load
ds_dir = '/scratch/gpfs/GILLES/mg6942/tmp/ppca_contrast_test_128/test_dataset'
cryos, sim_info, gt, nv = _load_simulated_dataset(
    _with_trailing_separator(ds_dir), 128, 50000, lazy=False)
vs = gt.volume_shape
gt_c = sim_info['per_image_contrast']

U_gt, s_gt, _ = gt.get_vol_svd()
gt_mean = gt.get_mean()
W_gt = U_gt[:, :10] * s_gt[:10]
W_gt_half = ftu.full_volume_to_half_volume(W_gt.T, vs).T

full_ds, ds_list = _normalize_experiment_datasets(cryos)
ref = full_ds if full_ds else ds_list[0]

# Get one batch of sufficient stats from JIT'd E-step
for batch_half, ctf_params, rots, trans, batch_idx in _iter_processed_batches_half(ref, 20):
    nv_half = ref.noise.get_half(batch_idx)
    out = _e_step_half_inner(
        batch_half, gt_mean, W_gt_half, ctf_params, rots, trans,
        ref.voxel_size, nv_half, ref.image_shape, vs,
        ref.ctf_evaluator, False)

    ez, smz = out[0], out[1]
    images_w, pmean_w, CTF_half = out[3], out[4], out[5]
    H, g, h, t, nu, y2 = out[7], out[8], out[9], out[10], out[11], out[12]

    bi = np.array(batch_idx).reshape(-1)
    gt_c_batch = gt_c[bi]

    print("=== Sufficient stats (first image) ===")
    print(f"H diag: {np.diag(np.array(H[0]))[:5]}")
    print(f"g:      {np.array(g[0])[:5]}")
    print(f"h:      {np.array(h[0])[:5]}")
    print(f"t={float(t[0]):.4e}  nu={float(nu[0]):.4e}  y2={float(y2[0]):.4e}")
    print(f"GT contrast for this image: {gt_c_batch[0]:.3f}")
    print()

    # Sanity check: t should be ~ c * nu if mean dominates signal
    # t = ỹ^T P̃μ,  ỹ = c·CTF·vol/σ + noise/σ
    # P̃μ = CTF·μ/σ
    # So t = c * ||CTF·μ/σ||² + c * (CTF·Wz/σ)^T (CTF·μ/σ) + noise^T(CTF·μ/σ)/σ
    # The first term is c*nu. Check if t/nu ≈ c:
    for i in range(min(10, len(gt_c_batch))):
        ratio = float(t[i]) / float(nu[i])
        print(f"  img {i}: c_gt={gt_c_batch[i]:.3f}  t/nu={ratio:.4f}")

    print()
    print("=== Profile scores for image 0 ===")
    Hi = np.array(H[0]); gi = np.array(g[0]); hi = np.array(h[0])
    ti = float(t[0]); nui = float(nu[0]); y2i = float(y2[0])

    c_vals = np.linspace(0.1, 2.0, 20)
    for c in [0.1, 0.5, 1.0, gt_c_batch[0], 1.5, 2.0]:
        r = y2i - 2*c*ti + c**2*nui
        q = c*gi - c**2*hi
        A = np.eye(10) + c**2 * Hi
        qAq = float(q @ np.linalg.solve(A, q))
        print(f"  c={c:.3f}: r={r:.2e}  qAq={qAq:.2e}  J=r-qAq={r-qAq:.2e}")

    break
