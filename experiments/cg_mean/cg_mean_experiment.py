#!/usr/bin/env python
r"""
CG mean estimation vs Wiener pipeline — correct implementation.

System (support-constrained Wiener, solved at full upsampled resolution):

    P  F^{-1}  diag(d_reg)  F  P  u   =   P  F^{-1}  c        (*)

where:
    d_reg  =  adjust_regularization_relion_style(d, tau)   — EXACT pipeline diagonal
    c      =  A^* b   (weighted back-projection, includes gridding kernel)
    P      =  real-space support mask, zero-padded to upsampled grid
    F      =  recovar's centered DFT  (fftshift-fftn-fftshift)

One matvec:  u  -->  P . idft( d_reg . dft( P . u ) )

Initial guess:  u0 = P . idft( c / d_reg )   (= masked unconstrained Wiener)

After CG convergence:
    1. crop u from (2N)^3 to N^3
    2. spherical mask  (soft_mask_outside_map, cosine_width=3)
    3. grid correction  (griddingCorrect_square, order=1)

Steps 2-3 are IDENTICAL to the pipeline's post-processing, ensuring the only
difference is mask-inside-CG vs mask-after-Wiener.

Uses PDB 5nrl spliceosome volume at 256^3 with 100k images.
"""

import logging, os, pickle, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger("cg_mean")

SCRATCH = "/scratch/gpfs/GILLES/mg6942"
WORKDIR = os.environ.get("CG_WORKDIR", f"{SCRATCH}/experiments/cg_mean")
os.makedirs(WORKDIR, exist_ok=True)

GRID_SIZE   = 512
N_IMAGES    = 100_000
VOXEL_SIZE  = 4.25 * 128 / 512  # = 1.0625 A/pix — default scaling so molecule fills box
NOISE_LEVEL = 0.5
UPSAMPLING  = 1    # no oversampling at 512^3 — gridding errors are small
BATCH_SIZE  = 512
SEED        = 42
CG_MAXITER  = 500
CG_TOL      = 1e-6

# ═══════════════════════════════════════════════════════════════════════════

def make_pdb_volume():
    from recovar.simulation.trajectory_generation import generate_trajectory_volumes
    from recovar import utils
    vol_dir = os.path.join(WORKDIR, "volumes")
    vol_path = os.path.join(vol_dir, "generated_volumes", "vol0000.mrc")
    if os.path.exists(vol_path):
        return utils.load_mrc(vol_path)
    logger.info("Generating 5nrl volume at %d^3 (voxel_size=%.4f)...", GRID_SIZE, VOXEL_SIZE)
    generate_trajectory_volumes(output_dir=vol_dir, grid_size=GRID_SIZE,
        n_volumes=1, voxel_size=VOXEL_SIZE, Bfactor=80, max_rotation_degrees=0)
    # Verify the volume fills the box reasonably
    v = utils.load_mrc(vol_path)
    nonzero_frac = np.sum(np.abs(v) > 0.01 * np.max(np.abs(v))) / v.size
    logger.info("Volume nonzero fraction: %.1f%%", nonzero_frac * 100)
    return utils.load_mrc(vol_path)

def generate_dataset(vol_gt):
    from recovar import utils
    from recovar.simulation import simulator
    sim_vol_dir = os.path.join(WORKDIR, "sim_volumes")
    os.makedirs(sim_vol_dir, exist_ok=True)
    utils.write_mrc(os.path.join(sim_vol_dir, "vol0000.mrc"), vol_gt, voxel_size=VOXEL_SIZE)
    dataset_dir = os.path.join(WORKDIR, "dataset")
    logger.info("Simulating %d images at %d^3...", N_IMAGES, GRID_SIZE)
    t0 = time.time()
    simulator.generate_synthetic_dataset(
        output_folder=dataset_dir, voxel_size=VOXEL_SIZE,
        volumes_path_root=os.path.join(sim_vol_dir, "vol"),
        n_images=N_IMAGES, grid_size=GRID_SIZE,
        volume_distribution=np.array([1.0]),
        dataset_params_option="uniform", noise_level=NOISE_LEVEL,
        noise_model="radial1", percent_outliers=0.0, volume_radius=0.9,
        trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0, contrast_std=0.1,
        disc_type="linear_interp", premultiplied_ctf=False,
        put_extra_particles=False)
    logger.info("Dataset generated in %.1fs", time.time()-t0)
    return dataset_dir

# ═══════════════════════════════════════════════════════════════════════════
# CG solver
# ═══════════════════════════════════════════════════════════════════════════

def cg_mean_solve(d_reg_3d, c_3d, mask_3d, maxiter=CG_MAXITER, tol=CG_TOL,
                   precondition=True):
    r"""Preconditioned CG for  P F^{-1} diag(d_reg) F P u = P F^{-1} c.

    BCCB preconditioner (Chan 1988, optimal circulant for Toeplitz):

        The Fourier-space diagonal of A = P F^{-1} D F P is not D itself
        but the convolution  c_k = (|p_hat|^2 * d)_k.  The circulant
        preconditioner uses  M^{-1} = F^{-1} diag(1/c) F, restricted
        to the support.

        This captures both the spectral weighting D and the mask's
        redistribution of spectral energy.  Cost per CG iteration:
        2 FFT pairs (one for matvec, one for preconditioner).

    References:
        Chan (1988) "An Optimal Circulant Preconditioner for Toeplitz Systems"
        Serra-Capizzano (1999) multilevel Toeplitz limitations
        Fessler & Booth (1999) DCD preconditioners for shift-variant systems
    """
    import jax, jax.numpy as jnp
    import recovar.core.fourier_transform_utils as ftu
    D = jnp.asarray(d_reg_3d); c = jnp.asarray(c_3d); P = jnp.asarray(mask_3d)

    @jax.jit
    def mv(u): return P * ftu.get_idft3(D * ftu.get_dft3(P * u)).real

    # ── Build BCCB preconditioner ──────────────────────────────────────
    if precondition:
        # Fourier diagonal of A at mode k = sum_{k'} |p_hat(k-k')|^2 d(k')
        # This is the convolution of |p_hat|^2 with d (both in Fourier space).
        # By convolution theorem: conv in Fourier = pointwise multiply in real.
        # So: precond_diag = DFT( IDFT(|p_hat|^2) * IDFT(d) )
        mask_ps = jnp.abs(ftu.get_dft3(P))**2          # |p_hat|^2 in Fourier
        mask_ps_real = ftu.get_idft3(mask_ps).real      # IDFT to real space
        d_real = ftu.get_idft3(D).real                  # IDFT(d) to real space
        # N^3 normalization: the centered DFT with norm="backward" has
        # DFT * IDFT = N^3 * identity. The convolution theorem for this
        # convention gives:  conv(f,g) = IDFT(DFT(f)*DFT(g)) / N^3
        # So:  sum_{k'} |p_hat(k-k')|^2 d(k') = DFT(IDFT(|p_hat|^2) * IDFT(d)) * N^3
        N3 = float(np.prod(D.shape))
        precond_diag = ftu.get_dft3(jnp.array(mask_ps_real * d_real)).real * N3
        precond_diag = jnp.maximum(precond_diag, jnp.max(precond_diag) * 1e-8)
        precond_inv = 1.0 / precond_diag

        @jax.jit
        def apply_M_inv(r):
            return P * ftu.get_idft3(precond_inv * ftu.get_dft3(P * r)).real

        logger.info("BCCB preconditioner built: diag range [%.2e, %.2e]",
                     float(precond_diag.min()), float(precond_diag.max()))
    else:
        apply_M_inv = lambda r: r

    # ── Initial guess and RHS ──────────────────────────────────────────
    rhs = P * ftu.get_idft3(c).real
    u   = P * ftu.get_idft3(c / D).real   # masked Wiener

    # ── Preconditioned CG ──────────────────────────────────────────────
    r = rhs - mv(u)
    z = apply_M_inv(r)
    p = z.copy()
    rz = float(jnp.sum(r * z))
    bn = float(jnp.sqrt(jnp.sum(rhs * rhs)))
    rr = float(jnp.sqrt(jnp.sum(r * r))) / max(bn, 1e-30)
    residuals = []
    logger.info("PCG: ||b||=%.4e  ||r0||/||b||=%.4e", bn, rr)

    for it in range(maxiter):
        Ap = mv(p)
        pAp = float(jnp.sum(p * Ap))
        if pAp < 1e-30: break
        alpha = rz / pAp
        u += alpha * p
        r -= alpha * Ap
        rr = float(jnp.sqrt(jnp.sum(r * r))) / max(bn, 1e-30)
        residuals.append(rr)
        if rr < tol:
            logger.info("PCG converged iter %d  rr=%.2e", it+1, rr); break
        z = apply_M_inv(r)
        rz_new = float(jnp.sum(r * z))
        beta = rz_new / max(abs(rz), 1e-30)
        p = z + beta * p
        rz = rz_new
        if (it+1) % 50 == 0:
            logger.info("PCG iter %3d: rr=%.2e", it+1, rr)

    if rr >= tol: logger.info("PCG maxiter=%d rr=%.2e", maxiter, rr)
    return np.asarray(u), residuals

def postprocess(u_up, N, up, use_sph_mask=True):
    """Pipeline-identical post-processing: crop + spherical mask + grid correction."""
    import jax.numpy as jnp
    from recovar.core import padding as pad_mod, mask as mask_mod
    from recovar.reconstruction import relion_functions
    up_n = N * up
    v = np.asarray(pad_mod.unpad_volume_spatial_domain(jnp.array(u_up), up_n - N))
    if use_sph_mask:
        v, _ = mask_mod.soft_mask_outside_map(jnp.array(v), cosine_width=3)
        v = np.asarray(v)
    v, _ = relion_functions.griddingCorrect_square(jnp.array(v), N, up, order=1)
    return np.asarray(v)

def compute_fsc(v1_ft, v2_ft, vol_shape):
    import jax.numpy as jnp
    from recovar.reconstruction.regularization import average_over_shells
    v1=jnp.asarray(v1_ft).reshape(-1); v2=jnp.asarray(v2_ft).reshape(-1)
    top = average_over_shells((jnp.conj(v1)*v2).real, vol_shape)
    b1  = average_over_shells(jnp.abs(v1)**2, vol_shape)
    b2  = average_over_shells(jnp.abs(v2)**2, vol_shape)
    return np.asarray(jnp.where(jnp.sqrt(b1*b2)>1e-20, top/jnp.sqrt(b1*b2), 0.0))

def compute_masked_fsc(v1_real, v2_real, mask, vol_shape):
    """FSC with mask applied to both volumes before DFT."""
    import jax.numpy as jnp
    import recovar.core.fourier_transform_utils as ftu
    f1 = ftu.get_dft3(jnp.asarray(v1_real * mask)).reshape(-1)
    f2 = ftu.get_dft3(jnp.asarray(v2_real * mask)).reshape(-1)
    return compute_fsc(f1, f2, vol_shape)

def fsc_res(fsc, vs, gs, thr=0.5):
    f = np.arange(len(fsc))/(gs*vs)
    a = np.where(fsc>=thr)[0]
    return 1./f[a[-1]] if len(a)>0 and f[a[-1]]>0 else np.inf

def corr(a,b,m):
    x,y=a*m,b*m; return float(np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-30))

# ═══════════════════════════════════════════════════════════════════════════

def main():
    import jax, jax.numpy as jnp
    import recovar.core.fourier_transform_utils as ftu
    import recovar.utils as utils
    from recovar.core import mask as mask_mod
    from recovar.data_io import cryoem_dataset
    from recovar.reconstruction import homogeneous, relion_functions

    np.random.seed(SEED)
    logger.info("JAX devices: %s", jax.devices())
    N = GRID_SIZE; vs = (N,)*3; up_n = N*UPSAMPLING; us = (up_n,)*3

    # ── 1. GT ──────────────────────────────────────────────────────────────
    logger.info("=== 1. GT volume (5nrl spliceosome) ===")
    vol_gt = make_pdb_volume()
    vol_gt_ft = np.asarray(ftu.get_dft3(jnp.array(vol_gt)))
    utils.write_mrc(f"{WORKDIR}/gt_vol.mrc", vol_gt, voxel_size=VOXEL_SIZE)

    # ── 2. Dataset ────────────────────────────────────────────────────────
    logger.info("=== 2. Dataset ===")
    dataset_dir = os.path.join(WORKDIR, "dataset")
    if not os.path.exists(os.path.join(dataset_dir, "particles.star")):
        generate_dataset(vol_gt)
    else:
        logger.info("Dataset exists, skipping")
    sim = utils.pickle_load(f"{dataset_dir}/simulation_info.pkl")
    noise_var = np.float32(np.median(sim["noise_variance"]))

    ds = cryoem_dataset.load_dataset(f"{dataset_dir}/particles.star",
        poses_file=f"{dataset_dir}/poses.pkl", ctf_file=f"{dataset_dir}/ctf.pkl", lazy=True)
    rng = np.random.RandomState(SEED); perm = rng.permutation(ds.n_images); h=ds.n_images//2
    ds.halfset_indices = [np.sort(perm[:h]).astype(np.int32),
                          np.sort(perm[h:]).astype(np.int32)]

    # ── 3. GT mask ────────────────────────────────────────────────────────
    logger.info("=== 3. GT mask ===")
    gt_mask = np.asarray(mask_mod.make_mask_from_gt(vol_gt, smax=3, iter=10, from_ft=False), dtype=np.float32)
    mask_frac = float(np.sum(gt_mask>0.5))/np.prod(vs)
    logger.info("Mask: %.1f%% of box", mask_frac*100)
    # Pad to upsampled size
    gt_mask_up = np.zeros(us, dtype=np.float32)
    s = (up_n-N)//2
    gt_mask_up[s:s+N, s:s+N, s:s+N] = gt_mask

    # ── 4. Pipeline mean ──────────────────────────────────────────────────
    logger.info("=== 4. Pipeline mean ===")
    t0 = time.time()
    means, mean_prior, _ = homogeneous.get_mean_conformation_relion(
        ds, BATCH_SIZE, noise_variance=noise_var,
        use_regularization=True, upsampling_factor=UPSAMPLING)
    t_pipe = time.time()-t0
    vol_pipe = np.asarray(ftu.get_idft3(jnp.array(means.combined.reshape(vs)))).real
    logger.info("Pipeline done in %.1fs", t_pipe)

    # ── 5. Full accumulation + regularized diagonal ───────────────────────
    logger.info("=== 5. Full accumulation ===")
    t0 = time.time()
    Ft_ctf, Ft_y = relion_functions.relion_style_triangular_kernel(
        ds, noise_var, BATCH_SIZE, upsampling_factor=UPSAMPLING)
    t_acc = time.time()-t0
    logger.info("Accumulation done in %.1fs", t_acc)

    c_up = np.asarray(Ft_y).reshape(us)

    # Use EXACT pipeline regularization (includes tau prior + floor + shell avg)
    d_reg = np.asarray(relion_functions.adjust_regularization_relion_style(
        jnp.array(Ft_ctf).reshape(-1), us, tau=jnp.array(mean_prior),
        padding_factor=UPSAMPLING)).reshape(us)
    logger.info("d_reg range: [%.4e, %.4e]", d_reg.min(), d_reg.max())

    # ── 6. CG with GT mask ───────────────────────────────────────────────
    logger.info("=== 6a. CG at %d^3 (GT mask) ===", up_n)
    t0 = time.time()
    u_cg, res_cg = cg_mean_solve(d_reg, c_up, gt_mask_up, precondition=False)
    t_cg = time.time()-t0
    logger.info("CG done in %.1fs (%d iters)", t_cg, len(res_cg))
    v_cg = postprocess(u_cg, N, UPSAMPLING)

    # ── 7. CG with spherical mask ─────────────────────────────────────────
    logger.info("=== 6b. CG at %d^3 (spherical mask) ===", up_n)
    sph_mask_up = np.asarray(mask_mod.get_radial_mask(us), dtype=np.float32)
    t0 = time.time()
    u_sph, res_sph = cg_mean_solve(d_reg, c_up, sph_mask_up, precondition=False)
    t_sph = time.time()-t0
    logger.info("CG sph done in %.1fs (%d iters)", t_sph, len(res_sph))
    v_sph = postprocess(u_sph, N, UPSAMPLING)

    # ── 8. Metrics ────────────────────────────────────────────────────────
    logger.info("=== 7. Metrics (masked FSC) ===")
    methods = {}
    def add(nm, vol, t=None):
        f = compute_masked_fsc(vol, vol_gt, gt_mask, vs)
        r = fsc_res(f, VOXEL_SIZE, N)
        c = corr(vol, vol_gt, gt_mask)
        methods[nm] = dict(vol=vol, fsc=f, res=r, corr=c, time=t)
        logger.info("  %-40s  FSC@0.5=%6.2fA  corr=%.5f", nm, r, c)

    add("A) Wiener pipeline",          vol_pipe,          t_pipe)
    add("B) Wiener + GT mask after",   vol_pipe * gt_mask)
    add("C) CG (GT mask inside)",      v_cg,              t_cg)
    add("D) CG (sph mask inside)",     v_sph,             t_sph)

    # ── 9. Plots ──────────────────────────────────────────────────────────
    logger.info("=== 8. Plots ===")
    mid = N//2
    names = list(methods.keys())
    freqs = np.arange(len(methods[names[0]]["fsc"])) / (N*VOXEL_SIZE)

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)

    # Row 0: central Z-slices (each auto-scaled)
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(vol_gt[:,:,mid].T, cmap="gray", origin="lower")
    ax.set_title("GT", fontsize=11); ax.axis("off")
    for i, nm in enumerate(names):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(methods[nm]["vol"][:,:,mid].T, cmap="gray", origin="lower")
        r = methods[nm]["res"]; c = methods[nm]["corr"]
        ax.set_title(f"{nm.split(')')[0]})  {r:.2f}A  c={c:.4f}", fontsize=8)
        ax.axis("off")

    # Row 1: scaled difference maps inside mask
    for i, nm in enumerate(names):
        ax = fig.add_subplot(gs[1, i])
        v = methods[nm]["vol"]
        scale = np.sum(v*vol_gt*gt_mask) / (np.sum(v*v*gt_mask) + 1e-30)
        diff = (scale*v - vol_gt) * gt_mask
        dm = max(np.abs(diff[:,:,mid]).max(), 1e-10)
        ax.imshow(diff[:,:,mid].T, cmap="RdBu_r", vmin=-dm, vmax=dm, origin="lower")
        ax.set_title(f"{nm.split(')')[0]})−GT (scaled)", fontsize=8); ax.axis("off")
    ax = fig.add_subplot(gs[1,4])
    ax.imshow(gt_mask[:,:,mid].T, cmap="gray", vmin=0, vmax=1, origin="lower")
    ax.set_title("GT mask", fontsize=10); ax.axis("off")

    # Row 2: FSC + convergence + line profile
    ax_fsc = fig.add_subplot(gs[2, 0:2])
    for nm in names:
        m = methods[nm]
        ax_fsc.plot(freqs, m["fsc"], label=f"{nm.split(')')[0]}) {m['res']:.2f}A", lw=1.5)
    ax_fsc.axhline(0.5, color="gray", ls="--", alpha=.5)
    ax_fsc.axhline(0.143, color="gray", ls=":", alpha=.5, label="0.143")
    ax_fsc.set_xlabel("Frequency (1/A)"); ax_fsc.set_ylabel("FSC")
    ax_fsc.set_title("FSC vs GT"); ax_fsc.legend(fontsize=7)
    ax_fsc.set_ylim(-.05, 1.05); ax_fsc.grid(True, alpha=.3)

    ax_cg = fig.add_subplot(gs[2, 2])
    ax_cg.semilogy(res_cg, label="GT mask", lw=1.5)
    ax_cg.semilogy(res_sph, label="sph mask", lw=1.5)
    ax_cg.set_xlabel("Iteration"); ax_cg.set_ylabel("||r||/||b||")
    ax_cg.set_title("CG convergence"); ax_cg.legend(); ax_cg.grid(True, alpha=.3)

    ax_lp = fig.add_subplot(gs[2, 3:5])
    ax_lp.plot(vol_gt[:, mid, mid], "k-", lw=2, label="GT")
    for nm in names:
        v = methods[nm]["vol"]
        sc = np.sum(v*vol_gt*gt_mask) / (np.sum(v*v*gt_mask) + 1e-30)
        ax_lp.plot(sc*v[:, mid, mid], lw=1, alpha=0.8,
                   label=f"{nm.split(')')[0]}) x{sc:.2f}")
    ax_lp.set_title("Line profile (y=z=mid, amplitude-matched)")
    ax_lp.legend(fontsize=7); ax_lp.grid(True, alpha=.3)

    plt.savefig(f"{WORKDIR}/cg_mean_comparison.png", dpi=150, bbox_inches="tight")
    logger.info("Saved: %s/cg_mean_comparison.png", WORKDIR)
    plt.close()

    # Save results
    res = {n: {k:v for k,v in m.items() if k!="vol"} for n,m in methods.items()}
    res["residuals_cg"] = res_cg; res["residuals_sph"] = res_sph
    with open(f"{WORKDIR}/results.pkl", "wb") as f: pickle.dump(res, f)
    for nm in names:
        tag = nm.split(")")[0].replace(" ","").replace("(","_").lower()
        utils.write_mrc(f"{WORKDIR}/mean_{tag}.mrc", methods[nm]["vol"], voxel_size=VOXEL_SIZE)
    logger.info("DONE")

if __name__ == "__main__":
    main()
