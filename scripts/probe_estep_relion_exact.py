"""ROUND 11 — RELION-exact custom scorer + BPref accumulator.

Bypasses recovar's run_em entirely. Computes the iter-1 BPref accumulator
by replicating RELION's storeWeightedSums semantics line-by-line in JAX:

  For each particle i:
    1. Preprocess image: ds.process_images(apply_image_mask=True) with
       RELION-exact mask geometry → Fimg.
    2. Compute Fimg_nomask: apply_image_mask=False → Fimg_nomask.
    3. Compute Fctf and Minvsigma2 from per-particle ctf_params + sigma2.
       Negate Fctf to absorb recovar↔RELION CTF sign convention.
    4. For each (orient, oversample, trans, otra) cell:
         Frefctf = -Fctf * project(volume, R)/N²       (bit-exact)
         Fimg_shifted = Fimg * exp(-2πi (kx*sx+ky*sy)/N)
         diff² = sum |Frefctf - Fimg_shifted|² * Minvsigma2 / 2
    5. exp_Mweight = exp(-(diff² - min_diff²)) (un-normalised, then /sum).
    6. BPref accumulate (RELION storeWeightedSums @ ml_optimiser.cpp:9554):
         For each cell with weight >= significance threshold:
           weight_norm = weight / sum_weight
           weightxinvsigma2 = weight_norm * Minvsigma2[k]
           Fimg_shift_nomask = Fimg_nomask * exp(-2πi shift)  per pixel
           bp_data[A^T (kx,ky,0)] += weightxinvsigma2 * Fimg_shift_nomask * CTF[k]
           bp_weight[A^T (kx,ky,0)] += weightxinvsigma2 * CTF[k]² (?? no, just CTF)
       Actually RELION's BPref.set2DFourierTransform applies CTF×CTF for weight.

Run on small fixture (500/64), compare to RELION's pipe_it1_c0_bp_data_pre_reweight.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run")
PARTICLES_STAR = Path(
    "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/"
    "slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/"
    "gpu_spa/test_dataset/particles.star"
)
DUMP = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small")


def _read_bin(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        nz, ny, nx = struct.unpack("qqq", f.read(24))
        pos = f.tell()
        f.seek(0, 2)
        rem = f.tell() - pos
        f.seek(pos)
        bp = rem // (nz * ny * nx) if nz * ny * nx else 8
        dt = np.complex128 if bp == 16 else np.float64
        return np.fromfile(f, dtype=dt, count=nz * ny * nx).reshape(nz, ny, nx)


def _cc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.real(np.vdot(af, bf)) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def _read_iter0_sigma2(n: int) -> np.ndarray:
    txt = (FIXTURE_DIR / "run_it000_model.star").read_text()
    m = re.search(r"data_model_optics_group_1\n(.*?)(?:\ndata_)", txt, re.DOTALL)
    v = np.zeros(n, dtype=np.float64)
    for line in m.group(1).strip().split("\n"):
        toks = line.split()
        if len(toks) == 3:
            try:
                v[int(toks[0])] = float(toks[2])
            except ValueError:
                pass
    return v


def euler_to_R(rot_d, tilt_d, psi_d):
    rot = np.deg2rad(rot_d)
    tilt = np.deg2rad(tilt_d)
    psi = np.deg2rad(psi_d)
    ca, sa = np.cos(rot), np.sin(rot)
    cb, sb = np.cos(tilt), np.sin(tilt)
    cg, sg = np.cos(psi), np.sin(psi)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    return np.array(
        [
            [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
            [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
            [sc, ss, cb],
        ]
    )


def main():
    import jax.numpy as jnp
    import mrcfile

    from recovar.core import fourier_transform_utils as ftu
    from recovar.core.relion_project import (
        centered_full_to_relion_half,
        gridding_correct_volume_real,
        relion_project_half,
    )
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.data_io.starfile import read_star

    # ------------------------------------------------------------------
    # 1. Load RELION's exact post-perturbation rotations + translations
    # ------------------------------------------------------------------
    with open(DUMP / "p0_oversampled_eulers.bin", "rb") as f:
        h = struct.unpack("qqq", f.read(24))
        eulers_full = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(h[0], h[1], h[2])
    with open(DUMP / "p0_oversampled_translations.bin", "rb") as f:
        h = struct.unpack("qqq", f.read(24))
        trans_full = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(h[0], h[1], h[2])

    n_orient, n_orot, _ = eulers_full.shape
    n_coarse_trans, n_otra, _ = trans_full.shape
    print(f"Grid: {n_orient} coarse orient × {n_orot} orot, {n_coarse_trans} coarse trans × {n_otra} otra")

    # Flatten to (n_rot=4608, 3) and (n_trans=116, 2)
    eulers = eulers_full.reshape(-1, 3)
    translations = trans_full[:, :, :2].reshape(-1, 2)
    n_rot = eulers.shape[0]
    n_trans = translations.shape[0]
    print(f"Total: {n_rot} rotations × {n_trans} translations = {n_rot * n_trans:,} cells/particle")

    # Pre-build all rotation matrices
    R_all = np.array([euler_to_R(*e) for e in eulers])

    # ------------------------------------------------------------------
    # 2. Build RELION-frame volume (gridding-corrected, half-complex)
    # ------------------------------------------------------------------
    vol_real_relion_frame = np.asarray(
        mrcfile.open(str(FIXTURE_DIR / "run_it000_class001.mrc"), permissive=True).data, dtype=np.float64
    )
    vol_corrected = np.asarray(gridding_correct_volume_real(jnp.asarray(vol_real_relion_frame), 64, 1))
    F_centered = np.asarray(ftu.get_dft3(jnp.asarray(vol_corrected)))
    vol_half = np.asarray(centered_full_to_relion_half(F_centered))
    N = 64
    print(f"Volume RELION-half shape: {vol_half.shape}")

    # ------------------------------------------------------------------
    # 3. Setup dataset + RELION-exact mask
    # ------------------------------------------------------------------
    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    backend = ds.image_source.backend
    backend.set_relion_image_mask(pixel_size=8.5, particle_diameter_ang=544.0, width_mask_edge_px=5.0)

    # RELION-sorted halfsets — h0 only
    main_in, _ = read_star(str(PARTICLES_STAR))
    mic_names = np.asarray(main_in["_rlnMicrographName"].tolist())
    sort_idx = np.argsort(mic_names, kind="stable")
    h0_ids = sort_idx[0::2]

    # Sigma2 per shell (matched to RELION's run_it000_model.star)
    sigma2 = _read_iter0_sigma2(N // 2 + 1)
    # Build 2D Minvsigma2 image (centered) and then convert to FFTW-natural windowed at current_size=28
    from recovar.reconstruction.noise import make_radial_noise

    # IMPORTANT: pass sigma2 directly (no N^4 scaling — that scaling was only
    # needed to compensate for run_em's normalisation; here we apply Minvsigma2
    # exactly as RELION does)
    nv_2d_centered = np.asarray(make_radial_noise(sigma2, (N, N))).reshape(N, N)
    Minvsigma2_full = np.where(nv_2d_centered > 0, 1.0 / nv_2d_centered, 0.0)
    # Window to current_size=28 in FFTW-natural layout (matches RELION's
    # exp_local_Minvsigma2 shape (28, 15))
    current_size = 28
    half_x = current_size // 2 + 1
    Minv_natural = np.fft.ifftshift(Minvsigma2_full, axes=(0, 1))
    keep_y = np.r_[np.arange(half_x), np.arange(N - (current_size - half_x), N)]
    Minvsigma2 = Minv_natural[keep_y][:, :half_x]
    print(f"Minvsigma2 shape: {Minvsigma2.shape}")

    # ------------------------------------------------------------------
    # 4. Per-cell trans phase grid (RELION-exact)
    # ------------------------------------------------------------------
    H_out, W_out = current_size, current_size // 2 + 1
    ky = np.fft.fftfreq(H_out) * H_out
    kx = np.arange(W_out)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    # phase[t, ky, kx] for each translation
    phases = np.exp(
        -2j * np.pi * (KX[None, ...] * translations[:, 0:1, None] + KY[None, ...] * translations[:, 1:2, None]) / N
    )  # (n_trans, H_out, W_out)

    # ------------------------------------------------------------------
    # 5. Build BPref output (Z, Y, X half) — match RELION's pad_size=ori=64
    # ------------------------------------------------------------------
    # RELION's BPref data shape from dump: read it
    target_bp_data = _read_bin(DUMP / "pipe_it1_c0_bp_data_pre_reweight.bin")
    target_bp_weight = _read_bin(DUMP / "pipe_it1_c0_bp_weight.bin")
    pad_size = target_bp_data.shape[0]  # 31 for r_max=14, pad=1
    pad_half = target_bp_data.shape[2]  # 16
    print(f"BPref shape: {target_bp_data.shape}")

    bp_data = np.zeros(target_bp_data.shape, dtype=np.complex128)
    bp_weight = np.zeros(target_bp_weight.shape, dtype=np.float64)

    # ------------------------------------------------------------------
    # 6. Loop over particles — score + accumulate
    # ------------------------------------------------------------------
    print(f"\nProcessing {len(h0_ids)} h0 particles...")
    n_particles = len(h0_ids)
    eps = 1e-12
    # CTF computation
    from recovar.core.forward import ForwardModelConfig

    config = ForwardModelConfig.from_dataset(ds, disc_type="linear_interp", process_fn=ds.process_images)

    # Pre-compute all projections once (rotations are the same for all particles)
    # n_rot × (H_out, W_out) complex128. With N=64 and current_size=28, projection is (28, 15).
    # Sign-negate to match RELION-frame convention (vol_relion = -transpose(vol_recovar)
    # already absorbed by using vol_real_relion_frame directly above; no extra sign needed).
    print(f"Pre-computing {n_rot} projections at coarse window...")
    projections = np.zeros((n_rot, H_out, W_out), dtype=np.complex128)
    # JAX-vmap: project all rotations in chunks
    from jax import jit

    project_fn = jit(
        lambda R: relion_project_half(
            jnp.asarray(vol_half, dtype=jnp.complex128), R, current_size, r_max=N // 2 - 1, padding_factor=1
        )
    )
    for r_start in range(0, n_rot, 200):
        r_end = min(r_start + 200, n_rot)
        for ri in range(r_start, r_end):
            projections[ri] = np.asarray(project_fn(jnp.asarray(R_all[ri], dtype=jnp.float64)))
        if r_start % 1000 == 0:
            print(f"  projected {r_end}/{n_rot}")
    projections /= N * N  # /N² FFT normalisation (matches RELION's amplitude scale)

    print("Pre-computed projections.")

    # Now per-particle scoring + accumulation
    for p_idx, particle_id in enumerate(h0_ids[:5]):  # FIRST 5 PARTICLES for diagnostic speed
        # Get the image
        ds_one = ds.subset([int(particle_id)])
        batch_data, _, _, ctf_params, _, _, _ = next(ds_one.iter_batches(1))

        # Masked Fimg (E-step input)
        Fimg_full = np.asarray(ds.process_images(jnp.asarray(batch_data), apply_image_mask=True))[0].reshape(N, N)
        Fimg_nat = np.fft.ifftshift(Fimg_full, axes=(0, 1))
        Fimg_masked = Fimg_nat[keep_y][:, :half_x] / (N * N)  # /N² normalised

        # Unmasked Fimg (M-step input)
        Fimg_nomask_full = np.asarray(ds.process_images(jnp.asarray(batch_data), apply_image_mask=False))[0].reshape(
            N, N
        )
        Fimg_nomask_nat = np.fft.ifftshift(Fimg_nomask_full, axes=(0, 1))
        Fimg_nomask = Fimg_nomask_nat[keep_y][:, :half_x] / (N * N)

        # CTF
        CTF_full = np.asarray(config.compute_ctf(jnp.asarray(ctf_params)))[0].reshape(N, N)
        CTF_nat = np.fft.ifftshift(CTF_full, axes=(0, 1))
        Fctf = -CTF_nat[keep_y][:, :half_x]  # negate to match RELION convention

        # Score all cells (n_rot × n_trans)
        scores = np.zeros((n_rot, n_trans), dtype=np.float64)
        for ri in range(n_rot):
            Frefctf = projections[ri] * Fctf  # (H, W)
            for ti in range(n_trans):
                Fimg_shifted = Fimg_masked * phases[ti]
                diff = Frefctf - Fimg_shifted
                diff2 = float(np.sum((diff.real**2 + diff.imag**2) * Minvsigma2 * 0.5))
                scores[ri, ti] = -diff2

        # Posterior: exp(scores - max), then normalize
        max_score = scores.max()
        weights = np.exp(scores - max_score)
        sum_weight = weights.sum()
        weights /= sum_weight

        # Accumulate into BPref via adjoint
        # RELION storeWeightedSums (line 9554-9559):
        #   weightxinvsigma2 = weight × Minvsigma2[k]
        #   bp_data[k] += weight_xinvsigma2 × Fimg_shift_nomask[k] × CTF[k] (via set2DFourierTransform)
        #   bp_weight[k] += weight_xinvsigma2 × CTF[k]²
        # Backproject for each cell — adjoint slice into bp_data
        # For now, only top-K cells per particle for speed (matches RELION significance pruning)
        K_TOP = min(2000, n_rot * n_trans)
        flat_idx = np.argsort(-weights.ravel())[:K_TOP]
        for fi in flat_idx:
            w = weights.flat[fi]
            if w < 1e-8:
                continue
            ri = fi // n_trans
            ti = fi % n_trans
            R = R_all[ri]
            shift = translations[ti]
            phase_t = phases[ti]
            Fimg_shifted_nomask = Fimg_nomask * phase_t
            wxinv = w * Minvsigma2  # (H, W)
            data_contrib = wxinv * Fimg_shifted_nomask * Fctf  # (H, W)
            weight_contrib = wxinv * (Fctf * Fctf).real  # (H, W) - CTF² contributes here

            # Adjoint slice: for each output pixel (h, x), at frequency (kx_o, ky_o, 0) in 2D,
            # the corresponding 3D point is R^T @ (kx_o, ky_o, 0). We deposit data_contrib[h,x]
            # at that 3D location in bp_data via trilinear adjoint (8 corners).
            Ainv = np.linalg.inv(R)
            for ho in range(H_out):
                ky_o = ho if ho <= H_out // 2 else ho - H_out
                for xo in range(W_out):
                    kx_o = xo
                    if kx_o * kx_o + ky_o * ky_o > (W_out - 1) ** 2:
                        continue
                    xp = Ainv[0, 0] * kx_o + Ainv[0, 1] * ky_o
                    yp = Ainv[1, 0] * kx_o + Ainv[1, 1] * ky_o
                    zp = Ainv[2, 0] * kx_o + Ainv[2, 1] * ky_o
                    if xp < 0:
                        xp, yp, zp = -xp, -yp, -zp
                        d_val = np.conj(data_contrib[ho, xo])
                    else:
                        d_val = data_contrib[ho, xo]
                    w_val = weight_contrib[ho, xo]

                    # Trilinear adjoint into bp_data (Xmipp origin, half-x)
                    x0 = int(np.floor(xp))
                    fx = xp - x0
                    y0 = int(np.floor(yp))
                    fy = yp - y0
                    z0 = int(np.floor(zp))
                    fz = zp - z0
                    y0r = y0 + pad_size // 2
                    z0r = z0 + pad_size // 2
                    if x0 < 0 or x0 + 1 >= pad_half or y0r < 0 or y0r + 1 >= pad_size or z0r < 0 or z0r + 1 >= pad_size:
                        continue
                    for dz in (0, 1):
                        for dy in (0, 1):
                            for dx in (0, 1):
                                wt_corner = (
                                    ((1 - fx) if dx == 0 else fx)
                                    * ((1 - fy) if dy == 0 else fy)
                                    * ((1 - fz) if dz == 0 else fz)
                                )
                                bp_data[z0r + dz, y0r + dy, x0 + dx] += wt_corner * d_val
                                bp_weight[z0r + dz, y0r + dy, x0 + dx] += wt_corner * w_val

        if (p_idx + 1) % 1 == 0:
            cc_data = _cc(bp_data, target_bp_data)
            cc_weight = _cc(bp_weight, target_bp_weight)
            print(f"  After particle {p_idx + 1}: bp_data CC = {cc_data:+.6f}, bp_weight CC = {cc_weight:+.6f}")

    cc_data = _cc(bp_data, target_bp_data)
    cc_weight = _cc(bp_weight, target_bp_weight)
    print(f"\nFinal: bp_data CC = {cc_data:+.6f}, bp_weight CC = {cc_weight:+.6f}")


if __name__ == "__main__":
    main()
