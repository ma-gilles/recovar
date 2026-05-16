"""Element-wise per-pose diff² comparison: recovar vs RELION (particle 0).

Inputs (RELION's dumped exact arrays):
  - p0_local_Fimg_shifted_t0.bin  (1, 28, 15) complex128 — Fimg shifted by trans[0]
  - p0_Fimg.bin                    (1, 28, 15) complex128 — Fimg pre-shift (windowed half)
  - p0_local_Fctf.bin              (1, 28, 15) float64    — CTF (windowed half)
  - p0_local_Minvsigma2.bin        (1, 28, 15) float64    — 1/σ² (windowed, half, RELION-remapped)
  - p0_oversampled_eulers.bin      (576, 8, 3)  float64    — RELION's 4608-rotation grid
  - p0_oversampled_translations.bin (29, 4, 3)  float64    — RELION's 116-translation grid
  - p0_exp_Mweight.bin              (1,48,12,29,8,4) float64 — exp_Mweight after convertSqDiff2Weights
  - p0_estep_meta.txt              — exp_min_diff2, exp_sum_weight, exp_max_weight

Compute recovar diff² for nonzero RELION cells using the same formula:
    diff²[r,t] = Σ |CTF_p · P_r(μ) - Fimg_shifted_t|² · Minvsigma2 / 2
where P_r is recovar's relion_project_half (verified bit-exact).

Recovered RELION diff²:
    log(exp_Mweight[i]) = -(diff²[i] - min_diff²) + log(pdf_orient × pdf_offset)
For a single (idir, ipsi, itrans) coarse cell, pdf_orient × pdf_offset is constant
across the 32 oversampled child cells, so within-cell relative diff² differences
are recoverable as:
    Δdiff²[a, b] = log(exp_Mweight[b] / exp_Mweight[a])

We compare recovar's diff² and RELION's recovered diff² on:
  (1) within-cell relative differences (clean test of formula+inputs)
  (2) absolute values up to a global additive constant (tests pdf_offset shape if any)
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
OUT = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/p0_diff2_diag")
OUT.mkdir(exist_ok=True, parents=True)


def read3(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        nz, ny, nx = struct.unpack("qqq", f.read(24))
        size_left = (Path(p).stat().st_size - 24) // (nz * ny * nx)
        dt = np.complex128 if size_left == 16 else np.float64
        return np.fromfile(f, dtype=dt, count=nz * ny * nx).reshape(nz, ny, nx)


def read_mweight(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        dims = struct.unpack("qqqqqq", f.read(48))
        n = int(np.prod(dims))
        data = np.fromfile(f, dtype=np.float64, count=n)
    return data.reshape(dims)


def read_meta(p: Path) -> dict:
    out = {}
    for line in p.read_text().strip().split("\n"):
        k, v = line.split("=")
        out[k] = v
    return out


def read_iter0_sigma2(n: int) -> np.ndarray:
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
    """RELION's Euler convention (matches Euler_angles2matrix)."""
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
    import jax
    import jax.numpy as jnp
    import mrcfile

    from recovar.core import fourier_transform_utils as ftu
    from recovar.core.relion_project import (
        centered_full_to_relion_half,
        gridding_correct_volume_real,
        relion_project_half,
    )

    print("== Loading RELION dumps ==")
    Fimg_local_shift_t0 = read3(DUMP / "p0_local_Fimg_shifted_t0.bin")[0]  # (28, 15)
    Fimg_pre_shift = read3(DUMP / "p0_Fimg.bin")[0]  # (28, 15)
    Fctf_local = read3(DUMP / "p0_local_Fctf.bin")[0]  # (28, 15)
    Minv = read3(DUMP / "p0_local_Minvsigma2.bin")[0]  # (28, 15)
    print(f"  Fimg shifted_t0: {Fimg_local_shift_t0.shape}, dtype {Fimg_local_shift_t0.dtype}")
    print(f"  Fimg pre-shift:  {Fimg_pre_shift.shape}, dtype {Fimg_pre_shift.dtype}")
    print(f"  Fctf:            {Fctf_local.shape}, dtype {Fctf_local.dtype}")
    print(f"  Minvsigma2:      {Minv.shape}, dtype {Minv.dtype}, min={Minv.min():.3e}, max={Minv.max():.3e}")

    eulers = read3(DUMP / "p0_oversampled_eulers.bin")  # (576, 8, 3)
    translations = read3(DUMP / "p0_oversampled_translations.bin")  # (29, 4, 3)
    n_orient, n_orot, _ = eulers.shape
    n_ctrans, n_otra, _ = translations.shape
    print(f"  oversampled eulers: {eulers.shape}")
    print(f"  oversampled translations: {translations.shape}")

    Mweight = read_mweight(DUMP / "p0_exp_Mweight.bin")  # (1,48,12,29,8,4)
    print(f"  exp_Mweight: shape={Mweight.shape}, sum={Mweight.sum():.3f}")
    meta = read_meta(DUMP / "p0_estep_meta.txt")
    min_diff2 = float(meta["exp_min_diff2"])
    sum_weight = float(meta["exp_sum_weight"])
    max_weight = float(meta["exp_max_weight"])
    print(f"  meta: min_diff2={min_diff2:.6f}, sum_weight={sum_weight:.4f}, max_weight={max_weight:.4e}")

    nonzero_mask = Mweight > 0
    print(f"  nonzero cells: {nonzero_mask.sum()} / {Mweight.size}")

    # ------------------------------------------------------------------
    # 1. Verify Fimg_shifted_t0 == Fimg_pre_shift × phase(translations[0,0])
    # ------------------------------------------------------------------
    N = 64
    # Pixel grid for the windowed (28, 15) half-image:
    #   ky = fft-natural in 0..H/2, -H/2..-1
    #   kx = 0..W/2 (rfft half)
    H_w, W_w = Minv.shape  # 28, 15
    # RELION's ky convention (fftw.cpp:882-906): y = i for i ∈ [0, XSIZE), y = i - YSIZE for i ∈ [XSIZE, YSIZE).
    # Differs from numpy fftfreq at i=N/2 (Nyquist): RELION has +N/2, numpy has -N/2.
    ky = np.array([i if i < W_w else i - H_w for i in range(H_w)], dtype=np.float64)
    kx = np.arange(W_w)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")

    t0 = translations[0, 0, :2]  # (sx, sy)
    print("\n== Verifying Fimg_shifted_t0 vs Fimg × phase ==")
    print(f"  translation[0, 0] = (x={t0[0]}, y={t0[1]})")
    # RELION's convention: shifted = Fimg × exp(-2πi (kx·sx + ky·sy)/N)
    phase_t0 = np.exp(-2j * np.pi * (KX * t0[0] + KY * t0[1]) / N)
    Fimg_shift_check = Fimg_pre_shift * phase_t0
    rel = np.linalg.norm(Fimg_shift_check - Fimg_local_shift_t0) / np.linalg.norm(Fimg_local_shift_t0)
    print(f"  rel_err(Fimg × phase vs dumped shift_t0): {rel:.3e}")
    if rel > 1e-6:
        # Try opposite sign
        phase_t0b = np.exp(+2j * np.pi * (KX * t0[0] + KY * t0[1]) / N)
        rel_b = np.linalg.norm(Fimg_pre_shift * phase_t0b - Fimg_local_shift_t0) / np.linalg.norm(Fimg_local_shift_t0)
        print(f"  rel_err(opposite-sign): {rel_b:.3e}")

    # ------------------------------------------------------------------
    # 2. Project iref at orient 0, oversampled rot 0 — compare to dumped Frefctf_orient0
    # ------------------------------------------------------------------
    print("\n== Loading iref + projecting orient 0 ==")
    vol_real_relion = np.asarray(
        mrcfile.open(str(FIXTURE_DIR / "run_it000_class001.mrc"), permissive=True).data, dtype=np.float64
    )
    vol_corrected = np.asarray(gridding_correct_volume_real(jnp.asarray(vol_real_relion), N, 1))
    F_centered = np.asarray(ftu.get_dft3(jnp.asarray(vol_corrected)))
    vol_half = np.asarray(centered_full_to_relion_half(F_centered))
    print(f"  iref FT half: {vol_half.shape}")

    R0 = euler_to_R(*eulers[0, 0])
    proj0 = np.array(
        relion_project_half(
            jnp.asarray(vol_half, dtype=jnp.complex128),
            jnp.asarray(R0, dtype=jnp.float64),
            H_w,
            r_max=N // 2 - 1,
            padding_factor=1,
        ),
        dtype=np.complex128,
    )
    proj0 = proj0 / (N * N)  # /N² FFT-norm to match RELION's amplitude scale

    # Compare to dumped p0_Frefctf_orient0 (10, 6) — that's the COARSE pass-1 windowed size, not 28×15.
    # We can't directly compare proj0 (28×15) to that. But we can compute Frefctf = -Fctf × proj at our 28×15 size.
    Frefctf_pose0 = -Fctf_local * proj0
    print(f"  Frefctf_pose0 norm: {np.linalg.norm(Frefctf_pose0):.4e}")

    # ------------------------------------------------------------------
    # 3. Compute recovar diff² for nonzero cells, compare to RELION
    # ------------------------------------------------------------------
    print(f"\n== Computing recovar diff² for {int(nonzero_mask.sum())} nonzero cells ==")

    # Pre-build phase shifts for all 116 oversampled translations (flatten ctrans×otra)
    trans_flat = translations[:, :, :2].reshape(-1, 2)  # (116, 2)
    n_trans = trans_flat.shape[0]
    phases = np.exp(
        -2j * np.pi * (KX[None, :, :] * trans_flat[:, 0:1, None] + KY[None, :, :] * trans_flat[:, 1:2, None]) / N
    )  # (116, H_w, W_w)
    print(f"  phases: {phases.shape}")

    # For each nonzero cell, project iref at the corresponding rotation, compute diff²
    # nonzero cells are indexed (1, idir, ipsi, itrans, iover_rot, iover_trans).
    # Flatten to (orient_idx = idir*12+ipsi, iover_rot, itrans, iover_trans).
    idxs = np.argwhere(nonzero_mask[0])  # shape (N_nz, 5): (idir, ipsi, itrans, iover_rot, iover_trans)
    print(f"  Nonzero cells to score: {len(idxs)}")

    # Project all unique (orient_idx, iover_rot) combinations encountered
    unique_rotations = set()
    for idir, ipsi, itrans, iover_rot, iover_trans in idxs:
        orient_idx = idir * 12 + ipsi
        unique_rotations.add((orient_idx, iover_rot))
    unique_rotations = sorted(unique_rotations)
    print(f"  Unique (orient, oversample_rot): {len(unique_rotations)}")

    # JIT-compiled projector
    proj_fn = jax.jit(
        lambda R: relion_project_half(
            jnp.asarray(vol_half, dtype=jnp.complex128), R, H_w, r_max=N // 2 - 1, padding_factor=1
        )
    )

    rot_to_proj = {}
    for k, (orient_idx, iover_rot) in enumerate(unique_rotations):
        idir = orient_idx // 12
        ipsi = orient_idx % 12
        eu = eulers[orient_idx, iover_rot]
        R = euler_to_R(*eu)
        p = np.asarray(proj_fn(jnp.asarray(R, dtype=jnp.float64)))
        rot_to_proj[(orient_idx, iover_rot)] = p / (N * N)
        if k < 3 or k % 100 == 0:
            print(f"    proj {k}/{len(unique_rotations)} (orient={orient_idx}, iover={iover_rot})")

    # Now compute diff² per nonzero cell
    recovar_diff2 = np.zeros(len(idxs), dtype=np.float64)
    relion_diff2_recovered = np.zeros(len(idxs), dtype=np.float64)
    for k, (idir, ipsi, itrans, iover_rot, iover_trans) in enumerate(idxs):
        orient_idx = idir * 12 + ipsi
        proj = rot_to_proj[(orient_idx, iover_rot)]
        Frefctf = -Fctf_local * proj
        # RELION cells use trans index = itrans * n_otra + iover_trans for flat phases
        flat_t = itrans * n_otra + iover_trans
        Fimg_t = Fimg_pre_shift * phases[flat_t]
        diff = Frefctf - Fimg_t
        recovar_diff2[k] = float(np.sum((diff.real**2 + diff.imag**2) * Minv * 0.5))
        relion_diff2_recovered[k] = -np.log(Mweight[0, idir, ipsi, itrans, iover_rot, iover_trans])

    # ------------------------------------------------------------------
    # 4. Analysis: the recovered RELION diff² has been shifted by min_diff² and
    #    contains an additive log(pdf_orient × pdf_offset / pdf_mean²) per cell.
    # ------------------------------------------------------------------
    print("\n== Analysis ==")
    print(f"  recovar diff² range: [{recovar_diff2.min():.4f}, {recovar_diff2.max():.4f}]")
    print(
        f"  RELION recovered diff²+const range: [{relion_diff2_recovered.min():.4f}, {relion_diff2_recovered.max():.4f}]"
    )

    # Normalize both to "relative to min in same coarse cell" — within a coarse
    # (idir, ipsi, itrans) cell, all 32 oversampled children share the same prior,
    # so the additive constant cancels.
    print("\n  Within coarse-cell relative diff² (recovar vs RELION):")
    cell_groups = {}
    for k, (idir, ipsi, itrans, iover_rot, iover_trans) in enumerate(idxs):
        cell_groups.setdefault((idir, ipsi, itrans), []).append(k)
    diffs_recovar = []
    diffs_relion = []
    n_groups_with_2plus = 0
    for cell, ks in cell_groups.items():
        if len(ks) < 2:
            continue
        n_groups_with_2plus += 1
        # use within-cell minimum as reference
        ref_recovar = recovar_diff2[ks].min()
        ref_relion = relion_diff2_recovered[ks].min()
        for k in ks:
            diffs_recovar.append(recovar_diff2[k] - ref_recovar)
            diffs_relion.append(relion_diff2_recovered[k] - ref_relion)
    diffs_recovar = np.asarray(diffs_recovar)
    diffs_relion = np.asarray(diffs_relion)
    print(f"  Groups with 2+ children: {n_groups_with_2plus}")
    print(f"  recovar Δ range: [{diffs_recovar.min():.4f}, {diffs_recovar.max():.4f}]")
    print(f"  RELION Δ range:  [{diffs_relion.min():.4f}, {diffs_relion.max():.4f}]")
    if len(diffs_recovar) > 1:
        # Linear regression: relion_diff = a × recovar_diff + b
        # Robust: just compute correlation
        ar = diffs_recovar - diffs_recovar.mean()
        br = diffs_relion - diffs_relion.mean()
        cc = float((ar * br).sum() / (np.linalg.norm(ar) * np.linalg.norm(br) + 1e-30))
        # Slope (forced through origin since both are zero at min):
        slope = float((diffs_recovar * diffs_relion).sum() / max((diffs_recovar * diffs_recovar).sum(), 1e-30))
        print(f"  Within-cell Δdiff² CC: {cc:+.6f}")
        print(f"  Within-cell Δdiff² slope (relion/recovar): {slope:.4f}")
        print(
            f"  Within-cell mean ratio (where recovar > 0.1): "
            f"{np.median(diffs_relion[diffs_recovar > 0.1] / diffs_recovar[diffs_recovar > 0.1]):.4f}"
        )

    # Save
    np.savez(
        OUT / "p0_diff2_compare.npz",
        idxs=idxs,
        recovar_diff2=recovar_diff2,
        relion_diff2_recovered=relion_diff2_recovered,
        min_diff2_relion=min_diff2,
    )
    print(f"\nSaved to {OUT}/p0_diff2_compare.npz")


if __name__ == "__main__":
    main()
