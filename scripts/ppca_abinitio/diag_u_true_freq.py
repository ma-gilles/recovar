"""Diagnostic: examine U_true frequency content for Ribosembly.

Question: does U_true have its energy concentrated in low frequencies
(in which case band-limiting could help) or spread across all freqs
(in which case band-limit zeros out signal)?

For each PC k of U_true, report the fraction of energy within
radial frequency r <= R for R in {2, 4, 6, 8, 10, 12, 14, 16}.
Volume side D=32, Nyquist at R=16.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.utils.helpers import load_mrc


def downsample_volume(vol_real, target_D):
    D = vol_real.shape[-1]
    if target_D == D:
        return vol_real.astype(np.float64)
    F = np.asarray(ftu.get_dft3(jnp.asarray(vol_real)), dtype=np.complex128)
    c = D // 2
    h = target_D // 2
    F_crop = F[c - h : c + h, c - h : c + h, c - h : c + h]
    out = np.array(np.asarray(ftu.get_idft3(jnp.asarray(F_crop))).real, dtype=np.float64, copy=True)
    out *= (target_D / D) ** 3
    return out


def main():
    D = 32
    q = 8
    root = Path("/home/mg6942/mytigress/cryobench2") / "Ribosembly"
    vol_dir = root / "vols" / "128_org"
    vol_files = sorted(vol_dir.glob("*.mrc"))
    print(f"loading {len(vol_files)} volumes from {vol_dir}", flush=True)
    vols_orig = np.stack([np.asarray(load_mrc(str(vf)), dtype=np.float64) for vf in vol_files])
    print(f"orig shape: {vols_orig.shape}", flush=True)
    vols = np.stack([downsample_volume(vols_orig[k], D) for k in range(vols_orig.shape[0])])
    print(f"downsampled shape: {vols.shape}", flush=True)

    # mu_true and U_true (as in synthetic.py)
    mu_real = vols.mean(axis=0)
    centered_flat = (vols - mu_real[None]).reshape(vols.shape[0], -1)
    _, S_svd, Vh = np.linalg.svd(centered_flat, full_matrices=False)
    U_real = Vh[:q].reshape((q, D, D, D))

    # Spectrum analysis: for each PC, compute its 3D FT and group by radial index
    print("\n=== U_true frequency content ===", flush=True)
    print(f"Singular values (top {q}): {S_svd[:q]}", flush=True)
    print(f"Total variance ratio: {(S_svd[:q] ** 2).sum() / (S_svd**2).sum():.4f}", flush=True)
    print()

    # Use centered FFT (full FT, not half) for clarity
    radii = [2, 4, 6, 8, 10, 12, 14, 16, 28]  # 28 ≈ corner of cube
    print(f"  {'PC':3s}  " + "  ".join([f"r<={r:>3d}" for r in radii]), flush=True)
    for k in range(q):
        v = U_real[k]
        F = np.fft.fftshift(np.fft.fftn(v))
        # Build radial index
        c = D // 2
        z = np.arange(D) - c
        ZZ, YY, XX = np.meshgrid(z, z, z, indexing="ij")
        R = np.sqrt(ZZ * ZZ + YY * YY + XX * XX)
        total_E = (np.abs(F) ** 2).sum()
        fracs = []
        for r in radii:
            mask = R <= r
            E_in = (np.abs(F[mask]) ** 2).sum()
            fracs.append(E_in / total_E)
        print(f"  PC{k:2d}  " + "  ".join([f"{f:>5.3f}" for f in fracs]), flush=True)

    # Also report for mu_real (compare against U)
    F_mu = np.fft.fftshift(np.fft.fftn(mu_real))
    total_E_mu = (np.abs(F_mu) ** 2).sum()
    fracs_mu = []
    for r in radii:
        mask = R <= r
        E_in = (np.abs(F_mu[mask]) ** 2).sum()
        fracs_mu.append(E_in / total_E_mu)
    print("  mu  " + "  ".join([f"{f:>5.3f}" for f in fracs_mu]), flush=True)

    # Where is most of U_true energy? Average across PCs
    avg_frac = []
    for r in radii:
        mask = R <= r
        e_in_total = sum((np.abs(np.fft.fftshift(np.fft.fftn(U_real[k])))[mask] ** 2).sum() for k in range(q))
        e_total = sum((np.abs(np.fft.fftshift(np.fft.fftn(U_real[k]))) ** 2).sum() for k in range(q))
        avg_frac.append(e_in_total / e_total)
    print("  avg " + "  ".join([f"{f:>5.3f}" for f in avg_frac]), flush=True)

    # What fraction of U_true energy is INSIDE the inscribed sphere (r<=D/2=16)?
    inside_sphere = avg_frac[7]  # r<=16
    print(f"\nFraction of U_true energy within inscribed sphere (r<=16): {inside_sphere:.4f}", flush=True)
    print(f"Fraction within k_max=N/4=8: {avg_frac[3]:.4f}", flush=True)
    print(f"Fraction within k_max=N/2=12 (3/4 nyquist): {avg_frac[5]:.4f}", flush=True)


if __name__ == "__main__":
    main()
