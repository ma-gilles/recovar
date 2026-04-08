"""Synthetic data harness for PPCA-ab-initio v0 stage gates.

Per spec Section 9, the harness must support five families. v0 ships
with the two graduation-gate families:

- Family A — **Null**. `s_true = 0`, no heterogeneity. PPCA must
  show no gain on this family.
- Family B — **Matched-grid heterogeneous**. Continuous low-rank
  heterogeneity, poses drawn from the inference grid.

Families C (misspecified pose), D (per-particle contrast), and E
(CTF-zero-localized heterogeneity) are tracked in TODO and will be
added when Stage 1C needs them.

Construction strategy
---------------------

To avoid materializing one 3D volume per image, the harness slices
`mu` and each row of `U` once through the random per-image
rotations, then composes the clean image as

    y_clean[i] = mean_proj[i] + Σ_k alpha[i, k] · u_proj[i, k]

For real-derived `mu` and `U`, this is exactly the projection of
`mu + Σ_k alpha[i, k] U[k]` through rotation `R_i`. The result is
Hermitian-symmetric (full-image FT layout), and noise is added in
real space and FT'd so that the noise term is also Hermitian.

Per spec Section 4.3, `noise_variance` is in **Fourier units**:
`noise_variance[k] = sigma_real² · N_full`.

Output dataclass
----------------

`SyntheticDataset` bundles everything the kernel needs to be called
on this batch:

- `mu_half_true`, `U_half_true`, `s_true` — ground-truth model state
- `batch_full` — `(n_img, N_full)` complex128 noisy images
- `rotations` — `(n_rot, 3, 3)` grid the kernel scores against
- `translations` — `(n_trans, 2)` grid
- `ctf_params`, `noise_variance_full` — kernel inputs
- `r_true_idx`, `t_true_idx` — per-image true grid indices
- `alpha_true` — per-image latent coordinates
- `volume_shape`, `image_shape`
- `train_idx`, `val_idx` — train/validation split (per spec Q6)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
import numpy as np

import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from recovar.core.slicing import slice_volume

from .types import FixedGridSpec


class SyntheticFamily(Enum):
    NULL = "A"
    MATCHED_GRID_HET = "B"
    MISSPECIFIED_POSE = "C"  # not yet implemented
    PER_PARTICLE_CONTRAST = "D"  # not yet implemented
    CTF_ZERO_HET = "E"  # not yet implemented


@dataclass
class SyntheticDataset:
    """Bundle of everything one Stage 0B/1A/1B/1C call needs.

    Field shapes are documented in the module docstring.
    """

    family: SyntheticFamily
    image_shape: tuple
    volume_shape: tuple

    mu_half_true: jnp.ndarray
    U_half_true: jnp.ndarray
    s_true: jnp.ndarray

    batch_full: jnp.ndarray
    ctf_params: jnp.ndarray
    noise_variance_full: jnp.ndarray
    rotations: jnp.ndarray
    translations: jnp.ndarray

    r_true_idx: np.ndarray
    t_true_idx: np.ndarray
    alpha_true: np.ndarray

    train_idx: np.ndarray
    val_idx: np.ndarray

    @property
    def n_img(self) -> int:
        return int(self.batch_full.shape[0])

    @property
    def n_rot(self) -> int:
        return int(self.rotations.shape[0])

    @property
    def n_trans(self) -> int:
        return int(self.translations.shape[0])

    @property
    def q(self) -> int:
        return int(self.U_half_true.shape[0])


# ---------------------------------------------------------------------------
# Real-space ground truth construction
# ---------------------------------------------------------------------------


def _gaussian_blob_volume(volume_shape, sigma=0.4):
    """A smooth Gaussian density centered in the volume."""
    N0, N1, N2 = volume_shape
    z = np.linspace(-1, 1, N0)
    y = np.linspace(-1, 1, N1)
    x = np.linspace(-1, 1, N2)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    return np.exp(-(Z**2 + Y**2 + X**2) / (2 * sigma**2)).astype(np.float64)


def _sinusoidal_pcs_real(volume_shape, q):
    """`q` orthogonal-ish low-frequency real-space PCs."""
    N0, N1, N2 = volume_shape
    z = np.linspace(-1, 1, N0)
    y = np.linspace(-1, 1, N1)
    x = np.linspace(-1, 1, N2)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    pcs = []
    for k in range(q):
        kz = (k % 2) + 1
        ky = ((k // 2) % 2) + 1
        kx = (k // 4) + 1
        pc = np.cos(np.pi * kz * Z) * np.cos(np.pi * ky * Y) * np.cos(np.pi * kx * X)
        # Normalize to unit L2 in real space
        pc = pc / np.sqrt(np.sum(pc**2))
        pcs.append(pc.astype(np.float64))
    return np.stack(pcs)  # (q, N0, N1, N2)


def _real_volume_to_half_flat(real_vol):
    return ftu.get_dft3_real(jnp.asarray(real_vol)).reshape(-1)


# ---------------------------------------------------------------------------
# Per-image image construction
# ---------------------------------------------------------------------------


def _slice_real_volumes_to_full_image(real_vols, rotations, image_shape, volume_shape):
    """Slice each volume in `real_vols[i]` through the rotation
    `rotations[i]`, returning `(n_img, full_image_size)` complex128
    in centered FT layout.

    For Stage 0B simplicity we use a Python loop and a JIT'd inner
    slice. n_img is small (≤ 256) so this is fine for v0.
    """
    n_img = real_vols.shape[0]
    out = []
    for i in range(n_img):
        vol_full_ft = ftu.get_dft3(jnp.asarray(real_vols[i]))
        rot_i = rotations[i : i + 1]
        proj = slice_volume(
            vol_full_ft.reshape(-1),
            rot_i,
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=False,
            half_image=False,
        )
        out.append(proj.reshape(-1))
    return jnp.asarray(jnp.stack(out), dtype=jnp.complex128)


def _slice_half_volume_through_rotations(half_vol_flat, rotations, image_shape, volume_shape):
    """Slice a single half-volume through `rotations`, returning
    `(n_rot, full_image_size)` complex128. Used to project `mu`
    and each `U` row to the (per-image) rotation set.
    """
    proj = slice_volume(
        half_vol_flat,
        rotations,
        image_shape,
        volume_shape,
        "linear_interp",
        half_volume=True,
        half_image=False,
    )
    return proj.astype(jnp.complex128)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def make_synthetic_fixed_grid_dataset(
    family: SyntheticFamily,
    *,
    volume_shape,
    image_shape,
    grid: FixedGridSpec,
    q: int,
    n_images_train: int,
    n_images_val: int,
    sigma_real: float = 1.0,
    seed: int = 0,
) -> SyntheticDataset:
    """Build a synthetic dataset for one of the v0 families.

    Parameters
    ----------
    family : SyntheticFamily
        A (null) or B (matched-grid heterogeneous). C/D/E raise
        NotImplementedError until they are added.
    volume_shape : 3-tuple
    image_shape : 2-tuple
    grid : FixedGridSpec
        The pose / translation grid the kernel will score against.
        True per-image poses are drawn from this grid for families
        A and B (matched-grid).
    q : int
        Latent dimensionality.
    n_images_train, n_images_val : int
        Sizes of the train/validation splits.
    sigma_real : float
        Real-space noise standard deviation per pixel. Translated
        to Fourier-units `noise_variance = sigma_real² · N_full`.
    seed : int
        Master RNG seed.
    """
    if family not in (SyntheticFamily.NULL, SyntheticFamily.MATCHED_GRID_HET):
        raise NotImplementedError(
            f"Family {family.value} not yet implemented in v0; only A (NULL) and B (MATCHED_GRID_HET) are supported."
        )

    rng = np.random.default_rng(seed)
    n_img = n_images_train + n_images_val

    # Ground-truth real-space mu and U
    mu_real = _gaussian_blob_volume(volume_shape, sigma=0.4)
    U_real = _sinusoidal_pcs_real(volume_shape, q)  # (q, N0, N1, N2)

    # Spectrum
    if family is SyntheticFamily.NULL:
        s_true = np.zeros(q, dtype=np.float64)
    else:
        s_true = (1.0 / (np.arange(q) + 1.0)).astype(np.float64)

    # Encode mu and U into half-volume layout
    mu_half = _real_volume_to_half_flat(mu_real)
    U_half = jnp.stack([_real_volume_to_half_flat(U_real[k]) for k in range(q)])

    # Sample per-image latents and pose indices
    if family is SyntheticFamily.NULL:
        alpha_true = np.zeros((n_img, q), dtype=np.float64)
    else:
        alpha_true = (rng.standard_normal((n_img, q)) * np.sqrt(s_true)).astype(np.float64)

    n_rot = int(grid.rotations.shape[0])
    n_trans = int(grid.translations.shape[0])
    r_true_idx = rng.integers(low=0, high=n_rot, size=n_img).astype(np.int32)
    t_true_idx = rng.integers(low=0, high=n_trans, size=n_img).astype(np.int32)

    # Compose clean projections per image:
    #   y_clean[i] = slice(mu_real + Σ_k alpha[i,k] U_real[k], R_{r_i})
    #              = slice(mu)[r_i] + Σ_k alpha[i,k] · slice(U_k)[r_i]
    # Slice mu and each U through the per-image rotation set.
    rotations_per_image = grid.rotations[r_true_idx]  # (n_img, 3, 3)
    mean_proj_full = _slice_half_volume_through_rotations(
        mu_half, rotations_per_image, image_shape, volume_shape
    )  # (n_img, N_full)
    u_proj_full = jnp.stack(
        [
            _slice_half_volume_through_rotations(U_half[k], rotations_per_image, image_shape, volume_shape)
            for k in range(q)
        ]
    )  # (q, n_img, N_full)
    u_proj_full_per_img = jnp.transpose(u_proj_full, (1, 0, 2))  # (n_img, q, N_full)

    # Linear combination
    alpha_jax = jnp.asarray(alpha_true)
    clean_full = mean_proj_full + jnp.einsum("iq,iqn->in", alpha_jax, u_proj_full_per_img)

    # Apply per-image translation in full-image
    translations_per_image = grid.translations[t_true_idx]  # (n_img, 2)
    # batch_trans_translate_images expects (n_img, n_trans, 2). We have n_trans=1
    # per image (one shift per image). Apply via the same primitive.
    trans_for_shift = translations_per_image[:, None, :]  # (n_img, 1, 2)
    shifted = core.batch_trans_translate_images(clean_full, trans_for_shift, image_shape)
    shifted = shifted[:, 0, :]  # drop the n_trans=1 axis → (n_img, N_full)

    # CTF: identity for v0
    # (no-op)

    # Noise: real-space white noise of variance sigma_real² → FT → Hermitian
    H, W = image_shape
    N_full = H * W
    noise_real = (sigma_real * rng.standard_normal((n_img, H, W))).astype(np.float64)
    noise_full = jnp.stack([ftu.get_dft2(jnp.asarray(noise_real[i])).reshape(-1) for i in range(n_img)])

    batch_full = jnp.asarray(shifted + noise_full, dtype=jnp.complex128)

    # CTF params: zero placeholders (the kernel test path uses identity CTF
    # but a real CryoEMDataset would carry per-image CTF params).
    ctf_params = jnp.zeros((n_img, 9), dtype=jnp.float64)

    # Noise variance in Fourier units
    noise_variance_full = jnp.full(N_full, sigma_real**2 * N_full, dtype=jnp.float64)

    # Train/val split
    perm = rng.permutation(n_img).astype(np.int32)
    train_idx = perm[:n_images_train]
    val_idx = perm[n_images_train:]

    return SyntheticDataset(
        family=family,
        image_shape=tuple(int(x) for x in image_shape),
        volume_shape=tuple(int(x) for x in volume_shape),
        mu_half_true=mu_half.astype(jnp.complex128),
        U_half_true=U_half.astype(jnp.complex128),
        s_true=jnp.asarray(s_true, dtype=jnp.float64),
        batch_full=batch_full,
        ctf_params=ctf_params,
        noise_variance_full=noise_variance_full,
        rotations=grid.rotations,
        translations=grid.translations,
        r_true_idx=np.asarray(r_true_idx),
        t_true_idx=np.asarray(t_true_idx),
        alpha_true=alpha_true,
        train_idx=np.asarray(train_idx),
        val_idx=np.asarray(val_idx),
    )
