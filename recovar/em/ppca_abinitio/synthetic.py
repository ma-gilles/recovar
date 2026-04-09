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

    # Optional per-image contrast factors. Only set for family D.
    contrast_true: np.ndarray | None = None

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
    """`q` orthonormal low-frequency real-space PCs.

    Built as Gaussian-windowed polynomials (Hermite-like basis)
    so that each PC's Fourier energy is concentrated near DC.
    This makes the synthetic harness compatible with the
    factor-update's default `k_max = grid_size // 4` band-limit
    without losing significant signal. Then real-space Gram-
    Schmidt / QR enforces mutual orthonormality under the
    standard inner product.

    Without the smooth basis, the previous cosine-based PCs had
    7%+ of their energy outside `r <= grid_size/4` for the
    default 8³ grid, which made the factor update's projection
    chain destroy `U_true` even from oracle init. Caught by the
    oracle-init factor-update diagnostic test on this branch.
    """
    N0, N1, N2 = volume_shape
    z = np.linspace(-1, 1, N0)
    y = np.linspace(-1, 1, N1)
    x = np.linspace(-1, 1, N2)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    sigma = 0.5  # Gaussian width relative to the volume half-extent
    G = np.exp(-(Z**2 + Y**2 + X**2) / (2 * sigma**2))

    raw_basis = [
        Z * G,
        Y * G,
        X * G,
        (Z * Y) * G,
        (Z * X) * G,
        (Y * X) * G,
        (Z**2 - 1.0 / 3.0) * G,
        (Y**2 - 1.0 / 3.0) * G,
    ]
    if q > len(raw_basis):
        raise ValueError(
            f"q={q} exceeds the number of synthetic PCs in the test "
            f"basis ({len(raw_basis)}); add more entries to raw_basis."
        )
    pcs = np.stack(raw_basis[:q]).astype(np.float64)

    # Gram-Schmidt / QR in real space.
    flat = pcs.reshape(q, -1)
    Q, _R = np.linalg.qr(flat.T)
    flat_orth = Q.T[:q]
    return flat_orth.reshape(q, N0, N1, N2)


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


def _random_small_rotation(rng, angle_min_deg, angle_max_deg):
    """Random rotation matrix about a uniform-direction axis with
    angle uniform in `[angle_min_deg, angle_max_deg]`. Used by
    family C to apply per-image off-grid rotation jitter."""
    axis = rng.standard_normal(3)
    axis = axis / np.linalg.norm(axis)
    angle = float(rng.uniform(angle_min_deg, angle_max_deg)) * np.pi / 180.0
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


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
    pose_jitter_deg_range: tuple = (1.0, 2.0),
    pose_jitter_trans_range: tuple = (0.25, 0.5),
    contrast_range: tuple = (0.8, 1.2),
) -> SyntheticDataset:
    """Build a synthetic dataset for one of the v0 families.

    Supported families:

    - **A** (NULL) — `s_true = 0`, no heterogeneity.
    - **B** (MATCHED_GRID_HET) — continuous low-rank heterogeneity,
      poses drawn from the inference grid.
    - **C** (MISSPECIFIED_POSE) — same as B, but per-image rotation
      and translation are perturbed by small off-grid jitters
      (defaults: rotation 1-2 degrees, translation 0.25-0.5 px).
      The kernel still scores against the grid, so the true pose
      is *not* representable. The recorded `r_true_idx` /
      `t_true_idx` point at the **nearest** grid pose.
    - **D** (PER_PARTICLE_CONTRAST) — per-image scalar contrast
      `c_i ∈ contrast_range` multiplies the clean projection
      before noise. Recorded in `contrast_true`.
    - **E** (CTF_ZERO_HET) — not implemented in v0.
    """
    if family is SyntheticFamily.CTF_ZERO_HET:
        raise NotImplementedError(
            f"Family {family.value} (CTF-zero-localized heterogeneity) is not yet implemented in v0."
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
    rotations_per_image_np = np.asarray(grid.rotations)[r_true_idx]  # (n_img, 3, 3)

    # Family C: jitter rotations off-grid
    if family is SyntheticFamily.MISSPECIFIED_POSE:
        jittered = np.empty_like(rotations_per_image_np)
        for i in range(n_img):
            R_eps = _random_small_rotation(rng, pose_jitter_deg_range[0], pose_jitter_deg_range[1])
            jittered[i] = R_eps @ rotations_per_image_np[i]
        rotations_per_image_np = jittered

    rotations_per_image = jnp.asarray(rotations_per_image_np, dtype=jnp.float64)

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

    # Linear combination (heterogeneity contribution)
    alpha_jax = jnp.asarray(alpha_true)
    clean_full = mean_proj_full + jnp.einsum("iq,iqn->in", alpha_jax, u_proj_full_per_img)

    # Family D: per-particle contrast multiply (BEFORE translation/noise so the
    # contrast scales the entire image including the heterogeneity contribution).
    contrast_true_arr = None
    if family is SyntheticFamily.PER_PARTICLE_CONTRAST:
        contrast_true_arr = rng.uniform(low=contrast_range[0], high=contrast_range[1], size=n_img).astype(np.float64)
        clean_full = clean_full * jnp.asarray(contrast_true_arr)[:, None]

    # Apply per-image translation in full-image
    translations_per_image_np = np.asarray(grid.translations)[t_true_idx].astype(np.float64)
    if family is SyntheticFamily.MISSPECIFIED_POSE:
        # Sub-pixel translation jitter — uniform direction, magnitude in
        # [pose_jitter_trans_range[0], pose_jitter_trans_range[1]].
        jitter_dir = rng.standard_normal((n_img, 2))
        jitter_dir = jitter_dir / np.linalg.norm(jitter_dir, axis=1, keepdims=True)
        jitter_mag = rng.uniform(pose_jitter_trans_range[0], pose_jitter_trans_range[1], size=n_img)
        translations_per_image_np = translations_per_image_np + jitter_dir * jitter_mag[:, None]
    translations_per_image = jnp.asarray(translations_per_image_np, dtype=jnp.float64)

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
        contrast_true=contrast_true_arr,
    )
