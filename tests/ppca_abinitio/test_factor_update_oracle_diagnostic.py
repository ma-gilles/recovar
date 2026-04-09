"""Diagnostic tests for the factor-update fixed-point at oracle init.

These tests pin two bugs that were fixed during this branch's
debugging session:

1. **Wirtinger gradient direction.** `jax.value_and_grad` for a
   real-valued loss `f(z)` with complex `z` returns the conjugate
   of the descent direction (Wirtinger calculus convention). The
   correct steepest-descent step is `z - lr * conj(grad)`, not
   `z - lr * grad`. The factor update originally used the latter,
   which sent the gradient in the *ascending* direction and
   destroyed `U` even from oracle init.

2. **Hermitian-symmetric gradient projection.** The free complex
   gradient does not respect the half-volume rfft layout's
   Hermitian-symmetry constraint. After the gradient step, `U`
   has random imaginary content in conjugate-symmetric pairs that
   the subsequent Cholesky orthonormalization rotates wildly.
   The fix is to apply `project_to_real_volume_subspace_batch`
   before the orthonormalize.

3. **Synthetic harness PC orthonormality.** `_sinusoidal_pcs_real`
   was building cosine-based PCs that were not mutually orthogonal
   in real space (~22% overlap on the default 8³ grid). Fixed
   via QR. Synthetic PCs are now Hermite-like Gaussian-windowed
   polynomials that have most of their energy at low frequencies.

Together these mean: at oracle init with very small `lr`, the
factor update should leave `U` essentially unchanged (modulo the
band-limit floor of ~0.155 at the toy 8³ scale). At a perturbed
init, the update should reduce projector error.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.factor_update import update_factor_one_outer_step
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    half_real_space_gram,
    make_half_volume_weights,
)
from recovar.em.ppca_abinitio.init import init_oracle, init_truth_perturbed
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(batch, apply_image_mask=False):
    return batch


class _SyntheticConfig(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, ctf_params, *, half_image=False):
        full = _identity_ctf(ctf_params, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, batch, apply_image_mask=False):
        return _identity_process(batch, apply_image_mask=apply_image_mask)


def _make_dataset(seed=0, sigma=0.001, n_train=128):
    grid = build_fixed_grid(healpix_order=0, max_shift=1)
    return make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=VOLUME_SHAPE,
        image_shape=IMAGE_SHAPE,
        grid=grid,
        q=2,
        n_images_train=n_train,
        n_images_val=2,
        sigma_real=sigma,
        seed=seed,
    )


def test_synthetic_U_true_is_real_space_orthonormal():
    """Pin the Gram-Schmidt fix in `_sinusoidal_pcs_real`. Without
    it, U_true had ~22% off-diagonal Gram entries."""
    ds = _make_dataset()
    weights = make_half_volume_weights(VOLUME_SHAPE)
    G = np.asarray(half_real_space_gram(ds.U_half_true, weights, int(np.prod(VOLUME_SHAPE))))
    np.testing.assert_allclose(G, np.eye(2), atol=1e-12)


def test_synthetic_U_true_concentrated_in_low_frequencies():
    """The new Hermite-like PCs should have ≥99% of their energy
    inside the toy-size band-limit radius `k_max=2.5`."""
    from recovar.em.ppca_abinitio.half_volume import half_volume_radial_index

    ds = _make_dataset()
    R = np.asarray(half_volume_radial_index(VOLUME_SHAPE))
    weights = np.asarray(make_half_volume_weights(VOLUME_SHAPE))
    inside = R <= 2.5
    for k in range(2):
        u = np.asarray(ds.U_half_true[k])
        e_in = float(np.sum(weights[inside] * np.abs(u[inside]) ** 2))
        e_total = float(np.sum(weights * np.abs(u) ** 2))
        assert e_in / e_total > 0.99, f"U_true[{k}] has only {100 * e_in / e_total:.2f}% energy inside r<=2.5"


def test_oracle_init_factor_update_at_tiny_lr_stays_at_band_limit_floor():
    """At oracle init with lr near 0, the factor update should leave
    U at the band-limit floor (the only loss is from the band limit
    + orthonormalize chain). For this synthetic harness the floor
    is ~0.155 in projector_frobenius_error."""
    ds = _make_dataset()
    cfg = _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
    )
    out = update_factor_one_outer_step(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        inner_steps=1,
        lr=1e-9,
        k_max=2.5,
    )
    err = projector_frobenius_error(out.U, ds.U_half_true, VOLUME_SHAPE)
    # Band-limit floor for q=2, k_max=2.5, with the Hermite-like PCs
    assert 0.10 <= err <= 0.20, (
        f"oracle-init projector_err = {err:.4f}, expected in "
        "[0.10, 0.20] (band-limit floor). Outside this band means "
        "either the gradient direction is wrong (Wirtinger fix) or "
        "the synthetic PCs have grown new high-frequency content."
    )


def test_oracle_init_factor_update_at_safe_lr_stays_near_floor():
    """At a small lr (1e-7), the oracle-init factor update should
    stay at the band-limit floor (~0.155 for the 8³ toy harness).
    Smaller lr is needed at toy size because the loss surface near
    the oracle is sharply curved; ECM with line search handles
    this automatically.

    Pre-Wirtinger-fix this would give 1.99 regardless of lr because
    the gradient was being applied in the wrong direction.
    """
    ds = _make_dataset()
    cfg = _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
    )
    out = update_factor_one_outer_step(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        inner_steps=3,
        lr=1e-7,
        k_max=2.5,
    )
    err = projector_frobenius_error(out.U, ds.U_half_true, VOLUME_SHAPE)
    assert err < 0.20, (
        f"oracle-init factor update at safe lr produced projector_err "
        f"= {err:.4f}; expected < 0.20 (band-limit floor ~0.155). "
        "A regression here means the gradient direction or projection "
        "is wrong."
    )


def test_perturbed_init_factor_update_actually_descends():
    """From a 0.3-perturbed init, the factor update should reduce
    projector error by at least 0.2 absolute. Before the Wirtinger
    fix, the update gave only ~0.12 of improvement; after, it gives
    ~0.30."""
    ds = _make_dataset(sigma=0.2)
    cfg = _SyntheticConfig(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=1.0)
    init = init_truth_perturbed(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=ds.s_true,
        volume_shape=VOLUME_SHAPE,
        eps_mu=0.0,
        eps_U=0.3,
        seed=0,
    )
    err0 = projector_frobenius_error(init.U, ds.U_half_true, VOLUME_SHAPE)
    out = update_factor_one_outer_step(
        cfg,
        init,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        inner_steps=3,
        lr=1e-3,
        k_max=2.5,
    )
    err1 = projector_frobenius_error(out.U, ds.U_half_true, VOLUME_SHAPE)
    improvement = err0 - err1
    assert improvement > 0.2, (
        f"factor update only improved projector by {improvement:.4f} (from {err0:.4f} to {err1:.4f}); expected > 0.2"
    )
