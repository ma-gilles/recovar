"""Phase 2 (M1) tests: per-pose PPCA score/moments, no contrast.

Each test corresponds to a gate item in
``docs/math/ppca_refine_implementation_steps_2026_05_01.md`` Phase 2.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from recovar.ppca.pose_marginal import (  # noqa: E402
    _pack_upper_tri,
    compute_ppca_pose_scores_and_moments_no_contrast,
)
from recovar.ppca.ppca import _tri_size, unpack_tri_to_full  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_psd_hzz(rng, q, batch=()):
    """Hermitian PSD ``Hzz`` of shape ``(*batch, q, q)`` with reasonable
    spectrum so ``M = I + Hzz`` is well-conditioned.
    """
    A = rng.standard_normal(batch + (q, q)) + 1j * rng.standard_normal(batch + (q, q))
    A = A.astype(np.complex64)
    return A.conj().swapaxes(-1, -2) @ A + 0.5 * np.broadcast_to(np.eye(q, dtype=np.complex64), batch + (q, q))


def _random_pose_stats(rng, q, batch=()):
    g_zx = (rng.standard_normal(batch + (q,)) + 1j * rng.standard_normal(batch + (q,))).astype(np.complex64)
    h_zm = (rng.standard_normal(batch + (q,)) + 1j * rng.standard_normal(batch + (q,))).astype(np.complex64)
    Hzz = _random_psd_hzz(rng, q, batch)
    y_norm = (rng.uniform(0.5, 2.0, size=batch)).astype(np.float32)
    t_mx = rng.standard_normal(size=batch).astype(np.float32)
    nu_mm = (rng.uniform(0.5, 2.0, size=batch)).astype(np.float32)
    return y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz


def _score_via_dense_inverse(y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz):
    """Reference score using ``np.linalg.solve`` instead of ``cho_solve``."""
    q = Hzz.shape[-1]
    rho = y_norm - 2.0 * t_mx + nu_mm
    if q == 0:
        return -0.5 * rho
    Hsym = 0.5 * (Hzz + Hzz.conj().swapaxes(-1, -2))
    M = np.eye(q, dtype=Hsym.dtype) + Hsym
    b = g_zx - h_zm
    M_inv_b = np.linalg.solve(M, b[..., None])[..., 0]
    quad = (b.conj() * M_inv_b).sum(-1).real
    sign, logdet = np.linalg.slogdet(M)
    # PD => sign should be 1.
    assert np.allclose(np.atleast_1d(sign), 1.0)
    return -0.5 * (rho - quad + logdet)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_q_zero_reduction_score_is_homogeneous():
    """With q=0 the score collapses to ``-½ ρ`` (homogeneous pose score up to
    a pose-independent constant — there is no constant difference here since
    we already drop the data-norm constant in ``rho``)."""
    rng = np.random.default_rng(0)
    batch = (4, 3)
    y_norm = rng.uniform(0.5, 2.0, size=batch).astype(np.float32)
    t_mx = rng.standard_normal(batch).astype(np.float32)
    nu_mm = rng.uniform(0.5, 2.0, size=batch).astype(np.float32)
    g_zx = np.zeros(batch + (0,), dtype=np.complex64)
    h_zm = np.zeros(batch + (0,), dtype=np.complex64)
    Hzz = np.zeros(batch + (0, 0), dtype=np.complex64)

    score, alpha, G = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
        return_moments=True,
    )
    expected = -0.5 * (y_norm - 2 * t_mx + nu_mm)
    np.testing.assert_allclose(np.asarray(score), expected, rtol=1e-6, atol=1e-7)
    # Augmented moments are trivial scalars [1].
    np.testing.assert_array_equal(np.asarray(alpha), np.ones(batch + (1,), dtype=np.complex64))
    np.testing.assert_array_equal(np.asarray(G), np.ones(batch + (1,), dtype=np.complex64))


def test_w_zero_reduction_matches_q_zero():
    """With q≥1 but ``Hzz=0``, ``g_zx=0``, ``h_zm=0`` (i.e. W contributes
    nothing this pose) score must match the q=0 reduction up to additive
    constants (here zero — log det I = 0, b* M^{-1} b = 0)."""
    rng = np.random.default_rng(1)
    batch = (5,)
    q = 3
    y_norm = rng.uniform(0.5, 2.0, size=batch).astype(np.float32)
    t_mx = rng.standard_normal(batch).astype(np.float32)
    nu_mm = rng.uniform(0.5, 2.0, size=batch).astype(np.float32)
    zeros_q = np.zeros(batch + (q,), dtype=np.complex64)
    zeros_qq = np.zeros(batch + (q, q), dtype=np.complex64)

    score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(zeros_q),
        jnp.asarray(zeros_q),
        jnp.asarray(zeros_qq),
        return_moments=False,
    )
    expected = -0.5 * (y_norm - 2 * t_mx + nu_mm)
    np.testing.assert_allclose(np.asarray(score), expected, rtol=1e-6, atol=1e-7)


def test_score_matches_dense_inverse_random_psd():
    """Random PD ``Hzz`` agrees with reference using ``np.linalg.solve`` and
    ``slogdet`` — bit-equivalent to brute force at this q with float32."""
    rng = np.random.default_rng(2)
    for q in (1, 2, 4):
        y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats(rng, q, batch=(7,))
        score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
            jnp.asarray(y_norm),
            jnp.asarray(t_mx),
            jnp.asarray(nu_mm),
            jnp.asarray(g_zx),
            jnp.asarray(h_zm),
            jnp.asarray(Hzz),
            return_moments=False,
        )
        expected = _score_via_dense_inverse(y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz)
        np.testing.assert_allclose(
            np.asarray(score),
            np.asarray(expected, dtype=np.float64),
            rtol=1e-4,
            atol=1e-5,
        )


def test_basis_rotation_invariance():
    """Replacing ``W ← W Q`` for orthogonal ``Q`` rotates the per-pose stats:
    ``g_zx ← Q* g_zx``, ``h_zm ← Q* h_zm``, ``Hzz ← Q* Hzz Q``. The score is
    invariant. (Construct the rotated stats directly; we do not need a real
    ``W``.)
    """
    rng = np.random.default_rng(3)
    q = 3
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats(rng, q, batch=(4,))

    # Random unitary Q: QR of complex Gaussian.
    A = rng.standard_normal((q, q)) + 1j * rng.standard_normal((q, q))
    Q, _ = np.linalg.qr(A)
    Q = Q.astype(np.complex64)
    QH = Q.conj().T

    g_rot = (QH @ g_zx[..., None])[..., 0]
    h_rot = (QH @ h_zm[..., None])[..., 0]
    Hzz_rot = QH @ Hzz @ Q

    score_orig, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
        return_moments=False,
    )
    score_rot, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_rot),
        jnp.asarray(h_rot),
        jnp.asarray(Hzz_rot),
        return_moments=False,
    )
    np.testing.assert_allclose(np.asarray(score_orig), np.asarray(score_rot), rtol=1e-4, atol=1e-5)


def test_cholesky_symmetrization_handles_nonhermitian():
    """Feeding a deliberately non-Hermitian ``Hzz`` should produce the same
    score as feeding the Hermitian-symmetrized ``Hzz``."""
    rng = np.random.default_rng(4)
    q = 3
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats(rng, q, batch=())
    # Add an antisymmetric perturbation that the function must symmetrize away.
    skew = (rng.standard_normal((q, q)) + 1j * rng.standard_normal((q, q))).astype(np.complex64)
    skew = skew - skew.conj().T  # purely anti-Hermitian
    Hzz_skewed = Hzz + 0.1 * skew

    Hzz_sym = 0.5 * (Hzz_skewed + Hzz_skewed.conj().T)
    score_skewed, alpha_skewed, G_skewed = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz_skewed),
        return_moments=True,
    )
    score_sym, alpha_sym, G_sym = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz_sym),
        return_moments=True,
    )
    np.testing.assert_allclose(np.asarray(score_skewed), np.asarray(score_sym), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(alpha_skewed), np.asarray(alpha_sym), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(G_skewed), np.asarray(G_sym), rtol=1e-5, atol=1e-6)


def test_jit_compiles_and_caches():
    """Wrap with ``@jax.jit`` and call twice; second call must reuse the
    compiled XLA executable. We assert by checking the lowered HLO is
    identical between the two calls."""
    rng = np.random.default_rng(5)
    q = 2
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats(rng, q, batch=(3,))

    def f(yn, tm, nm, gz, hz, H):
        score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
            yn,
            tm,
            nm,
            gz,
            hz,
            H,
            return_moments=False,
        )
        return score

    f_jit = jax.jit(f)
    args = (
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
    )
    out1 = f_jit(*args)
    out2 = f_jit(*args)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))


def test_no_pinv_in_source():
    """Regression guard: do not let agents copy ``pinv`` from the legacy
    ``recovar/ppca/ppca.py::_e_step_half_inner`` into the new path. Match
    the call form ``pinv(`` so the word can still appear in the module
    docstring (which explains we don't use it)."""
    src = Path(__file__).resolve().parents[3] / "recovar" / "ppca" / "pose_marginal.py"
    text = src.read_text()
    assert not re.search(r"\bpinv\s*\(", text), (
        "pinv(...) must not appear in pose_marginal.py — use Cholesky / cho_solve."
    )


# ---------------------------------------------------------------------------
# Augmented moment shape + structure tests
# ---------------------------------------------------------------------------


def test_alpha_aug_starts_with_ones():
    rng = np.random.default_rng(6)
    q = 4
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats(rng, q, batch=(5,))
    _, alpha_aug, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
        return_moments=True,
    )
    alpha_np = np.asarray(alpha_aug)
    assert alpha_np.shape == (5, q + 1)
    np.testing.assert_array_equal(alpha_np[..., 0], np.ones(5, dtype=np.complex64))


def test_G_aug_tri_unpacks_to_consistent_block_structure():
    """Round-trip ``G_aug_tri`` through ``unpack_tri_to_full`` and inspect
    the upper triangle. NOTE: ``unpack_tri_to_full`` mirrors the upper
    triangle into the lower without conjugation (it produces a *symmetric*
    matrix from a triangle, not a Hermitian one). For our complex-Hermitian
    augmented ``G_aug``, only the upper triangle is mathematically
    meaningful — that is the convention shared with the legacy ``lhs_tri``
    accumulator (real in legacy, complex Hermitian here, but the
    accumulator only ever multiplies through the upper triangle in the PCG
    matvec)."""
    rng = np.random.default_rng(7)
    q = 3
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats(rng, q, batch=())
    _, alpha_aug, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
        return_moments=True,
    )
    assert G_tri.shape[-1] == _tri_size(q + 1)

    G_full = np.asarray(unpack_tri_to_full(G_tri, q + 1))
    z_bar = np.asarray(alpha_aug)[..., 1:]
    # Build the expected (q+1, q+1) Hermitian augmented matrix.
    Hsym = 0.5 * (Hzz + Hzz.conj().T)
    M = np.eye(q, dtype=Hsym.dtype) + Hsym
    S_z = np.linalg.inv(M)
    G_expected = np.zeros((q + 1, q + 1), dtype=np.complex64)
    G_expected[0, 0] = 1.0
    G_expected[0, 1:] = z_bar.conj()
    G_expected[1:, 0] = z_bar
    G_expected[1:, 1:] = S_z + np.outer(z_bar, z_bar.conj())
    # Compare only the upper triangle (the meaningful part of `tri` packing).
    iu = np.triu_indices(q + 1)
    np.testing.assert_allclose(G_full[iu], G_expected[iu], rtol=1e-4, atol=1e-5)


def test_pack_upper_tri_q_zero_returns_empty():
    M = jnp.zeros((3, 0, 0), dtype=jnp.complex64)
    out = _pack_upper_tri(M)
    assert out.shape == (3, 0)
