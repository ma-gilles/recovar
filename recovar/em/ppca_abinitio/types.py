"""Typed data containers for the PPCA ab-initio v0 module.

See docs/math/plan_ppca_abinitio_v0.md Section 12 for the API
contract. All array fields are float64 / complex128; the dtype
contract is enforced at construction time so that float32
silently sneaking in fails loudly instead of producing wrong
numbers below the float32 precision floor (per Audit 2).

Pytree-relevant containers (PPCAInit, FixedGridSpec, PosteriorStats)
are equinox Modules so they can be passed through JIT'd functions.
PosteriorBlock contains slice objects and is yielded from a plain
Python generator, so it is a regular dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Dtype contract helpers
# ---------------------------------------------------------------------------


def _require_complex128(arr, name: str) -> jnp.ndarray:
    arr = jnp.asarray(arr)
    if arr.dtype != jnp.complex128:
        raise TypeError(
            f"{name}: expected complex128, got {arr.dtype}. "
            "PPCA-abinitio v0 runs strictly in float64 mode (see spec Section 0.1 / 5.2)."
        )
    return arr


def _require_float64(arr, name: str) -> jnp.ndarray:
    arr = jnp.asarray(arr)
    if arr.dtype != jnp.float64:
        raise TypeError(
            f"{name}: expected float64, got {arr.dtype}. "
            "PPCA-abinitio v0 runs strictly in float64 mode (see spec Section 0.1 / 5.2)."
        )
    return arr


# ---------------------------------------------------------------------------
# PPCAInit — current (μ, U, s) state
# ---------------------------------------------------------------------------


class PPCAInit(eqx.Module):
    """Current PPCA model state.

    Per spec Section 4.2, `mu` and each row of `U` correspond to
    real-space 3D volumes, stored in **half-volume rfft layout**
    (`(N0, N1, N2//2+1)` complex128, flattened). The half-volume
    layout makes Hermitian symmetry structural — there is no
    projection-back step. See
    `recovar.em.ppca_abinitio.half_volume` for the inner-product
    weighting (`make_half_volume_weights`) and the real-space
    orthonormalization (`real_volume_orthonormalize_half`).

    `volume_shape` is the **full** real-space volume shape
    `(N0, N1, N2)`. The flat half-volume size is
    `N0 * N1 * (N2//2 + 1)` and is stored separately as
    `half_volume_size` for downstream consumers.
    """

    mu: jnp.ndarray  # (half_volume_size,) complex128 — rfft-packed
    U: jnp.ndarray  # (q, half_volume_size) complex128 — rfft-packed
    s: jnp.ndarray  # (q,) float64
    volume_shape: tuple = eqx.field(static=True)

    def __init__(self, mu, U, s, volume_shape):
        import recovar.core.fourier_transform_utils as ftu

        mu = _require_complex128(mu, "PPCAInit.mu")
        U = _require_complex128(U, "PPCAInit.U")
        s = _require_float64(s, "PPCAInit.s")

        volume_shape = tuple(int(x) for x in volume_shape)
        if len(volume_shape) != 3:
            raise ValueError(f"volume_shape must have 3 dims, got {volume_shape}")
        half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        expected_half_size = int(np.prod(half_shape))

        if mu.ndim != 1 or mu.shape[0] != expected_half_size:
            raise ValueError(
                f"mu must be 1-D with half_volume_size={expected_half_size} "
                f"(from volume_shape={volume_shape}, half_shape={half_shape}), got shape {mu.shape}"
            )
        if U.ndim != 2 or U.shape[1] != expected_half_size:
            raise ValueError(f"U must be (q, half_volume_size={expected_half_size}), got shape {U.shape}")
        if s.ndim != 1 or s.shape[0] != U.shape[0]:
            raise ValueError(f"s must be (q={U.shape[0]},), got shape {s.shape}")

        self.mu = mu
        self.U = U
        self.s = s
        self.volume_shape = volume_shape

    @property
    def q(self) -> int:
        return int(self.U.shape[0])

    @property
    def half_volume_size(self) -> int:
        return int(self.mu.shape[0])

    @property
    def volume_size(self) -> int:
        """Total number of real-space voxels (`prod(volume_shape)`)."""
        return int(np.prod(self.volume_shape))


# ---------------------------------------------------------------------------
# FixedGridSpec — pose / translation grid
# ---------------------------------------------------------------------------


class FixedGridSpec(eqx.Module):
    """Pose and translation grid for the v0 fixed-grid loop.

    Per spec Q4, v0 supports HEALPix order 2 only. Order 3+ is
    Phase 4 and requires the streaming posterior API.

    `log_prior_rot` / `log_prior_trans` exist in the API even
    though v0 leaves them flat (uniform); the plumbing is here so
    that future non-uniform priors do not require an API break.
    """

    rotations: jnp.ndarray  # (n_rot, 3, 3) float64
    translations: jnp.ndarray  # (n_trans, 2) float64
    log_prior_rot: jnp.ndarray | None  # (n_rot,) or None
    log_prior_trans: jnp.ndarray | None  # (n_trans,) or None

    def __init__(self, rotations, translations, log_prior_rot=None, log_prior_trans=None):
        rotations = _require_float64(rotations, "FixedGridSpec.rotations")
        translations = _require_float64(translations, "FixedGridSpec.translations")
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(f"rotations must be (n_rot, 3, 3), got shape {rotations.shape}")
        if translations.ndim != 2 or translations.shape[1] != 2:
            raise ValueError(f"translations must be (n_trans, 2), got shape {translations.shape}")
        if log_prior_rot is not None:
            log_prior_rot = _require_float64(log_prior_rot, "FixedGridSpec.log_prior_rot")
            if log_prior_rot.shape != (rotations.shape[0],):
                raise ValueError(f"log_prior_rot shape {log_prior_rot.shape} != (n_rot={rotations.shape[0]},)")
        if log_prior_trans is not None:
            log_prior_trans = _require_float64(log_prior_trans, "FixedGridSpec.log_prior_trans")
            if log_prior_trans.shape != (translations.shape[0],):
                raise ValueError(f"log_prior_trans shape {log_prior_trans.shape} != (n_trans={translations.shape[0]},)")
        self.rotations = rotations
        self.translations = translations
        self.log_prior_rot = log_prior_rot
        self.log_prior_trans = log_prior_trans

    @property
    def n_rot(self) -> int:
        return int(self.rotations.shape[0])

    @property
    def n_trans(self) -> int:
        return int(self.translations.shape[0])


# ---------------------------------------------------------------------------
# PPCAConfig — loop hyperparameters
# ---------------------------------------------------------------------------


class PPCAConfig(eqx.Module):
    """Loop hyperparameters.

    Per spec Section 8.1, `update_s=True` is forbidden until Stage
    1D; the constructor enforces this.
    """

    n_iters: int = eqx.field(static=True)
    update_mu: bool = eqx.field(static=True)
    update_factor: bool = eqx.field(static=True)
    update_s: bool = eqx.field(static=True)
    factor_inner_steps: int = eqx.field(static=True)
    factor_lr: float = eqx.field(static=True)
    ridge_lambda: float = eqx.field(static=True)
    rot_block_size: int = eqx.field(static=True)
    trans_block_size: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)

    def __init__(
        self,
        n_iters: int,
        *,
        update_mu: bool = True,
        update_factor: bool = False,
        update_s: bool = False,
        factor_inner_steps: int = 3,
        factor_lr: float = 1e-2,
        ridge_lambda: float = 1e-4,
        rot_block_size: int = 256,
        trans_block_size: int = 16,
        seed: int = 0,
    ):
        if update_s:
            raise ValueError(
                "update_s=True is forbidden in v0 (spec Section 8.1, Q2). "
                "Stage 1D is the earliest stage where s may be updated, and "
                "1D is not part of v0 yet."
            )
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if factor_inner_steps < 0:
            raise ValueError(f"factor_inner_steps must be >= 0, got {factor_inner_steps}")
        if rot_block_size < 1 or trans_block_size < 1:
            raise ValueError("block sizes must be positive")
        self.n_iters = int(n_iters)
        self.update_mu = bool(update_mu)
        self.update_factor = bool(update_factor)
        self.update_s = False
        self.factor_inner_steps = int(factor_inner_steps)
        self.factor_lr = float(factor_lr)
        self.ridge_lambda = float(ridge_lambda)
        self.rot_block_size = int(rot_block_size)
        self.trans_block_size = int(trans_block_size)
        self.seed = int(seed)


# ---------------------------------------------------------------------------
# PosteriorStats — fully-materialized posterior tensors (test-only)
# ---------------------------------------------------------------------------


class PosteriorStats(eqx.Module):
    """Fully-materialized posterior tensors.

    For TINY problems and Stage 0A correctness checks only. Real
    workloads must use `iter_posterior_blocks` (spec Section 7.1)
    to avoid the (n_img, n_rot, n_trans, q) memory blow-up.
    """

    log_scores: jnp.ndarray  # (n_img, n_rot, n_trans) float64
    log_resp: jnp.ndarray  # (n_img, n_rot, n_trans) float64
    post_mean: jnp.ndarray  # (n_img, n_rot, n_trans, q) float64
    post_Hinv: jnp.ndarray  # (n_img, n_rot, q, q) float64

    def __init__(self, log_scores, log_resp, post_mean, post_Hinv):
        log_scores = _require_float64(log_scores, "PosteriorStats.log_scores")
        log_resp = _require_float64(log_resp, "PosteriorStats.log_resp")
        post_mean = _require_float64(post_mean, "PosteriorStats.post_mean")
        post_Hinv = _require_float64(post_Hinv, "PosteriorStats.post_Hinv")
        if log_scores.shape != log_resp.shape:
            raise ValueError(f"log_scores {log_scores.shape} and log_resp {log_resp.shape} must agree")
        if post_mean.ndim != 4 or post_mean.shape[:3] != log_scores.shape:
            raise ValueError(f"post_mean shape {post_mean.shape} inconsistent with log_scores {log_scores.shape}")
        if post_Hinv.ndim != 4 or post_Hinv.shape[:2] != log_scores.shape[:2]:
            raise ValueError(f"post_Hinv shape {post_Hinv.shape} inconsistent with log_scores {log_scores.shape}")
        if post_Hinv.shape[-1] != post_Hinv.shape[-2] or post_Hinv.shape[-1] != post_mean.shape[-1]:
            raise ValueError(
                f"post_Hinv must be (n_img, n_rot, q, q) with q={post_mean.shape[-1]}, got {post_Hinv.shape}"
            )
        self.log_scores = log_scores
        self.log_resp = log_resp
        self.post_mean = post_mean
        self.post_Hinv = post_Hinv


# ---------------------------------------------------------------------------
# PosteriorBlock — yielded by the streaming iterator
# ---------------------------------------------------------------------------


@dataclass
class PosteriorBlock:
    """One block of the streaming posterior iterator.

    `rot_slice` and `trans_slice` are Python slice objects, so this
    is a plain dataclass rather than an equinox Module — it is
    yielded from a Python generator, never crossed into JIT.

    `post_Hinv` does not carry a `n_trans` axis (Section 4.6 — H is
    translation-independent). M-step accumulators that need the
    full second moment form `C = Hinv + m m^T` on the fly per
    translation index.
    """

    rot_slice: slice
    trans_slice: slice
    log_scores: jnp.ndarray  # (n_img, len(rot_slice), len(trans_slice)) float64
    post_mean: jnp.ndarray  # (n_img, len(rot_slice), len(trans_slice), q) float64
    post_Hinv: jnp.ndarray  # (n_img, len(rot_slice), q, q) float64
