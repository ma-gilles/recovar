"""Post-EM postprocessing for ``recovar ppca-refine`` (Milestone 9).

Two functions: :func:`finalize_ppca_state` produces ``(U, S, W)`` from a
final ``PoseMarginalPPCAEMState`` via the existing
``recovar.ppca.ppca._orthonormalize_W_to_basis(_multimask)``; and
:func:`save_state` / :func:`load_state` provide pickleable
state-on-disk semantics for ``state.pkl`` restart.

Eigenvalue refit during EM is HARMFUL (memory entry
``project_ppca_eigenval_update_during_anneal_harmful.md``). Refit is
post-EM only — :func:`finalize_ppca_state` is the right home for that.
The actual refit reuses ``recovar.ppca.ppca_iterative_refitb`` driver
machinery; we expose only the post-EM path here.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from recovar.ppca.ppca import (
    _orthonormalize_W_to_basis,
    _orthonormalize_W_to_basis_multimask,
)

from .state import PoseMarginalPPCAEMState

__all__ = [
    "finalize_ppca_state",
    "save_state",
    "load_state",
]


def finalize_ppca_state(
    state: PoseMarginalPPCAEMState,
    *,
    volume_shape: tuple[int, int, int],
    pc_mask_assignment: Any | None = None,
):
    """Produce ``(U, S, W_half)`` from a final pose-marginal PPCA state.

    Parameters
    ----------
    state:
        Final :class:`PoseMarginalPPCAEMState` from the EM driver. We use
        ``state.W_score`` (the filtered halfset combine) as the
        loading matrix to SVD.
    volume_shape:
        ``(D, D, D)``. Required because the state stores W in real-space
        loading shape ``(q, *vs)`` (M5+ convention).
    pc_mask_assignment:
        Optional ``(q,)`` int — when provided, runs the multimask SVD
        path ``_orthonormalize_W_to_basis_multimask``.

    Returns
    -------
    U_real:
        ``(q, *vs)`` real32. Orthonormal real-space PPCA basis.
    S_squared:
        ``(q,)`` real32. Eigenvalues (squared singular values, Fourier
        convention).
    W_half:
        ``(half_vol, q)`` complex64. Half-Fourier loading matrix that
        produced the basis (re-derived from ``state.W_score``).
    """
    import recovar.core.fourier_transform_utils as ftu

    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))
    q = state.W_score.shape[0]

    # state.W_score is real-space (q, D, D, D); rfft to half-Fourier
    # (half_vol, q) — same axis order as legacy W_half.
    W_real_np = np.asarray(state.W_score)
    W_half_real = np.asarray(ftu.get_dft3_real(W_real_np))  # (q, *half_vs)
    W_half = W_half_real.reshape(q, half_vol).T.astype(np.complex64)  # (half_vol, q)

    if pc_mask_assignment is not None:
        U_real, S_squared, _Vt = _orthonormalize_W_to_basis_multimask(
            W_half,
            volume_shape,
            pc_mask_assignment,
        )
    else:
        U_real, S_squared, _Vt = _orthonormalize_W_to_basis(W_half, volume_shape)
    return U_real, S_squared, W_half


def save_state(state: PoseMarginalPPCAEMState, path: str | Path) -> Path:
    """Pickle ``state`` to disk for restart. Caller is responsible for
    ensuring ``state`` has been migrated to NumPy / safely-pickleable
    leaves (the frozen dataclass holds JAX arrays which pickle as
    DeviceArray; load_state converts back to JAX as needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert JAX leaves to NumPy for portability.
    payload = {}
    for f in state.__dataclass_fields__:
        v = getattr(state, f)
        payload[f] = _to_numpy_tree(v)
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    return path


def load_state(path: str | Path) -> PoseMarginalPPCAEMState:
    """Inverse of :func:`save_state`."""

    path = Path(path)
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    kwargs = {f: _to_jax_tree(payload[f]) for f in payload}
    # mu_half / W_half are tuples — preserve.
    if isinstance(kwargs.get("mu_half"), list):
        kwargs["mu_half"] = tuple(kwargs["mu_half"])
    if isinstance(kwargs.get("W_half"), list):
        kwargs["W_half"] = tuple(kwargs["W_half"])
    return PoseMarginalPPCAEMState(**kwargs)


def _to_numpy_tree(x):
    if isinstance(x, (tuple, list)):
        return type(x)(_to_numpy_tree(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return np.asarray(x)
    return x


def _to_jax_tree(x):
    import jax.numpy as jnp

    if isinstance(x, (tuple, list)):
        return type(x)(_to_jax_tree(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_jax_tree(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return jnp.asarray(x)
    return x
