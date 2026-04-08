"""High-level loop entry points for the PPCA-ab-initio v0 stages.

v0 ships:

- `run_score_diagnostic` — Stage 0B and Stage 1A score-only entry
  point. Runs the posterior helper twice (PPCA branch and
  homogeneous branch) on a `SyntheticDataset`, returns the per-stage
  metrics for both. No parameter updates.

`run_fixed_grid_ppca` (the iterative loop for Stage 1B/1C) is a
later milestone and is not in this commit.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .metrics import ScoreDiagnostic, score_diagnostic_one_seed
from .posterior import score_and_posterior_moments_eqx
from .synthetic import SyntheticDataset
from .types import PPCAInit


@dataclass
class ScoreDiagnosticResult:
    """Bundle returned by `run_score_diagnostic`."""

    family: str
    seed: int
    n_val: int
    diagnostic: ScoreDiagnostic
    log_resp_homog: jnp.ndarray
    log_resp_ppca: jnp.ndarray


def _make_homog_init(init: PPCAInit) -> PPCAInit:
    """Build a `homogeneous` PPCAInit by zeroing `U`. With `U=0`,
    the bHb correction reduces to `sum log s` (a g-independent
    constant) and after per-image normalization the resulting
    `log_resp` equals what the homogeneous score would produce.
    Verified by `tests/ppca_abinitio/test_compute_bHb_terms_correctness.py::test_bHb_zero_u_returns_minus_log_det_diag_inv_s`
    and `tests/ppca_abinitio/test_posterior_matches_production.py::test_half_image_kernel_zero_u_matches_homogeneous`.
    """
    return PPCAInit(
        mu=init.mu,
        U=jnp.zeros_like(init.U),
        s=init.s,
        volume_shape=init.volume_shape,
    )


def run_score_diagnostic(
    config,
    dataset: SyntheticDataset,
    init: PPCAInit,
    *,
    seed: int,
    n_bootstrap: int = 1000,
) -> ScoreDiagnosticResult:
    """Run one (family, seed) score-only diagnostic.

    Calls `score_and_posterior_moments_eqx` twice on the same
    dataset:
    - **PPCA branch**: with `init.U` and `init.s`.
    - **Homogeneous branch**: with `U = 0`, same `mu` and `s`.

    Reports the Stage 0B / 1A score metrics on the validation split.
    """
    # PPCA branch
    stats_ppca = score_and_posterior_moments_eqx(
        config,
        init.mu,
        init.U,
        init.s,
        dataset.batch_full,
        dataset.rotations,
        dataset.translations,
        dataset.ctf_params,
        dataset.noise_variance_full,
    )

    # Homogeneous branch
    homog_init = _make_homog_init(init)
    stats_homog = score_and_posterior_moments_eqx(
        config,
        homog_init.mu,
        homog_init.U,
        homog_init.s,
        dataset.batch_full,
        dataset.rotations,
        dataset.translations,
        dataset.ctf_params,
        dataset.noise_variance_full,
    )

    diagnostic = score_diagnostic_one_seed(
        log_resp_homog=stats_homog.log_resp,
        log_resp_ppca=stats_ppca.log_resp,
        r_true_idx=dataset.r_true_idx,
        t_true_idx=dataset.t_true_idx,
        val_idx=dataset.val_idx,
        family=dataset.family.value,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )

    return ScoreDiagnosticResult(
        family=dataset.family.value,
        seed=seed,
        n_val=int(len(dataset.val_idx)),
        diagnostic=diagnostic,
        log_resp_homog=stats_homog.log_resp,
        log_resp_ppca=stats_ppca.log_resp,
    )
