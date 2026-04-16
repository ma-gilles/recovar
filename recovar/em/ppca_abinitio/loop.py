"""High-level loop entry points for the PPCA-ab-initio v0 stages.

v0 ships:

- `run_score_diagnostic` — Stage 0B / Stage 1A score-only entry
  point. Runs the posterior helper twice (PPCA branch and
  homogeneous branch) on a `SyntheticDataset`. No parameter
  updates.
- `run_fixed_grid_ppca` — Stage 1B (and the skeleton of Stage 1C)
  iterative loop. Each iteration calls the posterior helper, runs
  the residualized mean update (and later the factor update), and
  records per-iteration metrics on the validation split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .mean_update import (
    MeanUpdateResult,
    update_mu_homogeneous,
    update_mu_residualized,
)
from .metrics import (
    ScoreDiagnostic,
    fourier_relative_error_mu,
    per_image_true_state_mass,
    score_diagnostic_one_seed,
)
from .posterior import score_and_posterior_moments_eqx
from .synthetic import SyntheticDataset, subset_synthetic_dataset
from .types import PPCAConfig, PPCAInit


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


# ---------------------------------------------------------------------------
# Stage 1B / 1C iterative loop
# ---------------------------------------------------------------------------


@dataclass
class IterationMetrics:
    """One iteration's metrics on the validation split."""

    iter: int
    fre_mu_val: float
    true_state_mass_val: float


@dataclass
class FixedGridPPCAResult:
    """Per-(family, init, run) loop output."""

    final_init: PPCAInit
    iter_metrics: list[IterationMetrics] = field(default_factory=list)


def _val_metrics_for_init(
    config,
    init: PPCAInit,
    dataset: SyntheticDataset,
    *,
    weights_half=None,
) -> tuple[float, float]:
    """Compute (FRE-mu, true_state_mass) on the validation split."""
    fre = fourier_relative_error_mu(init.mu, dataset.mu_half_true, weights_half=weights_half)

    stats = score_and_posterior_moments_eqx(
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
    val = np.asarray(dataset.val_idx)
    per_image = per_image_true_state_mass(
        np.asarray(stats.log_resp)[val],
        np.asarray(dataset.r_true_idx)[val],
        np.asarray(dataset.t_true_idx)[val],
    )
    return float(fre), float(np.mean(per_image))


def run_fixed_grid_ppca(
    config,
    dataset: SyntheticDataset,
    init: PPCAInit,
    cfg: PPCAConfig,
    *,
    use_residualized_mean: bool = True,
    weights_half=None,
    factor_update_fn: Callable | None = None,
) -> FixedGridPPCAResult:
    """Iterative fixed-grid PPCA loop.

    Stage 1B path (default):
        update_mu=True, update_factor=False, use_residualized_mean=True

    Stage 1C path:
        update_mu=True, update_factor=True, plus a `factor_update_fn`
        callable that takes `(config, init, dataset)` and returns a
        new `PPCAInit` with the factor updated. The factor update is
        kept as a callback so this loop module does not depend on
        `factor_update.py` directly. Stage 1C is added in a later
        commit.

    Per spec Section 5.3, the loop is self-contained: explicit
    accumulator reset every iteration, no cross-iteration hidden
    state beyond the current `(mu, U, s)` and the per-iteration
    metrics list.
    """
    if len(dataset.train_idx) == 0:
        raise ValueError("run_fixed_grid_ppca requires a non-empty training split")

    cur = init
    iter_metrics: list[IterationMetrics] = []
    train_dataset = subset_synthetic_dataset(dataset, dataset.train_idx)

    # Iter 0 — record metrics at the initialization
    fre_0, mass_0 = _val_metrics_for_init(config, cur, dataset, weights_half=weights_half)
    iter_metrics.append(IterationMetrics(iter=0, fre_mu_val=fre_0, true_state_mass_val=mass_0))

    for it in range(1, cfg.n_iters + 1):
        if cfg.update_mu:
            updater = update_mu_residualized if use_residualized_mean else update_mu_homogeneous
            mean_res: MeanUpdateResult = updater(
                config,
                cur,
                train_dataset.batch_full,
                train_dataset.rotations,
                train_dataset.translations,
                train_dataset.ctf_params,
                train_dataset.noise_variance_full,
                tau=cfg.ridge_lambda,
            )
            cur = PPCAInit(
                mu=mean_res.mu_half,
                U=cur.U,
                s=cur.s,
                volume_shape=cur.volume_shape,
            )

        if cfg.update_factor:
            if factor_update_fn is None:
                raise ValueError(
                    "cfg.update_factor=True requires a `factor_update_fn` callback. "
                    "Stage 1C plugs this in via factor_update.py."
                )
            cur = factor_update_fn(config, cur, train_dataset)

        fre_it, mass_it = _val_metrics_for_init(config, cur, dataset, weights_half=weights_half)
        iter_metrics.append(IterationMetrics(iter=it, fre_mu_val=fre_it, true_state_mass_val=mass_it))

    return FixedGridPPCAResult(final_init=cur, iter_metrics=iter_metrics)


def run_fixed_grid_homogeneous_baseline(
    config,
    dataset: SyntheticDataset,
    init: PPCAInit,
    cfg: PPCAConfig,
    *,
    weights_half=None,
) -> FixedGridPPCAResult:
    """Homogeneous baseline for the Stage 1B comparison.

    Same loop structure as `run_fixed_grid_ppca`, but the mean
    update uses `update_mu_homogeneous` (which still uses
    PPCA-shaped responsibilities — the difference is that no
    residualization is applied). And the kernel call inside the
    metric computation uses `U=0`, matching the homogeneous score
    convention from `run_score_diagnostic`.

    Per spec Section 8.2.1 / Q1, the unresidualized mean update is
    a debugging ablation only. Here we use it as the homogeneous
    *baseline* against which Stage 1B compares; the comparison is
    "PPCA-residualized vs PPCA-homogeneous", which isolates the
    effect of the residualization step.
    """
    homog_init = PPCAInit(
        mu=init.mu,
        U=jnp.zeros_like(init.U),
        s=init.s,
        volume_shape=init.volume_shape,
    )
    return run_fixed_grid_ppca(
        config,
        dataset,
        homog_init,
        cfg,
        use_residualized_mean=False,
        weights_half=weights_half,
        factor_update_fn=None,
    )
