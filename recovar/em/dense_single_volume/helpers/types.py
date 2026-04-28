"""Typed containers for the dense single-volume EM path."""

from typing import NamedTuple

import jax


class MeanStats(NamedTuple):
    """Accumulated M-step sufficient statistics.

    Both fields are additive over image batches and across devices,
    making this the natural unit for distributed all-reduce.

    Attributes:
        Ft_y: (volume_size,) complex -- weighted backprojected images.
        Ft_ctf: (volume_size,) real/complex -- weighted CTF^2 backprojection.
    """

    Ft_y: jax.Array
    Ft_ctf: jax.Array


class RelionStats(NamedTuple):
    """Per-image E-step statistics needed by the RELION-style refine loop.

    These fields are not additive like :class:`MeanStats`; they are emitted
    per iteration so convergence and current-size logic can reuse the exact
    normalization already computed inside ``run_em``.

    Attributes:
        log_evidence_per_image: Log normalizer ``log_Z`` for each image.
        best_log_score_per_image: Maximum unnormalized log-score per image.
        max_posterior_per_image: Maximum posterior probability per image.
        rotation_posterior_sums: Posterior mass accumulated per rotation over
            the processed image subset. This is additive across batches and can
            be collapsed to RELION-style ``pdf_direction`` updates.
    """

    log_evidence_per_image: jax.Array
    best_log_score_per_image: jax.Array
    max_posterior_per_image: jax.Array
    rotation_posterior_sums: jax.Array


class NoiseStats(NamedTuple):
    """Posterior-weighted noise shell statistics (RELION parity).

    Accumulated during the M-step pass of ``run_em`` when
    ``accumulate_noise=True``.  These are additive over image batches
    and across half-sets, matching RELION's ``wsum_sigma2_noise``.

    RELION formula::

        sigma2_noise[s] = (wsum_sigma2_noise[s] + wsum_img_power[s])
                          / (2 * sumw * Npix_per_shell[s])

    Attributes:
        wsum_sigma2_noise: (n_shells,) float -- accumulated
            ``sum_{i,r,t} w * (A2 - 2*XA)`` per shell (reference-dependent
            part of the residual).
        wsum_img_power: (n_shells,) float -- accumulated
            ``sum_i mass_i * |img_masked_i|^2`` per shell, where ``mass_i`` is
            the same significant-support posterior mass used for A2/XA.
            For ungated full-grid updates ``mass_i == 1``.
        wsum_sigma2_offset: float -- accumulated
            ``sum_{i,r,t} w * ||offset_{i,t} - prior_i||^2`` in Angstrom^2.
            This is RELION's sufficient statistic for updating
            ``sigma2_offset`` via ``wsum_sigma2_offset / (2 * sumw)``.
        sumw: float -- total posterior/support weight processed (equals the
            number of images when posteriors are normalised to sum to 1 per
            image and no significant-support pruning is active).
        wsum_noise_a2: optional diagnostic split of ``wsum_sigma2_noise``.
        wsum_noise_xa: optional diagnostic split of ``wsum_sigma2_noise``.
    """

    wsum_sigma2_noise: jax.Array
    wsum_img_power: jax.Array
    wsum_sigma2_offset: float
    sumw: float
    wsum_noise_a2: jax.Array | None = None
    wsum_noise_xa: jax.Array | None = None


class EMProfileStats(NamedTuple):
    """Host-side timing and work counters for one ``run_em`` call.

    These values are diagnostic only and do not affect numerical behavior.
    """

    batch_fetch_s: float
    preprocess_s: float
    score_prep_s: float
    pass1_projection_s: float
    pass1_score_s: float
    pass1_postprocess_s: float
    pass1_logsumexp_s: float
    pass2_skipmask_s: float
    pass2_projection_s: float
    pass2_score_s: float
    pass2_postprocess_s: float
    mstep_s: float
    window_scatter_s: float
    adjoint_y_s: float
    adjoint_ctf_s: float
    noise_s: float
    assignment_s: float
    stats_finalize_s: float
    host_stats_s: float
    solve_s: float
    accounted_s: float
    total_wall_s: float
    unattributed_s: float
    n_images: int
    n_trans: int
    n_rot: int
    n_rot_padded: int
    n_blocks: int
    n_windowed: int
    use_window: bool
    reused_pass1_projections: bool
    sparse_pass2_total_blocks: int
    sparse_pass2_skipped_blocks: int
    sparse_pass2_omitted_mass_upper_mean: float
    sparse_pass2_omitted_mass_upper_max: float
    sparse_pass2_omitted_mass_upper_sum: float
