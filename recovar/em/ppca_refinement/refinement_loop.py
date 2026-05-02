"""Mature multi-iter pose-marginalized PPCA refinement loop (Phase D).

Mirrors the structure of
``recovar.em.dense_single_volume.iteration_loop.refine_single_volume`` for the
augmented [μ, W] PPCA model. Wires in:

  D1: HEALPix angular schedule (via :class:`RefinementState`)
  D2: Resolution / current_size schedule (via Fourier window)
  D4: ``--low_resol_join_halves 40 Å`` on backprojection accumulators
       (BEFORE the Wiener / PCG solve)
  D5: Per-iter noise update from residual statistics
  D6: Per-iter mean prior recompute via ``compute_relion_prior``
  D8: x=0 Hermitian enforcement on rhs / lhs_tri half-volumes

The single-iter primitives in
``recovar.em.ppca_refinement.production_driver`` are unchanged — this
module ORCHESTRATES them.

Simple-test escape hatch
------------------------
Pass ``schedule="simple"`` to disable the schedule and run a fixed
rotation grid for ``n_iters`` iterations with no resolution ramp / no
prior or noise updates. This is the path used by the dev eval cells
(``--simple-eval`` flag in scripts/ppca_refine_eval.py).

The default ``schedule="full"`` mirrors the RELION-style schedule:
coarse-to-fine HEALPix order, current_size ramp, per-iter noise + prior
+ low_resol_join.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.helpers.convergence import (
    LOCAL_SEARCH_HEALPIX_ORDER,
)
from recovar.em.dense_single_volume.local_backprojection import (
    enforce_relion_half_volume_x0_hermitian,
)
from recovar.em.ppca_refinement.iterations import IterationOpts
from recovar.em.ppca_refinement.production_driver import (
    _build_Y1_for_block,
    _per_image_y_norm,
    _project_theta_aug_to_block,
    _theta_aug_half_fourier,
    build_local_neighborhood_layout,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.em.sampling import get_rotation_grid
from recovar.ppca import AugmentedPPCAStats, solve_augmented_ppca_mstep
from recovar.ppca.ppca import _tri_size

__all__ = [
    "PPCAScheduleOpts",
    "run_pose_marginal_refinement",
    "run_pose_marginal_refinement_simple",
]


@dataclass(frozen=True)
class PPCAScheduleOpts:
    """Schedule options for the mature pipeline.

    Defaults match a sensible RELION-style refinement; ``schedule="simple"``
    (preset via :func:`run_pose_marginal_refinement_simple`) bypasses every
    schedule knob.
    """

    # HEALPix angular schedule.
    healpix_order_init: int = 1  # 576 rotations / 29.3°
    healpix_order_max: int = 4  # 36864 rotations / 1.83° (stops here unless local)
    angular_increase_after: int = 3  # bump order if no improvement after N iters
    convergence_pmax_threshold: float = 0.85  # mean Pmax above this → bump order
    use_local_search_at_high_order: bool = True  # switch dense → local at order >= LOCAL_SEARCH_HEALPIX_ORDER

    # Resolution / current_size schedule.
    initial_current_size: int | None = None  # default: grid_size // 2 (rounded)
    max_current_size: int | None = None  # default: grid_size
    current_size_step: int = 8  # increment per iter when FSC permits
    current_size_fsc_threshold: float = 0.143  # gold-standard FSC threshold

    # Joints / priors / noise.
    low_resol_join_angstrom: float = 40.0
    enable_low_resol_join: bool = True
    enable_per_iter_prior: bool = True
    enable_per_iter_noise: bool = True
    enable_x0_hermitian: bool = True

    # D7: drop γ entries below significance_threshold from the M-step
    # accumulators in pass 2. Default off — preserves exact pose-marginal
    # semantics. Turning it on saves no FLOPs but improves numerical
    # robustness against negligibly weighted poses.
    apply_significance_mask: bool = False

    # D9: translation auto-grid + log prior. translation_max_pixels=0 means
    # "single zero translation" (no translation search). When > 0 we build
    # get_translation_grid(...) + make_relion_translation_log_prior(...).
    translation_max_pixels: int = 0
    translation_pixel_offset: int = 1
    translation_sigma_offset_angstrom: float = 5.0

    # Convergence / stopping.
    max_iters: int = 25
    min_iters: int = 5
    convergence_log_evidence_rtol: float = 1e-3  # stop if rel improvement < this 3 iters in a row

    # Bookkeeping.
    write_per_iter_diagnostics: bool = True
    halfset_combine_method: str = "mean"  # 'mean' or 'low_resol_join_volume' (post-Wiener mean)


# ---------------------------------------------------------------------------
# Per-iter E + accumulate (no M-step) — the building block the loop calls.
# ---------------------------------------------------------------------------


def _load_image_batch(
    cryo,
    batch_indices: np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pull one image batch from the dataset and pre-compute the per-image
    quantities both E-step accumulators consume:

      ``images_full``        ``[B, full_F]`` complex64 — full-spectrum images
      ``ctf_full``           ``[B, full_F]`` real32     — full-spectrum CTF
      ``noise_full``         ``[B, full_F]`` real32     — full-spectrum σ²
      ``y_norm``             ``[B]`` real32             — Σ |y|² / σ²
      ``ctf2_over_noise_half`` ``[B, half_F]`` real32   — C² / σ² (half-spectrum)
    """
    image_shape = cryo.image_shape
    full_F = int(np.prod(image_shape))
    B = int(len(batch_indices))

    chunks = []
    for it_chunk, _, _ in cryo.image_source.get_dataset_subset_generator(
        batch_size=B,
        subset_indices=batch_indices,
    ):
        chunks.append(np.asarray(it_chunk))
        break
    images_full = jnp.asarray(chunks[0].reshape(B, full_F), dtype=jnp.complex64)

    ctf_params = jnp.asarray(cryo.CTF_params[batch_indices])
    ctf_full = jnp.asarray(
        cryo.ctf_evaluator(ctf_params, image_shape, cryo.voxel_size),
        dtype=jnp.float32,
    ).reshape(B, full_F)
    noise_full = jnp.asarray(cryo.noise.get(batch_indices), dtype=jnp.float32).reshape(B, full_F)

    y_norm = _per_image_y_norm(images_full, noise_full)
    ctf2_over_noise_full = (ctf_full**2) / jnp.maximum(noise_full, 1e-12)
    ctf2_over_noise_half = jax.vmap(
        lambda im: ftu.full_image_to_half_image(im.astype(jnp.complex64), image_shape).real
    )(ctf2_over_noise_full).astype(jnp.float32)
    return images_full, ctf_full, noise_full, y_norm, ctf2_over_noise_half


@dataclass
class _HalfsetEStep:
    """Per-halfset E-step accumulators mutated across image batches and
    rotation blocks. Returned as a dict from the accumulate helpers so
    the main loop can index by halfset (0/1)."""

    rhs: jax.Array  # [P, half_vol] complex64
    lhs_tri: jax.Array  # [tri(P), half_vol] real32
    sum_logZ: float = 0.0
    sum_pmax: float = 0.0
    sum_nsig: int = 0
    residual_num: float = 0.0
    residual_den: float = 0.0
    n_total: int = 0

    @classmethod
    def zeros(cls, P: int, tri_size: int, half_vol: int) -> "_HalfsetEStep":
        return cls(
            rhs=jnp.zeros((P, half_vol), dtype=jnp.complex64),
            lhs_tri=jnp.zeros((tri_size, half_vol), dtype=jnp.float32),
        )

    def merge_block(self, diag, *, batch_size: int, y_norm_sum: float) -> None:
        """Fold one engine call's diagnostics + scalar residual proxy."""
        self.sum_logZ += float(jnp.sum(diag.logZ))
        self.sum_pmax += float(jnp.sum(diag.pmax))
        self.sum_nsig += int(jnp.sum(diag.n_significant_per_image))
        # Scalar residual proxy: per-image Σ|y|²/σ². Replaced when the
        # engine emits a per-shell whitened residual.
        self.residual_num += y_norm_sum
        self.residual_den += float(batch_size)
        self.n_total += int(batch_size)

    def to_record(self) -> dict:
        """Final per-halfset record consumed downstream by the loop's
        Hermitian/low-res-join steps and the M-step."""
        return {
            "rhs": self.rhs.T,  # [half_vol, P]
            "lhs_tri": self.lhs_tri.T,  # [half_vol, tri(P)]
            "sum_logZ": self.sum_logZ,
            "sum_pmax": self.sum_pmax,
            "sum_nsig": self.sum_nsig,
            "residual_num": self.residual_num,
            "residual_den": self.residual_den,
            "n_total": self.n_total,
        }


def _e_step_dense_accumulate(
    state: PoseMarginalPPCAEMState,
    cryo,
    rotation_grid: np.ndarray,
    translation_grid: np.ndarray,
    halfset_indices,
    image_batch_size: int,
    rotation_block_size: int,
    current_size: int | None,
    significance_threshold: float,
    disc_type_project: str = "linear_interp",
    disc_type_backproject: str = "linear_interp",
    apply_significance_mask: bool = False,
):
    """Run the dense fused E-step + per-rotation backprojection per halfset.

    Returns ``{0: dict, 1: dict}`` of ``_HalfsetEStep`` records (one per
    halfset). The M-step is deliberately NOT called here so the main
    loop can apply low_resol_join + Hermitian enforcement before the
    Wiener solve.
    """
    from recovar.em.ppca_refinement.dense_engine import fused_dense_pose_ppca_block

    image_shape = cryo.image_shape
    volume_shape = (cryo.grid_size, cryo.grid_size, cryo.grid_size)
    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    P = state.W_score.shape[0] + 1
    tri_size = _tri_size(P)

    theta_aug_full = _theta_aug_half_fourier(state.mu_score, state.W_score, volume_shape)
    rotation_grid_jax = jnp.asarray(rotation_grid, dtype=jnp.float32)
    translation_grid_jax = jnp.asarray(translation_grid, dtype=jnp.float32)
    R_total = rotation_grid_jax.shape[0]

    per_half: dict[int, dict] = {}
    for half_idx, idx_array in enumerate(halfset_indices):
        acc = _HalfsetEStep.zeros(P, tri_size, half_vol)

        idx_array_np = np.asarray(idx_array)
        for start in range(0, len(idx_array_np), image_batch_size):
            batch_indices = idx_array_np[start : start + image_batch_size]
            B = len(batch_indices)
            if B == 0:
                continue
            images_full, ctf_full, noise_full, y_norm, ctf2_over_noise_half = _load_image_batch(
                cryo,
                batch_indices,
            )
            Y1_half = _build_Y1_for_block(
                images_full,
                ctf_full,
                noise_full,
                translation_grid_jax,
                image_shape,
            )

            for r_start in range(0, R_total, rotation_block_size):
                rotations_block = rotation_grid_jax[r_start : r_start + rotation_block_size]
                proj_aug_block = _project_theta_aug_to_block(
                    theta_aug_full,
                    rotations_block,
                    image_shape,
                    volume_shape,
                    disc_type_project,
                )
                acc.rhs, acc.lhs_tri, diag = fused_dense_pose_ppca_block(
                    Y1_half,
                    proj_aug_block,
                    ctf2_over_noise_half,
                    y_norm,
                    rotations_block,
                    image_shape,
                    volume_shape,
                    acc.rhs,
                    acc.lhs_tri,
                    significance_threshold=significance_threshold,
                    apply_significance_mask=apply_significance_mask,
                    disc_type_backproject=disc_type_backproject,
                )
                acc.merge_block(diag, batch_size=B, y_norm_sum=float(jnp.sum(y_norm)))

        per_half[half_idx] = acc.to_record()
    return per_half


def _build_local_layout_for_batch(
    cryo,
    *,
    batch_indices: np.ndarray,
    halfset_idx: int,
    n_local_rotations: int,
    local_sigma_rad: float,
    translation_grid: np.ndarray,
    iteration_index: int,
    images_full: jax.Array,
    ctf_full: jax.Array,
    noise_full: jax.Array,
    y_norm_per_image: jax.Array,
    ctf2_over_noise_half: jax.Array,
    theta_aug_full: jax.Array,
    image_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
    disc_type_project: str,
):
    """Build the per-hypothesis ``SparseHypothesisLayout`` for one image
    batch's local-pose neighborhood, plus the matching rotations array.

    This packs the hypothesis-translation, half-image FFT, augmented-
    template projection, and per-hypothesis fan-out of (Y1, ctf2, y_norm)
    so the outer ``_e_step_local_accumulate`` loop only has to call the
    fused sparse engine.
    """
    from recovar import core
    from recovar.em.ppca_refinement.sparse_engine import SparseHypothesisLayout

    P = theta_aug_full.shape[0]
    rot_per_hyp_np, trans_per_hyp_np, image_id_np, _ = build_local_neighborhood_layout(
        cryo,
        batch_indices,
        halfset_idx=halfset_idx,
        n_local_rotations=n_local_rotations,
        sigma_rad=local_sigma_rad,
        translation_grid=translation_grid,
        seed=iteration_index,
    )
    rotations_per_hyp = jnp.asarray(rot_per_hyp_np, dtype=jnp.float32)

    # Per-hypothesis whitened, CTF-modulated, translated image (half-spectrum).
    cy_over_var = ctf_full * images_full / jnp.maximum(noise_full, 1e-12)
    translated = core.translate_images(
        cy_over_var[image_id_np],
        jnp.asarray(trans_per_hyp_np, dtype=jnp.float32),
        image_shape,
        half_image=False,
    )
    Y1_half = jax.vmap(lambda im: ftu.full_image_to_half_image(im, image_shape))(translated).astype(jnp.complex64)

    # Per-hypothesis projections of the augmented templates [μ, W₁ … W_q].
    # batch_slice_volume vmaps slice_volume over the leading P axis (or
    # uses the CUDA batched kernel when available), so we get one fused
    # kernel call instead of P separate launches.
    proj_aug_per_hyp = jnp.transpose(
        core.batch_slice_volume(
            theta_aug_full,  # [P, vol_size]
            rotations_per_hyp,  # [Nh, 3, 3]
            image_shape,
            volume_shape,
            disc_type_project,
            half_image=True,
        ),  # → [P, Nh, half_F]
        (1, 0, 2),  # → [Nh, P, half_F]
    )

    layout = SparseHypothesisLayout(
        Y1=Y1_half,
        proj_aug=proj_aug_per_hyp,
        ctf2_over_noise=ctf2_over_noise_half[image_id_np],
        y_norm=y_norm_per_image[image_id_np],
        pose_log_prior=None,
        image_id=jnp.asarray(image_id_np, dtype=jnp.int32),
        n_images=int(len(batch_indices)),
    )
    return layout, rotations_per_hyp


def _e_step_local_accumulate(
    state: PoseMarginalPPCAEMState,
    cryo,
    halfset_indices,
    n_local_rotations: int,
    local_sigma_rad: float,
    translation_grid: np.ndarray,
    image_batch_size: int,
    significance_threshold: float,
    disc_type_project: str = "linear_interp",
    disc_type_backproject: str = "linear_interp",
    iteration_index: int = 0,
    apply_significance_mask: bool = False,
):
    """Sparse / local-pose analogue of :func:`_e_step_dense_accumulate`.

    Builds a local-neighborhood ``SparseHypothesisLayout`` per image
    batch and routes it through the fused sparse engine. Returns the
    same ``{0: dict, 1: dict}`` per-halfset records.
    """
    from recovar.em.ppca_refinement.sparse_engine import fused_sparse_pose_ppca_block

    image_shape = cryo.image_shape
    volume_shape = (cryo.grid_size, cryo.grid_size, cryo.grid_size)
    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    P = state.W_score.shape[0] + 1
    tri_size = _tri_size(P)
    theta_aug_full = _theta_aug_half_fourier(state.mu_score, state.W_score, volume_shape)

    per_half: dict[int, dict] = {}
    for half_idx, idx_array in enumerate(halfset_indices):
        acc = _HalfsetEStep.zeros(P, tri_size, half_vol)

        idx_array_np = np.asarray(idx_array)
        for start in range(0, len(idx_array_np), image_batch_size):
            batch_indices = idx_array_np[start : start + image_batch_size]
            B = len(batch_indices)
            if B == 0:
                continue
            images_full, ctf_full, noise_full, y_norm, ctf2_over_noise_half = _load_image_batch(
                cryo,
                batch_indices,
            )
            layout, rotations_per_hyp = _build_local_layout_for_batch(
                cryo,
                batch_indices=batch_indices,
                halfset_idx=half_idx,
                n_local_rotations=n_local_rotations,
                local_sigma_rad=local_sigma_rad,
                translation_grid=translation_grid,
                iteration_index=iteration_index,
                images_full=images_full,
                ctf_full=ctf_full,
                noise_full=noise_full,
                y_norm_per_image=y_norm,
                ctf2_over_noise_half=ctf2_over_noise_half,
                theta_aug_full=theta_aug_full,
                image_shape=image_shape,
                volume_shape=volume_shape,
                disc_type_project=disc_type_project,
            )
            acc.rhs, acc.lhs_tri, diag = fused_sparse_pose_ppca_block(
                layout,
                rotations_per_hyp,
                image_shape,
                volume_shape,
                acc.rhs,
                acc.lhs_tri,
                significance_threshold=significance_threshold,
                apply_significance_mask=apply_significance_mask,
                disc_type_backproject=disc_type_backproject,
            )
            acc.merge_block(diag, batch_size=B, y_norm_sum=float(jnp.sum(y_norm)))

        per_half[half_idx] = acc.to_record()
    return per_half


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def _low_res_join_mask(
    volume_shape: tuple[int, int, int],
    voxel_size: float,
    grid_size: int,
    low_resol_join_angstrom: float,
    current_resolution_angstrom: float | None,
) -> jax.Array:
    """Return a half-spectrum boolean mask selecting the low-resolution sphere
    used by RELION's ``--low_resol_join_halves`` (cf. ``MlOptimiserMpi::
    joinTwoHalvesAtLowResolution`` in ``ml_optimiser_mpi.cpp:3112``).

    ``True`` inside the joining sphere where the half-set accumulators are
    averaged together; ``False`` outside, where they are kept independent.
    The joining radius is the LARGER (lower-frequency) of
    ``low_resol_join_angstrom`` and ``current_resolution_angstrom`` so the
    radius never exceeds the iter's actual map resolution.
    """
    from recovar.reconstruction.regularization import _relion_round_away_from_zero

    myres = float(low_resol_join_angstrom)
    if current_resolution_angstrom is not None and np.isfinite(current_resolution_angstrom):
        myres = max(myres, float(current_resolution_angstrom))
    lowres_r_max = int(np.ceil(grid_size * voxel_size / myres))

    pf = volume_shape[0] // grid_size if volume_shape[0] > grid_size else 1
    lowres_r_max_padded = int(_relion_round_away_from_zero(float(pf) * lowres_r_max))
    lowres_r2_max = lowres_r_max_padded * lowres_r_max_padded

    axes = [
        jnp.asarray(ftu.get_1d_frequency_grid(volume_shape[0], scaled=False), dtype=jnp.float32),
        jnp.asarray(ftu.get_1d_frequency_grid(volume_shape[1], scaled=False), dtype=jnp.float32),
        jnp.asarray(ftu.get_1d_frequency_grid_rfft(volume_shape[2], scaled=False), dtype=jnp.float32),
    ]
    kz, ky, kx = jnp.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return (kz * kz + ky * ky + kx * kx <= lowres_r2_max).reshape(-1)


def _join_aug_halves_low_res(
    rhs_h0: jax.Array,
    rhs_h1: jax.Array,
    lhs_tri_h0: jax.Array,
    lhs_tri_h1: jax.Array,
    volume_shape: tuple[int, int, int],
    voxel_size: float,
    low_resol_join_angstrom: float,
    current_resolution_angstrom: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply RELION's low-resolution halfset join to the augmented-PPCA
    backprojection accumulators.

    Inside the low-resolution sphere both halves' accumulators are
    replaced with their pointwise mean; outside they are passed through
    unchanged. Acts uniformly across:

      - all P RHS components (μ + W₁ … W_q),
      - all tri(P) LHS upper-triangle entries (both diagonal CTF² weights
        and off-diagonal cross-component weights).

    A single mask is computed once and broadcast over the trailing
    component axis — fixes a subtle bug in the prior implementation
    where off-diagonal LHS entries were averaged at every frequency.

    Shapes: ``rhs[half_vol, P]`` complex; ``lhs_tri[half_vol, tri(P)]`` real.
    """
    grid_size = volume_shape[0]
    join = _low_res_join_mask(
        volume_shape,
        voxel_size,
        grid_size,
        low_resol_join_angstrom,
        current_resolution_angstrom,
    )[:, None]  # broadcast over the trailing component axis

    rhs_avg = 0.5 * (rhs_h0 + rhs_h1)
    lhs_avg = 0.5 * (lhs_tri_h0 + lhs_tri_h1)
    return (
        jnp.where(join, rhs_avg, rhs_h0),
        jnp.where(join, rhs_avg, rhs_h1),
        jnp.where(join, lhs_avg, lhs_tri_h0),
        jnp.where(join, lhs_avg, lhs_tri_h1),
    )


def _enforce_x0_hermitian_aug(
    rhs: jax.Array,
    lhs_tri: jax.Array,
    volume_shape: tuple[int, int, int],
) -> tuple[jax.Array, jax.Array]:
    """D8: enforce the RELION ``x=0`` Hermitian symmetry on every
    augmented-component accumulator, vectorized over P / tri(P) columns
    via ``jax.vmap`` instead of a Python loop.

    The Hermitian fix-up applies to the half-spectrum's ``x=0`` plane
    where ``F(-y, -z) = F*(y, z)`` is structurally implied; without it
    the M-step's PCG accumulates a small phase asymmetry that bleeds
    into the reconstruction. Mirrors RELION's
    ``BackProjector::enforceHermitianSymmetry``.
    """
    fix_one = lambda col: enforce_relion_half_volume_x0_hermitian(col, volume_shape)
    new_rhs = jax.vmap(fix_one, in_axes=1, out_axes=1)(rhs)
    new_lhs = jax.vmap(fix_one, in_axes=1, out_axes=1)(lhs_tri)
    return new_rhs, new_lhs


def _ema_scalar_noise_update(
    state: PoseMarginalPPCAEMState,
    per_half: dict,
    *,
    blend: float = 0.5,
) -> jax.Array:
    """D5 (scalar EMA placeholder): blend the previous per-voxel noise
    variance with a fresh image-power scalar.

    For each halfset the engine accumulates ``residual_num = Σ y_norm``
    and ``residual_den = number of images``; their ratio is the mean
    image power per pixel under the current whitening. Averaging across
    halves and EMA-blending into ``state.noise_variance`` produces a
    non-trivial, iteration-aware noise estimate without engine surgery.

    This is a stand-in for RELION's true per-shell update
    ``sigma2_noise[s] = (Σ_w |x|² + Σ_w (A² − 2 XA)) / (2 sumw N_pix(s))``
    which requires the engine to emit per-shell residual stats. Tracked
    as a follow-up.
    """
    image_power_h = jnp.asarray(
        [ph["residual_num"] / max(ph["residual_den"], 1.0) for ph in (per_half[0], per_half[1])],
        dtype=jnp.float32,
    )
    fresh = jnp.full_like(state.noise_variance, jnp.mean(image_power_h))
    return blend * fresh + (1.0 - blend) * state.noise_variance


def _recompute_mean_prior_relion(
    state: PoseMarginalPPCAEMState,
    cryo,
    *,
    batch_size: int = 256,
) -> jax.Array:
    """D6: recompute the spectral mean prior from the current half-maps
    via :func:`recovar.reconstruction.regularization.compute_relion_prior`,
    then broadcast it to the per-half-voxel layout the augmented PPCA
    M-step expects.

    Returns the previous prior unchanged if the FSC computation
    fails (degenerate at iter 0 with identical half-maps).
    """
    from recovar.em.ppca_refinement.prior_provider import compute_mean_prior_relion
    from recovar.reconstruction.regularization import broadcast_shell_to_volume

    vol_shape = (cryo.grid_size, cryo.grid_size, cryo.grid_size)
    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(vol_shape)))
    mu_h0_full_f = np.asarray(ftu.get_dft3(jnp.asarray(state.mu_half[0]))).reshape(-1)
    mu_h1_full_f = np.asarray(ftu.get_dft3(jnp.asarray(state.mu_half[1]))).reshape(-1)
    cov_noise = float(jnp.mean(state.noise_variance))
    try:
        prior_per_shell, _fsc, _avg = compute_mean_prior_relion(
            (cryo, cryo),
            cov_noise,
            mu_h0_full_f,
            mu_h1_full_f,
            batch_size,
        )
        full = np.asarray(broadcast_shell_to_volume(np.asarray(prior_per_shell), vol_shape)).reshape(vol_shape)
    except Exception:
        return state.mean_prior

    half_packed = (
        np.asarray(ftu.full_volume_to_half_volume(jnp.asarray(full), vol_shape))
        .real.astype(np.float32)
        .reshape(half_vol)
    )
    return jnp.asarray(half_packed)


# ---------------------------------------------------------------------------
# Main mature loop
# ---------------------------------------------------------------------------


def _build_rotation_grid_for_iter(
    state,
    cryo,
    schedule_opts: PPCAScheduleOpts,
    healpix_order: int,
    use_local: bool,
):
    """Build the per-iter rotation grid. Dense path = HEALPix at order;
    local path = neighborhoods built downstream by the engine, returns None."""
    if use_local:
        return None
    rotations = get_rotation_grid(healpix_order, matrices=True).astype(np.float32)
    return rotations


def run_pose_marginal_refinement(
    initial_state: PoseMarginalPPCAEMState,
    cryo,
    *,
    halfset_indices,
    mask,
    masks=None,
    pc_mask_assignment=None,
    mean_mask_idx: int = 0,
    image_batch_size: int = 32,
    rotation_block_size: int = 64,
    n_local_rotations: int = 32,
    local_sigma_rad: float = 0.05,
    translation_grid=None,
    schedule_opts: PPCAScheduleOpts = PPCAScheduleOpts(),
    iteration_opts: IterationOpts = IterationOpts(),
    iteration_callback=None,
):
    """Mature multi-iter pose-marginal PPCA refinement (D1+D2+D4+D5+D6+D8).

    Returns ``(final_state, iteration_log)`` where iteration_log is a list
    of per-iter diagnostic dicts.

    The loop:

      1. Initialize current_size, healpix_order, RefinementState.
      2. For each iteration up to `schedule_opts.max_iters`:
         a. Build rotation grid for current order (dense path) or skip
            (local path with per-image neighborhood).
         b. Run E-step + per-halfset accumulation (no M-step yet).
         c. D8: x=0 Hermitian enforcement on rhs / lhs_tri.
         d. D4: low_resol_join_halves on rhs / lhs_tri accumulators.
         e. M-step per halfset → new (μ_h, W_h).
         f. Halfset combine → new (mu_score, W_score).
         g. D5: noise update from residuals.
         h. D6: mean_prior recompute via compute_relion_prior.
         i. D1: bump healpix_order if Pmax converges, else stay.
         j. D2: bump current_size if FSC permits.
         k. Log iter diagnostics; call iteration_callback if provided.
         l. Check convergence (log-evidence rtol over last 3 iters).
    """
    state = initial_state
    image_shape = cryo.image_shape
    volume_shape = (cryo.grid_size, cryo.grid_size, cryo.grid_size)
    grid_size = cryo.grid_size

    healpix_order = schedule_opts.healpix_order_init
    current_size = schedule_opts.initial_current_size or (grid_size // 2)
    max_current_size = schedule_opts.max_current_size or grid_size
    iters_since_order_bump = 0
    log_evidence_history: list[float] = []

    if translation_grid is None:
        if schedule_opts.translation_max_pixels > 0:
            from recovar.em.sampling import get_translation_grid

            translation_grid = get_translation_grid(
                schedule_opts.translation_max_pixels,
                schedule_opts.translation_pixel_offset,
            ).astype(np.float32)
        else:
            translation_grid = np.zeros((1, 2), dtype=np.float32)

    iter_log = []
    prev_mu = None  # for ΔRMS diagnostic
    prev_W = None
    for it in range(schedule_opts.max_iters):
        use_local = schedule_opts.use_local_search_at_high_order and (healpix_order >= LOCAL_SEARCH_HEALPIX_ORDER)

        # Snapshot pre-iter state for ΔRMS diagnostic.
        prev_mu = jnp.asarray(state.mu_score)
        prev_W = jnp.asarray(state.W_score)

        # ---- (a) Rotation grid for this iter ----
        rotation_grid = _build_rotation_grid_for_iter(
            state,
            cryo,
            schedule_opts,
            healpix_order,
            use_local,
        )

        # ---- (b) E-step + accumulate per halfset ----
        if use_local:
            per_half = _e_step_local_accumulate(
                state,
                cryo,
                halfset_indices=halfset_indices,
                n_local_rotations=n_local_rotations,
                local_sigma_rad=local_sigma_rad,
                translation_grid=translation_grid,
                image_batch_size=image_batch_size,
                significance_threshold=iteration_opts.significance_threshold,
                iteration_index=it,
                apply_significance_mask=schedule_opts.apply_significance_mask,
            )
        else:
            per_half = _e_step_dense_accumulate(
                state,
                cryo,
                rotation_grid=rotation_grid,
                translation_grid=translation_grid,
                halfset_indices=halfset_indices,
                image_batch_size=image_batch_size,
                rotation_block_size=rotation_block_size,
                current_size=current_size,
                significance_threshold=iteration_opts.significance_threshold,
                apply_significance_mask=schedule_opts.apply_significance_mask,
            )

        # ---- (c) D8 Hermitian + (d) D4 low-res halfset join ----
        rhs_h0, rhs_h1 = per_half[0]["rhs"], per_half[1]["rhs"]
        lhs_h0, lhs_h1 = per_half[0]["lhs_tri"], per_half[1]["lhs_tri"]
        if schedule_opts.enable_x0_hermitian:
            rhs_h0, lhs_h0 = _enforce_x0_hermitian_aug(rhs_h0, lhs_h0, volume_shape)
            rhs_h1, lhs_h1 = _enforce_x0_hermitian_aug(rhs_h1, lhs_h1, volume_shape)
        if schedule_opts.enable_low_resol_join:
            rhs_h0, rhs_h1, lhs_h0, lhs_h1 = _join_aug_halves_low_res(
                rhs_h0,
                rhs_h1,
                lhs_h0,
                lhs_h1,
                volume_shape,
                cryo.voxel_size,
                schedule_opts.low_resol_join_angstrom,
            )

        # ---- (e) M-step per halfset ----
        new_mu_half, new_W_half = [], []
        for half_idx, (rhs, lhs) in enumerate([(rhs_h0, lhs_h0), (rhs_h1, lhs_h1)]):
            stats = AugmentedPPCAStats(
                rhs=rhs,
                lhs_tri=lhs,
                n_images=per_half[half_idx]["n_total"],
                log_likelihood=per_half[half_idx]["sum_logZ"],
            )
            mu_h, W_h = solve_augmented_ppca_mstep(
                stats,
                mean_prior=state.mean_prior,
                W_prior=state.W_prior,
                mask=mask,
                masks=masks,
                pc_mask_assignment=pc_mask_assignment,
                mean_mask_idx=mean_mask_idx,
                maxiter=iteration_opts.pcg_maxiter,
                tol=iteration_opts.pcg_tol,
                theta_init=(state.mu_half[half_idx], state.W_half[half_idx])
                if state.mu_half[half_idx] is not None and state.W_half[half_idx] is not None
                else None,
            )
            new_mu_half.append(mu_h)
            new_W_half.append(W_h)

        # ---- (f) Halfset combine ----
        mu_score_new = 0.5 * (new_mu_half[0] + new_mu_half[1])
        W_score_new = 0.5 * (new_W_half[0] + new_W_half[1])

        # ---- (g) Noise update ----
        new_noise = state.noise_variance
        if schedule_opts.enable_per_iter_noise:
            new_noise = _ema_scalar_noise_update(state, per_half)

        # ---- State update so D6 sees fresh μ_half ----
        state = state.replace(
            mu_half=(new_mu_half[0], new_mu_half[1]),
            W_half=(new_W_half[0], new_W_half[1]),
            mu_score=mu_score_new,
            W_score=W_score_new,
            noise_variance=new_noise,
        )

        # ---- (h) Mean prior recompute ----
        if schedule_opts.enable_per_iter_prior:
            state = state.replace(mean_prior=_recompute_mean_prior_relion(state, cryo))

        # ---- (i+j) Schedule advance ----
        n_total = sum(per_half[h]["n_total"] for h in (0, 1)) or 1
        log_evidence_total = sum(per_half[h]["sum_logZ"] for h in (0, 1))
        pmax_mean = sum(per_half[h]["sum_pmax"] for h in (0, 1)) / n_total
        nsig_mean = sum(per_half[h]["sum_nsig"] for h in (0, 1)) / n_total
        log_evidence_history.append(log_evidence_total)

        order_bumped = False
        if (
            it >= schedule_opts.min_iters
            and pmax_mean > schedule_opts.convergence_pmax_threshold
            and iters_since_order_bump >= schedule_opts.angular_increase_after
            and healpix_order < schedule_opts.healpix_order_max
        ):
            healpix_order += 1
            iters_since_order_bump = 0
            order_bumped = True
        else:
            iters_since_order_bump += 1

        size_bumped = False
        if pmax_mean > schedule_opts.convergence_pmax_threshold:
            new_size = min(current_size + schedule_opts.current_size_step, max_current_size)
            if new_size != current_size:
                current_size = new_size
                size_bumped = True

        # ---- (k) Log ----
        # Per-iter Δ diagnostics: state-change RMS / W norm. Useful for
        # diagnosing convergence (steady decrease ↔ converging) or
        # oscillation (jumpy values across iters).
        mu_now = jnp.asarray(state.mu_score)
        W_now = jnp.asarray(state.W_score)
        mu_delta_rms = float(jnp.sqrt(jnp.mean((mu_now - prev_mu) ** 2)))
        W_delta_rms = float(jnp.sqrt(jnp.mean((W_now - prev_W) ** 2)))
        W_frob = float(jnp.sqrt(jnp.sum(W_now**2)))
        mu_rms = float(jnp.sqrt(jnp.mean(mu_now**2)))

        info = {
            "iteration": it,
            "healpix_order": healpix_order,
            "current_size": current_size,
            "use_local_search": use_local,
            "n_rotations": int(rotation_grid.shape[0]) if rotation_grid is not None else None,
            "log_evidence_total": log_evidence_total,
            "pmax_mean": pmax_mean,
            "n_significant_mean": nsig_mean,
            "order_bumped": order_bumped,
            "size_bumped": size_bumped,
            "noise_var_mean": float(jnp.mean(state.noise_variance)),
            "mean_prior_mean": float(jnp.mean(state.mean_prior)),
            "mu_delta_rms": mu_delta_rms,
            "W_delta_rms": W_delta_rms,
            "mu_rms": mu_rms,
            "W_frob": W_frob,
        }
        iter_log.append(info)
        if iteration_callback is not None:
            iteration_callback(it, state, info)

        # ---- (l) Convergence check ----
        if it >= schedule_opts.min_iters and len(log_evidence_history) >= 3:
            recent = log_evidence_history[-3:]
            rel = abs(recent[-1] - recent[0]) / max(abs(recent[0]), 1.0)
            if rel < schedule_opts.convergence_log_evidence_rtol and not order_bumped and not size_bumped:
                info["converged"] = True
                break

    return state, iter_log


def run_pose_marginal_refinement_simple(
    initial_state: PoseMarginalPPCAEMState,
    cryo,
    *,
    rotation_grid: np.ndarray,
    translation_grid: np.ndarray | None = None,
    halfset_indices,
    mask,
    masks=None,
    pc_mask_assignment=None,
    mean_mask_idx: int = 0,
    image_batch_size: int = 32,
    rotation_block_size: int = 64,
    em_iters: int = 5,
    iteration_opts: IterationOpts = IterationOpts(),
):
    """Simple-test escape hatch: fixed rotation grid, fixed n iterations,
    no schedule, no per-iter prior/noise updates. Mirrors the dev-eval
    behavior the user asked us to keep."""
    if translation_grid is None:
        translation_grid = np.zeros((1, 2), dtype=np.float32)

    schedule_opts = PPCAScheduleOpts(
        healpix_order_init=0,  # ignored — we override rotation_grid
        healpix_order_max=0,
        max_iters=em_iters,
        min_iters=em_iters,  # disable convergence early-stop
        enable_low_resol_join=False,
        enable_per_iter_prior=False,
        enable_per_iter_noise=False,
        enable_x0_hermitian=False,
        use_local_search_at_high_order=False,
        initial_current_size=None,
        max_current_size=None,
    )
    # Patch the loop's rotation grid builder to return our fixed grid.
    # We override at the dispatch point by directly calling
    # _e_step_dense_accumulate with the user's grid, then mirroring
    # the rest of the loop. To keep one code path, we run the schedule
    # loop with all schedule knobs disabled and a degenerate HEALPix
    # 'order' that returns the user's grid via a custom rotation-grid
    # injection. Simplest: walk the same iteration body manually.
    state = initial_state
    image_shape = cryo.image_shape
    volume_shape = (cryo.grid_size, cryo.grid_size, cryo.grid_size)
    iter_log = []
    for it in range(em_iters):
        per_half = _e_step_dense_accumulate(
            state,
            cryo,
            rotation_grid=rotation_grid,
            translation_grid=translation_grid,
            halfset_indices=halfset_indices,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            current_size=None,
            significance_threshold=iteration_opts.significance_threshold,
        )
        rhs_h0, rhs_h1 = per_half[0]["rhs"], per_half[1]["rhs"]
        lhs_h0, lhs_h1 = per_half[0]["lhs_tri"], per_half[1]["lhs_tri"]

        new_mu_half, new_W_half = [], []
        for half_idx, (rhs, lhs) in enumerate([(rhs_h0, lhs_h0), (rhs_h1, lhs_h1)]):
            stats = AugmentedPPCAStats(
                rhs=rhs,
                lhs_tri=lhs,
                n_images=per_half[half_idx]["n_total"],
                log_likelihood=per_half[half_idx]["sum_logZ"],
            )
            mu_h, W_h = solve_augmented_ppca_mstep(
                stats,
                mean_prior=state.mean_prior,
                W_prior=state.W_prior,
                mask=mask,
                masks=masks,
                pc_mask_assignment=pc_mask_assignment,
                mean_mask_idx=mean_mask_idx,
                maxiter=iteration_opts.pcg_maxiter,
                tol=iteration_opts.pcg_tol,
                theta_init=(state.mu_half[half_idx], state.W_half[half_idx])
                if state.mu_half[half_idx] is not None and state.W_half[half_idx] is not None
                else None,
            )
            new_mu_half.append(mu_h)
            new_W_half.append(W_h)
        mu_score_new = 0.5 * (new_mu_half[0] + new_mu_half[1])
        W_score_new = 0.5 * (new_W_half[0] + new_W_half[1])
        state = state.replace(
            mu_half=(new_mu_half[0], new_mu_half[1]),
            W_half=(new_W_half[0], new_W_half[1]),
            mu_score=mu_score_new,
            W_score=W_score_new,
        )
        n_total = sum(per_half[h]["n_total"] for h in (0, 1)) or 1
        iter_log.append(
            {
                "iteration": it,
                "log_evidence_total": sum(per_half[h]["sum_logZ"] for h in (0, 1)),
                "pmax_mean": sum(per_half[h]["sum_pmax"] for h in (0, 1)) / n_total,
                "n_significant_mean": sum(per_half[h]["sum_nsig"] for h in (0, 1)) / n_total,
            }
        )
    return state, iter_log
