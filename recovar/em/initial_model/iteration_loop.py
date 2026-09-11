"""VDAM iteration loop `run_vdam_iterations`.

Mirrors `MlOptimiser::iterate` (ml_optimiser.cpp:3458-3550) for the
gradient-refine branch:

  for iter in 1 .. nr_iter:
      schedule_update(state, iter)        # stepsize, tau2_fudge, subset
      do_grad = ...                       # drop grad at the EM tail
      pseudo_halfsets = do_grad
      select_subset_for_iter(state, ...)  # shuffle + prefix + stable-sort
      update_current_resolution(state)    # FSC-driven from iter 2
      expectation_step(state, ...)        # E-step adapter -> posteriors
      maximisation_step(state, ...)       # VDAM M-step
      post_mstep_update(state, ...)       # masks / other post-M-step hooks
      write_iter_artifacts(state, iter)

The E-step adapter (`expectation_step`) is a callback supplied by the
caller because it requires dense-path kernels + real particle data. This
module is the pure orchestrator.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Callable, Literal, Sequence

import numpy as np

from recovar.em.initial_model.estep_meta_updates import (
    update_noise_from_estep_meta,
    update_probabilities_from_estep_meta,
)
from recovar.em.initial_model.subset_schedule import (
    _resolve_phase_lengths,
    select_subset_for_iter,
)

from ..dense_single_volume.mean_helpers import _relion_optimizer_average_pmax
from .m_step import vdam_m_step
from .schedules import (
    DEFAULT_GRAD_EM_ITERS,
    DEFAULT_GRAD_MU,
    VdamPhaseLengths,
    _relion_round,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
)
from .state import InitialModelState
from .subset import (
    RndUnifFn,
)

# Callback signatures
ExpectationStepFn = Callable[
    # (state, particle_ids, halfset_ids) -> (posterior_accumulators, posterior_meta)
    [InitialModelState, np.ndarray, np.ndarray],
    tuple,  # (List[VdamAccumulator], dict)
]
"""E-step callback. Must return `(accumulators, meta)` where accumulators
holds 2K entries when pseudo_halfsets is on (halfset-0 first, then
halfset-1) and meta is a free-form dict written into per-iter STAR output
(Pmax, nr_significant, best_class, best_euler, best_trans).
"""

IterArtifactSink = Callable[[InitialModelState, int, dict], None]
PostMstepUpdateFn = Callable[[InitialModelState, int, dict], InitialModelState]


def refresh_tau2_from_projector_power(
    state: InitialModelState,
    *,
    padding_factor: int = 1,
    interpolator: int = 1,
) -> InitialModelState:
    """``MlModel::setFourierTransformMaps(!fix_tau)``."""
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import recovar_volume_to_relion

    current_size = int(state.current_size if state.current_size > 0 else state.ori_size)
    new_tau2 = np.asarray(state.tau2_class, dtype=np.float64).copy()
    for k in range(int(state.K)):
        new_tau2[k] = np.asarray(
            bind.vdam_projector_power_spectrum(
                np.ascontiguousarray(
                    recovar_volume_to_relion(np.asarray(state.Iref[k], dtype=np.float64))
                ),
                int(state.ori_size),
                int(padding_factor),
                int(interpolator),
                current_size,
                True,
                2,
            ),
            dtype=np.float64,
        )
    out = replace(state)
    out.tau2_class = new_tau2
    return out


def default_schedule_update(
    state: InitialModelState,
    iter: int,
    phase_lengths: VdamPhaseLengths,
    *,
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    nr_particles: int,
    tau2_fudge_arg: float,
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS,
    grad_stepsize: float | None = None,
) -> InitialModelState:
    """Apply the three VDAM schedules to `state.iter=iter`."""
    subset_size = compute_subset_size(
        iter=iter,
        phase_lengths=phase_lengths,
        grad_ini_subset_size=grad_ini_subset_size,
        grad_fin_subset_size=grad_fin_subset_size,
        nr_particles=nr_particles,
        nr_iter=state.nr_iter,
        grad_em_iters=grad_em_iters,
        has_converged=state.has_converged,
        grad_has_converged=state.grad_has_converged,
        nr_classes=state.K,
    )
    stepsize = compute_stepsize(
        iter=iter,
        phase_lengths=phase_lengths,
        is_3d_model=True,
        ref_dim=3,
        grad_stepsize=grad_stepsize,
    )
    tau2_fudge = compute_tau2_fudge(
        iter=iter,
        phase_lengths=phase_lengths,
        is_3d_model=True,
        ref_dim=3,
        tau2_fudge_arg=tau2_fudge_arg,
    )
    new_state = replace(state)
    new_state.iter = iter
    new_state.subset_size = subset_size
    new_state.grad_current_stepsize = stepsize
    new_state.tau2_fudge_factor = tau2_fudge
    return new_state


def _resolution_shell_from_data_vs_prior(data_vs_prior: np.ndarray, ori_size: int) -> int:
    """RELION updateCurrentResolution shell scan for one class."""
    dvp = np.asarray(data_vs_prior, dtype=np.float64)
    limit = min(int(ori_size) // 2, int(dvp.size))
    ires = 1
    while ires < limit:
        if float(dvp[ires]) < 1.0:
            break
        ires += 1
    return max(0, ires - 1)


def update_current_resolution_from_data_vs_prior(
    state: InitialModelState,
    *,
    minres_map: int = 5,
) -> InitialModelState:
    """Mirror RELION ``updateCurrentResolution`` for InitialModel/VDAM.

    Gradient InitialModel uses ``data_vs_prior_class`` produced by
    ``BackProjector::updateSSNRarrays``. The resulting ``current_resolution``
    is written in this iteration and converted to the next iteration's
    ``current_size`` when expectation setup calls
    ``updateImageSizeAndResolutionPointers``.
    """
    maxres = 0
    for k in range(int(state.K)):
        maxres = max(maxres, _resolution_shell_from_data_vs_prior(state.data_vs_prior_class[k], state.ori_size))
    maxres = max(maxres, int(minres_map))

    new_state = replace(state)
    new_state.current_resolution_shell = int(maxres)
    new_state.current_resolution = float(maxres) / (float(state.pixel_size) * float(state.ori_size))
    return new_state


def update_image_size_and_resolution_pointers(state: InitialModelState) -> InitialModelState:
    """Mirror the current-size part of RELION ``updateImageSizeAndResolutionPointers``."""
    maxres = _relion_round(float(state.current_resolution) * float(state.pixel_size) * float(state.ori_size))
    if float(state.ave_Pmax) > 0.1 and bool(state.has_high_fsc_at_limit):
        maxres += _relion_round(0.25 * float(state.ori_size) / 2.0)
    else:
        maxres += int(state.incr_size)
    current_size = min(2 * maxres, int(state.ori_size))
    if current_size < 2:
        current_size = 2
    if current_size % 2:
        current_size += 1
    current_size = min(current_size, int(state.ori_size))

    new_state = replace(state)
    new_state.current_size = int(current_size)
    return new_state


def _ave_pmax_from_meta(meta: dict) -> float | None:
    pmax = meta.get("max_posterior_per_image")
    if pmax is not None:
        arr = np.asarray(pmax, dtype=np.float32)
        if arr.size:
            # RELION accumulates Pmax over all VDAM pseudo-halfsets, then
            # divides by the retained M-step posterior mass (rather than the
            # particle count).  Treat the two pseudo-halfsets as one optimiser
            # population when reusing the Class3D normalization helper.
            normalization_mass = meta.get("class_posterior_sums")
            if normalization_mass is not None:
                normalization_mass = float(np.sum(np.asarray(normalization_mass, dtype=np.float64)))
            elif meta.get("noise_sumw") is not None:
                normalization_mass = float(meta["noise_sumw"])
            if normalization_mass is not None:
                _, average, _ = _relion_optimizer_average_pmax([arr], [normalization_mass])
                return average
            return float(np.mean(arr, dtype=np.float64))

    weighted_sum = 0.0
    count = 0
    for key, value in meta.items():
        if key.endswith("_pmax_mean"):
            prefix = key[: -len("_pmax_mean")]
            n_key = f"{prefix}_n_images"
            n = int(meta.get(n_key, 1))
            weighted_sum += float(value) * n
            count += n
    if count:
        return weighted_sum / float(count)

    if "pmax_mean" in meta:
        return float(meta["pmax_mean"])
    return None


def relion_solvent_mask(
    *,
    ori_size: int,
    pixel_size: float,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
) -> np.ndarray:
    """Return RELION's centered spherical ``solventFlatten`` mask."""
    if particle_diameter_ang <= 0.0:
        raise ValueError(f"particle_diameter_ang must be positive, got {particle_diameter_ang}")
    if width_mask_edge_px < 0.0:
        raise ValueError(f"width_mask_edge_px must be non-negative, got {width_mask_edge_px}")
    if pixel_size <= 0.0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")

    n = int(ori_size)
    radius = float(particle_diameter_ang) / (2.0 * float(pixel_size))
    width = float(width_mask_edge_px)
    radius_p = radius + width

    coords = np.arange(-(n // 2), n - (n // 2), dtype=np.float64)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)

    mask = np.zeros((n, n, n), dtype=np.float64)
    mask[r < radius] = 1.0
    if width > 0.0:
        edge = (r >= radius) & (r <= radius_p)
        mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / width)

    return mask


def relion_solvent_flatten_state(
    state: InitialModelState,
    *,
    particle_diameter_ang: float | None = None,
    width_mask_edge_px: float | None = None,
    mask: np.ndarray | None = None,
    compute_dtype: Literal["float32", "float64"] = "float64",
) -> InitialModelState:
    """Apply RELION's spherical ``solventFlatten`` mask to all references (post-maximization)."""
    if compute_dtype not in {"float32", "float64"}:
        raise ValueError(f"Unknown solvent compute_dtype: {compute_dtype!r}")
    iref = np.asarray(state.Iref)
    if compute_dtype == "float32" and iref.dtype != np.dtype(np.float32):
        raise ValueError("float32 solvent multiplication requires float32 state.Iref")
    if iref.ndim != 4 or iref.shape[1:] != (state.ori_size,) * 3:
        raise ValueError(f"state.Iref must have shape (K, {state.ori_size}, ...), got {iref.shape}")
    if mask is None:
        if particle_diameter_ang is None or width_mask_edge_px is None:
            raise ValueError("particle_diameter_ang and width_mask_edge_px are required when mask is not provided")
        mask = relion_solvent_mask(
            ori_size=int(state.ori_size),
            pixel_size=float(state.pixel_size),
            particle_diameter_ang=float(particle_diameter_ang),
            width_mask_edge_px=float(width_mask_edge_px),
        )
    mask = np.asarray(mask, dtype=np.dtype(compute_dtype))
    if mask.shape != (state.ori_size,) * 3:
        raise ValueError(f"mask must have shape ({state.ori_size},)*3, got {mask.shape}")

    new_state = replace(state)
    new_state.Iref = (iref * mask[None, :, :, :]).astype(iref.dtype, copy=False)
    return new_state


def run_vdam_iterations(
    state: InitialModelState,
    *,
    nr_particles: int,
    optics_group_by_particle: Sequence[int],
    grad_ini_subset_size: int,
    grad_fin_subset_size: int,
    tau2_fudge_arg: float,
    grad_em_iters: int,
    random_seed: int,
    rnd_unif_factory: Callable[[int], RndUnifFn],
    expectation_step: ExpectationStepFn,
    iter_artifact_sink: IterArtifactSink = lambda *args, **kw: None,
    post_mstep_update: PostMstepUpdateFn | None = None,
    particle_order: Sequence[int] | None = None,
    grad_ini_frac: float = 0.3,
    grad_fin_frac: float = 0.2,
    phase_lengths: VdamPhaseLengths | None = None,
    grad_stepsize: float | None = None,
    mu: float = DEFAULT_GRAD_MU,
    refresh_tau2_from_projector: bool = True,
    projector_refresh_fn: Callable[..., InitialModelState] | None = None,
    projector_padding_factor: int = 1,
    mstep_backend: str = "native",
    mstep_compute_dtype: Literal["float32", "float64"] = "float64",
    projector_interpolator: int = 1,
    start_iteration: int = 0,
    diagnostic_stop_after_iteration: int | None = None,
) -> InitialModelState:
    """Full VDAM loop; ``state`` must come from ``initialise_denovo_state`` + ``seed_noise_from_mavg``."""
    phase_lengths = _resolve_phase_lengths(
        int(state.nr_iter),
        float(grad_ini_frac),
        float(grad_fin_frac),
        phase_lengths,
    )
    start_iteration = int(start_iteration)
    if start_iteration < 0 or start_iteration >= int(state.nr_iter):
        raise ValueError("start_iteration must be between 0 and state.nr_iter - 1")
    if int(state.iter) != start_iteration:
        raise ValueError(
            f"state.iter must equal start_iteration ({int(state.iter)} != {start_iteration})"
        )
    final_iteration = int(state.nr_iter)
    if diagnostic_stop_after_iteration is not None:
        final_iteration = int(diagnostic_stop_after_iteration)
        if final_iteration <= start_iteration or final_iteration > int(state.nr_iter):
            raise ValueError(
                "diagnostic_stop_after_iteration must be greater than start_iteration "
                "and no greater than state.nr_iter"
            )
    current = state
    profile_iterations = bool(os.environ.get("RECOVAR_INITIAL_MODEL_PROFILE"))

    for it in range(start_iteration + 1, final_iteration + 1):
        iteration_started = time.perf_counter()
        stage_started = iteration_started
        iteration_profile: dict[str, float] = {}

        def _record_stage(name: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            iteration_profile[f"{name}_time_s"] = float(now - stage_started)
            stage_started = now

        do_grad = ((state.nr_iter - it) >= grad_em_iters) and not current.has_converged

        current = default_schedule_update(
            current,
            iter=it,
            phase_lengths=phase_lengths,
            grad_ini_subset_size=grad_ini_subset_size,
            grad_fin_subset_size=grad_fin_subset_size,
            nr_particles=nr_particles,
            tau2_fudge_arg=tau2_fudge_arg,
            grad_em_iters=grad_em_iters,
            grad_stepsize=grad_stepsize,
        )
        if profile_iterations:
            _record_stage("schedule")

        current = select_subset_for_iter(
            current,
            iter=it,
            nr_particles=nr_particles,
            optics_group_by_particle=optics_group_by_particle,
            rnd_unif_factory=rnd_unif_factory,
            random_seed=random_seed,
            do_grad=do_grad,
            particle_order=particle_order,
        )
        if profile_iterations:
            _record_stage("subset")

        current = update_image_size_and_resolution_pointers(current)
        if refresh_tau2_from_projector:
            refresh = projector_refresh_fn or refresh_tau2_from_projector_power
            current = refresh(
                current,
                padding_factor=projector_padding_factor,
                interpolator=projector_interpolator,
            )
        if profile_iterations:
            _record_stage("projector_refresh")

        # E-step: caller-supplied closure over the data loader + dense kernels
        accumulators, meta = expectation_step(
            current,
            current.subset_particle_ids,
            current.subset_halfset_ids,
        )
        if profile_iterations:
            _record_stage("expectation")

        # M-step
        current = vdam_m_step(
            current,
            accumulators=accumulators,
            grad_current_stepsize=current.grad_current_stepsize,
            tau2_fudge_factor=current.tau2_fudge_factor,
            padding_factor=projector_padding_factor,
            mstep_backend=mstep_backend,
            mstep_compute_dtype=mstep_compute_dtype,
        )
        if profile_iterations:
            _record_stage("mstep")
        current = update_probabilities_from_estep_meta(current, meta, do_grad=do_grad, mu=mu)
        current = update_noise_from_estep_meta(current, meta, do_grad=do_grad, mu=mu)
        ave_pmax = _ave_pmax_from_meta(meta)
        if ave_pmax is not None:
            current = replace(current, ave_Pmax=float(ave_pmax))
        if post_mstep_update is not None and not current.has_converged:
            current = post_mstep_update(current, it, meta)

        current = update_current_resolution_from_data_vs_prior(current)
        if profile_iterations:
            _record_stage("state_update")
        meta = dict(meta)
        meta.update(
            {
                "current_size": int(current.current_size),
                "current_resolution": float(current.current_resolution),
                "current_resolution_shell": int(current.current_resolution_shell),
                "ave_Pmax": float(current.ave_Pmax),
                "subset_size": int(current.subset_size),
            }
        )
        if profile_iterations:
            iteration_profile["pre_artifact_time_s"] = float(time.perf_counter() - iteration_started)
            meta["vdam_iteration_profile_summary"] = iteration_profile
            stage_started = time.perf_counter()
        iter_artifact_sink(current, it, meta)
        if profile_iterations:
            _record_stage("artifact")
            iteration_profile["total_time_s"] = float(time.perf_counter() - iteration_started)
            print(f"VDAM iteration {it} profile: {iteration_profile}", flush=True)

        # RECOVAR_CLEAR_JAX_CACHES_PER_ITER=1: release scratch buffers to avoid
        # CUFFT_ALLOC_FAILED at 50k×256² (forces next-iter recompile).
        if os.environ.get("RECOVAR_CLEAR_JAX_CACHES_PER_ITER", "") in ("1", "true", "TRUE"):
            import gc

            import jax

            jax.clear_caches()
            gc.collect()

    return current
