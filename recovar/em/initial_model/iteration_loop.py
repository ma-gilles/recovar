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
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .m_step import vdam_m_step
from .schedules import (
    DEFAULT_GRAD_EM_ITERS,
    DEFAULT_GRAD_MU,
    VdamPhaseLengths,
    _relion_round,
    compute_phase_lengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
)
from .state import InitialModelState
from .subset import (
    RndUnifFn,
    pseudo_halfsets_active,
    randomise_particles_order,
    select_vdam_subset,
)

MIN_SIGMA2_OFFSET_ANGSTROM2: float = 2.0

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

    current_size = int(state.current_size if state.current_size > 0 else state.ori_size)
    new_tau2 = np.asarray(state.tau2_class, dtype=np.float64).copy()
    for k in range(int(state.K)):
        new_tau2[k] = np.asarray(
            bind.vdam_projector_power_spectrum(
                np.asarray(state.Iref[k], dtype=np.float64),
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


def _halfset_values(meta: dict, key: str) -> list:
    suffix = f"_{key}"
    return [meta[name] for name in sorted(meta) if name.startswith("halfset_") and name.endswith(suffix)]


def _posterior_sums_from_meta(meta: dict, key: str) -> np.ndarray | None:
    if (value := meta.get(key)) is not None:
        return np.asarray(value, dtype=np.float64)
    values = [np.asarray(v, dtype=np.float64) for v in _halfset_values(meta, key)]
    return sum(values[1:], values[0].copy()) if values else None


def _scalar_sum_from_meta(meta: dict, key: str) -> float | None:
    if (value := meta.get(key)) is not None:
        return float(value)
    values = _halfset_values(meta, key)
    return float(sum(float(v) for v in values)) if values else None


def _my_mu(mu: float, do_grad: bool, subset_size: int) -> float:
    my_mu = float(mu) if do_grad and subset_size != -1 else 0.0
    if my_mu < 0.0 or my_mu > 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")
    return my_mu


def _array_finite_summary(name: str, value: object, *, max_indices: int = 5) -> str:
    arr = np.asarray(value)
    bad = np.argwhere(~np.isfinite(arr))
    if bad.size == 0:
        values = arr.astype(np.float64, copy=False).reshape(-1)
        if values.size == 0:
            return f"{name}: shape={arr.shape}, empty"
        return f"{name}: shape={arr.shape}, all finite, min={float(np.min(values)):.6g}, max={float(np.max(values)):.6g}"
    finite_values = arr[np.isfinite(arr)].astype(np.float64, copy=False)
    finite_range = (
        f"finite_min={float(np.min(finite_values)):.6g}, finite_max={float(np.max(finite_values)):.6g}"
        if finite_values.size
        else "no finite values"
    )
    sample_indices = [tuple(int(x) for x in idx) for idx in bad[:max_indices]]
    return f"{name}: shape={arr.shape}, nonfinite={int(bad.shape[0])}/{arr.size}, {finite_range}, first_bad={sample_indices}"


def _dump_noise_failure_meta(state: InitialModelState, meta: dict, summaries: Sequence[str]) -> str | None:
    dump_root = os.environ.get("RECOVAR_INITIALMODEL_NOISE_FAILURE_DUMP_DIR")
    if not dump_root:
        return None
    path = Path(dump_root)
    path.mkdir(parents=True, exist_ok=True)
    dump_path = path / f"noise_failure_iter_{getattr(state, 'iter', 'unknown')}.npz"
    payload: dict[str, np.ndarray] = {
        "state_iter": np.asarray([getattr(state, "iter", -1)], dtype=np.int64),
        "state_subset_size": np.asarray([getattr(state, "subset_size", -1)], dtype=np.int64),
        "state_ori_size": np.asarray([getattr(state, "ori_size", -1)], dtype=np.int64),
        "summaries": np.asarray(list(summaries), dtype=str),
    }
    for key, value in sorted(meta.items()):
        if not any(token in key for token in ("noise", "wsum")):
            continue
        try:
            payload[key] = np.asarray(value)
        except Exception:
            payload[f"{key}_repr"] = np.asarray([repr(value)], dtype=str)
    np.savez_compressed(dump_path, **payload)
    return str(dump_path)


def update_noise_from_estep_meta(
    state: InitialModelState,
    meta: dict,
    *,
    do_grad: bool,
    mu: float = DEFAULT_GRAD_MU,
) -> InitialModelState:
    """Update ``sigma2_noise`` from E-step weighted sums (engine units → RELION /N⁴)."""
    wsum_sigma2_noise = _posterior_sums_from_meta(meta, "wsum_sigma2_noise")
    wsum_img_power = _posterior_sums_from_meta(meta, "wsum_img_power")
    noise_sumw = _scalar_sum_from_meta(meta, "noise_sumw")
    if wsum_sigma2_noise is None or wsum_img_power is None or noise_sumw is None:
        return state
    if noise_sumw <= 0.0 or not np.isfinite(noise_sumw):
        return state
    my_mu = _my_mu(mu, do_grad, state.subset_size)

    if wsum_sigma2_noise.shape != wsum_img_power.shape:
        raise ValueError(
            f"wsum_sigma2_noise and wsum_img_power shape mismatch: {wsum_sigma2_noise.shape} vs {wsum_img_power.shape}"
        )
    expected_shells = int(state.ori_size) // 2 + 1
    if wsum_sigma2_noise.shape != (expected_shells,):
        raise ValueError(f"noise weighted sums must have shape ({expected_shells},), got {wsum_sigma2_noise.shape}")
    if not np.all(np.isfinite(wsum_sigma2_noise)) or not np.all(np.isfinite(wsum_img_power)):
        summaries = [
            _array_finite_summary("wsum_sigma2_noise", wsum_sigma2_noise),
            _array_finite_summary("wsum_img_power", wsum_img_power),
            f"noise_sumw={noise_sumw!r}",
        ]
        if dump_path := _dump_noise_failure_meta(state, meta, summaries):
            summaries.append(f"dump={dump_path}")
        raise ValueError("noise weighted sums must be finite: " + "; ".join(summaries))

    from recovar.reconstruction import noise

    sigma2_relion_units = np.asarray(
        noise.normalize_wsum_to_sigma2_noise(
            wsum_sigma2_noise, wsum_img_power, float(noise_sumw), (int(state.ori_size), int(state.ori_size))
        ),
        dtype=np.float64,
    ) / float(int(state.ori_size) ** 4)
    if not np.all(np.isfinite(sigma2_relion_units)) or np.any(sigma2_relion_units <= 0.0):
        raise ValueError("updated sigma2_noise must be positive and finite")

    new_state = replace(state)
    new_sigma2 = np.asarray(state.sigma2_noise, dtype=np.float64).copy()
    if new_sigma2.ndim != 2 or new_sigma2.shape[1] != expected_shells:
        raise ValueError(f"sigma2_noise must have shape (G, {expected_shells}), got {new_sigma2.shape}")
    new_state.sigma2_noise = new_sigma2 * my_mu + (1.0 - my_mu) * sigma2_relion_units[None, :]
    return new_state


def update_probabilities_from_estep_meta(
    state: InitialModelState,
    meta: dict,
    *,
    do_grad: bool,
    mu: float = DEFAULT_GRAD_MU,
) -> InitialModelState:
    """``MlOptimiser::maximizationOtherParameters`` for pdf_class / pdf_direction / sigma2_offset."""
    class_sums = _posterior_sums_from_meta(meta, "class_posterior_sums")
    if class_sums is None:
        return state
    class_sums = np.asarray(class_sums, dtype=np.float64)
    if class_sums.shape != (state.K,):
        raise ValueError(f"class_posterior_sums must have shape ({state.K},), got {class_sums.shape}")
    if not np.all(np.isfinite(class_sums)) or np.any(class_sums < 0.0):
        raise ValueError("class_posterior_sums must be non-negative and finite")
    sum_weight = float(np.sum(class_sums))
    if sum_weight <= 0.0:
        return state
    my_mu = _my_mu(mu, do_grad, state.subset_size)

    new_state = replace(state)
    new_pdf_class = np.asarray(state.pdf_class, dtype=np.float64) * my_mu
    new_pdf_class += (1.0 - my_mu) * class_sums / sum_weight
    pdf_class_sum = float(np.sum(new_pdf_class))
    if pdf_class_sum > 0.0:
        new_pdf_class /= pdf_class_sum
    new_state.pdf_class = new_pdf_class

    direction_sums = _posterior_sums_from_meta(meta, "class_direction_posterior_sums")
    if direction_sums is not None and state.pdf_direction is not None:
        direction_sums = np.asarray(direction_sums, dtype=np.float64)
        if direction_sums.ndim != 2 or direction_sums.shape[0] != state.K:
            raise ValueError(
                f"class_direction_posterior_sums must have shape ({state.K}, n_directions), got {direction_sums.shape}"
            )
        if not np.all(np.isfinite(direction_sums)) or np.any(direction_sums < 0.0):
            raise ValueError("class_direction_posterior_sums must be non-negative and finite")
        pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64)
        if pdf_direction.shape != direction_sums.shape:
            # RELION resizes pdf_direction to the new sampling.NrDirections()
            # and fills it uniformly when angular sampling changes.
            pdf_direction = np.full(direction_sums.shape, 1.0 / float(state.K * direction_sums.shape[1]))
        new_pdf_direction = pdf_direction * my_mu
        new_pdf_direction += (1.0 - my_mu) * direction_sums / sum_weight
        new_state.pdf_direction = new_pdf_direction

    wsum_sigma2_offset = meta.get("wsum_sigma2_offset")
    if wsum_sigma2_offset is not None:
        wsum_sigma2_offset = float(wsum_sigma2_offset)
        if not np.isfinite(wsum_sigma2_offset) or wsum_sigma2_offset < 0.0:
            raise ValueError("wsum_sigma2_offset must be non-negative and finite")
        sigma2_offset = float(state.sigma2_offset) * my_mu
        # RELION divides by 2*sum_weight for 2D particle translations.
        sigma2_offset += (1.0 - my_mu) * wsum_sigma2_offset / (2.0 * sum_weight)
        new_state.sigma2_offset = max(float(sigma2_offset), MIN_SIGMA2_OFFSET_ANGSTROM2)

    return new_state


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
        arr = np.asarray(pmax, dtype=np.float64)
        if arr.size:
            return float(np.mean(arr))

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


def select_subset_for_iter(
    state: InitialModelState,
    iter: int,
    nr_particles: int,
    optics_group_by_particle: Sequence[int],
    rnd_unif_factory: Callable[[int], RndUnifFn],
    random_seed: int,
    do_grad: bool,
    particle_order: Sequence[int] | None = None,
) -> InitialModelState:
    """Select RELION's per-iteration VDAM subset.

    RELION skips particle-order randomization entirely when ``random_seed`` is
    zero. Otherwise it shuffles RELION's current ``sorted_idx`` particle list
    with seed ``random_seed + iter``, takes the first ``subset_size`` (or all
    particles if ``-1``), stable-sorts by optics group, and assigns pseudo-
    halfset ids.
    """
    if particle_order is None:
        base_order = np.arange(int(nr_particles), dtype=np.int64)
        base_halfset_ids = base_order
    else:
        base_order = np.asarray(particle_order, dtype=np.int64)
        if base_order.shape != (int(nr_particles),):
            raise ValueError(f"particle_order must have shape ({int(nr_particles)},), got {base_order.shape}")
        if (
            np.unique(base_order).size != int(nr_particles)
            or np.any(base_order < 0)
            or np.any(base_order >= nr_particles)
        ):
            raise ValueError("particle_order must be a permutation of particle ids [0, nr_particles)")
        # RELION's InitialModel pseudo-halfset routing uses the global particle
        # table row from ``mydata.sorted_idx``:
        # ``iproj_offset = (part_id % 2) * nr_classes`` in storeWeightedSums.
        # ``particle_order`` is RECOVAR's copy of that sorted_idx list, so use
        # the dataset-row parity rather than the sorted-position parity.
        base_halfset_ids = base_order

    if int(random_seed) == 0:
        shuffled = base_order.copy()
        shuffled_halfset_ids = base_halfset_ids.copy()
    else:
        # C++ binding does std::shuffle byte-exact vs RELION; Python is a fallback.
        try:
            from recovar.relion_bind import _relion_bind_core as _bind

            permutation = np.asarray(
                _bind.vdam_randomise_particles_order(int(nr_particles), int(random_seed + iter)), dtype=np.int64
            )
        except (ImportError, AttributeError):
            permutation = randomise_particles_order(nr_particles, rnd_unif_factory(random_seed + iter))
        shuffled = base_order[permutation]
        shuffled_halfset_ids = base_halfset_ids[permutation]
    subset_size = state.subset_size if state.subset_size != -1 else nr_particles
    # `-1` (all particles) still needs to be translated via select_vdam_subset
    pseudo = do_grad and pseudo_halfsets_active(gradient_refine=True, do_split_random_halves=False)
    plan = select_vdam_subset(
        shuffled_particle_ids=shuffled,
        subset_size=subset_size,
        optics_group_by_particle=optics_group_by_particle,
        pseudo_halfsets=pseudo,
        halfset_particle_ids=shuffled_halfset_ids,
    )
    new_state = replace(state)
    new_state.subset_particle_ids = plan.particle_ids
    new_state.subset_halfset_ids = plan.halfset_ids
    new_state.pseudo_halfsets = pseudo
    return new_state


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
) -> InitialModelState:
    """Apply RELION's spherical ``solventFlatten`` mask to all references (post-maximization)."""
    iref = np.asarray(state.Iref)
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
    mask = np.asarray(mask, dtype=np.float64)
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
    mu: float = DEFAULT_GRAD_MU,
    refresh_tau2_from_projector: bool = True,
    projector_padding_factor: int = 1,
    projector_interpolator: int = 1,
) -> InitialModelState:
    """Full VDAM loop; ``state`` must come from ``initialise_denovo_state`` + ``seed_noise_from_mavg``."""
    phase_lengths = compute_phase_lengths(state.nr_iter, grad_ini_frac, grad_fin_frac)
    current = state

    for it in range(1, state.nr_iter + 1):
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
        )

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

        current = update_image_size_and_resolution_pointers(current)
        if refresh_tau2_from_projector:
            current = refresh_tau2_from_projector_power(
                current,
                padding_factor=projector_padding_factor,
                interpolator=projector_interpolator,
            )

        # E-step: caller-supplied closure over the data loader + dense kernels
        accumulators, meta = expectation_step(
            current,
            current.subset_particle_ids,
            current.subset_halfset_ids,
        )

        # M-step
        current = vdam_m_step(
            current,
            accumulators=accumulators,
            grad_current_stepsize=current.grad_current_stepsize,
            tau2_fudge_factor=current.tau2_fudge_factor,
        )
        current = update_probabilities_from_estep_meta(current, meta, do_grad=do_grad, mu=mu)
        current = update_noise_from_estep_meta(current, meta, do_grad=do_grad, mu=mu)
        ave_pmax = _ave_pmax_from_meta(meta)
        if ave_pmax is not None:
            current = replace(current, ave_Pmax=float(ave_pmax))
        if post_mstep_update is not None and not current.has_converged:
            current = post_mstep_update(current, it, meta)

        current = update_current_resolution_from_data_vs_prior(current)
        meta = dict(meta)
        meta.update(
            {
                "current_size": int(current.current_size),
                "current_resolution": float(current.current_resolution),
                "current_resolution_shell": int(current.current_resolution_shell),
                "ave_Pmax": float(current.ave_Pmax),
            }
        )
        iter_artifact_sink(current, it, meta)

        # RECOVAR_CLEAR_JAX_CACHES_PER_ITER=1: release scratch buffers to avoid
        # CUFFT_ALLOC_FAILED at 50k×256² (forces next-iter recompile).
        if os.environ.get("RECOVAR_CLEAR_JAX_CACHES_PER_ITER", "") in ("1", "true", "TRUE"):
            import gc

            import jax

            jax.clear_caches()
            gc.collect()

    return current
