"""Model and particle-state updates from the InitialModel E-step metadata.

RELION's ``MlOptimiser::maximization`` noise (sigma2) and class-probability
(pdf_class) updates for the native VDAM InitialModel, computed from the
E-step accumulator metadata. Optional reports live in diagnostics.vdam_noise.
``iteration_loop`` calls these once per iteration.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from recovar.em.diagnostics import vdam_noise
from recovar.em.helpers.orientation_priors import relion_round_away_from_zero
from recovar.em.vdam.schedules import DEFAULT_GRAD_MU
from recovar.em.vdam.state import InitialModelState, NativeParticleState

MIN_SIGMA2_OFFSET_ANGSTROM2: float = 2.0


def _posterior_sums_from_meta(meta: dict, key: str) -> np.ndarray | None:
    value = meta.get(key)
    return None if value is None else np.asarray(value, dtype=np.float64)


def _my_mu(mu: float, do_grad: bool, subset_size: int) -> float:
    my_mu = float(mu) if do_grad and subset_size != -1 else 0.0
    if my_mu < 0.0 or my_mu > 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")
    return my_mu


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
    wsum_noise_a2 = _posterior_sums_from_meta(meta, "wsum_noise_a2")
    wsum_noise_xa = _posterior_sums_from_meta(meta, "wsum_noise_xa")
    noise_sumw = meta.get("noise_sumw")
    if noise_sumw is not None:
        noise_sumw = float(noise_sumw)
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
            vdam_noise._array_finite_summary("wsum_sigma2_noise", wsum_sigma2_noise),
            vdam_noise._array_finite_summary("wsum_img_power", wsum_img_power),
            f"noise_sumw={noise_sumw!r}",
        ]
        if dump_path := vdam_noise._dump_noise_failure_meta(state, meta, summaries):
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
    vdam_noise._maybe_dump_noise_update_boundary(
        state,
        new_state,
        wsum_sigma2_noise=wsum_sigma2_noise,
        wsum_img_power=wsum_img_power,
        noise_sumw=float(noise_sumw),
        wsum_noise_a2=wsum_noise_a2,
        wsum_noise_xa=wsum_noise_xa,
    )
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
        sigma2_offset_sumw = float(meta.get("sigma2_offset_sumw", sum_weight))
        if not np.isfinite(sigma2_offset_sumw) or sigma2_offset_sumw <= 0.0:
            raise ValueError("sigma2_offset_sumw must be positive and finite")
        sigma2_offset = float(state.sigma2_offset) * my_mu
        # RELION divides by 2*sum_weight for 2D particle translations.
        # Its sum_weight is accumulated from the same significant-pruned
        # reconstruction weights as wsum_sigma2_offset, rather than from the
        # unpruned per-image class responsibilities.
        sigma2_offset += (1.0 - my_mu) * wsum_sigma2_offset / (2.0 * sigma2_offset_sumw)
        new_state.sigma2_offset = max(float(sigma2_offset), MIN_SIGMA2_OFFSET_ANGSTROM2)

    return new_state


def _ensure_field(arr: np.ndarray | None, shape: tuple, dtype, fill=0) -> np.ndarray:
    if arr is None or arr.shape != shape:
        return np.full(shape, fill, dtype=dtype) if fill != 0 else np.zeros(shape, dtype=dtype)
    return arr


def _update_particle_state_from_estep_meta(
    particle_state: NativeParticleState,
    meta: dict,
    translations: np.ndarray,
) -> None:
    selected = meta.get("selected_particle_ids")
    if selected is None:
        return
    ids = np.asarray(selected, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return
    N = particle_state.translation_offsets.shape[0]
    if np.any(ids < 0) or np.any(ids >= N):
        raise ValueError("selected_particle_ids contains entries outside the particle state table")
    if particle_state.visited is None or np.asarray(particle_state.visited).shape != (N,):
        particle_state.visited = np.zeros(N, dtype=bool)
    particle_state.visited[ids] = True

    if (pose := meta.get("pose_assignments")) is not None:
        assignments = np.asarray(pose, dtype=np.int64).reshape(-1)
        trans = np.asarray(translations, dtype=np.float64)
        translation_ids = np.mod(assignments, int(trans.shape[0]))
        base = relion_round_away_from_zero(particle_state.translation_offsets[ids])
        particle_state.translation_offsets[ids] = base + trans[translation_ids, :2]
        particle_state.pose_assignments = _ensure_field(particle_state.pose_assignments, (N,), np.int32, -1)
        particle_state.pose_assignments[ids] = assignments.astype(np.int32, copy=False)

    if (rot := meta.get("best_pose_rotations")) is not None:
        particle_state.best_pose_rotations = _ensure_field(particle_state.best_pose_rotations, (N, 3, 3), np.float32)
        particle_state.best_pose_rotations[ids] = np.asarray(rot, dtype=np.float32)

    source_eulers = meta.get("best_pose_eulers_deg")
    if rot is not None or source_eulers is not None:
        particle_state.best_pose_eulers_valid = _ensure_field(particle_state.best_pose_eulers_valid, (N,), bool, False)
        particle_state.best_pose_eulers_valid[ids] = False
    if source_eulers is not None:
        eulers = np.asarray(source_eulers)
        if eulers.dtype != np.float64 or eulers.shape != (ids.size, 3) or not np.all(np.isfinite(eulers)):
            raise ValueError("source Euler metadata must be finite float64 [selected_particles, 3]")
        particle_state.best_pose_eulers_deg = _ensure_field(particle_state.best_pose_eulers_deg, (N, 3), np.float64)
        particle_state.best_pose_eulers_deg[ids] = eulers
        valid = np.asarray(meta.get("best_pose_eulers_valid", np.ones(ids.size, dtype=bool)))
        if valid.dtype != bool or valid.shape != (ids.size,):
            raise ValueError("source Euler validity must be boolean [selected_particles]")
        particle_state.best_pose_eulers_valid[ids] = valid

    if (bt := meta.get("best_pose_translations")) is not None:
        particle_state.best_pose_translations = _ensure_field(particle_state.best_pose_translations, (N, 2), np.float32)
        particle_state.best_pose_translations[ids] = np.asarray(bt, dtype=np.float32)

    if (rid := meta.get("best_pose_rotation_ids")) is not None:
        particle_state.best_pose_rotation_ids = _ensure_field(particle_state.best_pose_rotation_ids, (N,), np.int32, -1)
        particle_state.best_pose_rotation_ids[ids] = np.asarray(rid, dtype=np.int32).reshape(-1)
        particle_state.best_pose_rotation_orders = _ensure_field(
            particle_state.best_pose_rotation_orders, (N,), np.int32, -1
        )
        particle_state.best_pose_rotation_orders[ids] = int(meta.get("healpix_order", 0)) + int(
            meta.get("oversampling", 0)
        )

    if (cls := meta.get("class_assignments")) is not None:
        particle_state.class_assignments[ids] = np.asarray(cls, dtype=np.int32).reshape(-1)

    if (pmax := meta.get("max_posterior_per_image")) is not None:
        particle_state.max_posterior[ids] = np.asarray(pmax, dtype=np.float32).reshape(-1)
