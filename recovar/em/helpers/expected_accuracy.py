"""RELION-exact expected angular and translational accuracy estimation."""

from __future__ import annotations

import logging
import multiprocessing
import traceback
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from recovar.core.ctf import CTFParamIndex

logger = logging.getLogger("recovar.em.dense_single_volume.helpers.expected_accuracy")


@dataclass(frozen=True)
class ExpectedAccuracy:
    acc_rot: float
    acc_trans_angstrom: float
    acc_rot_per_class: np.ndarray
    acc_trans_per_class_angstrom: np.ndarray
    class_counts: np.ndarray
    trial_local_indices: np.ndarray
    trial_particle_ids: np.ndarray


def estimate_relion_expected_accuracy_from_prepared_inputs(
    *,
    references_relion,
    trial_eulers_deg,
    trial_local_indices,
    trial_class_ids,
    class_weights,
    sigma2_noise_relion,
    defocus_u,
    defocus_v,
    defocus_angle,
    phase_shift,
    voltage: float,
    spherical_aberration: float,
    amplitude_contrast: float,
    pixel_size: float,
    ori_size: int,
    current_image_size: int,
    padding_factor: int,
    sigma2_fudge: float,
    random_seed: int,
    do_ctf_correction: bool,
    random_seed_particle_ids,
) -> ExpectedAccuracy:
    """Call RELION's expected-accuracy binding from prepared native inputs.

    Supplied-map EM and InitialModel have different state containers and noise
    conventions, but the native calculation itself must have one owner.
    Callers prepare their layouts, then cross this shared binding boundary.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    trial_local = np.asarray(trial_local_indices, dtype=np.int64).reshape(-1)
    trial_particles = np.asarray(random_seed_particle_ids, dtype=np.int64).reshape(-1)
    eulers = np.asarray(trial_eulers_deg, dtype=np.float64)
    classes = np.asarray(trial_class_ids, dtype=np.int32).reshape(-1)
    if eulers.shape != (trial_local.size, 3):
        raise ValueError(
            "trial_eulers_deg must have shape "
            f"({trial_local.size}, 3), got {eulers.shape}"
        )
    if classes.shape != trial_local.shape:
        raise ValueError(
            f"trial_class_ids must have shape {trial_local.shape}, got {classes.shape}"
        )
    if trial_particles.shape != trial_local.shape:
        raise ValueError(
            "random_seed_particle_ids must have shape "
            f"{trial_local.shape}, got {trial_particles.shape}"
        )

    out = bind.vdam_expected_angular_errors(
        np.ascontiguousarray(references_relion, dtype=np.float64),
        np.ascontiguousarray(eulers),
        np.ascontiguousarray(trial_local),
        np.ascontiguousarray(classes),
        np.ascontiguousarray(class_weights, dtype=np.float64),
        np.ascontiguousarray(sigma2_noise_relion, dtype=np.float64),
        np.ascontiguousarray(defocus_u, dtype=np.float64),
        np.ascontiguousarray(defocus_v, dtype=np.float64),
        np.ascontiguousarray(defocus_angle, dtype=np.float64),
        np.ascontiguousarray(phase_shift, dtype=np.float64),
        float(voltage),
        float(spherical_aberration),
        float(amplitude_contrast),
        float(pixel_size),
        int(ori_size),
        int(current_image_size),
        int(padding_factor),
        1,
        float(sigma2_fudge),
        int(random_seed),
        bool(do_ctf_correction),
        False,
        np.ascontiguousarray(trial_particles),
    )
    return ExpectedAccuracy(
        acc_rot=float(out["acc_rot"]),
        acc_trans_angstrom=float(out["acc_trans"]),
        acc_rot_per_class=np.asarray(out["acc_rot_class"], dtype=np.float64),
        acc_trans_per_class_angstrom=np.asarray(out["acc_trans_class"], dtype=np.float64),
        class_counts=np.asarray(out["class_counts"], dtype=np.int64),
        trial_local_indices=trial_local.copy(),
        trial_particle_ids=trial_particles.copy(),
    )


def _estimate_relion_expected_accuracy_spawn_worker(
    connection,
    kwargs: dict[str, Any],
) -> None:
    """Run the authoritative binding behind an isolated spawn boundary."""
    try:
        result = estimate_relion_expected_accuracy_from_prepared_inputs(**kwargs)
        connection.send(("ok", result))
    except BaseException:  # pragma: no cover - exercised through the parent error
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def estimate_relion_expected_accuracy_in_spawned_process_from_prepared_inputs(
    **kwargs,
) -> ExpectedAccuracy:
    """Evaluate expected accuracy without mutating the caller's process state.

    The numerical owner remains
    :func:`estimate_relion_expected_accuracy_from_prepared_inputs`.  This
    wrapper exists for trajectory diagnostics where RELION's process-global
    RNG, FFT, or allocator state must not leak from the temporary
    expected-accuracy projectors into the production E/M-step process.
    ``spawn`` is deliberate: forking a process after JAX has created worker
    threads or a CUDA context is unsafe.
    """
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_estimate_relion_expected_accuracy_spawn_worker,
        args=(child_connection, kwargs),
        name="recovar-relion-expected-accuracy",
    )
    process.start()
    child_connection.close()
    try:
        status, payload = parent_connection.recv()
    except EOFError as error:
        process.join()
        raise RuntimeError(
            "isolated RELION expected-accuracy worker exited without a result "
            f"(exit code {process.exitcode})"
        ) from error
    finally:
        parent_connection.close()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            "isolated RELION expected-accuracy worker failed after returning "
            f"a result (exit code {process.exitcode})"
        )
    if status != "ok":
        raise RuntimeError(f"isolated RELION expected-accuracy worker failed:\n{payload}")
    if not isinstance(payload, ExpectedAccuracy):
        raise RuntimeError(
            "isolated RELION expected-accuracy worker returned an invalid payload"
        )
    return payload


def relion_auto_refine_half_orders(
    random_subsets,
    random_seed: int,
    first_iteration: int = 1,
    *,
    optics_group_ids=None,
    shuffle_algorithm: Literal["legacy", "mt19937"] = "legacy",
) -> tuple[np.ndarray, np.ndarray]:
    """Return RELION's fresh AutoRefine particle-row order for both halves.

    Legacy RELION uses libc ``rand``/``random_shuffle``; f2c1a384 instead
    uses ``mt19937``/``shuffle``. Both shuffle half 1 and then half 2 without
    reseeding, and stable-sort each half by numeric optics group. The legacy
    default preserves existing pinned fixtures during modern-oracle validation.
    Returned values index the supplied full particle table, including the
    first 100 trials consumed by the expected-accuracy estimator.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    subsets = np.asarray(random_subsets, dtype=np.int64).reshape(-1)
    if set(subsets.tolist()) != {1, 2}:
        raise ValueError("random_subsets must contain exactly RELION half labels 1 and 2")
    base_orders = (
        np.flatnonzero(subsets == 1).astype(np.int64),
        np.flatnonzero(subsets == 2).astype(np.int64),
    )
    binding_names = {
        "legacy": "auto_refine_randomise_half_orders",
        "mt19937": "auto_refine_randomise_half_orders_mt19937",
    }
    if shuffle_algorithm not in binding_names:
        raise ValueError(f"unknown shuffle_algorithm: {shuffle_algorithm!r}")
    shuffle = getattr(bind, binding_names[shuffle_algorithm], None)
    if shuffle is None:
        raise RuntimeError(
            f"RELION binding lacks {binding_names[shuffle_algorithm]}; rebuild recovar/relion_bind"
        )
    positions = shuffle(
        int(base_orders[0].size),
        int(base_orders[1].size),
        int(random_seed) + int(first_iteration),
    )
    orders = [
        base[np.asarray(permutation, dtype=np.int64)]
        for base, permutation in zip(base_orders, positions, strict=True)
    ]

    if optics_group_ids is not None:
        optics = np.asarray(optics_group_ids, dtype=np.int64).reshape(-1)
        if optics.shape != subsets.shape:
            raise ValueError(
                f"optics_group_ids must have shape {subsets.shape}, got {optics.shape}"
            )
        orders = [
            order[np.argsort(optics[order], kind="stable")]
            for order in orders
        ]
    return orders[0], orders[1]


def relion_half1_trial_order(
    n_particles: int,
    random_seed: int,
    first_iteration: int = 1,
    *,
    base_order_local=None,
    optics_group_ids=None,
) -> np.ndarray:
    """Return RELION's one-time randomized half-1 local particle order.

    ``Experiment::randomiseParticlesOrder`` uses ``srand(random_seed + iter)``
    followed by ``std::random_shuffle``.  Full-data AutoRefine performs that
    shuffle only once per process; a fresh refinement therefore uses
    ``first_iteration=1``.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    if not hasattr(bind, "auto_refine_randomise_half_order"):
        raise RuntimeError("RELION binding lacks auto_refine_randomise_half_order; rebuild recovar/relion_bind")
    shuffled_positions = np.asarray(
        bind.auto_refine_randomise_half_order(int(n_particles), int(random_seed) + int(first_iteration)),
        dtype=np.int64,
    )
    if base_order_local is None:
        base_order = np.arange(int(n_particles), dtype=np.int64)
    else:
        base_order = np.asarray(base_order_local, dtype=np.int64).reshape(-1)
        if base_order.shape != (int(n_particles),):
            raise ValueError(
                f"base_order_local must have shape ({n_particles},), got {base_order.shape}",
            )
        if not np.array_equal(np.sort(base_order), np.arange(int(n_particles), dtype=np.int64)):
            raise ValueError("base_order_local must be a permutation of half-local particle indices")
    order = base_order[shuffled_positions]
    if optics_group_ids is not None:
        optics = np.asarray(optics_group_ids, dtype=np.int64).reshape(-1)
        if optics.shape != (int(n_particles),):
            raise ValueError(f"optics_group_ids must have shape ({n_particles},), got {optics.shape}")
        # Experiment::randomiseParticlesOrder stable-sorts the already
        # shuffled half by numeric optics group.
        order = order[np.argsort(optics[order], kind="stable")]
    return order


def prepare_relion_half1_trial_order(
    *,
    expected_accuracy,
    half1_dataset,
    optimizer_random_seed,
    init_relion_iteration,
    log,
) -> np.ndarray | None:
    """Resolve the controller's half-1 expected-accuracy trial order.

    Explicit metadata must be a full local permutation. Native setup failures
    retain the existing warning/None fallback, so unavailable exact accuracy
    cannot trigger convergence. The caller supplies its logging context.
    """
    expected_accuracy_trial_order = None
    if expected_accuracy.half1_trial_order_local is not None:
        expected_accuracy_trial_order = np.asarray(
            expected_accuracy.half1_trial_order_local,
            dtype=np.int64,
        ).reshape(-1)
        expected_trial_count = int(half1_dataset.n_units)
        if expected_accuracy_trial_order.shape != (expected_trial_count,):
            raise ValueError(
                "expected_accuracy.half1_trial_order_local must have shape "
                f"({expected_trial_count},), got {expected_accuracy_trial_order.shape}"
            )
        if not np.array_equal(
            np.sort(expected_accuracy_trial_order),
            np.arange(expected_trial_count, dtype=np.int64),
        ):
            raise ValueError(
                "expected_accuracy.half1_trial_order_local must be a permutation "
                "of half-1 local particle indices"
            )
        log.info(
            "RELION expected accuracy consumes an explicit physical trial order "
            "(%d particles)",
            expected_trial_count,
        )
    elif optimizer_random_seed is not None and int(half1_dataset.n_units) > 0:
        try:
            expected_accuracy_trial_order = relion_half1_trial_order(
                int(half1_dataset.n_units),
                int(optimizer_random_seed),
                first_iteration=max(1, int(init_relion_iteration) + 1),
                base_order_local=expected_accuracy.half1_base_order_local,
                optics_group_ids=expected_accuracy.half1_optics_group_ids,
            )
            if (
                expected_accuracy.half1_optics_group_ids is not None
                and np.unique(np.asarray(expected_accuracy.half1_optics_group_ids)).size > 1
            ):
                raise NotImplementedError(
                    "exact expected accuracy currently supports one RELION optics group; "
                    "per-optics image size/noise/CTF scaling is not yet implemented",
                )
        except Exception as exc:
            expected_accuracy_trial_order = None
            log.warning("RELION exact expected-accuracy particle order unavailable: %s", exc)

    return expected_accuracy_trial_order


def _constant_selected(values: np.ndarray, indices: np.ndarray, name: str) -> float:
    selected = np.asarray(values, dtype=np.float64).reshape(-1)[indices]
    if selected.size == 0:
        raise ValueError(f"cannot estimate accuracy without selected {name} values")
    if not np.allclose(selected, selected[0], rtol=0.0, atol=1e-6):
        raise NotImplementedError(f"RELION expected-accuracy binding currently requires one optics-group {name}")
    return float(selected[0])


def estimate_relion_expected_accuracy(
    *,
    reference_fourier,
    volume_shape: tuple[int, int, int],
    best_eulers_deg,
    class_ids,
    class_weights,
    sigma2_noise_native,
    dataset,
    trial_order_local,
    current_image_size: int,
    padding_factor: int,
    sigma2_fudge: float,
    random_seed: int,
    random_seed_particle_ids=None,
    ctf_params_override=None,
    do_ctf_correction: bool | None = None,
    max_trials: int = 100,
) -> ExpectedAccuracy:
    """Evaluate RELION ``calculateExpectedAngularErrors`` on half 1.

    The binding consumes RELION map/noise conventions.  RECOVAR's real-space
    maps already have the same physical scale, while its native FFT noise
    variance is larger by ``ori_size**4``.
    """
    from recovar.core import fourier_transform_utils
    from recovar.utils.helpers import recovar_volume_to_relion

    eulers = np.asarray(best_eulers_deg, dtype=np.float64)
    if eulers.ndim != 2 or eulers.shape[1] != 3:
        raise ValueError(f"best_eulers_deg must have shape (N, 3), got {eulers.shape}")
    n_particles = int(eulers.shape[0])
    order = np.asarray(trial_order_local, dtype=np.int64).reshape(-1)
    if order.shape != (n_particles,):
        raise ValueError(f"trial_order_local must have shape ({n_particles},), got {order.shape}")
    trial_local = order[: min(int(max_trials), n_particles)]
    if trial_local.size == 0:
        raise ValueError("expected-accuracy estimation requires at least one particle")

    if random_seed_particle_ids is None:
        particle_ids = np.asarray(dataset.original_image_indices_from_local(), dtype=np.int64).reshape(-1)
    else:
        particle_ids = np.asarray(random_seed_particle_ids, dtype=np.int64).reshape(-1)
    if particle_ids.shape != (n_particles,):
        raise ValueError(f"random-seed particle IDs have shape {particle_ids.shape}, expected ({n_particles},)")
    trial_particle_ids = particle_ids[trial_local]

    refs_ft = np.asarray(reference_fourier)
    if refs_ft.ndim == 1:
        refs_ft = refs_ft[None, :]
    expected_voxels = int(np.prod(volume_shape))
    if refs_ft.ndim != 2 or refs_ft.shape[1] != expected_voxels:
        raise ValueError(f"reference_fourier must have shape (K, {expected_voxels}), got {refs_ft.shape}")
    refs_relion = []
    for ref_ft in refs_ft:
        ref_real = np.asarray(
            fourier_transform_utils.get_idft3(ref_ft.reshape(volume_shape)).real,
            dtype=np.float64,
        )
        refs_relion.append(np.asarray(recovar_volume_to_relion(ref_real), dtype=np.float64))
    references = np.ascontiguousarray(np.stack(refs_relion, axis=0))

    classes = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    if classes.shape != (n_particles,):
        raise ValueError(f"class_ids must have shape ({n_particles},), got {classes.shape}")
    weights = np.asarray(class_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (references.shape[0],):
        raise ValueError(f"class_weights must have shape ({references.shape[0]},), got {weights.shape}")

    ctf_source = dataset.CTF_params if ctf_params_override is None else ctf_params_override
    ctf = np.asarray(ctf_source, dtype=np.float64)
    if ctf.shape[0] != n_particles:
        raise ValueError(f"CTF parameter rows {ctf.shape[0]} do not match particles {n_particles}")
    voltage = _constant_selected(ctf[:, CTFParamIndex.VOLT], trial_local, "voltage")
    cs = _constant_selected(ctf[:, CTFParamIndex.CS], trial_local, "spherical aberration")
    amplitude_contrast = _constant_selected(ctf[:, CTFParamIndex.W], trial_local, "amplitude contrast")
    if do_ctf_correction is None:
        # RECOVAR's simulator uses negative amplitude contrast as its explicit
        # identity/no-CTF sentinel.  Canonical RELION parity runs should pass
        # rlnDoCorrectCtf from the optimiser instead of relying on this fallback.
        do_ctf_correction = amplitude_contrast >= 0.0

    ori_size = int(volume_shape[0])
    sigma2_noise_relion = np.asarray(sigma2_noise_native, dtype=np.float64).reshape(-1) / float(ori_size**4)
    return estimate_relion_expected_accuracy_from_prepared_inputs(
        references_relion=references,
        trial_eulers_deg=eulers[trial_local],
        trial_local_indices=trial_local,
        trial_class_ids=classes[trial_local],
        class_weights=weights,
        sigma2_noise_relion=sigma2_noise_relion,
        defocus_u=ctf[:, CTFParamIndex.DFU],
        defocus_v=ctf[:, CTFParamIndex.DFV],
        defocus_angle=ctf[:, CTFParamIndex.DFANG],
        phase_shift=ctf[:, CTFParamIndex.PHASE_SHIFT],
        voltage=voltage,
        spherical_aberration=cs,
        amplitude_contrast=amplitude_contrast,
        pixel_size=float(dataset.voxel_size),
        ori_size=ori_size,
        current_image_size=int(current_image_size),
        padding_factor=int(padding_factor),
        sigma2_fudge=float(sigma2_fudge),
        random_seed=int(random_seed),
        do_ctf_correction=bool(do_ctf_correction),
        random_seed_particle_ids=trial_particle_ids,
    )
