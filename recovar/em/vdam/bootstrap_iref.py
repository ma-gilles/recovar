"""Denovo Iref seeding (RELION ``--pad 1`` parity).

Production path is ``compute_bootstrap_iref_via_cpp`` (C++ binding mirrors
``calculateSumOfPowerSpectraAndAverageImage`` ml_optimiser.cpp:3127-3205 +
reconstruct :3265 + ``initialLowPassFilterReferences`` :3336-3372).
Parity target: ``run_it000_class001.mrc`` (|CC|>0.998).
"""

from __future__ import annotations

import os

import numpy as np

from recovar.em.relion import initial_model_io
from recovar.em.relion.initial_model_io import _experiment_read_order
from recovar.em.relion.initial_noise import _image_sigma2_iter, compute_avg_unaligned_and_sigma2
from recovar.em.vdam import output
from recovar.em.vdam.init import initialise_data_vs_prior_from_references, initialise_denovo_state, seed_noise_from_mavg
from recovar.em.vdam.native_options import NativeInitialModelOptions
from recovar.em.vdam.native_sampling import _n_directions_for_healpix_order
from recovar.em.vdam.state import InitialModelState


def compute_bootstrap_iref_via_cpp(
    *,
    images: np.ndarray,
    defU: np.ndarray,
    defV: np.ndarray,
    defAngle: np.ndarray,
    phase_shift: np.ndarray,
    voltage: float,
    Cs: float,
    Q0: float,
    pixel_size: float,
    ori_size: int,
    nr_classes: int,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_zero_mask: bool,
    do_ctf_correction: bool,
    random_seed: int,
    padding_factor: int = 1,
    current_size: int = -1,
    minimum_nr_particles: int = 1000,
    particle_seed_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Run the full RELION InitialModel bootstrap in C++; returns Iref in recovar frame."""
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import relion_volume_to_recovar

    if current_size <= 0:
        # RELION wsum_model.current_size = ROUND(0.07 * ori_size) (shell count, not Å).
        current_size = int(np.floor(0.07 * ori_size + 0.5))
    seed_ids = None if particle_seed_ids is None else np.ascontiguousarray(particle_seed_ids, dtype=np.int64)

    iref_relion = np.asarray(
        bind.vdam_bootstrap_iref(
            np.ascontiguousarray(images.astype(np.float64)),
            np.ascontiguousarray(defU.astype(np.float64)),
            np.ascontiguousarray(defV.astype(np.float64)),
            np.ascontiguousarray(defAngle.astype(np.float64)),
            np.ascontiguousarray(phase_shift.astype(np.float64)),
            voltage,
            Cs,
            Q0,
            pixel_size,
            ori_size,
            nr_classes,
            particle_diameter_ang,
            width_mask_edge_px,
            do_zero_mask,
            do_ctf_correction,
            random_seed,
            padding_factor,
            1,  # TRILINEAR
            current_size,
            minimum_nr_particles,
            seed_ids,
        )
    )
    return np.asarray([relion_volume_to_recovar(vol) for vol in iref_relion], dtype=np.float64)


def postprocess_bootstrap_iref_via_cpp(
    Iref: np.ndarray,
    *,
    pixel_size: float,
    ini_high_ang: float,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_init_blobs: bool = True,
    is_helical_segment: bool = False,
) -> np.ndarray:
    """Apply RELION's post-bootstrap blobs+LP+softMask pipeline (ml_optimiser.cpp:2940-2980).

    Call immediately after ``compute_bootstrap_iref_via_cpp`` to preserve RELION's
    global ``rand()`` state for the blob draws.
    """
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    arr = np.asarray(Iref, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1] != arr.shape[2] or arr.shape[2] != arr.shape[3]:
        raise ValueError(f"Iref must have shape (K, N, N, N), got {arr.shape}")

    iref_relion = np.asarray([recovar_volume_to_relion(vol) for vol in arr], dtype=np.float64)
    post_relion = np.asarray(
        bind.vdam_postprocess_initial_iref(
            np.ascontiguousarray(iref_relion),
            float(pixel_size),
            float(ini_high_ang),
            float(particle_diameter_ang),
            float(width_mask_edge_px),
            bool(do_init_blobs),
            bool(is_helical_segment),
        ),
        dtype=np.float64,
    )
    return np.asarray([relion_volume_to_recovar(vol) for vol in post_relion], dtype=np.float64)


def _load_raw_images(dataset, image_indices: np.ndarray, *, batch_size: int) -> np.ndarray:
    """Load raw real-space particle images through ``CryoEMDataset`` I/O."""

    images: list[np.ndarray] = []
    for batch_images, _particle_indices, _local_indices in dataset.image_source.iter_batches(
        batch_size=batch_size,
        batch_mode="images",
        subset_indices=np.asarray(image_indices, dtype=np.int64),
    ):
        images.append(np.asarray(batch_images))
    if not images:
        return np.empty((0, dataset.grid_size, dataset.grid_size), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(images, axis=0))


def _initial_state_from_particles(
    dataset,
    main_star,
    optics_star,
    opts: NativeInitialModelOptions,
) -> tuple[InitialModelState, np.ndarray]:
    profile = output._StageProfile()

    ori_size = int(dataset.grid_size)
    pixel_size = float(dataset.voxel_size)
    order = _experiment_read_order(main_star)
    optics_group_by_particle = initial_model_io._optics_group_indices(main_star)
    nr_optics_groups = int(np.unique(optics_group_by_particle).size)
    if nr_optics_groups != 1:
        raise NotImplementedError("native InitialModel currently supports one optics group")
    profile.record("setup")

    Mavg, sigma2_per_group = compute_avg_unaligned_and_sigma2(
        _image_sigma2_iter(
            dataset,
            order,
            optics_group_by_particle,
            batch_size=max(1, int(opts.image_batch_size)),
        ),
        ori_size=ori_size,
        pixel_size=pixel_size,
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=int(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        nr_optics_groups=nr_optics_groups,
        minimum_nr_particles=int(opts.sigma2_min_particles),
    )
    profile.record("average_unaligned")

    bootstrap_count = min(len(order), int(opts.bootstrap_min_particles))
    bootstrap_order = order[:bootstrap_count]
    images = _load_raw_images(dataset, bootstrap_order, batch_size=max(1, int(opts.image_batch_size)))
    profile.record("raw_images")
    sorted_star = main_star.iloc[bootstrap_order]
    voltage, Cs, Q0, pixel_size = initial_model_io._single_optics_scalars(sorted_star, optics_star, dataset)
    profile.record("optics_metadata")

    iref = compute_bootstrap_iref_via_cpp(
        images=images,
        defU=np.asarray(sorted_star["_rlnDefocusU"].astype(float).to_numpy(), dtype=np.float64),
        defV=np.asarray(sorted_star["_rlnDefocusV"].astype(float).to_numpy(), dtype=np.float64),
        defAngle=np.asarray(sorted_star["_rlnDefocusAngle"].astype(float).to_numpy(), dtype=np.float64),
        phase_shift=initial_model_io._phase_shift(sorted_star),
        voltage=voltage,
        Cs=Cs,
        Q0=Q0,
        pixel_size=pixel_size,
        ori_size=ori_size,
        nr_classes=int(opts.nr_classes),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        do_ctf_correction=bool(opts.do_ctf_correction),
        random_seed=int(opts.random_seed),
        padding_factor=int(opts.padding_factor),
        current_size=-1,
        minimum_nr_particles=int(opts.bootstrap_min_particles),
    )
    profile.record("bootstrap")

    state = initialise_denovo_state(
        ori_size=ori_size,
        pixel_size=pixel_size,
        K=int(opts.nr_classes),
        nr_iter=int(opts.nr_iter),
        n_directions=_n_directions_for_healpix_order(int(opts.healpix_order)),
        nr_optics_groups=nr_optics_groups,
        pseudo_halfsets=True,
        padding_factor=int(opts.padding_factor),
    )
    state = seed_noise_from_mavg(state, sigma2_per_group)
    init_sigma_offset_angstrom = (
        opts.translation_sigma_angstrom if opts.translation_sigma_angstrom is not None else 10.0
    )
    state.sigma2_offset = float(init_sigma_offset_angstrom) ** 2
    state.Mavg = Mavg
    profile.record("state_init")
    # RECOVAR_INITIAL_IREF_OVERRIDE lets a parity caller swap in RELION's
    # iter000 ref directly when isolating E/M-step behavior from bootstrap.
    override_path = os.environ.get("RECOVAR_INITIAL_IREF_OVERRIDE")
    if override_path:
        # Parity hook: load Iref directly. Comma-separated paths for K-class,
        # single path broadcast across K, or a "{k}" template expanded k=1..K.
        from recovar.utils.helpers import load_relion_volume

        K = int(opts.nr_classes)
        paths = [p.strip() for p in override_path.split(",") if p.strip()]
        if len(paths) == 1 and "{k" in paths[0]:
            paths = [paths[0].format(k=k + 1) for k in range(K)]
        if len(paths) not in (1, K):
            raise ValueError(f"RECOVAR_INITIAL_IREF_OVERRIDE expects 1 or K={K} paths, got {len(paths)}")
        vols = np.stack(
            [np.asarray(load_relion_volume(p), dtype=np.float64) for p in paths],
            axis=0,
        )
        if vols.shape[1:] != (ori_size, ori_size, ori_size):
            raise ValueError(f"RECOVAR_INITIAL_IREF_OVERRIDE volume shape {vols.shape[1:]} != {(ori_size,) * 3}")
        state.Iref = np.broadcast_to(vols, (K, ori_size, ori_size, ori_size)).copy() if len(paths) == 1 else vols
    else:
        state.Iref = postprocess_bootstrap_iref_via_cpp(
            iref,
            pixel_size=pixel_size,
            ini_high_ang=float(state.ini_high),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
            do_init_blobs=True,
            is_helical_segment=False,
        )
    profile.record("initial_reference")
    state = initialise_data_vs_prior_from_references(
        state,
        nr_particles=len(main_star),
        fix_tau=False,
    )
    profile.record("data_vs_prior")
    profile.report("initial state")
    return state, optics_group_by_particle
