"""Images of several RELION optics groups, each on its own pixel size and box.

Shared by the RELION 5 tomo writer (:mod:`recovar.simulation.relion_tomo`) and the
RELION SPA writer (:mod:`recovar.simulation.relion_spa`). A group's volumes are the
reference volumes Fourier-resampled to its pixel size and cropped or padded in real
space to its box, with the atomic-volume transform applied on that grid. Every
group gets the same per-pixel noise variance, calibrated so that the first group's
mean noise-free signal power over the noise power is ``snr``, times the group's
``noise_scale``.
"""

import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.utils as utils
from recovar.data_io import cryoem_dataset
from recovar.simulation import simulator, solvent_contrast

DEFAULT_OPTICS_GROUPS = (
    {"voltage": 300.0, "cs": 2.7, "amp_contrast": 0.1, "noise_scale": 1.0},
    {"voltage": 200.0, "cs": 1.4, "amp_contrast": 0.07, "noise_scale": 1.5},
)


def group_volumes(volumes_path_root, trailing_zero_format_in_vol_name, voxel_size, grid_size, pixel_size, box_size):
    """Fourier volumes resampled to ``pixel_size`` and cropped or padded in real space to ``box_size``."""
    resampled_grid = grid_size * voxel_size / pixel_size
    if not np.isclose(resampled_grid, round(resampled_grid)) or round(resampled_grid) % 2:
        raise ValueError(f"grid_size * voxel_size / pixel_size = {resampled_grid} must be an even integer")
    resampled_grid = int(round(resampled_grid))
    volumes = simulator.load_volumes_from_folder(
        volumes_path_root, resampled_grid, trailing_zero_format_in_vol_name, normalize=False
    )
    if resampled_grid == box_size:
        return volumes
    out = np.zeros((volumes.shape[0], box_size, box_size, box_size))
    lo_in, lo_out = max(0, (resampled_grid - box_size) // 2), max(0, (box_size - resampled_grid) // 2)
    n = min(resampled_grid, box_size)
    for i, vol in enumerate(volumes):
        real = np.real(np.asarray(fourier_transform_utils.get_idft3(vol.reshape((resampled_grid,) * 3))))
        out[i, lo_out : lo_out + n, lo_out : lo_out + n, lo_out : lo_out + n] = real[
            lo_in : lo_in + n, lo_in : lo_in + n, lo_in : lo_in + n
        ]
    return np.stack([np.asarray(fourier_transform_utils.get_dft3(v)).reshape(-1) for v in out])


def simulate_optics_groups(
    volumes,
    optics_groups,
    row_optics,
    rots,
    ctf_params,
    row_volume,
    row_contrast,
    *,
    ctf_evaluator,
    volumes_path_root,
    trailing_zero_format_in_vol_name,
    voxel_size,
    grid_size,
    scale_vol,
    solvent_record,
    snr,
    noise_model,
    n_probe,
    seed,
    disc_type,
    premultiplied_ctf,
    row_translations=None,
):
    """Simulate every row's image on its optics group's grid.

    ``volumes`` are the reference volumes on the ``(voxel_size, grid_size)`` grid,
    already scaled by ``scale_vol`` and transformed by ``solvent_record``. Row ``r``
    belongs to group ``row_optics[r]`` (an index into ``optics_groups``, dicts with
    ``pixel_size``, ``box_size`` and ``noise_scale``), shows volume ``row_volume[r]``
    with contrast ``row_contrast[r]`` at rotation ``rots[r]`` and CTF parameters
    ``ctf_params[r]`` evaluated by ``ctf_evaluator`` (or by ``ctf_evaluator[g]`` when it is
    a sequence with one evaluator per group). The first ``n_probe`` rows of
    each group calibrate the noise.

    Returns the per-row images (a list, each on its group's box) and the per-pixel
    noise variance of each group (``None`` for a group without rows).
    ``row_translations`` are per-row image shifts in pixels of the row's group,
    in the convention of translations read from a RELION STAR (``+rlnOrigin``);
    None means zero.
    """

    row_images = [None] * len(row_optics)
    noise_variances = []
    target_noise_power = None
    for g, og in enumerate(optics_groups):
        pixel_size, box_size = og["pixel_size"], og["box_size"]
        group_ctf_evaluator = ctf_evaluator[g] if isinstance(ctf_evaluator, (list, tuple)) else ctf_evaluator
        if pixel_size == voxel_size and box_size == grid_size:
            volumes_g = volumes
        else:
            volumes_g = scale_vol * group_volumes(
                volumes_path_root, trailing_zero_format_in_vol_name, voxel_size, grid_size, pixel_size, box_size
            )
            if solvent_record["enabled"]:
                volumes_g = solvent_contrast.apply_solvent_contrast(
                    volumes_g,
                    (box_size,) * 3,
                    pixel_size,
                    solvent_record["a"],
                    solvent_record["B"],
                    solvent_record["B_atomic"],
                )
        batch_size = int(5 * utils.get_image_batch_size(box_size, utils.get_gpu_memory_total()))

        def simulate(rows, noise_variance, contrast, noise_scale, seed_offset):
            dataset = cryoem_dataset.CryoEMDataset(
                None,
                pixel_size,
                cryoem_dataset.ImageMetadata(
                    rots[rows],
                    np.zeros((rows.size, 2)) if row_translations is None else row_translations[rows],
                    ctf_params[rows],
                ),
                ctf_evaluator=group_ctf_evaluator,
                grid_size=box_size,
            )
            return simulator.simulate_data(
                dataset,
                volumes_g,
                noise_variance,
                batch_size,
                row_volume[rows],
                contrast,
                noise_scale,
                seed=seed + seed_offset,
                disc_type=disc_type,
                premultiplied_ctf=premultiplied_ctf,
            )

        rows = np.nonzero(row_optics == g)[0]
        if rows.size == 0:
            noise_variances.append(None)
            continue
        noise_shape = simulator.get_noise_model(noise_model, box_size)
        probe = rows[:n_probe]
        ones, zeros = np.ones(probe.size), np.zeros(probe.size)
        if target_noise_power is None:
            target_noise_power = np.mean(simulate(probe, 0 * noise_shape, ones, ones, 3 * g + 1) ** 2) / snr
        unit_noise_power = np.mean(simulate(probe, noise_shape, zeros, ones, 3 * g + 2) ** 2)
        noise_variance = noise_shape * target_noise_power / unit_noise_power
        noise_variances.append(noise_variance.astype(np.float32))

        images = simulate(rows, noise_variance, row_contrast[rows], og["noise_scale"] * np.ones(rows.size), 3 * g)
        for row, image in zip(rows, images):
            row_images[row] = image
    return row_images, noise_variances


def _block_seed(*parts):
    """A 32-bit JAX seed derived from ``parts`` (one independent noise stream per block and call)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def simulate_per_particle_volumes(
    particle_volume_fn,
    volume_grid_size,
    optics_groups,
    row_optics,
    row_particle,
    rots,
    ctf_params,
    row_contrast,
    *,
    ctf_evaluator,
    voxel_size,
    grid_size,
    snr,
    noise_model,
    n_probe_particles,
    seed,
    disc_type,
    premultiplied_ctf,
    row_translations=None,
    max_volume_bytes=2**31,
):
    """Simulate every row's image from its own particle's volume, as :func:`simulate_optics_groups` does for shared volumes.

    ``particle_volume_fn(particle_indices)`` returns the volumes of those particles,
    shape ``(len(particle_indices), volume_grid_size**3)``: flat, centred Fourier volumes
    in recovar's frame (:func:`recovar.core.fourier_transform_utils.get_dft3` of a real
    array whose axes 0, 1, 2 are RELION's reference x, y, z) at ``voxel_size``. Row ``r``
    shows particle ``row_particle[r]``. Volumes are built in blocks of at most
    ``max_volume_bytes`` and only for the particles a call needs, so no file or array
    holds every particle's volume.

    ``volume_grid_size`` may exceed the image box ``grid_size`` (same voxel size): each
    image is then formed on the larger box (projection, CTF on the 2x padded image,
    noise) and cropped to the central ``grid_size`` pixels. This lets content outside
    the image box (e.g. a filament that continues past the box) project into tilted
    images. White noise cropped this way keeps its per-pixel variance.

    Noise follows :func:`simulate_optics_groups`: one per-pixel variance shared by every
    group, ``noise_scale`` times it, calibrated so that the mean noise-free signal power
    of the probe rows of the first group with rows over the noise power is ``snr``. The
    probe rows are every row of ``n_probe_particles`` particles spread evenly over that
    group (not the first rows, so the probe does not hinge on one filament). ``snr`` of
    None or inf writes noise-free images. Every group's ``pixel_size`` and ``box_size``
    must equal ``voxel_size`` and ``grid_size``.

    Returns the per-row images (``(n_rows, grid_size, grid_size)``) and the per-pixel
    noise variance of each group on the ``volume_grid_size`` grid (``None`` for a group
    without rows), plus a dict with the calibration record.
    """
    for og in optics_groups:
        if og["pixel_size"] != voxel_size or og["box_size"] != grid_size:
            raise ValueError(
                "per-particle volumes need every optics group on the reference grid "
                f"(pixel_size {voxel_size}, box {grid_size}), got {og['pixel_size']}, {og['box_size']}"
            )
    volume_grid_size = int(volume_grid_size)
    if volume_grid_size < grid_size or (volume_grid_size - grid_size) % 2:
        raise ValueError(
            f"volume_grid_size {volume_grid_size} must be >= grid_size {grid_size} with an even difference"
        )
    lo = (volume_grid_size - grid_size) // 2
    row_optics = np.asarray(row_optics)
    row_particle = np.asarray(row_particle)
    noiseless = snr is None or np.isinf(snr)
    particles_per_block = max(1, int(max_volume_bytes // (16 * volume_grid_size**3)))
    batch_size = int(5 * utils.get_image_batch_size(volume_grid_size, utils.get_gpu_memory_total()))

    def zero_volumes(particles):
        return np.zeros((len(particles), volume_grid_size**3), dtype=np.complex128)

    def simulate(rows, volume_fn, noise_variance, contrast, noise_scale, group_ctf_evaluator, seed_parts):
        out = np.empty((rows.size, grid_size, grid_size), dtype=np.float32)  # simulate_data's image dtype
        particles = np.unique(row_particle[rows])
        for b, start in enumerate(range(0, particles.size, particles_per_block)):
            block = particles[start : start + particles_per_block]
            sel = np.nonzero(np.isin(row_particle[rows], block))[0]
            block_rows = rows[sel]
            volumes = volume_fn(block)
            if tuple(volumes.shape) != (block.size, volume_grid_size**3):
                raise ValueError(
                    f"particle_volume_fn returned shape {tuple(volumes.shape)}, "
                    f"expected {(block.size, volume_grid_size**3)}"
                )
            dataset = cryoem_dataset.CryoEMDataset(
                None,
                voxel_size,
                cryoem_dataset.ImageMetadata(
                    rots[block_rows],
                    np.zeros((block_rows.size, 2)) if row_translations is None else row_translations[block_rows],
                    ctf_params[block_rows],
                ),
                ctf_evaluator=group_ctf_evaluator,
                grid_size=volume_grid_size,
            )
            images = simulator.simulate_data(
                dataset,
                volumes,
                noise_variance,
                batch_size,
                np.searchsorted(block, row_particle[block_rows]),
                contrast[sel],
                noise_scale[sel],
                seed=_block_seed(seed, *seed_parts, b),
                disc_type=disc_type,
                premultiplied_ctf=premultiplied_ctf,
            )
            out[sel] = np.asarray(images)[:, lo : lo + grid_size, lo : lo + grid_size]
        return out

    row_images = np.empty((row_optics.size, grid_size, grid_size), dtype=np.float32)
    noise_variances = []
    target_noise_power = None
    calibration = {"noiseless": bool(noiseless), "probe_particles": None, "probe_signal_power": None}
    for g, og in enumerate(optics_groups):
        group_ctf_evaluator = ctf_evaluator[g] if isinstance(ctf_evaluator, (list, tuple)) else ctf_evaluator
        rows = np.nonzero(row_optics == g)[0]
        if rows.size == 0:
            noise_variances.append(None)
            continue
        noise_shape = simulator.get_noise_model(noise_model, volume_grid_size)
        if noiseless:
            noise_variance = 0 * noise_shape
        else:
            if target_noise_power is None:
                group_particles = np.unique(row_particle[rows])
                probe_particles = group_particles[
                    np.unique(np.linspace(0, group_particles.size - 1, n_probe_particles).round().astype(int))
                ]
                probe = rows[np.isin(row_particle[rows], probe_particles)]
                ones = np.ones(probe.size)
                signal_power = np.mean(
                    simulate(probe, particle_volume_fn, 0 * noise_shape, ones, ones, group_ctf_evaluator, (g, 1)) ** 2
                )
                target_noise_power = signal_power / snr
                calibration.update(probe_particles=probe_particles.tolist(), probe_signal_power=float(signal_power))
            probe = rows[np.isin(row_particle[rows], np.unique(row_particle[rows])[:n_probe_particles])]
            unit_noise_power = np.mean(
                simulate(
                    probe,
                    zero_volumes,
                    noise_shape,
                    np.zeros(probe.size),
                    np.ones(probe.size),
                    group_ctf_evaluator,
                    (g, 2),
                )
                ** 2
            )
            noise_variance = noise_shape * target_noise_power / unit_noise_power
        noise_variances.append(np.asarray(noise_variance, dtype=np.float32))
        row_images[rows] = simulate(
            rows,
            particle_volume_fn,
            noise_variance,
            row_contrast[rows],
            og["noise_scale"] * np.ones(rows.size),
            group_ctf_evaluator,
            (g, 0),
        )
    calibration["target_noise_power"] = None if target_noise_power is None else float(target_noise_power)
    return row_images, noise_variances, calibration
