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
):
    """Simulate every row's image on its optics group's grid.

    ``volumes`` are the reference volumes on the ``(voxel_size, grid_size)`` grid,
    already scaled by ``scale_vol`` and transformed by ``solvent_record``. Row ``r``
    belongs to group ``row_optics[r]`` (an index into ``optics_groups``, dicts with
    ``pixel_size``, ``box_size`` and ``noise_scale``), shows volume ``row_volume[r]``
    with contrast ``row_contrast[r]`` at rotation ``rots[r]`` and CTF parameters
    ``ctf_params[r]`` evaluated by ``ctf_evaluator``. The first ``n_probe`` rows of
    each group calibrate the noise.

    Returns the per-row images (a list, each on its group's box) and the per-pixel
    noise variance of each group (``None`` for a group without rows).
    """

    row_images = [None] * len(row_optics)
    noise_variances = []
    target_noise_power = None
    for g, og in enumerate(optics_groups):
        pixel_size, box_size = og["pixel_size"], og["box_size"]
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
                cryoem_dataset.ImageMetadata(rots[rows], np.zeros((rows.size, 2)), ctf_params[rows]),
                ctf_evaluator=ctf_evaluator,
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
