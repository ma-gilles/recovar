"""Shared source-precision RELION CTF operands for coarse, local and sparse EM.

The source STAR and native CTF cache retain their original per-process lifetime.
Host callers can gather planned pixels before stacking; the device accessor
places the full operand once. Native RFLOAT values, centered half-spectrum
coordinates, signs and caller-selected cast boundaries are preserved.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.batch_fetch import original_image_indices

_RELION_EXACT_CTF_SOURCE_CACHE: dict[tuple[str, tuple[int, int]], dict] = {}


def _star_column(table, name: str):
    """Return a RELION STAR column with or without its legacy underscore."""

    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return table[candidate]
    raise ValueError(f"RELION source STAR column {name} is missing")


def _relion_exact_ctf_source_star(experiment_dataset) -> Path:
    """Resolve the immutable source STAR for exact RELION CTF evaluation."""

    source_star = os.environ.get("RECOVAR_K1_RELION_EXACT_CTF_STAR", "").strip()
    if not source_star:
        dataset_source = getattr(experiment_dataset, "particles_file", None)
        if dataset_source and Path(dataset_source).suffix.lower() == ".star":
            source_star = str(dataset_source)
    if not source_star:
        raise ValueError(
            "exact RELION operands require a STAR-backed dataset or "
            "RECOVAR_K1_RELION_EXACT_CTF_STAR"
        )
    return Path(source_star).expanduser().resolve()


def _relion_exact_ctf_half_from_source_star_host(
    experiment_dataset,
    image_indices,
    image_shape,
    *,
    pixel_indices=None,
):
    """Evaluate source-precision SPA CTFs into one host-native operand.

    The result uses RECOVAR's centered-y half-spectrum coordinates and sign.
    The source STAR is mandatory because the ordinary dataset metadata has
    already been rounded to float32 before pass 2.  RELION's binding and the
    source cache are host-native; callers that must pad on the image axis use
    this helper so they place the final operand exactly once. Optional host
    pixel indices gather the requested columns before stacking full CTF rows;
    their order and duplicates are preserved without changing source precision.
    """

    source_path = _relion_exact_ctf_source_star(experiment_dataset)
    cache_key = (str(source_path), tuple(int(size) for size in image_shape))
    cache = _RELION_EXACT_CTF_SOURCE_CACHE.get(cache_key)
    if cache is None:
        from recovar.data_io.starfile import read_star
        from recovar.relion_bind import _relion_bind_core as relion_bind

        particles, optics = read_star(str(source_path))
        if optics is None:
            raise ValueError(f"RELION source STAR has no optics table: {source_path}")
        optics_ids = np.asarray(_star_column(optics, "rlnOpticsGroup"), dtype=np.int64)
        if np.unique(optics_ids).size != optics_ids.size:
            raise ValueError(f"RELION source STAR has duplicate optics groups: {source_path}")
        cache = {
            "particles": particles,
            "optics": {
                int(group): optics.iloc[row]
                for row, group in enumerate(optics_ids)
            },
            "relion_bind": relion_bind,
            "images": {},
        }
        _RELION_EXACT_CTF_SOURCE_CACHE[cache_key] = cache

    original_indices = original_image_indices(
        experiment_dataset,
        np.asarray(image_indices, dtype=np.int64),
    )
    image_h, image_w = (int(size) for size in image_shape)
    if image_h != image_w:
        raise ValueError("exact RELION CTF replay currently requires square images")
    if pixel_indices is not None:
        if not isinstance(pixel_indices, np.ndarray):
            raise TypeError("CTF pixel indices must already be a host NumPy array")
        if pixel_indices.ndim != 1 or pixel_indices.dtype.kind not in "iu":
            raise ValueError("CTF pixel indices must be a one-dimensional integer array")
        if np.any(pixel_indices < 0) or np.any(pixel_indices >= image_h * (image_w // 2 + 1)):
            raise ValueError("CTF pixel indices are outside the full half-spectrum")
    ctf_rows = []
    for original_index in original_indices:
        original_index = int(original_index)
        cached_image = cache["images"].get(original_index)
        if cached_image is None:
            particle = cache["particles"].iloc[original_index]
            optics_group = int(
                particle["rlnOpticsGroup"]
                if "rlnOpticsGroup" in particle
                else particle["_rlnOpticsGroup"]
            )
            optics = cache["optics"][optics_group]

            def particle_value(name: str) -> float:
                return float(
                    particle[name] if name in particle else particle[f"_{name}"]
                )

            def optics_value(name: str) -> float:
                return float(optics[name] if name in optics else optics[f"_{name}"])

            native = np.asarray(
                cache["relion_bind"].get_ctf_image(
                    particle_value("rlnDefocusU"),
                    particle_value("rlnDefocusV"),
                    particle_value("rlnDefocusAngle"),
                    optics_value("rlnVoltage"),
                    optics_value("rlnSphericalAberration"),
                    optics_value("rlnAmplitudeContrast"),
                    0.0,
                    optics_value("rlnImagePixelSize"),
                    image_w,
                    image_h,
                    False,
                    False,
                    False,
                    particle_value("rlnPhaseShift"),
                    1.0,
                ),
                dtype=np.float64,
            )
            # RELION/FFTW stores y in standard order and uses the opposite CTF
            # sign from RECOVAR's forward-model convention.
            cached_image = (-np.fft.fftshift(native, axes=0)).reshape(-1)
            cache["images"][original_index] = cached_image
        ctf_rows.append(cached_image if pixel_indices is None else cached_image[pixel_indices])
    return np.asarray(np.stack(ctf_rows, axis=0), dtype=np.float64)


def _relion_exact_ctf_half_from_source_star(
    experiment_dataset,
    image_indices,
    image_shape,
):
    """Return the shared source-precision CTF operand on the JAX device.

    Device-first EM callers reuse this single binary64 placement for their
    float32 score and reconstruction operands.  Host-padding callers should
    use :func:`_relion_exact_ctf_half_from_source_star_host` to avoid a
    device-to-host-to-device round trip.
    """

    return jnp.asarray(
        _relion_exact_ctf_half_from_source_star_host(
            experiment_dataset,
            image_indices,
            image_shape,
        ),
        dtype=jnp.float64,
    )
