"""Shared source-precision RELION CTF operands for coarse, local and sparse EM.

The source STAR and native CTF cache retain their original per-process lifetime.
Host callers can gather planned pixels before stacking; the device accessor
places the full operand once. Native RFLOAT values, centered half-spectrum
coordinates, signs and caller-selected cast boundaries are preserved.
"""

from __future__ import annotations

import collections
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.batch_fetch import original_image_indices

_RELION_EXACT_CTF_SOURCE_CACHE: dict[tuple[str, tuple[int, int]], dict] = {}

# Assembled-operand memo. The per-image rows below are already cached, but the
# stack that turns them into one operand is rebuilt on every call: pass 2 asks
# for a whole half (4966 rows of 33024 float64 = 1.31 GB of host memcpy, 250 ms
# per half per iteration), and the coarse pass asks for one image batch of
# planned columns. Both requests repeat with identical inputs every iteration,
# so the assembled array is a pure function of the key and the memo returns the
# very same object: bit-for-bit by construction, not by re-derivation.
_EXACT_CTF_RESULT_CACHE: "collections.OrderedDict[tuple, tuple]" = collections.OrderedDict()
_EXACT_CTF_RESULT_BYTES = 0
_EXACT_CTF_CACHE_GB_ENV = "RECOVAR_RELION_EXACT_CTF_CACHE_GB"


def _exact_ctf_result_cache_budget_bytes() -> int:
    """Host bytes the assembled-operand memo may hold; 0 disables it.

    The default is 4 GB since the P4-I merge, which holds the two 1.31 GB
    whole-half operands of the 256x256 fixture and costs about 2.5 GB of host
    memory for 2.0 s of a steady hp3 iteration. Set the variable to 0 to
    restore the unmemoized assembly, which stays the oracle.

    The memo trades host memory for the repeated stack: a whole-half operand is
    1.31 GB at 256x256, so a budget below that disables the pass-2 entry while
    still holding the much smaller per-batch coarse entries.
    """

    token = os.environ.get(_EXACT_CTF_CACHE_GB_ENV, "4").strip()
    try:
        budget = float(token)
    except ValueError as exc:
        raise ValueError(f"{_EXACT_CTF_CACHE_GB_ENV} must be a number of gigabytes, got {token!r}") from exc
    if budget < 0:
        raise ValueError(f"{_EXACT_CTF_CACHE_GB_ENV} must not be negative, got {token!r}")
    return int(budget * (1024 ** 3))


def _exact_ctf_result_key(source_path, image_shape, original_indices, pixel_indices):
    """A content key for the assembled operand, with the arrays kept for verification."""

    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(original_indices).view(np.uint8))
    if pixel_indices is None:
        pixel_tag = b"none"
    else:
        pixel_tag = np.ascontiguousarray(pixel_indices).view(np.uint8).tobytes()
        digest.update(b"|pixels|")
        digest.update(pixel_tag)
    return (
        str(source_path),
        tuple(int(size) for size in image_shape),
        int(original_indices.size),
        None if pixel_indices is None else (str(pixel_indices.dtype), int(pixel_indices.size)),
        digest.digest(),
    )


def _exact_ctf_result_lookup(key, original_indices, pixel_indices):
    """Return the memoized operand for an exactly equal request, else None."""

    entry = _EXACT_CTF_RESULT_CACHE.get(key)
    if entry is None:
        return None
    cached_indices, cached_pixels, result = entry
    # A digest match is not trusted on its own: compare the actual selectors, so
    # a collision can never serve one image set's CTFs for another's.
    if not np.array_equal(cached_indices, original_indices):
        return None
    if (cached_pixels is None) != (pixel_indices is None):
        return None
    if cached_pixels is not None and not np.array_equal(cached_pixels, pixel_indices):
        return None
    _EXACT_CTF_RESULT_CACHE.move_to_end(key)
    return result


def _exact_ctf_result_store(key, original_indices, pixel_indices, result):
    """Memoize one assembled operand, evicting least-recently-used entries."""

    global _EXACT_CTF_RESULT_BYTES

    budget = _exact_ctf_result_cache_budget_bytes()
    if budget <= 0 or result.nbytes > budget:
        return result
    # The memo hands the same object to every caller, so make it read-only: no
    # caller mutates it today (each wraps it in np.asarray and uploads), and a
    # future one must not silently corrupt a shared operand.
    result.setflags(write=False)
    _EXACT_CTF_RESULT_CACHE[key] = (
        np.array(original_indices, copy=True),
        None if pixel_indices is None else np.array(pixel_indices, copy=True),
        result,
    )
    _EXACT_CTF_RESULT_BYTES += result.nbytes
    while _EXACT_CTF_RESULT_BYTES > budget and len(_EXACT_CTF_RESULT_CACHE) > 1:
        _, evicted = _EXACT_CTF_RESULT_CACHE.popitem(last=False)
        _EXACT_CTF_RESULT_BYTES -= evicted[2].nbytes
    return result


def clear_exact_ctf_result_cache() -> None:
    """Drop the assembled-operand memo (tests and long-lived processes)."""

    global _EXACT_CTF_RESULT_BYTES

    _EXACT_CTF_RESULT_CACHE.clear()
    _EXACT_CTF_RESULT_BYTES = 0


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
    cache_key = _exact_ctf_result_key(source_path, image_shape, original_indices, pixel_indices)
    memoized = _exact_ctf_result_lookup(cache_key, original_indices, pixel_indices)
    if memoized is not None:
        return memoized

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
    assembled = np.asarray(np.stack(ctf_rows, axis=0), dtype=np.float64)
    return _exact_ctf_result_store(cache_key, original_indices, pixel_indices, assembled)


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
