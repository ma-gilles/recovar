"""The cached RELION CTF batch gather must not depend on how it is cached.

`_relion_exact_ctf_half_from_source_star_host` used to hold one NumPy row per
particle in a dict and rebuild each batch with a Python loop, one small gather
per image. On a K=1 100k/256 run that loop was 38.5 s of self time, 9.4% of the
whole run, so the rows moved into a single block gathered in one indexing
operation. These tests pin the observable behaviour that change has to preserve:
row identity, order, duplicates, and the selected pixel columns.

The cache is pre-populated so the tests never reach the RELION binding, which is
built per run root rather than into the checkout.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.relion import relion_ctf

PIXELS = 12  # a 4x4 image half-spectrum is 4 * (4 // 2 + 1)
PARTICLES = 5


@pytest.fixture
def populated_cache(monkeypatch, tmp_path):
    star = tmp_path / "particles.star"
    star.write_text("")
    monkeypatch.setenv("RECOVAR_K1_RELION_EXACT_CTF_STAR", str(star))

    rows = np.arange(PARTICLES * PIXELS, dtype=np.float64).reshape(PARTICLES, PIXELS)
    key = (str(star.resolve()), (4, 4))
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE,
        key,
        {
            "particles": None,
            "optics": {},
            "relion_bind": None,
            "slots": np.arange(PARTICLES, dtype=np.int64),
            "rows": rows,
            "n_cached": PARTICLES,
        },
    )
    return SimpleNamespace(particles_file=str(star)), rows


@pytest.mark.unit
def test_cached_ctf_batch_preserves_order_and_duplicates(populated_cache):
    dataset, rows = populated_cache
    indices = np.asarray([3, 0, 3, 1], dtype=np.int64)

    out = relion_ctf._relion_exact_ctf_half_from_source_star_host(
        dataset, indices, (4, 4),
    )

    assert out.dtype == np.float64
    assert out.shape == (indices.size, PIXELS)
    np.testing.assert_array_equal(out, rows[indices])


@pytest.mark.unit
def test_cached_ctf_batch_selects_requested_pixels(populated_cache):
    dataset, rows = populated_cache
    indices = np.asarray([2, 2, 4], dtype=np.int64)
    # Unsorted, with a repeat: column order and duplication must be preserved.
    pixel_indices = np.asarray([7, 0, 7, 11], dtype=np.int64)

    out = relion_ctf._relion_exact_ctf_half_from_source_star_host(
        dataset, indices, (4, 4), pixel_indices=pixel_indices,
    )

    assert out.shape == (indices.size, pixel_indices.size)
    np.testing.assert_array_equal(out, rows[indices][:, pixel_indices])


@pytest.mark.unit
def test_cached_ctf_batch_fails_closed_on_an_unevaluated_row(populated_cache, monkeypatch):
    """A slot that was never filled must raise, not return another particle's CTF."""

    dataset, _ = populated_cache
    key = next(iter(relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE))
    cache = relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE[key]
    # Leave the slot unset and make evaluating it a no-op, which is what a
    # silently-skipped particle would look like.
    cache["slots"][1] = -1
    monkeypatch.setattr(
        relion_ctf, "original_image_indices", lambda dataset, indices: np.asarray(indices),
    )

    with pytest.raises(Exception):
        relion_ctf._relion_exact_ctf_half_from_source_star_host(
            dataset, np.asarray([1], dtype=np.int64), (4, 4),
        )
