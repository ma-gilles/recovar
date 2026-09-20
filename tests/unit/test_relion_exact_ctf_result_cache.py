"""The assembled exact-CTF operand memo must return the same bytes, or nothing.

`_relion_exact_ctf_half_from_source_star_host` already caches each image's CTF
row; what repeats every iteration is the stack that turns those rows into one
operand. Pass 2 asks for a whole half and the coarse pass for one image batch of
planned columns, both with inputs that do not change between iterations, so the
memo returns the previously assembled array. These tests pin that the memo is
transparent: identical values to the unmemoized path, never a different image
set's CTFs, off by default, and bounded.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.em.relion import relion_ctf

pytestmark = pytest.mark.unit

IMAGE_SHAPE = (8, 8)
HALF_PIXELS = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
N_IMAGES = 12


@pytest.fixture
def prepared_cache(monkeypatch):
    """A warm per-image row cache, so no STAR file or RELION binding is needed."""

    rng = np.random.default_rng(17)
    # The per-image rows live in one 2-D block addressed by a slot table, so the
    # assembled operand is one gather rather than a Python loop; see the commit
    # that introduced `slots`/`rows`. Warm every slot so this fixture still needs
    # no STAR file or RELION binding.
    block = rng.standard_normal((N_IMAGES, HALF_PIXELS)).astype(np.float64)
    rows = {index: block[index] for index in range(N_IMAGES)}
    source = "/nonexistent/source.star"
    key = (source, tuple(int(size) for size in IMAGE_SHAPE))
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE,
        key,
        {
            "particles": None,
            "optics": {},
            "relion_bind": None,
            "slots": np.arange(N_IMAGES, dtype=np.int64),
            "rows": block,
            "n_cached": N_IMAGES,
        },
    )
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_source_star", lambda dataset: source)
    monkeypatch.setattr(
        relion_ctf,
        "original_image_indices",
        lambda dataset, indices: np.asarray(indices, dtype=np.int64),
    )
    relion_ctf.clear_exact_ctf_result_cache()
    yield rows
    relion_ctf.clear_exact_ctf_result_cache()


def _call(indices, pixel_indices=None):
    return relion_ctf._relion_exact_ctf_half_from_source_star_host(
        None,
        np.asarray(indices, dtype=np.int64),
        IMAGE_SHAPE,
        pixel_indices=pixel_indices,
    )


def test_memo_can_be_disabled_and_then_hands_back_fresh_arrays(prepared_cache, monkeypatch):
    """With the memo off, two calls agree and neither is the other's array.

    The budget defaults to 4 GB since the P4-I merge; ``0`` restores the
    unmemoized assembly, which is the oracle the memo is measured against.
    """

    monkeypatch.setenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", "0")
    first = _call(range(N_IMAGES))
    second = _call(range(N_IMAGES))
    np.testing.assert_array_equal(first, second)
    assert first is not second
    assert first.flags.writeable


def test_memo_budget_defaults_to_four_gigabytes(monkeypatch):
    """The default is the candidate configuration, not off."""

    from recovar.em.relion import relion_ctf

    monkeypatch.delenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", raising=False)
    assert relion_ctf._exact_ctf_result_cache_budget_bytes() == 4 * (1024 ** 3)
    monkeypatch.setenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", "0")
    assert relion_ctf._exact_ctf_result_cache_budget_bytes() == 0


def test_memo_returns_the_identical_bytes(prepared_cache, monkeypatch):
    monkeypatch.delenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", raising=False)
    control = _call(range(N_IMAGES))
    monkeypatch.setenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", "1")
    first = _call(range(N_IMAGES))
    second = _call(range(N_IMAGES))
    assert first is second, "a repeated request must not rebuild the operand"
    assert not first.flags.writeable, "a shared operand must not be mutable"
    np.testing.assert_array_equal(first, control)
    assert first.dtype == control.dtype and first.shape == control.shape


def test_memo_distinguishes_index_sets_and_pixel_plans(prepared_cache, monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", "1")
    pixels_a = np.array([0, 3, 7, 11], dtype=np.int32)
    pixels_b = np.array([1, 3, 7, 11], dtype=np.int32)
    whole = _call(range(N_IMAGES))
    first_half = _call(range(N_IMAGES // 2))
    gathered_a = _call(range(N_IMAGES), pixel_indices=pixels_a)
    gathered_b = _call(range(N_IMAGES), pixel_indices=pixels_b)

    assert whole.shape == (N_IMAGES, HALF_PIXELS)
    assert first_half.shape == (N_IMAGES // 2, HALF_PIXELS)
    assert gathered_a.shape == (N_IMAGES, pixels_a.size)
    np.testing.assert_array_equal(first_half, whole[: N_IMAGES // 2])
    np.testing.assert_array_equal(gathered_a, whole[:, pixels_a])
    np.testing.assert_array_equal(gathered_b, whole[:, pixels_b])
    assert gathered_a is not gathered_b

    # A permuted index set is a different operand, not a cache hit.
    permuted = _call(list(reversed(range(N_IMAGES))))
    np.testing.assert_array_equal(permuted, whole[::-1])
    assert permuted is not whole


def test_memo_repeats_every_shape_and_stays_bitwise(prepared_cache, monkeypatch):
    monkeypatch.delenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", raising=False)
    plans = [
        (range(N_IMAGES), None),
        (range(3, 9), None),
        (range(N_IMAGES), np.array([2, 4, 6], dtype=np.int32)),
        (range(0, N_IMAGES, 2), np.array([0, 1], dtype=np.int64)),
    ]
    controls = [_call(indices, pixels) for indices, pixels in plans]
    monkeypatch.setenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", "1")
    for (indices, pixels), control in zip(plans, controls):
        for _ in range(2):
            candidate = _call(indices, pixels)
            assert candidate.dtype == control.dtype
            assert candidate.shape == control.shape
            np.testing.assert_array_equal(candidate, control)


def test_memo_respects_its_budget(prepared_cache, monkeypatch):
    # One whole-half operand is N_IMAGES * HALF_PIXELS * 8 bytes; a budget of
    # one such operand must hold one entry and evict the older one.
    one_operand = N_IMAGES * HALF_PIXELS * 8
    monkeypatch.setenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", str(one_operand / (1024 ** 3)))
    whole = _call(range(N_IMAGES))
    assert _call(range(N_IMAGES)) is whole
    _call(list(reversed(range(N_IMAGES))))
    assert relion_ctf._EXACT_CTF_RESULT_BYTES <= one_operand
    # The evicted entry is rebuilt with the identical bytes.
    rebuilt = _call(range(N_IMAGES))
    np.testing.assert_array_equal(rebuilt, whole)


def test_memo_rejects_an_unparsable_budget(prepared_cache, monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", "lots")
    with pytest.raises(ValueError, match="RECOVAR_RELION_EXACT_CTF_CACHE_GB"):
        _call(range(N_IMAGES))
    monkeypatch.setenv("RECOVAR_RELION_EXACT_CTF_CACHE_GB", "-1")
    with pytest.raises(ValueError, match="RECOVAR_RELION_EXACT_CTF_CACHE_GB"):
        _call(range(N_IMAGES))


def test_prefetch_depth_defaults_to_four_and_validates(monkeypatch):
    """The loader's queue depth is selectable and fails closed on a bad token.

    The default is 4 since the P4-I merge; 2 restores the previous behaviour.
    """

    from recovar.data_io import image_backends

    monkeypatch.delenv(image_backends.PREFETCH_DEPTH_ENV, raising=False)
    assert image_backends.prefetch_depth() == image_backends.DEFAULT_PREFETCH_DEPTH == 4
    monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, "2")
    assert image_backends.prefetch_depth() == 2
    monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, "6")
    assert image_backends.prefetch_depth() == 6
    assert image_backends._PrefetchIterator(iter(()))._buffer_size == 6
    monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, "0")
    with pytest.raises(ValueError, match=image_backends.PREFETCH_DEPTH_ENV):
        image_backends.prefetch_depth()
    monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, "deep")
    with pytest.raises(ValueError, match=image_backends.PREFETCH_DEPTH_ENV):
        image_backends.prefetch_depth()


def test_prefetch_yields_the_same_items_at_every_depth(monkeypatch):
    """Depth changes the buffering, never the sequence."""

    from recovar.data_io import image_backends

    payload = [("batch", index) for index in range(17)]
    monkeypatch.delenv(image_backends.PREFETCH_DEPTH_ENV, raising=False)
    control = list(image_backends._PrefetchIterator(iter(payload)))
    for depth in ("1", "2", "4", "8"):
        monkeypatch.setenv(image_backends.PREFETCH_DEPTH_ENV, depth)
        assert list(image_backends._PrefetchIterator(iter(payload))) == control == payload
