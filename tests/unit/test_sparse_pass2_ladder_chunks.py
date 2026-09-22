"""``RECOVAR_SPARSE_PASS2_LADDER_CHUNKS``: power-of-two bucket chunking so K-class bucket shapes repeat."""

import numpy as np
import pytest

from recovar.em.scoring import sparse_bucket_arrays as sba


def test_default_chunking_is_consecutive_caps(monkeypatch):
    monkeypatch.delenv(sba.LADDER_CHUNKS_ENV, raising=False)
    assert sba.bucket_chunk_bounds(639, 256) == [(0, 256), (256, 512), (512, 639)]
    assert sba.bucket_chunk_bounds(5, 256) == [(0, 5)]
    assert sba.bucket_chunk_bounds(0, 256) == []


@pytest.mark.parametrize("n_images", [0, 1, 15, 16, 17, 31, 32, 100, 443, 639, 2048, 2049])
def test_ladder_chunks_cover_images_once_with_repeating_sizes(n_images):
    bounds = sba.bucket_chunk_bounds(n_images, 2048, ladder=True)
    covered = [i for start, stop in bounds for i in range(start, stop)]
    assert covered == list(range(n_images))
    sizes = [stop - start for start, stop in bounds]
    for size in sizes[:-1]:
        assert size >= sba.LADDER_CHUNK_FLOOR and size & (size - 1) == 0
    if sizes:
        last = sizes[-1]
        assert last < sba.LADDER_CHUNK_FLOOR or (last & (last - 1) == 0)


def test_ladder_respects_cap_and_env(monkeypatch):
    assert sba.bucket_chunk_bounds(639, 100, ladder=True) == [(0, 64), (64, 128), (128, 192), (192, 256), (256, 320), (320, 384), (384, 448), (448, 512), (512, 576), (576, 640 - 1)] or all(
        stop - start <= 64 for start, stop in sba.bucket_chunk_bounds(639, 100, ladder=True)
    )
    monkeypatch.setenv(sba.LADDER_CHUNKS_ENV, "1")
    assert sba.ladder_chunks_enabled() is True
    assert sba.bucket_chunk_bounds(639, 2048) == [(0, 512), (512, 576), (576, 608), (608, 624), (624, 639)]
    monkeypatch.setenv(sba.LADDER_CHUNKS_ENV, "2")
    with pytest.raises(ValueError):
        sba.ladder_chunks_enabled()


def test_k_class_planner_uses_ladder(monkeypatch):
    monkeypatch.setenv(sba.LADDER_CHUNKS_ENV, "1")
    rng = np.random.default_rng(0)
    n = 300
    per_image = [{"oversampled_rots": [np.zeros((int(c), 3, 3), dtype=np.float32) for c in rng.integers(1, 40, n)]} for _ in range(2)]
    buckets = sba._bucket_sparse_k_class_pass2_inputs(per_image, 4)
    seen = np.concatenate([b["image_indices"] for b in buckets])
    assert sorted(seen.tolist()) == list(range(n))
    for b in buckets:
        size = int(b["image_indices"].shape[0])
        assert size < sba.LADDER_CHUNK_FLOOR or size & (size - 1) == 0


@pytest.mark.parametrize("cap", [1, 2, 7, 15, 16, 100, 2048])
@pytest.mark.parametrize("n_images", [1, 15, 16, 17, 31, 100, 443, 639, 2049])
def test_ladder_never_exceeds_the_callers_cap(n_images, cap):
    """The cap comes from gather / prepare / dense-M-step byte budgets, so exceeding it overshoots memory."""
    bounds = sba.bucket_chunk_bounds(n_images, cap, ladder=True)
    covered = [i for start, stop in bounds for i in range(start, stop)]
    assert covered == list(range(n_images))
    assert all(stop - start <= cap for start, stop in bounds)
