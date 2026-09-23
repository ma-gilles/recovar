"""Image-chunked RELION Wavg triplet stage of the sparse bucketed pass 2.

``_relion_wavg_add_triplet_pixels_chunked`` must give every image the same
float32 Wavg ``[XA, AA, diff2]`` operands as translating the whole bucket's
rectangle at once (the composition the bucket loop used before), while the
complex64 ``[images, translations, pixels]`` rectangle only exists for one
image chunk at a time. Before, the pipelined tail could hold that rectangle
and its exact gather for up to three buckets (the K1 100k/256 completion run,
job 14282511, failed allocating one: [136, 116, 9303] complex64).
"""

import json
import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.helpers.fourier_window import make_fourier_window_indices_np
from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed
from recovar.em.sparse_pass2.sparse_pass2_wavg import (
    _make_relion_wavg_rectangle,
    _relion_cuda_translate_wavg_norm_images,
    _relion_wavg_add_triplet_pixels_chunked,
    _relion_wavg_atomic_triplet_terms,
    _relion_wavg_chunk_bytes_per_image,
    _relion_wavg_image_chunk_ranges,
    _relion_wavg_rectangle_triplet_terms,
    _relion_wavg_sequential_triplet_terms,
)

pytestmark = pytest.mark.unit

GIB = 1 << 30


# ---------------------------------------------------------------------------
# Chunk planning and the per-bucket bound (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch", "bytes_per_image", "budget"),
    [(1, 10, 100), (13, 10, 35), (13, 10, 10), (13, 10, 9), (136, 7, 1 << 20), (7, 1 << 40, 1)],
)
def test_image_chunk_ranges_cover_the_batch_within_budget(batch, bytes_per_image, budget):
    ranges = _relion_wavg_image_chunk_ranges(batch, bytes_per_image, budget)

    assert ranges[0][0] == 0 and ranges[-1][1] == batch
    assert all(stop == next_start for (_, stop), (next_start, _) in zip(ranges, ranges[1:]))
    sizes = [stop - start for start, stop in ranges]
    assert min(sizes) >= 1
    for size in sizes:
        assert size == 1 or size * bytes_per_image <= budget
    # Chunks are as large as the budget allows, so only the last is short.
    assert all(size == sizes[0] for size in sizes[:-1])
    assert sizes[0] == min(batch, max(1, budget // bytes_per_image))


@pytest.mark.parametrize(("bucket_images", "rotations"), [(136, 16), (93, 32), (126, 16)])
def test_failing_completion_bucket_wavg_chunks_fit_the_noise_block_budget(bucket_images, rotations):
    """The 14282511 shapes: current_size 154, 116 translations, 12012/9303 pixels."""

    translations, rectangle_pixels, exact_pixels = 116, 12012, 9303
    per_image = _relion_wavg_chunk_bytes_per_image(translations, rotations, rectangle_pixels, exact_pixels)
    # The translated rectangle alone was the failing allocation's parent.
    assert per_image >= translations * rectangle_pixels * 8
    # Unchunked, one bucket's Wavg stage is several GiB ...
    assert bucket_images * per_image > 3 * GIB
    # ... chunked it stays inside the 1 GiB noise-block budget of that run.
    for start, stop in _relion_wavg_image_chunk_ranges(bucket_images, per_image, GIB):
        assert (stop - start) * per_image <= GIB


def test_bucket_loop_snapshots_only_the_image_for_the_wavg_stage():
    names = set(bucketed._BUCKET_TAIL_SNAPSHOT_NAMES)

    assert "processed_score_half_for_noise" in names
    assert "relion_wavg_rectangle" in names
    assert "raw_translated_wavg_rectangle" not in names
    assert "raw_translated_wavg_for_atomic" not in names


# ---------------------------------------------------------------------------
# Chunked vs whole-bucket Wavg operands (GPU, RECOVAR CUDA library)
# ---------------------------------------------------------------------------


def _wavg_case(*, batch, translations, rotations, image_size, current_size, seed):
    image_shape = (image_size, image_size)
    exact_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        include_dc=True,
        exact_radius=True,
    )
    rectangle = _make_relion_wavg_rectangle(image_shape, current_size, exact_indices)
    exact = int(rectangle.exact_positions.size)
    half_pixels = image_size * (image_size // 2 + 1)
    rng = np.random.default_rng(seed)

    def complex_normal(*shape):
        return jnp.asarray(
            (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64),
        )

    posterior = rng.random((batch, rotations, translations)).astype(np.float32)
    posterior[rng.random(posterior.shape) < 0.3] = 0.0
    posterior /= np.maximum(posterior.sum(axis=(1, 2), keepdims=True), np.float32(1e-6))
    ctf_posterior = rng.random((batch, rotations, exact)).astype(np.float32)
    ctf_posterior[rng.random(ctf_posterior.shape) < 0.2] = 0.0
    return dict(
        image_shape=image_shape,
        rectangle=rectangle,
        processed_score_half=complex_normal(batch, half_pixels),
        translation_angles=jnp.asarray(rng.uniform(-0.2, 0.2, (translations, 2)).astype(np.float32)),
        proj=complex_normal(batch, rotations, exact),
        proj_abs2=jnp.asarray(rng.random((batch, rotations, exact)).astype(np.float32)),
        summed_shifted=complex_normal(batch, rotations, exact),
        ctf_posterior=jnp.asarray(ctf_posterior),
        noise_variance=jnp.asarray(rng.uniform(0.5, 2.0, exact).astype(np.float32)),
        scale=jnp.asarray(rng.uniform(0.5, 1.5, batch).astype(np.float32)),
        raw_ctf=jnp.asarray(rng.uniform(-1.0, 1.0, (batch, exact)).astype(np.float32)),
        posterior=jnp.asarray(posterior),
    )


def _whole_bucket_rectangle_terms(case, *, sequential):
    """The pre-chunking bucket-loop composition, over the whole bucket."""

    rectangle = case["rectangle"]
    raw_rectangle = _relion_cuda_translate_wavg_norm_images(
        case["processed_score_half"],
        case["translation_angles"],
        rectangle.centered_indices,
        case["image_shape"],
    )
    raw_exact = raw_rectangle[:, :, rectangle.exact_positions]
    if sequential:
        exact_terms = _relion_wavg_sequential_triplet_terms(
            case["proj"],
            case["raw_ctf"],
            case["scale"],
            raw_exact,
            case["posterior"],
        )
    else:
        exact_terms = _relion_wavg_atomic_triplet_terms(
            case["proj"],
            case["proj_abs2"],
            case["summed_shifted"],
            case["ctf_posterior"],
            case["noise_variance"],
            case["scale"],
            raw_exact,
            case["posterior"],
        )
    return _relion_wavg_rectangle_triplet_terms(
        exact_terms,
        raw_rectangle,
        case["posterior"],
        rectangle.exact_positions,
    )


def _chunked(case, accumulator, *, sequential, max_block_bytes):
    return _relion_wavg_add_triplet_pixels_chunked(
        accumulator,
        processed_score_half=case["processed_score_half"],
        translation_angles=case["translation_angles"],
        rectangle=case["rectangle"],
        image_shape=case["image_shape"],
        proj=case["proj"],
        proj_abs2=case["proj_abs2"],
        summed_shifted=case["summed_shifted"],
        ctf_posterior=case["ctf_posterior"],
        noise_variance=case["noise_variance"],
        scale=case["scale"],
        raw_ctf=case["raw_ctf"] if sequential else None,
        posterior=case["posterior"],
        max_block_bytes=max_block_bytes,
    )


def _per_image_bytes(case):
    batch, rotations, translations = case["posterior"].shape
    return _relion_wavg_chunk_bytes_per_image(
        translations,
        rotations,
        case["rectangle"].centered_indices.size,
        case["rectangle"].exact_positions.size,
    )


def _bits(values):
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)


@pytest.mark.gpu
@pytest.mark.parametrize("sequential", [True, False], ids=["sequential", "algebraic"])
@pytest.mark.parametrize(
    ("shape", "images_per_chunk"),
    [
        (dict(batch=13, translations=5, rotations=3, image_size=64, current_size=24), 1),
        (dict(batch=13, translations=5, rotations=3, image_size=64, current_size=24), 4),
        (dict(batch=13, translations=5, rotations=3, image_size=64, current_size=24), 13),
        # 116 translations and a 64-pixel window: the regime in which a batched
        # GEMM over translations changes its result with the batch count.
        (dict(batch=17, translations=116, rotations=16, image_size=128, current_size=64), 1),
        (dict(batch=17, translations=116, rotations=16, image_size=128, current_size=64), 5),
    ],
)
def test_chunked_wavg_operands_are_bitwise_the_whole_bucket_operands(monkeypatch, sequential, shape, images_per_chunk):
    from recovar import cuda_backproject

    case = _wavg_case(seed=images_per_chunk, **shape)
    expected = np.asarray(_whole_bucket_rectangle_terms(case, sequential=sequential))

    issued = []

    def capture_atomic_add(terms, accumulator):
        issued.append(np.asarray(terms))
        return accumulator + jnp.sum(terms, axis=1)

    monkeypatch.setattr(cuda_backproject, "relion_wavg_rotation_atomic_triplet_add_f32", capture_atomic_add)
    batch = shape["batch"]
    accumulator = jnp.zeros((batch, case["rectangle"].centered_indices.size, 3), dtype=jnp.float32)
    _chunked(
        case,
        accumulator,
        sequential=sequential,
        max_block_bytes=images_per_chunk * _per_image_bytes(case),
    )

    assert [terms.shape[0] for terms in issued] == [
        stop - start
        for start, stop in _relion_wavg_image_chunk_ranges(
            batch, _per_image_bytes(case), images_per_chunk * _per_image_bytes(case)
        )
    ]
    assert len(issued) == -(-batch // images_per_chunk)
    actual = np.concatenate(issued, axis=0)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.mark.gpu
@pytest.mark.parametrize("sequential", [True, False], ids=["sequential", "algebraic"])
def test_chunked_wavg_atomic_accumulator_is_bitwise_per_image(sequential):
    """One rotation per image: each accumulator cell receives one real atomic add."""

    from recovar import cuda_backproject

    case = _wavg_case(batch=9, translations=7, rotations=1, image_size=64, current_size=24, seed=3)
    rng = np.random.default_rng(4)
    accumulator = jnp.asarray(
        rng.standard_normal((9, case["rectangle"].centered_indices.size, 3)).astype(np.float32),
    )
    expected = cuda_backproject.relion_wavg_rotation_atomic_triplet_add_f32(
        _whole_bucket_rectangle_terms(case, sequential=sequential),
        accumulator,
    )
    actual = _chunked(case, accumulator, sequential=sequential, max_block_bytes=2 * _per_image_bytes(case))

    np.testing.assert_array_equal(_bits(actual), _bits(expected))


_PEAK_PROBE = textwrap.dedent(
    """
    import json, sys
    import jax, jax.numpy as jnp
    import numpy as np
    sys.path.insert(0, {tests_unit!r})
    from test_wavg_chunked_triplet_pixels import _chunked, _per_image_bytes, _wavg_case, _whole_bucket_rectangle_terms
    from recovar import cuda_backproject

    mode, images_per_chunk = sys.argv[1], int(sys.argv[2])
    case = _wavg_case(batch=48, translations=116, rotations=8, image_size=256, current_size=154, seed=0)
    accumulator = jnp.zeros((48, case["rectangle"].centered_indices.size, 3), dtype=jnp.float32)
    jax.block_until_ready([value for value in case.values() if isinstance(value, jax.Array)] + [accumulator])
    device = jax.devices()[0]
    before = device.memory_stats()
    if mode == "chunked":
        out = _chunked(case, accumulator, sequential=True, max_block_bytes=images_per_chunk * _per_image_bytes(case))
    else:
        out = cuda_backproject.relion_wavg_rotation_atomic_triplet_add_f32(
            _whole_bucket_rectangle_terms(case, sequential=True), accumulator
        )
    jax.block_until_ready(out)
    after = device.memory_stats()
    print(json.dumps(dict(
        before_in_use=int(before["bytes_in_use"]),
        before_peak=int(before["peak_bytes_in_use"]),
        peak=int(after["peak_bytes_in_use"]),
        per_image=int(_per_image_bytes(case)),
        rectangle_pixels=int(case["rectangle"].centered_indices.size),
        exact_pixels=int(case["rectangle"].exact_positions.size),
    )))
    """
)


def _probe_peak(mode, images_per_chunk=0):
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    script = _PEAK_PROBE.format(tests_unit=os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-c", script, mode, str(images_per_chunk)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    # The probe must have set the process peak itself, or the delta measures nothing.
    assert stats["peak"] > stats["before_peak"], stats
    stats["delta"] = stats["peak"] - stats["before_in_use"]
    return stats


@pytest.mark.gpu
def test_chunked_wavg_device_peak_is_bounded_by_the_chunk_budget():
    """Measured allocator peak of one bucket's Wavg stage, current_size 154 geometry.

    48 images x 116 translations x 12012 rectangle pixels: the whole-bucket
    complex64 rectangle is 0.54 GB and its exact gather 0.41 GB. Chunked, the
    rectangle exists per chunk only; the whole-bucket operands are its float32
    power and the exact gather.
    """

    batch, translations, rotations = 48, 116, 8
    images_per_chunk = 2
    chunked = _probe_peak("chunked", images_per_chunk)
    whole = _probe_peak("whole")
    rectangle_pixels, exact_pixels = chunked["rectangle_pixels"], chunked["exact_pixels"]
    assert rectangle_pixels == 154 * 78

    assert chunked["per_image"] == _relion_wavg_chunk_bytes_per_image(
        translations, rotations, rectangle_pixels, exact_pixels
    )
    budget = images_per_chunk * chunked["per_image"]
    # Whole-bucket residents: float32 rectangle power, complex64 exact gather
    # and the copy of it the sequential reducer's loop carries, that reducer's
    # working set (reference re/im, three accumulators, loop temporaries,
    # stacked terms), the rectangle image power, the result accumulator and
    # the assembled output.
    residents = (
        batch * translations * rectangle_pixels * 4
        + 2 * batch * translations * exact_pixels * 8
        + batch * rotations * exact_pixels * 4 * 11
        + batch * rotations * rectangle_pixels * 4
        + 2 * batch * rectangle_pixels * 3 * 4
    )
    slack = 64 << 20
    assert chunked["delta"] <= residents + budget + slack, (chunked, residents, budget)
    # The whole-bucket composition, which also holds the complex64 rectangle,
    # does not fit that bound.
    assert whole["delta"] >= batch * translations * (rectangle_pixels + exact_pixels) * 8, whole
    assert whole["delta"] > residents + budget + slack, (whole, residents, budget)
