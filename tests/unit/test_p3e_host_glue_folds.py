"""P3-E: the merged head's remaining EM-only host glue, folded into programs.

Ticket: ``em_parity_tickets_20260918/P3E_merged_head_census_and_glue.md``.

Every fold here is pure data movement or host arithmetic, so every assertion
is bitwise against the expression it replaces rather than against a tolerance.
The eager dispatch counts come from the same
``EvalTrace.process_primitive`` hook T19's census uses, so the "no eager
dispatch" claims are the claims the census measures, on CPU.

Covered here:

* two per-chunk host scalars reach the device without an eager
  ``convert_element_type``, and the M-step carry's real-part dtype is NumPy
  promotion rather than a 0-d device allocation read for its dtype;
* the four zero accumulators of the M-step carry are one program per
  capacity class;
* the chunk operand permutation, reshape and padded-slot mask are one
  program;
* the per-image-batch window gather and the per-half concatenate-and-reorder
  are one program each.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.sparse_pass2 import resident_operands as ro
from recovar.em.sparse_pass2 import resident_pass2 as rp

pytestmark = pytest.mark.unit


class _DispatchCounter:
    """Count eager JAX primitive dispatches, the way the T19 census does."""

    def __init__(self):
        self.count = 0
        self.by_primitive: dict[str, int] = {}

    def __enter__(self):
        from jax._src import core

        self._core = core
        self._original = core.EvalTrace.process_primitive
        counter = self

        def process_primitive(trace, primitive, args, params):
            counter.count += 1
            counter.by_primitive[primitive.name] = (
                counter.by_primitive.get(primitive.name, 0) + 1
            )
            return counter._original(trace, primitive, args, params)

        core.EvalTrace.process_primitive = process_primitive
        return self

    def __exit__(self, *exc):
        self._core.EvalTrace.process_primitive = self._original
        return False


def _same(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(left, right, equal_nan=True)
    )


# ------------------------------------------------------------ host scalars ---


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64])
def test_scalar_operand_matches_jnp_asarray_bitwise(dtype):
    value = np.int32(4231)
    folded = rp._scalar_operand(value, dtype)
    loose = jnp.asarray(value, dtype=dtype)
    assert folded.dtype == loose.dtype
    assert folded.shape == loose.shape == ()
    assert folded.weak_type == loose.weak_type
    assert int(folded) == int(loose) == 4231


def test_scalar_operand_issues_no_eager_dispatch():
    value = np.int32(17)
    rp._scalar_operand(value, jnp.int32)  # warm any first-call machinery
    with _DispatchCounter() as folded:
        rp._scalar_operand(value, jnp.int32)
    with _DispatchCounter() as loose:
        jnp.asarray(value, dtype=jnp.int32)
    assert folded.count == 0, folded.by_primitive
    assert loose.count > 0, "the expression this replaces did dispatch"


# -------------------------------------------------------- carry real dtype ---


@pytest.mark.parametrize(
    "cross_dtype", [jnp.complex64, jnp.complex128, jnp.float32, jnp.float64]
)
def test_real_part_dtype_is_numpy_promotion(cross_dtype):
    """The host form gives the dtype the device form gave, for every operand."""

    assert (
        np.zeros((), dtype=cross_dtype).real.dtype
        == jnp.zeros((), dtype=cross_dtype).real.dtype
    )


# ----------------------------------------------------- zero accumulators -----


def test_zero_block_partials_match_jnp_zeros_bitwise():
    key = (
        ((7, 5, 3), jnp.dtype(jnp.float32)),
        ((9,), jnp.dtype(jnp.float64)),
        ((7,), jnp.dtype(jnp.float64)),
        ((7,), jnp.dtype(jnp.float32)),
    )
    folded = rp._zero_block_partials(key)()
    loose = tuple(jnp.zeros(shape, dtype=dtype) for shape, dtype in key)
    assert len(folded) == len(loose)
    for got, want in zip(folded, loose):
        assert _same(got, want)


def test_zero_block_partials_issue_no_eager_dispatch_and_return_fresh_buffers():
    key = (((4, 2, 3), jnp.dtype(jnp.float32)), ((5,), jnp.dtype(jnp.float64)))
    build = rp._zero_block_partials(key)
    build()  # compile
    with _DispatchCounter() as counter:
        first = build()
        second = build()
    assert counter.count == 0, counter.by_primitive
    # The M-step block program donates the carry, so every call must hand back
    # buffers of its own rather than a cached constant.
    assert first[0].unsafe_buffer_pointer() != second[0].unsafe_buffer_pointer()


def test_zero_block_partials_reuses_one_program_per_class():
    key = (((6,), jnp.dtype(jnp.float32)),)
    assert rp._zero_block_partials(key) is rp._zero_block_partials(key)
    other = (((7,), jnp.dtype(jnp.float32)),)
    assert rp._zero_block_partials(other) is not rp._zero_block_partials(key)


# -------------------------------------------------- chunk operand row prep ---


def _chunk_operand_case(seed=0, image_capacity=6, n_valid=4, n_fine_trans=3, pixels=5):
    rng = np.random.default_rng(seed)
    shape = (image_capacity, pixels)
    arrays = rp._ChunkOperandRowInputs(
        score_input=jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
                                dtype=jnp.complex64),
        corr_img_score=jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32),
        highres_xi2_half=jnp.asarray(rng.standard_normal((image_capacity,)), dtype=jnp.float32),
        shifted_recon=jnp.asarray(
            rng.standard_normal((image_capacity, n_fine_trans * pixels)), dtype=jnp.complex64
        ),
        shifted_noise=jnp.asarray(
            rng.standard_normal((image_capacity, n_fine_trans * pixels)), dtype=jnp.complex64
        ),
        ctf2_over_nv_recon=jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32),
        direct_ctf_rfloat_recon=None,
        processed_score_half_for_noise=jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32),
        relion_norm_high_shell=None,
        raw_translated_wavg_rectangle=jnp.asarray(
            rng.standard_normal((image_capacity, n_fine_trans, pixels)), dtype=jnp.float32
        ),
    )
    permutation = jnp.asarray(rng.permutation(image_capacity), dtype=jnp.int32)
    valid_images = jnp.asarray(np.arange(image_capacity) < n_valid, dtype=bool)
    exact_positions = jnp.asarray([0, 2, 4], dtype=jnp.int32)
    return arrays, permutation, valid_images, exact_positions, image_capacity, n_fine_trans


def _chunk_operand_rows_loose(arrays, permutation, valid_images, exact_positions,
                              image_capacity, n_fine_trans):
    """The expression the fold replaces, statement for statement."""

    def take(values):
        return None if values is None else rp._zero_padded_images(
            jnp.asarray(values)[permutation], valid_images
        )

    raw = take(arrays.raw_translated_wavg_rectangle)
    return (
        take(arrays.score_input),
        take(arrays.corr_img_score),
        None if arrays.highres_xi2_half is None else take(arrays.highres_xi2_half),
        take(arrays.shifted_recon.reshape(image_capacity, n_fine_trans, -1)),
        take(arrays.shifted_noise.reshape(image_capacity, n_fine_trans, -1)),
        take(arrays.ctf2_over_nv_recon),
        take(arrays.direct_ctf_rfloat_recon),
        take(arrays.processed_score_half_for_noise),
        take(arrays.relion_norm_high_shell),
        raw,
        raw[:, :, exact_positions],
    )


def test_chunk_operand_rows_match_the_loose_dispatch_bitwise():
    case = _chunk_operand_case()
    arrays, permutation, valid_images, exact_positions, capacity, n_trans = case
    folded = rp._chunk_operand_rows(
        arrays, permutation, valid_images, exact_positions,
        image_capacity=capacity, n_fine_trans=n_trans,
    )
    loose = _chunk_operand_rows_loose(*case)
    assert len(folded) == len(loose)
    for got, want in zip(folded, loose):
        assert _same(got, want)


def test_chunk_operand_rows_zero_the_padded_slots():
    case = _chunk_operand_case(seed=3, image_capacity=6, n_valid=4)
    arrays, permutation, valid_images, exact_positions, capacity, n_trans = case
    folded = rp._chunk_operand_rows(
        arrays, permutation, valid_images, exact_positions,
        image_capacity=capacity, n_fine_trans=n_trans,
    )
    for value in folded:
        if value is None:
            continue
        assert np.all(np.asarray(value)[4:] == 0)


def test_chunk_operand_rows_issue_no_eager_dispatch():
    case = _chunk_operand_case(seed=5)
    arrays, permutation, valid_images, exact_positions, capacity, n_trans = case
    rp._chunk_operand_rows(arrays, permutation, valid_images, exact_positions,
                           image_capacity=capacity, n_fine_trans=n_trans)
    with _DispatchCounter() as folded:
        rp._chunk_operand_rows(arrays, permutation, valid_images, exact_positions,
                               image_capacity=capacity, n_fine_trans=n_trans)
    with _DispatchCounter() as loose:
        _chunk_operand_rows_loose(*case)
    assert folded.count == 0, folded.by_primitive
    assert loose.count >= 30, loose.by_primitive


# ------------------------------------------------- per-batch window gather ---


def _window_case(seed=0, n_images=4, half_pixels=11, score=5, recon=4, bpref=True, rfloat=True):
    rng = np.random.default_rng(seed)

    def complex_half():
        real = rng.standard_normal((n_images, half_pixels))
        imag = rng.standard_normal((n_images, half_pixels))
        return jnp.asarray(real + 1j * imag, dtype=jnp.complex64)

    def real_half():
        return jnp.asarray(rng.standard_normal((n_images, half_pixels)), dtype=jnp.float32)

    arrays = ro._BatchWindowInputs(
        ctf2_over_nv_half=real_half(),
        sparse_score_input_half=complex_half(),
        processed_score_half_for_noise=real_half(),
        recon_input_half=complex_half(),
        weighted_ctf_half=real_half() if bpref else None,
        score_weighted_half=complex_half(),
        ctf2_over_nv_recon_half=real_half(),
        ctf_half_rfloat=real_half() if rfloat else None,
        dc_mask=jnp.asarray(np.arange(half_pixels) == 0, dtype=bool),
        score_indices=jnp.asarray(rng.choice(half_pixels, size=score, replace=False),
                                  dtype=jnp.int32),
        recon_indices=jnp.asarray(rng.choice(half_pixels, size=recon, replace=False),
                                  dtype=jnp.int32),
    )
    return arrays


def _window_loose(arrays, mask_dc, score_real_dtype, score_complex_dtype, acc_real_dtype):
    """The expression the fold replaces, statement for statement."""

    ctf2_score = arrays.ctf2_over_nv_half
    if mask_dc:
        ctf2_score = jnp.where(arrays.dc_mask[None, :], 0.0, ctf2_score)
    batch = {
        "score_input": arrays.sparse_score_input_half[:, arrays.score_indices],
        "corr_img_score": ctf2_score[:, arrays.score_indices].astype(score_real_dtype),
        "processed_image_half": arrays.processed_score_half_for_noise,
    }
    batch["recon_image"] = jnp.asarray(
        arrays.recon_input_half[:, arrays.recon_indices], dtype=score_complex_dtype
    )
    if arrays.weighted_ctf_half is not None:
        batch["recon_weight"] = jnp.asarray(
            arrays.weighted_ctf_half[:, arrays.recon_indices], dtype=acc_real_dtype
        )
    batch["noise_image"] = jnp.asarray(
        arrays.score_weighted_half[:, arrays.recon_indices], dtype=score_complex_dtype
    )
    batch["ctf2_over_nv_recon"] = arrays.ctf2_over_nv_recon_half[:, arrays.recon_indices]
    if arrays.ctf_half_rfloat is not None:
        batch["direct_ctf_rfloat_recon"] = arrays.ctf_half_rfloat[:, arrays.recon_indices]
    return batch


@pytest.mark.parametrize("mask_dc", [True, False])
@pytest.mark.parametrize("bpref,rfloat", [(True, True), (False, False), (True, False)])
def test_batch_window_operands_match_the_loose_dispatch_bitwise(mask_dc, bpref, rfloat):
    arrays = _window_case(seed=1, bpref=bpref, rfloat=rfloat)
    kwargs = dict(
        mask_dc=mask_dc,
        score_real_dtype=jnp.dtype(jnp.float32),
        score_complex_dtype=jnp.dtype(jnp.complex64),
        acc_real_dtype=jnp.dtype(jnp.float32),
    )
    folded = ro._batch_window_operands(arrays, **kwargs)
    loose = _window_loose(arrays, **kwargs)
    assert set(folded) == set(loose)
    for name in loose:
        assert _same(folded[name], loose[name]), name


def test_batch_window_operands_issue_no_eager_dispatch():
    arrays = _window_case(seed=2)
    kwargs = dict(
        mask_dc=True,
        score_real_dtype=jnp.dtype(jnp.float32),
        score_complex_dtype=jnp.dtype(jnp.complex64),
        acc_real_dtype=jnp.dtype(jnp.float32),
    )
    ro._batch_window_operands(arrays, **kwargs)
    with _DispatchCounter() as folded:
        ro._batch_window_operands(arrays, **kwargs)
    with _DispatchCounter() as loose:
        _window_loose(arrays, **kwargs)
    assert folded.count == 0, folded.by_primitive
    # Each ``values[:, indices]`` is five eager primitives, not one: JAX
    # normalizes the index array before every gather.
    assert loose.count >= 25, loose.by_primitive


# ------------------------------------------------ concatenate and reorder ----


def test_concatenate_and_reorder_matches_the_loose_dispatch_bitwise():
    rng = np.random.default_rng(7)
    parts = tuple(
        jnp.asarray(rng.standard_normal((size, 3)), dtype=jnp.float32) for size in (4, 4, 2)
    )
    reorder = jnp.asarray(rng.permutation(10), dtype=jnp.int64)
    folded = ro._concatenate_and_reorder(parts, reorder)
    loose = jnp.concatenate(list(parts), axis=0)[reorder]
    assert _same(folded, loose)


def test_concatenate_and_reorder_issues_no_eager_dispatch():
    rng = np.random.default_rng(8)
    parts = tuple(jnp.asarray(rng.standard_normal((3, 2)), dtype=jnp.float32) for _ in range(4))
    reorder = jnp.asarray(rng.permutation(12), dtype=jnp.int64)
    ro._concatenate_and_reorder(parts, reorder)
    with _DispatchCounter() as folded:
        ro._concatenate_and_reorder(parts, reorder)
    with _DispatchCounter() as loose:
        jnp.concatenate(list(parts), axis=0)[reorder]
    assert folded.count == 0, folded.by_primitive
    assert loose.count >= 2, loose.by_primitive
