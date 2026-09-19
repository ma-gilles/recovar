"""Segmented CUDA sparse pass-2 posterior: wrapper contracts and bitwise parity.

The segmented handlers take the scores of one chunk as a flat cell array in
which image ``i`` owns ``[segment_offsets[i], segment_offsets[i + 1])`` instead
of a padded rectangular row.  Their acceptance criterion is bitwise equality
with the rectangular handlers on the same candidate values, so every GPU test
here builds a rectangular oracle and compares with ``assert_array_equal``.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb

pytestmark = pytest.mark.unit

_ADAPTIVE_FRACTIONS = (0.999, 1.0)


def make_scores(shape, seed, *, all_inf_row=False, nan=False):
    """Random pass-2 scores with masked (-inf) cells, as the fine scorer emits."""

    rng = np.random.default_rng(seed)
    scores = (rng.normal(size=shape) * 30 - 200).astype(np.float32)
    scores[rng.random(shape) < 0.3] = -np.inf
    if nan:
        scores[tuple(0 for _ in shape)] = np.nan
    if all_inf_row and shape[0] > 1:
        scores[1] = -np.inf
    for row in range(shape[0]):
        if all_inf_row and row == 1:
            continue
        flat = scores[row].reshape(-1)
        if not np.isfinite(flat).any():
            flat[0] = -150.0
    return scores


def _gpu_library_has(symbol):
    cb._ensure_ffi()
    return hasattr(cb._get_lib(), symbol)


def _require_segmented_gpu():
    assert jax.default_backend() == "gpu"
    if not _gpu_library_has("SparsePass2SegmentedPosteriorF32"):
        pytest.skip("loaded CUDA library lacks SparsePass2SegmentedPosteriorF32")
    assert cb.custom_cuda_requested()


_OUTPUT_NAMES = (
    "log_z",
    "best_log_score",
    "best_cell_index",
    "max_posterior",
    "probs",
    "normalized_weights",
    "reconstruction_probs",
    "mask",
    "n_significant",
    "sum_weight",
    "threshold",
)


def _rectangular(scores_2d, log_z, external, *, adaptive_fraction, keep_all):
    """Rectangular oracle: one row per image, named like the segmented outputs."""

    use_external = external is not None
    outputs = cb.sparse_pass2_posterior_f32(
        jnp.asarray(scores_2d, dtype=jnp.float32),
        jnp.asarray(log_z, dtype=jnp.float64),
        jnp.asarray(
            np.ones(scores_2d.shape[0], np.float32) if external is None else external,
            dtype=jnp.float32,
        ),
        adaptive_fraction=float(adaptive_fraction),
        keep_all=bool(keep_all),
        use_external_sum_weight=use_external,
    )
    return {name: np.asarray(value) for name, value in zip(_OUTPUT_NAMES, outputs)}


def _segmented(scores_flat, offsets, n_valid, log_z, external, *, adaptive_fraction, keep_all):
    segments = len(offsets) - 1
    outputs = cb.sparse_pass2_segmented_posterior_f32(
        jnp.asarray(scores_flat, dtype=jnp.float32),
        jnp.asarray(offsets, dtype=jnp.int32),
        jnp.asarray(n_valid, dtype=jnp.int32),
        jnp.asarray(log_z, dtype=jnp.float64),
        jnp.asarray(
            np.ones(segments, np.float32) if external is None else external,
            dtype=jnp.float32,
        ),
        adaptive_fraction=float(adaptive_fraction),
        keep_all=bool(keep_all),
        use_external_sum_weight=external is not None,
    )
    return {name: np.asarray(value) for name, value in zip(_OUTPUT_NAMES, outputs)}


def _assert_same(actual, expected, context=""):
    for name in _OUTPUT_NAMES:
        got, want = actual[name], expected[name]
        assert got.dtype == want.dtype, f"{name} dtype {context}"
        assert got.shape == want.shape, f"{name} shape {context}"
        np.testing.assert_array_equal(got, want, err_msg=f"{name} {context}")


# ---------------------------------------------------------------------------
# Contracts that hold on any backend.
# ---------------------------------------------------------------------------


def test_segment_state_bytes_match_header():
    """The Python scratch size tracks sizeof(SegmentState) in the CUDA header."""

    header = os.path.join(os.path.dirname(cb.__file__), "cuda", "sparse_pass2_posterior.cuh")
    body = open(header).read().split("struct SegmentState", 1)[1].split("};", 1)[0]
    sizes = {"RowState": cb._SPARSE_PASS2_ROW_STATE_BYTES, "int": 4, "int64_t": 8}
    fields = [line.split()[0] for line in body.splitlines() if line.strip() and line.strip()[0] not in "{/"]
    total = 0
    for kind in fields:
        size = sizes[kind]
        align = min(size, 8)
        total = (total + align - 1) // align * align + size
    total = (total + 7) // 8 * 8
    assert total == cb._SPARSE_PASS2_SEGMENT_STATE_BYTES


def test_segmented_targets_are_optional_abi():
    """Both handlers register lazily, like the other optional CUDA targets."""

    for target in (
        cb._TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64,
        cb._TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32,
    ):
        assert target in cb._OPTIONAL_FFI_REGISTRATIONS
        assert target not in {name for name, _symbol in cb._FFI_REGISTRATIONS}
    assert isinstance(cb.sparse_pass2_segmented_supported(), bool)


@pytest.mark.parametrize(
    "case",
    [
        "scores_dtype",
        "scores_empty",
        "offsets_dtype",
        "offsets_rank",
        "offsets_short",
        "n_valid_dtype",
        "n_valid_size",
        "log_z_dtype",
        "log_z_shape",
        "external_dtype",
        "static_bool",
    ],
)
def test_wrapper_rejects_bad_operands(case):
    scores = jnp.zeros((12,), jnp.float32)
    offsets = jnp.asarray([0, 6, 12], jnp.int32)
    n_valid = jnp.asarray(2, jnp.int32)
    log_z = jnp.zeros((2,), jnp.float64)
    external = jnp.ones((2,), jnp.float32)
    kwargs = dict(adaptive_fraction=0.999, keep_all=False, use_external_sum_weight=False)
    if case == "scores_dtype":
        scores = scores.astype(jnp.float64)
    if case == "scores_empty":
        scores = jnp.zeros((0,), jnp.float32)
    if case == "offsets_dtype":
        offsets = offsets.astype(jnp.int64)
    if case == "offsets_rank":
        offsets = offsets.reshape(3, 1)
    if case == "offsets_short":
        offsets = jnp.asarray([0], jnp.int32)
    if case == "n_valid_dtype":
        n_valid = n_valid.astype(jnp.float32)
    if case == "n_valid_size":
        n_valid = jnp.asarray([2, 2], jnp.int32)
    if case == "log_z_dtype":
        log_z = log_z.astype(jnp.float32)
    if case == "log_z_shape":
        log_z = jnp.zeros((3,), jnp.float64)
    if case == "external_dtype":
        external = external.astype(jnp.float64)
    if case == "static_bool":
        kwargs["keep_all"] = 1
    with pytest.raises((TypeError, ValueError)):
        cb.sparse_pass2_segmented_posterior_f32(
            scores, offsets, n_valid, log_z, external, **kwargs
        )


def test_wrapper_requires_gpu_backend():
    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only contract")
    scores = jnp.zeros((12,), jnp.float32)
    offsets = jnp.asarray([0, 6, 12], jnp.int32)
    n_valid = jnp.asarray(2, jnp.int32)
    with pytest.raises(RuntimeError, match="GPU backend"):
        cb.sparse_pass2_segmented_log_z_f64(scores, offsets, n_valid)
    with pytest.raises(RuntimeError, match="GPU backend"):
        cb.sparse_pass2_segmented_posterior_f32(
            scores,
            offsets,
            n_valid,
            jnp.zeros((2,), jnp.float64),
            jnp.ones((2,), jnp.float32),
            adaptive_fraction=0.999,
            keep_all=False,
            use_external_sum_weight=False,
        )


# ---------------------------------------------------------------------------
# GPU parity against the rectangular handlers.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (4, 257, 84), (2, 1000, 3), (6, 4096, 21)])
@pytest.mark.parametrize("variant", ["plain", "nan_and_inf_row", "ties"])
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
def test_flattened_rows_match_rectangular(shape, variant, adaptive_fraction):
    """Each rectangular row flattened into a segment reproduces it bitwise."""

    _require_segmented_gpu()
    images, rows, translations = shape
    cells = rows * translations
    scores = make_scores(shape, 5, all_inf_row=variant == "nan_and_inf_row", nan=variant == "nan_and_inf_row")
    if variant == "ties":
        scores[:, 0] = scores.reshape(images, -1).max(axis=1)[:, None]
    scores_2d = scores.reshape(images, cells)
    log_z = np.array(
        cb.sparse_pass2_log_z_f64(jnp.asarray(scores_2d, jnp.float32)), dtype=np.float64
    )
    if variant == "nan_and_inf_row" and images > 2:
        log_z[2] = np.inf
    offsets = (np.arange(images + 1) * cells).astype(np.int32)

    segmented_log_z = np.asarray(
        cb.sparse_pass2_segmented_log_z_f64(
            jnp.asarray(scores_2d.reshape(-1), jnp.float32),
            jnp.asarray(offsets, jnp.int32),
            jnp.asarray(images, jnp.int32),
        )
    )
    rectangular_log_z = np.asarray(cb.sparse_pass2_log_z_f64(jnp.asarray(scores_2d, jnp.float32)))
    assert segmented_log_z.dtype == rectangular_log_z.dtype
    np.testing.assert_array_equal(segmented_log_z, rectangular_log_z, err_msg="segmented log_z")

    rng = np.random.default_rng(3)
    for keep_all in (False, True):
        for external in (None, (np.abs(rng.normal(size=images)) + 0.5).astype(np.float32)):
            expected = _rectangular(
                scores_2d, log_z, external, adaptive_fraction=adaptive_fraction, keep_all=keep_all
            )
            actual = _segmented(
                scores_2d.reshape(-1),
                offsets,
                images,
                log_z,
                external,
                adaptive_fraction=adaptive_fraction,
                keep_all=keep_all,
            )
            expected = {
                name: value.reshape(-1) if value.shape == scores_2d.shape else value
                for name, value in expected.items()
            }
            context = f"keep_all={keep_all} external={external is not None}"
            _assert_same(actual, expected, context)


@pytest.mark.gpu
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
@pytest.mark.parametrize("keep_all", [False, True])
@pytest.mark.parametrize("use_external", [False, True])
def test_ragged_segments_match_per_segment_rectangular(adaptive_fraction, keep_all, use_external):
    """Segments of different lengths each reproduce their own rectangular row."""

    _require_segmented_gpu()
    rng = np.random.default_rng(17)
    translations = 12
    row_counts = [1, 3, 40, 2, 173, 7, 512]
    lengths = [count * translations for count in row_counts]
    segments = len(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    cells = int(offsets[-1])
    scores = make_scores((cells,), 23).astype(np.float32)
    # Guarantee at least one finite candidate per segment.
    for index in range(segments):
        segment = scores[offsets[index] : offsets[index + 1]]
        if not np.isfinite(segment).any():
            segment[0] = -150.0
    log_z = np.empty(segments, np.float64)
    for index in range(segments):
        segment = scores[offsets[index] : offsets[index + 1]]
        log_z[index] = np.asarray(
            cb.sparse_pass2_log_z_f64(jnp.asarray(segment.reshape(1, -1), jnp.float32))
        )[0]
    external = (np.abs(rng.normal(size=segments)) + 0.5).astype(np.float32) if use_external else None

    segmented_log_z = np.asarray(
        cb.sparse_pass2_segmented_log_z_f64(
            jnp.asarray(scores, jnp.float32),
            jnp.asarray(offsets, jnp.int32),
            jnp.asarray(segments, jnp.int32),
        )
    )
    np.testing.assert_array_equal(segmented_log_z, log_z, err_msg="ragged segmented log_z")

    actual = _segmented(
        scores, offsets, segments, log_z, external,
        adaptive_fraction=adaptive_fraction, keep_all=keep_all,
    )
    for index in range(segments):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        expected = _rectangular(
            scores[begin:end].reshape(1, -1),
            log_z[index : index + 1],
            None if external is None else external[index : index + 1],
            adaptive_fraction=adaptive_fraction,
            keep_all=keep_all,
        )
        per_segment = {}
        for name in _OUTPUT_NAMES:
            value = actual[name]
            per_segment[name] = (
                value[begin:end].reshape(1, -1) if value.shape == (cells,) else value[index : index + 1]
            )
        _assert_same(per_segment, expected, f"segment {index} rows={row_counts[index]}")


@pytest.mark.gpu
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
@pytest.mark.parametrize("keep_all", [False, True])
@pytest.mark.parametrize("use_external", [False, True])
def test_empty_segments_invalid_images_and_padding(adaptive_fraction, keep_all, use_external):
    """Empty and out-of-range segments equal an all -inf row; padding cells are zero."""

    _require_segmented_gpu()
    translations = 9
    #             live   empty  live   invalid (past n_valid_images)   empty
    lengths = [4 * translations, 0, 11 * translations, 6 * translations, 0]
    segments = len(lengths)
    n_valid = 3
    padding_cells = 5 * translations
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    cells = int(offsets[-1]) + padding_cells
    scores = make_scores((cells,), 41).astype(np.float32)
    for index in range(segments):
        segment = scores[offsets[index] : offsets[index + 1]]
        if segment.size and not np.isfinite(segment).any():
            segment[0] = -150.0
    log_z = np.zeros(segments, np.float64)
    for index in range(segments):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        log_z[index] = (
            np.asarray(cb.sparse_pass2_log_z_f64(jnp.asarray(scores[begin:end].reshape(1, -1), jnp.float32)))[0]
            if end > begin
            else -np.inf
        )
    rng = np.random.default_rng(5)
    external = (np.abs(rng.normal(size=segments)) + 0.5).astype(np.float32) if use_external else None

    segmented_log_z = np.asarray(
        cb.sparse_pass2_segmented_log_z_f64(
            jnp.asarray(scores, jnp.float32),
            jnp.asarray(offsets, jnp.int32),
            jnp.asarray(n_valid, jnp.int32),
        )
    )
    assert np.all(np.isneginf(segmented_log_z[[1, 3, 4]])), segmented_log_z
    np.testing.assert_array_equal(segmented_log_z[[0, 2]], log_z[[0, 2]])

    actual = _segmented(
        scores, offsets, n_valid, log_z, external,
        adaptive_fraction=adaptive_fraction, keep_all=keep_all,
    )
    for index in range(segments):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        length = max(end - begin, 1)
        live = index < n_valid and end > begin
        oracle_scores = (
            scores[begin:end].reshape(1, -1) if live else np.full((1, length), -np.inf, np.float32)
        )
        expected = _rectangular(
            oracle_scores,
            log_z[index : index + 1] if live else np.zeros(1, np.float64),
            None if external is None else external[index : index + 1],
            adaptive_fraction=adaptive_fraction,
            keep_all=keep_all,
        )
        per_segment = {}
        for name in _OUTPUT_NAMES:
            value = actual[name]
            if value.shape != (cells,):
                per_segment[name] = value[index : index + 1]
            elif end > begin:
                per_segment[name] = value[begin:end].reshape(1, -1)
            else:
                # An empty segment has no cells; the oracle row is length 1.
                per_segment[name] = expected[name]
        _assert_same(per_segment, expected, f"segment {index} live={live}")

    tail = int(offsets[-1])
    assert np.all(actual["probs"][tail:] == 0.0)
    assert np.all(actual["normalized_weights"][tail:] == 0.0)
    assert np.all(actual["reconstruction_probs"][tail:] == 0.0)
    assert not np.any(actual["mask"][tail:])


@pytest.mark.gpu
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
def test_capacity_launch_is_independent_of_occupancy(adaptive_fraction):
    """Padding a chunk up to a larger capacity class changes no live output.

    The device-resident driver pads every chunk to a capacity class, so the same
    images are handed to the handler with different segment counts and different
    cell counts from one run to the next.  The handler reads ``n_valid_images``
    only on the device and sizes its scratch from the static cell count, so the
    live segments must come back bitwise identical whatever the padding is.
    """

    _require_segmented_gpu()
    translations = 7
    row_counts = [5, 1, 64, 12]
    lengths = [count * translations for count in row_counts]
    n_valid = len(lengths)
    live_offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    live_cells = int(live_offsets[-1])
    scores = make_scores((live_cells,), 71).astype(np.float32)
    for index in range(n_valid):
        segment = scores[live_offsets[index] : live_offsets[index + 1]]
        if not np.isfinite(segment).any():
            segment[0] = -150.0
    log_z_live = np.empty(n_valid, np.float64)
    for index in range(n_valid):
        segment = scores[live_offsets[index] : live_offsets[index + 1]]
        log_z_live[index] = np.asarray(
            cb.sparse_pass2_log_z_f64(jnp.asarray(segment.reshape(1, -1), jnp.float32))
        )[0]

    reference = None
    for image_capacity, row_capacity in ((n_valid, sum(row_counts)), (16, 256), (64, 1024)):
        cells = row_capacity * translations
        padded_scores = np.full(cells, -np.inf, np.float32)
        padded_scores[:live_cells] = scores
        offsets = np.full(image_capacity + 1, live_cells, np.int32)
        offsets[: n_valid + 1] = live_offsets[: n_valid + 1]
        log_z = np.full(image_capacity, -np.inf, np.float64)
        log_z[:n_valid] = log_z_live

        segmented_log_z = np.asarray(
            cb.sparse_pass2_segmented_log_z_f64(
                jnp.asarray(padded_scores, jnp.float32),
                jnp.asarray(offsets, jnp.int32),
                jnp.asarray(n_valid, jnp.int32),
            )
        )
        np.testing.assert_array_equal(
            segmented_log_z[:n_valid], log_z_live, err_msg=f"log_z at capacity {image_capacity}"
        )

        actual = _segmented(
            padded_scores, offsets, n_valid, log_z, None,
            adaptive_fraction=adaptive_fraction, keep_all=False,
        )
        live = {
            name: (
                value[:live_cells] if value.shape == (cells,) else value[:n_valid]
            )
            for name, value in actual.items()
        }
        if reference is None:
            reference = live
            continue
        for name in _OUTPUT_NAMES:
            np.testing.assert_array_equal(
                live[name], reference[name], err_msg=f"{name} at capacity {image_capacity}"
            )


@pytest.mark.gpu
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
def test_offsets_past_n_valid_images_do_not_leak(adaptive_fraction):
    """Nonempty segments past ``n_valid_images`` stay all ``-inf`` rows.

    The sort/scan loop no longer reads ``n_valid_images`` back to the host, so
    it sorts those segments' scratch as well; nothing they write may reach a
    live segment or their own outputs.
    """

    _require_segmented_gpu()
    translations = 5
    lengths = [6 * translations, 9 * translations, 4 * translations]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    cells = int(offsets[-1])
    scores = make_scores((cells,), 97).astype(np.float32)
    for index in range(len(lengths)):
        segment = scores[offsets[index] : offsets[index + 1]]
        if not np.isfinite(segment).any():
            segment[0] = -150.0
    log_z = np.empty(len(lengths), np.float64)
    for index in range(len(lengths)):
        segment = scores[offsets[index] : offsets[index + 1]]
        log_z[index] = np.asarray(
            cb.sparse_pass2_log_z_f64(jnp.asarray(segment.reshape(1, -1), jnp.float32))
        )[0]

    full = _segmented(
        scores, offsets, len(lengths), log_z, None,
        adaptive_fraction=adaptive_fraction, keep_all=False,
    )
    clipped = _segmented(
        scores, offsets, 2, log_z, None,
        adaptive_fraction=adaptive_fraction, keep_all=False,
    )
    live_cells = int(offsets[2])
    for name in _OUTPUT_NAMES:
        value, reference = clipped[name], full[name]
        if value.shape == (cells,):
            np.testing.assert_array_equal(
                value[:live_cells], reference[:live_cells], err_msg=f"{name} live cells"
            )
        else:
            np.testing.assert_array_equal(value[:2], reference[:2], err_msg=f"{name} live segments")

    invalid = _rectangular(
        np.full((1, lengths[2]), -np.inf, np.float32),
        np.zeros(1, np.float64),
        None,
        adaptive_fraction=adaptive_fraction,
        keep_all=False,
    )
    begin = int(offsets[2])
    per_segment = {
        name: (
            clipped[name][begin : begin + lengths[2]].reshape(1, -1)
            if clipped[name].shape == (cells,)
            else clipped[name][2:3]
        )
        for name in _OUTPUT_NAMES
    }
    _assert_same(per_segment, invalid, "segment past n_valid_images")
