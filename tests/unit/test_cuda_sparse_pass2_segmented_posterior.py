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


# Handler scratch, appended by ``return_scratch``: the exponentiated weights,
# their per-segment ascending sort and its inclusive scan.  The sort/scan modes
# are compared on these directly, because the significance boundary is defined
# on them.
_SCRATCH_NAMES = ("raw_weights", "sorted", "cumulative")

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


def _segmented(
    scores_flat,
    offsets,
    n_valid,
    log_z,
    external,
    *,
    adaptive_fraction,
    keep_all,
    sort_scan_mode=cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED_SORT,
    scratch=False,
):
    # The bitwise oracle tests pin a mode that is bitwise with the rectangular
    # handler by construction, so they keep testing the arithmetic rather than
    # whatever the capacity-class default happens to pick for their shape.
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
        sort_scan_mode=sort_scan_mode,
        return_scratch=scratch,
    )
    names = _OUTPUT_NAMES + (_SCRATCH_NAMES if scratch else ())
    return {name: np.asarray(value) for name, value in zip(names, outputs)}


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


@pytest.mark.parametrize(
    "cells, segments, expected",
    [
        (1376256, 128, cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED),   # 10752 cells/segment
        (5505024, 128, cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED),   # 43008
        (22020096, 512, cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED),  # 43008
        (22020096, 128, cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED),  # 172032
        (22020096, 32, cb.SPARSE_PASS2_SORT_SCAN_PER_SEGMENT),  # 688128
        (0, 0, cb.SPARSE_PASS2_SORT_SCAN_PER_SEGMENT),
    ],
)
def test_auto_mode_follows_the_measured_crossover(cells, segments, expected):
    """The default picks per capacity class, and only from the buffer shapes.

    One block per segment beats one sort per segment while a segment holds tens
    of thousands of cells and loses once it holds hundreds of thousands.  The
    resident classes at this commit are row capacities 8192/32768/131072 times
    168 fine translations over 32/128/512 images; only the widest rows on the
    narrowest image capacity fall on the per-segment side.
    """

    assert cb.sparse_pass2_segmented_auto_mode(cells, segments) == expected


def test_auto_mode_is_the_default_and_the_environment_names_every_mode():
    assert cb._SPARSE_PASS2_SORT_SCAN_DEFAULT == cb.SPARSE_PASS2_SORT_SCAN_AUTO
    assert set(cb._SPARSE_PASS2_SORT_SCAN_NAMES.values()) == {
        cb.SPARSE_PASS2_SORT_SCAN_AUTO,
        cb.SPARSE_PASS2_SORT_SCAN_PER_SEGMENT,
        cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED_SORT,
        cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED,
        cb.SPARSE_PASS2_SORT_SCAN_PARTITIONED_SORT,
        cb.SPARSE_PASS2_SORT_SCAN_PARTITIONED,
    }


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
        "sort_scan_mode",
        "return_scratch",
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
    if case == "sort_scan_mode":
        kwargs["sort_scan_mode"] = 9
    if case == "return_scratch":
        kwargs["return_scratch"] = 1
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


# ---------------------------------------------------------------------------
# Sort/scan modes on chunks shaped like the device-resident classes.
# ---------------------------------------------------------------------------

# (image capacity, row capacity, fine translations, row occupancy) of the
# resident chunk classes the matched hp3 and early states run.  nsys records
# image capacities 32 and 128 and cell counts 1376256 / 2752512 / 5505024 /
# 11010048, which are row capacities 8192-65536 times 168 fine translations.
_PRODUCTION_CHUNKS = (
    pytest.param(128, 8192, 168, 0.76, id="128img_8192rows"),
    pytest.param(32, 32768, 168, 0.95, id="32img_32768rows"),
)

# Bounds on what the device segmented scan (mode 2) may move relative to the
# per-segment CUB scan (mode 0), measured on these fixtures by
# /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_t17_posterior_sort_20260919/
# measure/segmented_scan_agreement.py and recorded in that root's REPORT.md.
# They are properties of a new opt-in path, not a scientific tolerance.
_SINGLE_SCAN_SUM_WEIGHT_ULP = 8
_SINGLE_SCAN_SIGNIFICANCE_SHIFT = 2
_SINGLE_SCAN_FLIPPED_MASS = 1e-5


def make_production_chunk(images, rows, translations, occupancy, seed):
    """A ragged chunk: image ``i`` owns ``rows_i * translations`` cells."""

    rng = np.random.default_rng(seed)
    cells = rows * translations
    share = rng.random(images) + 0.3
    per_image = np.maximum(
        1, np.floor(share / share.sum() * round(occupancy * rows))
    ).astype(np.int64)
    while per_image.sum() > rows:
        per_image[int(np.argmax(per_image))] -= 1
    offsets = np.zeros(images + 1, np.int64)
    offsets[1:] = np.cumsum(per_image * translations)
    live = int(offsets[-1])
    scores = np.full(cells, -np.inf, np.float32)
    body = (rng.normal(size=live) * 30.0 - 200.0).astype(np.float32)
    body[rng.random(live) < 0.25] = -np.inf
    scores[:live] = body
    for index in range(images):
        segment = scores[offsets[index] : offsets[index + 1]]
        if segment.size and not np.isfinite(segment).any():
            segment[0] = -150.0
    return scores, offsets.astype(np.int32)


def _chunk_log_z(scores, offsets, images):
    return np.asarray(
        cb.sparse_pass2_segmented_log_z_f64(
            jnp.asarray(scores, jnp.float32),
            jnp.asarray(offsets, jnp.int32),
            jnp.asarray(images, jnp.int32),
        )
    )


def _float32_ulps(left, right):
    """Distance in float32 ULP, on the monotone integer image of float32."""

    def order(values):
        bits = np.asarray(values, np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, np.int64(-2147483648) - bits, bits)

    return np.abs(order(left) - order(right))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode",
    [cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED_SORT, cb.SPARSE_PASS2_SORT_SCAN_PARTITIONED_SORT],
)
@pytest.mark.parametrize("images, rows, translations, occupancy", _PRODUCTION_CHUNKS)
def test_segmented_sort_is_bitwise_on_production_chunks(
    images, rows, translations, occupancy, mode
):
    """One segmented sort per chunk reproduces the per-segment sorts bitwise.

    A radix sort is an exact permutation of its keys, so mode 1 changes the
    launch structure and nothing the sort itself produces.  The scan that
    follows is the same per-segment CUB call in both modes, but CUB's
    decoupled-lookback float32 scan is not reproducible run to run once a
    segment spans many tiles, so the scan-fed outputs are compared against a
    same-source band: two mode-0 arms in this same process.
    """

    _require_segmented_gpu()
    scores, offsets = make_production_chunk(images, rows, translations, occupancy, 5)
    log_z = _chunk_log_z(scores, offsets, images)
    arms = [
        _segmented(
            scores, offsets, images, log_z, None,
            adaptive_fraction=0.999, keep_all=False, sort_scan_mode=mode, scratch=True,
        )
        for mode in (
            cb.SPARSE_PASS2_SORT_SCAN_PER_SEGMENT,
            cb.SPARSE_PASS2_SORT_SCAN_PER_SEGMENT,
            mode,
        )
    ]
    reference, band, candidate = arms
    for name in ("raw_weights", "sorted", "log_z", "best_log_score", "best_cell_index", "probs"):
        np.testing.assert_array_equal(
            candidate[name], reference[name], err_msg=f"{name} segmented sort vs mode 0"
        )
    for name in _OUTPUT_NAMES + _SCRATCH_NAMES:
        if np.array_equal(band[name], reference[name]):
            np.testing.assert_array_equal(
                candidate[name], reference[name], err_msg=f"{name} segmented sort vs mode 0"
            )
            continue
        # The oracle did not reproduce itself on this field; the candidate only
        # has to stay inside the band the two mode-0 arms span.
        assert np.count_nonzero(candidate[name] != reference[name]) <= np.count_nonzero(
            band[name] != reference[name]
        ), f"{name} segmented sort outside the mode-0 band"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode", [cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED, cb.SPARSE_PASS2_SORT_SCAN_PARTITIONED]
)
@pytest.mark.parametrize("images, rows, translations, occupancy", _PRODUCTION_CHUNKS)
def test_single_scan_keeps_keys_and_moves_only_near_ties(
    images, rows, translations, occupancy, mode
):
    """Mode 2 keeps the sorted keys and everything the scan does not feed.

    The device segmented scan sums each segment in a different float32 order, so
    ``sum_weight`` and the threshold can move by a few ULP and an image whose
    significance boundary sits between two adjacent sorted weights can keep a
    different number of candidates.  What must not move: the sorted keys (an
    exact permutation), the raw weights, the log-Z, the posterior probabilities
    and the best candidate, none of which the scan feeds.  A candidate that
    flips must carry a negligible share of the image's weight, which is what
    "near-tie" means for this boundary.
    """

    _require_segmented_gpu()
    scores, offsets = make_production_chunk(images, rows, translations, occupancy, 5)
    log_z = _chunk_log_z(scores, offsets, images)
    reference = _segmented(
        scores, offsets, images, log_z, None,
        adaptive_fraction=0.999, keep_all=False,
        sort_scan_mode=cb.SPARSE_PASS2_SORT_SCAN_PER_SEGMENT, scratch=True,
    )
    candidate = _segmented(
        scores, offsets, images, log_z, None,
        adaptive_fraction=0.999, keep_all=False, sort_scan_mode=mode, scratch=True,
    )
    for name in ("raw_weights", "sorted", "log_z", "best_log_score", "best_cell_index", "probs"):
        np.testing.assert_array_equal(
            candidate[name], reference[name], err_msg=f"{name} device scan vs mode 0"
        )

    # sum_weight is the scan's last element, so it moves by ULP; the threshold
    # is a sorted weight, so it moves by whole candidates and is bounded below
    # by the count instead.
    assert (
        _float32_ulps(
            candidate["sum_weight"][:images], reference["sum_weight"][:images]
        ).max()
        <= _SINGLE_SCAN_SUM_WEIGHT_ULP
    )
    significance_shift = np.abs(
        candidate["n_significant"][:images].astype(np.int64)
        - reference["n_significant"][:images].astype(np.int64)
    )
    assert significance_shift.max() <= _SINGLE_SCAN_SIGNIFICANCE_SHIFT

    # Every candidate whose significance flips lies between the two thresholds
    # and carries a negligible share of the image's mass.
    for index in range(images):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        if end <= begin:
            continue
        flips = np.nonzero(reference["mask"][begin:end] != candidate["mask"][begin:end])[0]
        if flips.size == 0:
            continue
        raw = reference["raw_weights"][begin:end][flips]
        low = min(reference["threshold"][index], candidate["threshold"][index])
        high = max(reference["threshold"][index], candidate["threshold"][index])
        assert np.all((raw >= low) & (raw < high)), f"image {index} flipped outside the band"
        mass = float(raw.sum() / reference["sum_weight"][index])
        assert mass <= _SINGLE_SCAN_FLIPPED_MASS, f"image {index} flipped mass {mass}"


@pytest.mark.gpu
@pytest.mark.parametrize("mode", [1, 2, 3, 4])
@pytest.mark.parametrize("adaptive_fraction", _ADAPTIVE_FRACTIONS)
def test_modes_agree_on_empty_and_invalid_segments(mode, adaptive_fraction):
    """Empty segments and images past ``n_valid_images`` are mode-independent.

    Those segments hold no cells on the device, so no sort and no scan touches
    them and every mode must reproduce the oracle bitwise, padding included.
    """

    _require_segmented_gpu()
    translations = 5
    lengths = [6 * translations, 0, 9 * translations, 4 * translations]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    cells = int(offsets[-1]) + 3 * translations  # trailing padding cells
    scores = make_scores((cells,), 131).astype(np.float32)
    offsets = np.concatenate([offsets, [offsets[-1]]]).astype(np.int32)
    n_valid = 3
    for index in range(len(offsets) - 1):
        segment = scores[offsets[index] : offsets[index + 1]]
        if segment.size and not np.isfinite(segment).any():
            segment[0] = -150.0
    log_z = _chunk_log_z(scores, offsets, n_valid)
    reference = _segmented(
        scores, offsets, n_valid, log_z, None,
        adaptive_fraction=adaptive_fraction, keep_all=False,
        sort_scan_mode=cb.SPARSE_PASS2_SORT_SCAN_PER_SEGMENT, scratch=True,
    )
    candidate = _segmented(
        scores, offsets, n_valid, log_z, None,
        adaptive_fraction=adaptive_fraction, keep_all=False,
        sort_scan_mode=mode, scratch=True,
    )
    empty_and_invalid = [1, 3, 4]
    for name in _OUTPUT_NAMES:
        if reference[name].shape == (len(offsets) - 1,):
            np.testing.assert_array_equal(
                candidate[name][empty_and_invalid],
                reference[name][empty_and_invalid],
                err_msg=f"{name} on empty/invalid segments, mode {mode}",
            )
    tail = int(offsets[3])
    for name in ("normalized_weights", "reconstruction_probs", "mask", "probs"):
        np.testing.assert_array_equal(
            candidate[name][tail:], reference[name][tail:],
            err_msg=f"{name} past n_valid_images, mode {mode}",
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode", [cb.SPARSE_PASS2_SORT_SCAN_SEGMENTED, cb.SPARSE_PASS2_SORT_SCAN_PARTITIONED]
)
def test_single_scan_clamps_a_malformed_offset_table_on_the_device(mode):
    """Modes 1 and 2 hand the offsets to CUB, so the device clamps them.

    The host no longer reads the table back in mode 2, so a decreasing entry
    must be repaired on the device instead of indexing outside the buffers: the
    handler must behave as it does for the monotone clamp of the same table,
    where the offending segment is empty.
    """

    _require_segmented_gpu()
    translations = 4
    lengths = [5 * translations, 7 * translations, 6 * translations]
    good = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    cells = int(good[-1])
    scores = make_scores((cells,), 157).astype(np.float32)
    for index in range(len(lengths)):
        segment = scores[good[index] : good[index + 1]]
        if not np.isfinite(segment).any():
            segment[0] = -150.0
    images = len(lengths)
    # Segment 1 runs backwards; the monotone clamp turns it into an empty
    # segment that starts where segment 0 ended.
    malformed = good.copy()
    malformed[2] = good[1] - translations
    clamped = good.copy()
    clamped[2] = good[1]

    log_z = _chunk_log_z(scores, clamped, images)
    expected = _segmented(
        scores, clamped, images, log_z, None,
        adaptive_fraction=0.999, keep_all=False,
        sort_scan_mode=cb.SPARSE_PASS2_SORT_SCAN_PER_SEGMENT,
    )
    actual = _segmented(
        scores, malformed, images, _chunk_log_z(scores, malformed, images), None,
        adaptive_fraction=0.999, keep_all=False, sort_scan_mode=mode,
    )
    for name in ("log_z", "best_log_score", "n_significant", "sum_weight", "threshold"):
        np.testing.assert_array_equal(
            actual[name][:2], expected[name][:2], err_msg=f"{name} before the malformed entry"
        )
