"""Host contract for low-cardinality exact VDAM Fourier-window shapes."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_stable_fourier_window_shape_plan,
    stable_fourier_window_current_size,
)
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    crop_relion_x_half_accumulator,
)
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    plan_coarse_gemm_certificate_topology,
)
from recovar.em.dense_single_volume.helpers.significance import (
    _plan_coarse_gaussian_square_layout,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _make_relion_wavg_rectangle,
    _make_stable_relion_wavg_rectangle,
)
from recovar.em.dense_single_volume.shape_buckets import pad_axis

pytestmark = pytest.mark.unit

_IMAGE_SHAPE = (128, 128)
_N_HALF = 128 * 65
_GF46_CURRENT_SIZES_THROUGH_80 = (
    30,
    32,
    34,
    38,
    44,
    46,
    48,
    50,
    56,
    60,
    62,
    66,
    68,
    70,
    76,
    78,
    84,
    86,
    88,
    90,
    98,
    104,
    106,
    110,
    114,
    116,
    122,
    126,
    128,
)


def _plan(
    current_size,
    *,
    enabled=True,
    reconstruction_current_size=None,
    quantum=8,
):
    return make_stable_fourier_window_shape_plan(
        _IMAGE_SHAPE,
        current_size,
        _N_HALF,
        enabled=enabled,
        reconstruction_current_size=reconstruction_current_size,
        quantum=quantum,
    )


def _relion_lane_tree_sum(storage, logical_count):
    """Host float32 replay of the fine scorer's 256-lane issue order."""

    block_size = 256
    lanes = np.zeros(block_size, dtype=np.float32)
    for pixel in range(int(logical_count)):
        lane = pixel % block_size
        lanes[lane] = np.float32(lanes[lane] + np.float32(storage[pixel]))
    width = block_size // 2
    while width:
        lanes[:width] = np.float32(lanes[:width] + lanes[width : 2 * width])
        width //= 2
    return lanes[0]


def test_stable_window_size_uses_eight_pixel_classes_and_isolates_full_box():
    assert stable_fourier_window_current_size(30, 128) == 32
    assert stable_fourier_window_current_size(34, 128) == 40
    assert stable_fourier_window_current_size(56, 128) == 56
    assert stable_fourier_window_current_size(70, 128) == 72
    assert stable_fourier_window_current_size(84, 128) == 88
    assert stable_fourier_window_current_size(122, 128) == 126
    assert stable_fourier_window_current_size(126, 128) == 126
    assert stable_fourier_window_current_size(128, 128) == 128


@pytest.mark.parametrize(
    ("current_size", "image_size", "quantum"),
    ((0, 128, 8), (31, 128, 8), (130, 128, 8), (32, 127, 8), (32, 128, 3)),
)
def test_stable_window_size_rejects_invalid_shapes(current_size, image_size, quantum):
    with pytest.raises(ValueError):
        stable_fourier_window_current_size(current_size, image_size, quantum=quantum)


def test_stable_window_runtime_quantum_is_diagnostic_and_fail_closed(monkeypatch):
    from recovar.em.dense_single_volume.local_em_engine import (
        DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
        STABLE_FOURIER_WINDOW_QUANTUM_ENV,
        _stable_fourier_window_quantum,
    )

    monkeypatch.delenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raising=False)
    assert _stable_fourier_window_quantum() == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM == 8

    monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, "16")
    assert _stable_fourier_window_quantum() == 16

    for invalid in ("0", "3", "nope"):
        monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, invalid)
        with pytest.raises(ValueError, match=STABLE_FOURIER_WINDOW_QUANTUM_ENV):
            _stable_fourier_window_quantum()


def _active_coarse_score_indices(current_size: int) -> np.ndarray:
    plan = make_stable_fourier_window_shape_plan(
        _IMAGE_SHAPE,
        current_size,
        _N_HALF,
        enabled=False,
    )
    return np.asarray(plan.logical_spec.score_indices_np, dtype=np.int32)


def test_stable_coarse_square_preserves_logical_issue_prefix(monkeypatch):
    monkeypatch.setenv(
        "RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM",
        "32",
    )
    logical_size = 70
    layout = _plan_coarse_gaussian_square_layout(
        _IMAGE_SHAPE,
        logical_size,
        _active_coarse_score_indices(logical_size),
        stable_fourier_window_shapes=True,
    )
    logical_indices, logical_count = make_fourier_window_indices_np(
        _IMAGE_SHAPE,
        logical_size,
        square=True,
        include_dc=True,
    )
    logical_lookup = _relion_cuda_fine_full_to_compact_lookup(
        _IMAGE_SHAPE,
        logical_size,
        logical_indices,
    )

    assert layout.physical_current_size == 96
    assert layout.logical_square_count == logical_count == 70 * 36
    assert layout.physical_square_count == 96 * 49
    np.testing.assert_array_equal(
        layout.score_indices_np[:logical_count],
        logical_indices,
    )
    np.testing.assert_array_equal(
        layout.full_to_compact_np[:logical_count],
        logical_lookup,
    )
    np.testing.assert_array_equal(
        layout.full_to_compact_np[logical_count:],
        np.arange(logical_count, layout.physical_square_count, dtype=np.int32),
    )
    assert not np.any(layout.score_active_mask_np[logical_count:])

    topology = plan_coarse_gemm_certificate_topology(
        layout.full_to_compact_np,
        compact_pixel_count=layout.physical_square_count,
        translation_count=29,
    )
    assert topology.full_position_count == layout.physical_square_count


def test_stable_coarse_square_has_one_shape_across_q32_class(monkeypatch):
    monkeypatch.setenv(
        "RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM",
        "32",
    )
    layouts = [
        _plan_coarse_gaussian_square_layout(
            _IMAGE_SHAPE,
            current_size,
            _active_coarse_score_indices(current_size),
            stable_fourier_window_shapes=True,
        )
        for current_size in (70, 72, 76, 78, 84, 96)
    ]

    assert {layout.physical_current_size for layout in layouts} == {96}
    assert {layout.score_indices_np.shape for layout in layouts} == {(96 * 49,)}
    assert {layout.full_to_compact_np.shape for layout in layouts} == {(96 * 49,)}
    assert len({layout.logical_square_count for layout in layouts}) == 6


def test_disabled_coarse_square_layout_is_legacy_exact(monkeypatch):
    monkeypatch.setenv(
        "RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM",
        "32",
    )
    current_size = 70
    active = _active_coarse_score_indices(current_size)
    layout = _plan_coarse_gaussian_square_layout(
        _IMAGE_SHAPE,
        current_size,
        active,
        stable_fourier_window_shapes=False,
    )
    legacy_indices, legacy_count = make_fourier_window_indices_np(
        _IMAGE_SHAPE,
        current_size,
        square=True,
        include_dc=True,
    )
    legacy_lookup = _relion_cuda_fine_full_to_compact_lookup(
        _IMAGE_SHAPE,
        current_size,
        legacy_indices,
    )

    assert layout.logical_current_size == layout.physical_current_size == current_size
    assert layout.logical_square_count == layout.physical_square_count == legacy_count
    np.testing.assert_array_equal(layout.score_indices_np, legacy_indices)
    np.testing.assert_array_equal(layout.score_active_mask_np, np.isin(legacy_indices, active))
    np.testing.assert_array_equal(layout.full_to_compact_np, legacy_lookup)


def test_shape_policy_is_default_off():
    plan = _plan(70, enabled=False)

    assert plan.logical_current_size == 70
    assert plan.physical_current_size == 70
    assert plan.logical_loop_bounds == (
        plan.physical_score_pixels,
        plan.physical_reconstruction_pixels,
        plan.physical_projection_pixels,
        plan.physical_rectangle_pixels,
    )


def test_adjacent_cutoffs_share_one_physical_signature_but_keep_logical_bounds():
    plans = [_plan(current_size) for current_size in (68, 70, 72)]

    assert {plan.physical_current_size for plan in plans} == {72}
    assert len({plan.physical_signature for plan in plans}) == 1
    assert len({plan.logical_loop_bounds for plan in plans}) == 3


def test_physical_class_boundary_stays_on_runtime_bound_engine_route():
    lower = _plan(50)
    boundary = _plan(56)

    assert lower.physical_signature == boundary.physical_signature
    assert lower.logical_spec.use_window is True
    assert boundary.logical_spec.use_window is True
    engine_source = (
        Path(__file__).resolve().parents[3]
        / "recovar"
        / "em"
        / "dense_single_volume"
        / "local_em_engine.py"
    ).read_text()
    assert (
        "stable_fourier_window_shapes and "
        "stable_window_plan.logical_spec.use_window"
    ) in engine_source


def test_stable_bpref_has_one_fail_closed_source_operand_route():
    engine_source = (
        Path(__file__).resolve().parents[3]
        / "recovar"
        / "em"
        / "dense_single_volume"
        / "local_em_engine.py"
    ).read_text()

    assert "if stable_window_active and not (" in engine_source
    assert "or return_deferred_source_vdam_operands" in engine_source
    assert engine_source.count("stable_dense_positions=(") == 2


def test_score_and_reconstruction_cutoffs_are_bucketed_independently():
    plan = _plan(70, reconstruction_current_size=84)

    assert plan.logical_current_size == 70
    assert plan.physical_current_size == 72
    assert plan.logical_reconstruction_current_size == 84
    assert plan.physical_reconstruction_current_size == 88
    assert plan.physical_projection_pixels >= plan.logical_projection_pixels


@pytest.mark.parametrize(
    ("current_size", "physical_size", "logical_projection", "physical_projection"),
    (
        (56, 56, 1276, 1276),
        (70, 72, 1980, 2093),
        (84, 88, 2835, 3105),
        (128, 128, 8320, 8320),
    ),
)
def test_gf46_checkpoint_projection_capacity(
    current_size,
    physical_size,
    logical_projection,
    physical_projection,
):
    plan = _plan(current_size)

    assert plan.physical_current_size == physical_size
    assert plan.logical_projection_pixels == logical_projection
    assert plan.physical_projection_pixels == physical_projection


def test_gf46_shape_trajectory_collapses_from_29_signatures_to_14():
    plans = [_plan(current_size) for current_size in _GF46_CURRENT_SIZES_THROUGH_80]

    assert len({plan.logical_current_size for plan in plans}) == 29
    assert len({plan.physical_signature for plan in plans}) == 14
    assert sorted({plan.physical_current_size for plan in plans}) == [
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        126,
        128,
    ]


@pytest.mark.parametrize(
    ("quantum", "expected_physical_sizes"),
    (
        (16, [32, 48, 64, 80, 96, 112, 126, 128]),
        (32, [32, 64, 96, 126, 128]),
    ),
)
def test_larger_diagnostic_quantums_reduce_trajectory_shape_cardinality(
    quantum,
    expected_physical_sizes,
):
    plans = [
        _plan(current_size, quantum=quantum)
        for current_size in _GF46_CURRENT_SIZES_THROUGH_80
    ]

    assert sorted({plan.physical_current_size for plan in plans}) == expected_physical_sizes
    assert all(
        plan.physical_projection_pixels >= plan.logical_projection_pixels
        for plan in plans
    )


def test_padding_appends_storage_without_changing_score_pixel_order():
    plan = _plan(70)
    logical_indices, _ = make_fourier_window_indices_np(_IMAGE_SHAPE, 70)
    padded_indices = pad_axis(
        logical_indices,
        0,
        plan.physical_score_pixels,
        value=0,
    )

    np.testing.assert_array_equal(
        padded_indices[: plan.logical_score_pixels],
        logical_indices,
    )
    assert np.all(padded_indices[plan.logical_score_pixels :] == 0)


def test_packed_capacity_reorders_interleaved_physical_support_behind_logical_prefix():
    plan = _plan(70)
    logical = plan.logical_spec.score_indices_np
    physical_sorted = plan.physical_spec.score_indices_np

    # Enlarging a radial window inserts new flat-grid indices throughout the
    # sorted support; slicing the physical spec would therefore change order.
    assert not np.array_equal(physical_sorted[: logical.size], logical)

    packed = plan.packed_indices_np("score")
    np.testing.assert_array_equal(packed[: logical.size], logical)
    np.testing.assert_array_equal(
        np.sort(packed[logical.size :]),
        np.setdiff1d(physical_sorted, logical, assume_unique=True),
    )


@pytest.mark.parametrize("name", ("score", "recon"))
def test_logical_projection_takes_stay_at_front_of_packed_capacity(name):
    plan = _plan(70, reconstruction_current_size=84)
    packed_projection = plan.packed_indices_np("projection")
    packed_support = plan.packed_indices_np(name)
    packed_take = plan.packed_projection_take_np(name)
    logical_count = getattr(plan, f"logical_{'reconstruction' if name == 'recon' else name}_pixels")

    np.testing.assert_array_equal(
        packed_projection[packed_take[:logical_count]],
        packed_support[:logical_count],
    )


def test_runtime_rectangle_stride_is_the_logical_not_physical_width():
    plan = _plan(70)
    first_pixel_on_second_logical_row = plan.logical_current_size // 2 + 1
    logical_half_width = plan.logical_current_size // 2 + 1
    physical_half_width = plan.physical_current_size // 2 + 1

    assert divmod(first_pixel_on_second_logical_row, logical_half_width) == (1, 0)
    assert divmod(first_pixel_on_second_logical_row, physical_half_width) == (0, 36)


def test_runtime_logical_bound_preserves_fine_lane_reduction_bitwise():
    plan = _plan(70)
    rng = np.random.default_rng(17)
    logical = rng.standard_normal(plan.logical_rectangle_pixels).astype(np.float32)
    padded = pad_axis(
        logical,
        0,
        plan.physical_rectangle_pixels,
        value=np.float32(1.25e5),
    )

    baseline = _relion_lane_tree_sum(logical, plan.logical_rectangle_pixels)
    stable_shape = _relion_lane_tree_sum(padded, plan.logical_rectangle_pixels)
    wrong_physical_bound = _relion_lane_tree_sum(padded, plan.physical_rectangle_pixels)

    assert baseline.view(np.uint32) == stable_shape.view(np.uint32)
    assert baseline.view(np.uint32) != wrong_physical_bound.view(np.uint32)


def test_runtime_logical_bound_preserves_bpref_issue_sequence():
    plan = _plan(84)
    logical_issues = np.arange(plan.logical_reconstruction_pixels, dtype=np.int32)
    padded_issues = pad_axis(
        logical_issues,
        0,
        plan.physical_reconstruction_pixels,
        value=-1,
    )

    np.testing.assert_array_equal(
        padded_issues[: plan.logical_reconstruction_pixels],
        logical_issues,
    )
    assert np.all(padded_issues[plan.logical_reconstruction_pixels :] == -1)


def test_packed_physical_spec_keeps_every_logical_stream_as_its_prefix():
    plan = _plan(70)
    packed = plan.packed_physical_spec()

    assert packed.n_score == plan.physical_score_pixels
    assert packed.n_recon == plan.physical_reconstruction_pixels
    assert packed.n_projection == plan.physical_projection_pixels
    for name, logical_count in (
        ("score", plan.logical_score_pixels),
        ("recon", plan.logical_reconstruction_pixels),
        ("projection", plan.logical_projection_pixels),
    ):
        np.testing.assert_array_equal(
            getattr(packed, f"{name}_indices_np")[:logical_count],
            getattr(plan.logical_spec, f"{name}_indices_np"),
        )
    np.testing.assert_array_equal(
        packed.score_projection_take_np[: plan.logical_score_pixels],
        plan.logical_spec.score_projection_take_np,
    )
    np.testing.assert_array_equal(
        packed.recon_projection_take_np[: plan.logical_reconstruction_pixels],
        plan.logical_spec.recon_projection_take_np,
    )


def test_stable_wavg_rectangle_preserves_logical_fftw_order_and_poison_tail():
    plan = _plan(70)
    logical = _make_relion_wavg_rectangle(
        _IMAGE_SHAPE,
        plan.logical_current_size,
        plan.logical_spec.recon_indices_np,
    )
    stable = _make_stable_relion_wavg_rectangle(_IMAGE_SHAPE, plan)
    logical_rectangle_count = plan.logical_rectangle_pixels
    logical_recon_count = plan.logical_reconstruction_pixels

    assert stable.centered_indices.size == plan.physical_rectangle_pixels
    np.testing.assert_array_equal(
        stable.centered_indices[:logical_rectangle_count],
        logical.centered_indices,
    )
    np.testing.assert_array_equal(
        stable.exact_positions[:logical_recon_count],
        logical.exact_positions,
    )
    np.testing.assert_array_equal(
        stable.shell_indices[:logical_rectangle_count],
        logical.shell_indices,
    )
    assert np.all(stable.shell_indices[logical_rectangle_count:] == -1)
    assert not np.intersect1d(
        stable.centered_indices[:logical_rectangle_count],
        stable.centered_indices[logical_rectangle_count:],
    ).size

    poison = np.full(plan.physical_rectangle_pixels, np.float32(np.nan))
    logical_values = np.arange(logical_rectangle_count, dtype=np.float32)
    poison[:logical_rectangle_count] = logical_values
    np.testing.assert_array_equal(poison[:logical_rectangle_count], logical_values)
    assert np.isnan(poison[logical_rectangle_count:]).all()


def test_stable_wavg_reconstruction_tail_cannot_alias_logical_rectangle():
    plan = _plan(70)
    stable = _make_stable_relion_wavg_rectangle(_IMAGE_SHAPE, plan)

    assert np.all(
        stable.exact_positions[plan.logical_reconstruction_pixels :]
        >= plan.logical_rectangle_pixels
    )
    assert np.unique(stable.exact_positions).size == stable.exact_positions.size


def test_every_gf46_stable_window_has_enough_inert_wavg_tail_capacity():
    for current_size in _GF46_CURRENT_SIZES_THROUGH_80:
        plan = _plan(current_size)
        if plan.physical_current_size == plan.logical_current_size:
            continue
        stable = _make_stable_relion_wavg_rectangle(_IMAGE_SHAPE, plan)
        logical_recon_count = plan.logical_reconstruction_pixels
        np.testing.assert_array_equal(
            stable.exact_positions[:logical_recon_count],
            _make_relion_wavg_rectangle(
                _IMAGE_SHAPE,
                current_size,
                plan.logical_spec.recon_indices_np,
            ).exact_positions,
        )
        assert np.all(
            stable.exact_positions[logical_recon_count:]
            >= plan.logical_rectangle_pixels
        )
        assert np.all(
            stable.exact_positions[logical_recon_count:]
            < plan.physical_rectangle_pixels
        )


def test_crop_relion_x_half_accumulator_excludes_physical_poison_bitwise():
    physical_shape = (11, 11, 11)
    logical_shape = (7, 7, 7)
    physical_half_width = physical_shape[2] // 2 + 1
    logical_half_width = logical_shape[2] // 2 + 1
    physical = np.full(
        (physical_shape[0], physical_shape[1], physical_half_width),
        np.uint32(0x7FC00001),
        dtype=np.uint32,
    )
    expected = np.arange(np.prod((*logical_shape[:2], logical_half_width)), dtype=np.uint32).reshape(
        (*logical_shape[:2], logical_half_width)
    )
    start = (physical_shape[0] - logical_shape[0]) // 2
    physical[
        start : start + logical_shape[0],
        start : start + logical_shape[1],
        :logical_half_width,
    ] = expected

    cropped = crop_relion_x_half_accumulator(
        physical.reshape(-1),
        physical_shape,
        logical_shape,
    )

    np.testing.assert_array_equal(cropped.reshape(expected.shape), expected)
    assert not np.any(cropped == np.uint32(0x7FC00001))


@pytest.mark.parametrize(
    ("physical_shape", "logical_shape"),
    (
        ((10, 10, 10), (7, 7, 7)),
        ((11, 11, 11), (8, 8, 8)),
        ((9, 11, 9), (7, 7, 7)),
        ((7, 7, 7), (9, 9, 9)),
    ),
)
def test_crop_relion_x_half_accumulator_rejects_unsupported_topology(
    physical_shape,
    logical_shape,
):
    with pytest.raises(ValueError, match="nested odd cubic shapes"):
        crop_relion_x_half_accumulator(
            np.zeros(1, dtype=np.float32),
            physical_shape,
            logical_shape,
        )


def test_runtime_cuda_kernels_are_separate_from_default_primitives():
    source = (
        Path(__file__).resolve().parents[3]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    powerclass_start = source.index("void relion_powerclass_spectrum_highres_f32_kernel(")
    powerclass = source[
        powerclass_start : source.index(
            "cudaError_t launch_relion_powerclass_spectrum_highres_f32(",
            powerclass_start,
        )
    ]
    assert "runtime_resolution_limit" not in powerclass
    assert "shell >= resolution_limit" in powerclass
    runtime_powerclass_start = source.index(
        "void relion_powerclass_spectrum_highres_runtime_f32_kernel("
    )
    runtime_powerclass = source[
        runtime_powerclass_start : source.index(
            "cudaError_t launch_relion_powerclass_spectrum_highres_runtime_f32(",
            runtime_powerclass_start,
        )
    ]
    assert "runtime_resolution_limit[0]" in runtime_powerclass

    wavg_start = source.index("relion_wavg_rotation_atomic_triplet_f32_kernel(")
    wavg = source[
        wavg_start : source.index(
            "cudaError_t launch_relion_wavg_rotation_atomic_triplet_add_f32(",
            wavg_start,
        )
    ]
    assert "runtime_logical_pixel_count" not in wavg
    runtime_wavg_start = source.index(
        "relion_wavg_rotation_atomic_runtime_triplet_f32_kernel("
    )
    runtime_wavg = source[
        runtime_wavg_start : source.index(
            "cudaError_t launch_relion_wavg_rotation_atomic_runtime_triplet_add_f32(",
            runtime_wavg_start,
        )
    ]
    assert "pixel < logical_pixel_count" in runtime_wavg
    assert "pixel_capacity + pixel" in runtime_wavg


def test_stable_bpref_uses_capacity_stride_but_logical_native_issue_count():
    source = (
        Path(__file__).resolve().parents[3]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()
    launcher_start = source.index(
        "cudaError_t launch_relion_vdam_mstep_fused_projector_x_half("
    )
    launcher = source[
        launcher_start : source.index(
            "__device__ __forceinline__ float relion_fine_diff2_update_f32",
            launcher_start,
        )
    ]

    assert "const int64_t image_stride = pixel_capacity;" in launcher
    assert "static_cast<unsigned>(pixel_count)" in launcher
    assert "image_real + particle * image_stride" in launcher
    assert "ctf + particle * image_stride" in launcher
    assert "minvsigma2 + particle * image_stride" in launcher
    assert "launch_relion_vdam_mstep_denominator_f32(" in launcher
    assert "pixel_count," in launcher
    assert "pixel_capacity,\n            runtime_current_size);" in launcher

    handler_start = source.rindex("ffi::Error RelionVdamMstepFusedProjectorXHalfCommon(")
    handler = source[
        handler_start : source.index(
            "XLA_FFI_DEFINE_HANDLER_SYMBOL(",
            handler_start,
        )
    ]
    assert "const int64_t pixel_count = image_h * image_w;" in handler
    assert "image_dims[1] != pixel_capacity" in handler
    assert "denominator_dims[2] != pixel_capacity" in handler


def test_stable_bpref_wrapper_packs_logical_rows_and_poison_tail(monkeypatch):
    import jax.numpy as jnp

    from recovar import cuda_backproject

    image_shape = (8, 8)
    plan = make_stable_fourier_window_shape_plan(
        image_shape,
        4,
        image_shape[0] * (image_shape[1] // 2 + 1),
        enabled=True,
    )
    rectangle = _make_stable_relion_wavg_rectangle(image_shape, plan)
    compact_count = plan.physical_reconstruction_pixels
    capacity = plan.physical_rectangle_pixels
    observed = {}

    def fake_ffi_call(target, result_types, **options):
        observed["target"] = target
        observed["result_types"] = result_types
        observed["options"] = options

        def call(*args, **attrs):
            observed["args"] = args
            observed["attrs"] = attrs
            denominator = jnp.broadcast_to(
                jnp.arange(capacity, dtype=jnp.float32),
                (1, 2, capacity),
            )
            return args[14], args[15], args[16], denominator

        return call

    monkeypatch.delenv("RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY", raising=False)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    images = jnp.arange(1, compact_count + 1, dtype=jnp.float32).astype(
        jnp.complex64
    )[None, :]
    volume_shape = (9, 9, 9)
    volume_count = volume_shape[0] * volume_shape[1] * (
        volume_shape[2] // 2 + 1
    )

    _, _, compact_denominator = (
        cuda_backproject.relion_vdam_mstep_fused_projector_x_half.__wrapped__(
            jnp.zeros(volume_count, dtype=jnp.complex64),
            jnp.zeros(volume_count, dtype=jnp.float32),
            images,
            jnp.ones(images.shape, dtype=jnp.float32),
            jnp.ones(images.shape, dtype=jnp.float32),
            jnp.ones((1, 2, 1), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.arange(compact_count, dtype=jnp.int32),
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (1, 2, 3, 3)),
            image_shape,
            volume_shape,
            3.0,
            2,
            1,
            stable_dense_positions=jnp.asarray(
                rectangle.exact_positions,
                dtype=jnp.int32,
            ),
            logical_current_size=4,
        )
    )

    assert observed["target"] == (
        cuda_backproject._TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_RUNTIME_X_HALF
    )
    assert observed["attrs"]["image_h"] == plan.physical_current_size
    assert observed["attrs"]["image_w"] == plan.physical_current_size // 2 + 1
    assert observed["attrs"]["pixel_capacity"] == capacity
    assert np.asarray(observed["args"][-1]).item() == 4
    dense_images = np.asarray(observed["args"][1])
    np.testing.assert_array_equal(
        dense_images[0, rectangle.exact_positions],
        np.asarray(images)[0],
    )
    occupied = np.zeros(capacity, dtype=bool)
    occupied[rectangle.exact_positions] = True
    np.testing.assert_array_equal(dense_images[0, ~occupied], 0.0)
    np.testing.assert_array_equal(
        np.asarray(compact_denominator)[0, 0],
        rectangle.exact_positions.astype(np.float32),
    )


def test_runtime_bpref_ffi_abi_keeps_default_static_target_separate():
    root = Path(__file__).resolve().parents[3]
    python_source = (root / "recovar" / "cuda_backproject.py").read_text()
    cuda_source = (root / "recovar" / "cuda" / "cuda_backproject.cu").read_text()

    assert "cuda_relion_vdam_mstep_fused_projector_x_half" in python_source
    assert "cuda_relion_vdam_mstep_fused_projector_runtime_x_half" in python_source
    assert "RelionVdamMstepFusedProjectorXHalfCommon(" in cuda_source
    assert "RelionVdamMstepFusedProjectorRuntimeXHalfImpl(" in cuda_source
    static_binding_start = cuda_source.index(
        "XLA_FFI_DEFINE_HANDLER_SYMBOL(\n    RelionVdamMstepFusedProjectorXHalf,"
    )
    runtime_binding_start = cuda_source.index(
        "XLA_FFI_DEFINE_HANDLER_SYMBOL(\n"
        "    RelionVdamMstepFusedProjectorRuntimeXHalf,"
    )
    static_binding = cuda_source[static_binding_start:runtime_binding_start]
    runtime_binding = cuda_source[
        runtime_binding_start : cuda_source.index(
            "ffi::Error RelionCoarseDiff2RectangularF32Impl(",
            runtime_binding_start,
        )
    ]
    assert static_binding.count(".Arg<ffi::AnyBuffer>()") == 17
    assert runtime_binding.count(".Arg<ffi::AnyBuffer>()") == 18
    runtime_handler = cuda_source[
        cuda_source.index("ffi::Error RelionVdamMstepFusedProjectorRuntimeXHalfImpl(") :
        cuda_source.index(
            "XLA_FFI_DEFINE_HANDLER_SYMBOL(\n    RelionVdamMstepFusedProjectorXHalf,"
        )
    ]
    assert "ffi::AnyBuffer logical_current_size" in runtime_handler
    assert "&logical_current_size" in runtime_handler
    common_start = cuda_source.rindex(
        "ffi::Error RelionVdamMstepFusedProjectorXHalfCommon("
    )
    common = cuda_source[
        common_start : cuda_source.index(
            "XLA_FFI_DEFINE_HANDLER_SYMBOL(\n    RelionVdamMstepFusedProjectorXHalf,",
            common_start,
        )
    ]
    assert "const ffi::AnyBuffer* runtime_current_size" in common
    assert "logical_current_size must be an S32 scalar" in common
    assert "runtime_current_size->untyped_data()" in common
    assert "runtime_current_size != nullptr &&" in cuda_source
    assert "exact_native_ptx_requested || exact_wavg_predecessor_requested" in cuda_source
    assert 'denominator[output] = nanf("")' in cuda_source


def test_runtime_bpref_lowering_and_jit_cache_ignore_logical_size(monkeypatch):
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject

    observed = []

    def fake_ffi_call(target, result_types, **options):
        def call(*args, **attrs):
            observed.append((target, options, attrs, len(args)))
            logical_current_size = args[-1]
            denominator = jnp.broadcast_to(
                logical_current_size.astype(jnp.float32),
                result_types[-1].shape,
            )
            return args[14], args[15], args[16], denominator

        return call

    monkeypatch.delenv("RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY", raising=False)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    function = cuda_backproject.relion_vdam_mstep_fused_projector_x_half
    function.clear_cache()

    image_shape = (128, 128)
    physical_current_size = 72
    capacity = physical_current_size * (physical_current_size // 2 + 1)
    volume_shape = (75, 75, 75)
    volume_count = volume_shape[0] * volume_shape[1] * (
        volume_shape[2] // 2 + 1
    )
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (1, 2, 3, 3))
    arguments = (
        jnp.zeros(volume_count, dtype=jnp.complex64),
        jnp.zeros(volume_count, dtype=jnp.float32),
        jnp.ones((1, capacity), dtype=jnp.complex64),
        jnp.ones((1, capacity), dtype=jnp.float32),
        jnp.ones((1, capacity), dtype=jnp.float32),
        jnp.ones((1, 2, 1), dtype=jnp.float32),
        jnp.zeros((1, 2), dtype=jnp.float32),
        jnp.arange(capacity, dtype=jnp.int32),
        jnp.zeros((5, 5, 5), dtype=jnp.complex64),
        rotations,
        image_shape,
        volume_shape,
        float(physical_current_size // 2),
        2,
        1,
    )
    dynamic_arguments = {
        "stable_dense_positions": jnp.arange(capacity, dtype=jnp.int32),
    }

    lowered_hlo = []
    for logical_current_size in (68, 70, 72):
        lowered = function.lower(
            *arguments,
            **dynamic_arguments,
            logical_current_size=jnp.asarray(logical_current_size, dtype=jnp.int32),
        )
        lowered_hlo.append(str(lowered.compiler_ir("stablehlo")))
        result = function(
            *arguments,
            **dynamic_arguments,
            logical_current_size=jnp.asarray(logical_current_size, dtype=jnp.int32),
        )
        jax.block_until_ready(result)
        if logical_current_size == 68:
            first_cache_size = function._cache_size()
        else:
            assert function._cache_size() == first_cache_size

    assert lowered_hlo[0] == lowered_hlo[1]
    assert observed
    assert {
        target for target, _options, _attrs, _operand_count in observed
    } == {
        cuda_backproject._TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_RUNTIME_X_HALF
    }
    assert {operand_count for _target, _options, _attrs, operand_count in observed} == {18}
    for _target, _options, attrs, _operand_count in observed:
        assert attrs["image_h"] == physical_current_size
        assert attrs["image_w"] == physical_current_size // 2 + 1
        assert attrs["pixel_capacity"] == capacity
        assert "logical_current_size" not in attrs


def test_stable_bpref_helper_rejects_noninline_projector():
    from recovar.em.dense_single_volume.local_em_engine import (
        _accumulate_relion_vdam_physical_particle_grid,
    )

    with pytest.raises(ValueError, match="requires the inline RELION projector"):
        _accumulate_relion_vdam_physical_particle_grid(
            np.zeros((1, 1), dtype=np.complex64),
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1, 1), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 1, 1), dtype=np.complex64),
            np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3),
            np.ones((1, 1), dtype=bool),
            np.zeros(1, dtype=np.complex64),
            np.zeros(1, dtype=np.float32),
            pixel_indices=np.zeros(1, dtype=np.int32),
            image_shape=(1, 1),
            volume_shape=(1, 1, 1),
            max_r=0.0,
            stable_dense_positions=np.zeros(1, dtype=np.int32),
            logical_current_size=1,
        )


def test_stable_window_engine_fails_closed_without_exact_vdam_topology():
    from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact

    with pytest.raises(ValueError, match="K=1 exact-local VDAM topology"):
        run_local_em_exact(
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=70,
            stable_fourier_window_shapes=True,
        )


@pytest.mark.parametrize("disabled_adjoint", ("disable_adjoint_y", "disable_adjoint_ctf"))
def test_stable_window_engine_requires_complete_bpref_accumulators(disabled_adjoint):
    from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact

    kwargs = {
        disabled_adjoint: True,
        "relion_exact_fine_diff2": True,
        "relion_exact_bpref_operands": True,
        "relion_wavg_sequential_cuda": True,
        "accumulate_noise": True,
        "mstep_relion_x_half": True,
        "preserve_bpref_particle_order": True,
        "relion_projector_half": object(),
    }
    with pytest.raises(ValueError, match="K=1 exact-local VDAM topology"):
        run_local_em_exact(
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=70,
            stable_fourier_window_shapes=True,
            **kwargs,
        )


def test_stable_window_engine_rejects_external_host_replay(monkeypatch):
    from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact

    monkeypatch.setenv("RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY", "/tmp/replay.so")
    with pytest.raises(ValueError, match="do not support external VDAM host replay"):
        run_local_em_exact(
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=70,
            stable_fourier_window_shapes=True,
            relion_exact_fine_diff2=True,
            relion_exact_bpref_operands=True,
            relion_wavg_sequential_cuda=True,
            accumulate_noise=True,
            mstep_relion_x_half=True,
            preserve_bpref_particle_order=True,
            relion_projector_half=object(),
        )


def test_stable_window_engine_rejects_distinct_score_and_reconstruction_sizes():
    from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact

    dataset = SimpleNamespace(
        image_shape=(128, 128),
        volume_shape=(128, 128, 128),
    )
    with pytest.raises(ValueError, match="identical score and reconstruction"):
        run_local_em_exact(
            dataset,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=70,
            reconstruction_current_size=72,
            stable_fourier_window_shapes=True,
            relion_exact_fine_diff2=True,
            relion_exact_bpref_operands=True,
            relion_wavg_sequential_cuda=True,
            accumulate_noise=True,
            mstep_relion_x_half=True,
            preserve_bpref_particle_order=True,
            relion_projector_half=object(),
        )
