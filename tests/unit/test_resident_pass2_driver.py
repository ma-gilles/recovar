"""Integration tests for the device-resident K=1 sparse pass-2 driver (T9b).

Ticket: ``em_parity_tickets_20260918/T9b_resident_pass2_integration.md``.
Design: ``em_device_resident_pass2_design_20260918.md``.

What the CPU tests cover
------------------------
Every stage of the resident driver that is CUDA-only skips on CPU: T6 scoring
(the flat-row fused-translate kernel), T7's segmented posterior, T8's flat-row
Wavg reducer and the x-half backprojection. What remains CPU-testable, and is
tested here, is:

* the production-configuration gate, one refusal per unsupported knob;
* the chunk segment offsets the segmented posterior is driven with;
* the flat-row twins of the three rectangular host helpers the driver replaces
  (``compute_local_mstep_sums``, ``_relion_wavg_atomic_triplet_terms`` and
  ``_relion_wavg_rectangle_triplet_terms``).

The whole-driver comparison against ``compute_pass2_stats_sparse_bucketed``
needs a GPU and the custom CUDA library; it is the GPU test at the end.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.local.local_backprojection import compute_local_mstep_sums
from recovar.em.sparse_pass2 import resident_pass2 as rp
from recovar.em.sparse_pass2.resident_candidates import (
    CapacityChunk,
    ResidentCandidateTables,
)
from recovar.em.sparse_pass2.sparse_pass2_wavg import (
    _relion_wavg_atomic_triplet_terms,
    _relion_wavg_rectangle_triplet_terms,
)

pytestmark = pytest.mark.unit


def _ulp32(a, b):
    a = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


def _production_gate_kwargs(**overrides):
    kwargs = dict(
        relion_x_half_mstep=True,
        relion_exact_fine_gaussian=True,
        relion_firstiter_score_mode="gaussian",
        use_float64_scoring=False,
        relion_firstiter_winner_take_all=False,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        return_score_log_z_only=False,
        accumulate_noise=True,
        mstep_subtract_ctf_projection=False,
        normalization_log_z=None,
        normalization_other_score_log_z=None,
        relion_f32_normalization_sum_weight=None,
        relion_coarse_hard_assignment=None,
        preserve_bpref_particle_order=False,
        soft_posterior_block_bpref=False,
        fine_rotations_override=np.zeros((2, 3, 3), dtype=np.float32),
        fine_rotation_parent_override=np.zeros(2, dtype=np.int32),
        n_coarse_trans=4,
        use_window=True,
        projection_cache_available=True,
        relion_wavg_atomic_scale_aa=True,
        relion_wavg_atomic_direct_noise=True,
        relion_wavg_atomic_direct_norm=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_production_configuration_is_accepted():
    rp.require_resident_production_configuration(**_production_gate_kwargs())


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"relion_x_half_mstep": False}, "x-half M-step"),
        ({"use_float64_scoring": True}, "float64 scoring"),
        ({"relion_firstiter_winner_take_all": True}, "winner-take-all"),
        ({"disable_adjoint_y": True, "disable_adjoint_ctf": True}, "score-only"),
        ({"accumulate_noise": False}, "noise statistics"),
        ({"normalization_log_z": np.zeros(3)}, "externally supplied log-Z"),
        (
            {"normalization_other_score_log_z": np.zeros(3)},
            "finite cross-class score normalization",
        ),
        ({"relion_f32_normalization_sum_weight": np.ones(3)}, "zero-oversampling"),
        ({"preserve_bpref_particle_order": True}, "per-particle BPref launches"),
        ({"fine_rotations_override": None}, "fine_rotations_override"),
        ({"n_coarse_trans": 33}, "one uint32"),
        ({"use_window": False}, "Nyquist row"),
        ({"projection_cache_available": False}, "projection cache"),
        ({"relion_wavg_atomic_scale_aa": False}, "atomic Wavg triplet"),
        ({"relion_wavg_atomic_direct_noise": False}, "direct low-shell residual"),
        ({"relion_wavg_atomic_direct_norm": True}, "stopped diagnostic"),
        ({"relion_firstiter_score_mode": "normalized_cc"}, "fine Gaussian"),
        ({"mstep_subtract_ctf_projection": True}, "projected reference"),
    ],
)
def test_gate_names_the_missing_piece(override, expected):
    with pytest.raises(NotImplementedError, match=expected):
        rp.require_resident_production_configuration(**_production_gate_kwargs(**override))


def test_gate_accepts_production_bpref_order_with_the_block_prototype():
    """``preserve_bpref_particle_order`` is production; the block prototype makes it block-wise."""

    rp.require_resident_production_configuration(
        **_production_gate_kwargs(
            preserve_bpref_particle_order=True, soft_posterior_block_bpref=True
        )
    )


def test_gate_refuses_a_diagnostic_dump(monkeypatch):
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", "/tmp/does-not-matter")
    with pytest.raises(NotImplementedError, match="RECOVAR_PASS2_DUMP_DIR"):
        rp.require_resident_production_configuration(**_production_gate_kwargs())


def _tables(row_counts):
    row_counts = np.asarray(row_counts, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
    n_rows = int(offsets[-1])
    n_images = int(row_counts.size)
    return ResidentCandidateTables(
        n_images=n_images,
        n_rows=n_rows,
        n_fine_trans=4,
        n_coarse_trans=2,
        row_offsets=offsets,
        row_image=np.repeat(np.arange(n_images, dtype=np.int32), row_counts),
        row_fine_rot=np.zeros(n_rows, dtype=np.int32),
        row_parent_local=np.zeros(n_rows, dtype=np.int32),
        row_log_prior=np.zeros(n_rows, dtype=np.float32),
        mask_mode=np.zeros(n_images, dtype=np.int8),
        parent_offsets=np.zeros(n_images + 1, dtype=np.int32),
        parent_trans_bits=np.zeros(0, dtype=np.uint32),
    )


def test_chunk_segment_offsets_cover_each_image_once_and_pad_empty():
    tables = _tables([3, 5, 2, 7])
    chunk = CapacityChunk(
        image_start=1,
        image_stop=3,
        row_start=3,
        row_stop=10,
        row_capacity=16,
        image_capacity=4,
    )
    offsets = rp._chunk_segment_offsets(tables, chunk, n_fine_trans=4)
    assert offsets.dtype == np.int32
    assert offsets.shape == (5,)
    # Image 1 owns 5 rows, image 2 owns 2; both are contiguous from cell 0.
    np.testing.assert_array_equal(offsets, np.asarray([0, 20, 28, 28, 28], dtype=np.int32))
    assert np.all(np.diff(offsets) >= 0)
    assert int(offsets[-1]) == chunk.n_valid_rows * 4
    # Padded slots are empty segments; the rows past n_valid_rows are covered
    # by no segment at all, which the handler treats as an all -inf row.
    assert int(offsets[3]) == int(offsets[4])


def test_flat_row_weighted_sums_agree_with_the_rectangular_mstep_sums():
    """The flat-row weighted sums reproduce ``compute_local_mstep_sums``.

    Both call ``compute_local_weighted_sums`` with its pinned
    ``Precision.HIGHEST``; the only change is that each flat row gathers its
    own image tile, which gives the translation contraction a singleton
    rotation axis. That reassociates a float32 GEMM, so the two are not
    bitwise equal on GPU. What is asserted is what matters: every output must
    agree with the rectangular one, and the contracted sums with a float64
    reference, to float32 relative accuracy. On an
    A100 the measured relative L2 against float64 is 1.8e-7 for the
    rectangular layout and 7.5e-8 for the flat-row layout at production
    shapes, so the flat-row layout is the slightly more accurate of the two.
    The per-element maximum relative difference reaches 5e-4, but only where
    the summed value itself has cancelled to near zero.
    """

    rng = np.random.default_rng(20260918)
    batch, n_rot, n_trans, n_pix = 8, 32, 12, 24
    probs = np.abs(rng.normal(size=(batch, n_rot, n_trans))).astype(np.float32)
    probs[0, 2, :] = 0.0  # a row with no posterior mass exercises the != 0 guard
    shifted = (
        rng.normal(size=(batch, n_trans, n_pix)) + 1j * rng.normal(size=(batch, n_trans, n_pix))
    ).astype(np.complex64)
    noise = (
        rng.normal(size=(batch, n_trans, n_pix)) + 1j * rng.normal(size=(batch, n_trans, n_pix))
    ).astype(np.complex64)
    ctf = np.abs(rng.normal(size=(batch, n_pix))).astype(np.float32)

    summed_rect, ctf_rect = compute_local_mstep_sums(
        jnp.asarray(probs),
        jnp.asarray(shifted),
        jnp.asarray(ctf),
        relion_x_half=True,
        sequential_translation_reduction=False,
    )
    masked_rect = compute_local_mstep_sums(
        jnp.asarray(probs),
        jnp.asarray(noise),
        jnp.asarray(ctf),
        relion_x_half=True,
        sequential_translation_reduction=False,
    )[0]

    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)
    summed, summed_masked, ctf_probs, probs_sum_t = rp._resident_block_weighted_sums(
        jnp.asarray(probs.reshape(batch * n_rot, n_trans)),
        jnp.asarray(row_image),
        jnp.asarray(shifted),
        jnp.asarray(noise),
        jnp.asarray(ctf),
    )

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    for flat, rect, tile in (
        (summed, summed_rect, shifted),
        (summed_masked, masked_rect, noise),
    ):
        flat = np.asarray(flat)
        rect = np.asarray(rect).reshape(batch * n_rot, n_pix)
        reference = np.einsum(
            "brt,btp->brp", probs.astype(np.float64), tile.astype(np.complex128)
        ).reshape(batch * n_rot, n_pix)
        assert rel_l2(rect, flat) < 1e-6
        assert rel_l2(reference, flat) <= rel_l2(reference, rect) * 2.0
        assert rel_l2(reference, flat) < 1e-6

    # The rotation-posterior sum reduces the same translation axis, so it is
    # reassociated by the same shape change: measured at 2.2e-7 relative on an
    # A100. The CTF sum is that value times an elementwise row, so it inherits
    # the same bound rather than being bitwise.
    np.testing.assert_allclose(
        np.asarray(probs_sum_t),
        np.asarray(jnp.sum(jnp.asarray(probs), axis=-1)).reshape(batch * n_rot),
        rtol=1e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(ctf_probs),
        np.asarray(ctf_rect).reshape(batch * n_rot, n_pix),
        rtol=1e-6,
        atol=0.0,
    )


def test_flat_row_algebraic_wavg_terms_match_the_rectangular_helper():
    """The flat-row algebraic Wavg triplet reproduces the rectangular helper.

    ``xa`` and ``aa`` are elementwise, so they must be bitwise equal. The
    ``diff2`` channel carries RELION's image-power contraction, whose float32
    einsum is shape-dependent, so it is compared as a ULP distribution and the
    measured worst case is asserted rather than assumed. See the T9b report:
    at production shapes this contraction is the one place the two layouts
    disagree, and the ``image_power + aa - 2*xa`` cancellation amplifies it.
    """

    rng = np.random.default_rng(31)
    batch, n_rot, n_trans, n_pix = 3, 4, 5, 9
    proj = (
        rng.normal(size=(batch, n_rot, n_pix)) + 1j * rng.normal(size=(batch, n_rot, n_pix))
    ).astype(np.complex64)
    proj_abs2 = np.abs(proj) ** 2
    summed = (
        rng.normal(size=(batch, n_rot, n_pix)) + 1j * rng.normal(size=(batch, n_rot, n_pix))
    ).astype(np.complex64)
    ctf_probs = np.abs(rng.normal(size=(batch, n_rot, n_pix))).astype(np.float32)
    ctf_probs[1, 0, :] = 0.0
    noise_variance = np.abs(rng.normal(size=n_pix)).astype(np.float32) + 0.1
    scale = np.abs(rng.normal(size=batch)).astype(np.float32) + 0.5
    raw_shifted = (
        rng.normal(size=(batch, n_trans, n_pix)) + 1j * rng.normal(size=(batch, n_trans, n_pix))
    ).astype(np.complex64)
    posterior = np.abs(rng.normal(size=(batch, n_rot, n_trans))).astype(np.float32)

    rect = np.asarray(
        _relion_wavg_atomic_triplet_terms(
            jnp.asarray(proj),
            jnp.asarray(proj_abs2),
            jnp.asarray(summed),
            jnp.asarray(ctf_probs),
            jnp.asarray(noise_variance),
            jnp.asarray(scale),
            jnp.asarray(raw_shifted),
            jnp.asarray(posterior),
        )
    ).reshape(batch * n_rot, n_pix, 3)
    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)
    flat = np.asarray(
        rp._resident_block_wavg_algebraic_terms(
            jnp.asarray(proj.reshape(batch * n_rot, n_pix)),
            jnp.asarray(proj_abs2.reshape(batch * n_rot, n_pix)),
            jnp.asarray(summed.reshape(batch * n_rot, n_pix)),
            jnp.asarray(ctf_probs.reshape(batch * n_rot, n_pix)),
            jnp.asarray(noise_variance),
            jnp.asarray(scale),
            jnp.asarray(raw_shifted),
            jnp.asarray(posterior.reshape(batch * n_rot, n_trans)),
            jnp.asarray(row_image),
        )
    )
    np.testing.assert_array_equal(flat[:, :, 0], rect[:, :, 0])  # XA
    np.testing.assert_array_equal(flat[:, :, 1], rect[:, :, 1])  # AA
    assert int(_ulp32(flat[:, :, 2], rect[:, :, 2]).max()) <= 4


def test_flat_row_wavg_rectangle_terms_match_the_rectangular_helper():
    """The rectangle embedding places the same terms at the same positions."""

    rng = np.random.default_rng(97)
    batch, n_rot, n_trans, n_rect, n_exact = 2, 3, 4, 10, 6
    exact_positions = np.sort(
        rng.choice(n_rect, size=n_exact, replace=False).astype(np.int32)
    )
    exact_terms = rng.normal(size=(batch, n_rot, n_exact, 3)).astype(np.float32)
    raw_rect = (
        rng.normal(size=(batch, n_trans, n_rect)) + 1j * rng.normal(size=(batch, n_trans, n_rect))
    ).astype(np.complex64)
    posterior = np.abs(rng.normal(size=(batch, n_rot, n_trans))).astype(np.float32)

    rect = np.asarray(
        _relion_wavg_rectangle_triplet_terms(
            jnp.asarray(exact_terms),
            jnp.asarray(raw_rect),
            jnp.asarray(posterior),
            jnp.asarray(exact_positions),
        )
    ).reshape(batch * n_rot, n_rect, 3)
    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)
    flat = np.asarray(
        rp._resident_block_wavg_rectangle_terms(
            jnp.asarray(exact_terms.reshape(batch * n_rot, n_exact, 3)),
            jnp.asarray(raw_rect),
            jnp.asarray(posterior.reshape(batch * n_rot, n_trans)),
            jnp.asarray(row_image),
            jnp.asarray(exact_positions),
        )
    )
    # The exact positions carry the supplied terms verbatim in both layouts.
    np.testing.assert_array_equal(flat[:, exact_positions, :], rect[:, exact_positions, :])
    other = np.setdiff1d(np.arange(n_rect), exact_positions)
    np.testing.assert_array_equal(flat[:, other, 0], rect[:, other, 0])
    np.testing.assert_array_equal(flat[:, other, 1], rect[:, other, 1])
    assert int(_ulp32(flat[:, other, 2], rect[:, other, 2]).max()) <= 4


@pytest.mark.parametrize(
    "batch,n_rot,n_trans,n_rect,n_exact",
    [(2, 3, 4, 10, 6), (5, 64, 84, 97, 61), (1, 17, 3, 8, 8), (7, 1, 9, 33, 2)],
)
def test_wavg_power_per_image_is_bitwise_against_the_per_row_path(
    batch, n_rot, n_trans, n_rect, n_exact
):
    """P4-G phase 2: squaring before the gather must change no bit.

    ``|x|^2`` is elementwise, so squaring the chunk rectangle once per image and
    gathering the float32 result is the same value as gathering the complex
    rectangle and squaring once per row. The contraction that follows sees the
    same shapes and the same translation axis, so the whole triplet is bitwise.
    Shapes cover the production ratio (many rows over few images), one row per
    image, a single rotation, and an all-exact rectangle.
    """

    rng = np.random.default_rng(4207 + n_rect)
    exact_positions = np.sort(
        rng.choice(n_rect, size=n_exact, replace=False).astype(np.int32)
    )
    rows = batch * n_rot
    exact_terms = rng.normal(size=(rows, n_exact, 3)).astype(np.float32)
    raw_rect = (
        rng.normal(size=(batch, n_trans, n_rect))
        + 1j * rng.normal(size=(batch, n_trans, n_rect))
    ).astype(np.complex64)
    posterior = np.abs(rng.normal(size=(rows, n_trans))).astype(np.float32)
    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)

    def run(power_per_image):
        return np.asarray(
            rp._resident_block_wavg_rectangle_terms(
                jnp.asarray(exact_terms),
                jnp.asarray(raw_rect),
                jnp.asarray(posterior),
                jnp.asarray(row_image),
                jnp.asarray(exact_positions),
                power_per_image=power_per_image,
            )
        )

    per_row, per_image = run(False), run(True)
    np.testing.assert_array_equal(
        per_image.view(np.uint32), per_row.view(np.uint32)
    )


def test_wavg_power_per_image_flag_defaults_on_and_reaches_the_spec(monkeypatch):
    """The flag is the default and the block body reads it from the spec.

    P4-G shipped the hoist opt-in and measured the two forms bitwise on GPU at
    production shapes; it is the default from the P4-F/P4-G merge, with ``0``
    restoring the per-row square as the oracle.
    """

    monkeypatch.delenv(rp._WAVG_POWER_PER_IMAGE_ENV, raising=False)
    assert rp._wavg_power_per_image_enabled() is True
    monkeypatch.setenv(rp._WAVG_POWER_PER_IMAGE_ENV, "0")
    assert rp._wavg_power_per_image_enabled() is False
    monkeypatch.setenv(rp._WAVG_POWER_PER_IMAGE_ENV, "1")
    assert rp._wavg_power_per_image_enabled() is True
    import dataclasses

    fields = {f.name for f in dataclasses.fields(rp._ChunkProgramSpec)}
    assert "wavg_power_per_image" in fields


def test_wavg_shifted_power_commutes_with_a_row_gather():
    """The identity the hoist rests on, stated on its own.

    ``square(gather(x)) == gather(square(x))`` in float32, including the
    optimization barrier that keeps the two products from contracting.
    """

    from recovar.em.sparse_pass2.sparse_pass2_wavg import _relion_wavg_shifted_power

    rng = np.random.default_rng(5150)
    rect = (
        rng.normal(size=(6, 11, 29)) + 1j * rng.normal(size=(6, 11, 29))
    ).astype(np.complex64)
    take = rng.integers(0, 6, size=97).astype(np.int32)
    gather_then_square = np.asarray(
        _relion_wavg_shifted_power(jnp.asarray(rect)[jnp.asarray(take)])
    )
    square_then_gather = np.asarray(
        _relion_wavg_shifted_power(jnp.asarray(rect))[jnp.asarray(take)]
    )
    np.testing.assert_array_equal(
        square_then_gather.view(np.uint32), gather_then_square.view(np.uint32)
    )


def test_default_row_ladder_is_dense_enough_to_bound_padding():
    """Consecutive classes must not more than double, so occupancy floors at 50%.

    The chunker rounds a chunk's row count up to the next class, so the gap
    between classes is the worst-case padding. The design's (8192, 32768,
    131072) left hp3 iteration 0 at 49% occupancy, which the pixel-axis
    gather and the per-cell posterior pay for.
    """

    ladder = rp._DEFAULT_ROW_CAPACITY_LADDER
    assert list(ladder) == sorted(ladder)
    for small, large in zip(ladder, ladder[1:]):
        assert large <= 2 * small, (small, large)


def test_mstep_block_rows_divides_every_row_capacity():
    ladder = rp._DEFAULT_ROW_CAPACITY_LADDER
    block = rp._resolve_mstep_block_rows(
        n_recon_pixels=4324, max_block_bytes=513124859, row_capacity_ladder=ladder
    )
    assert block > 0 and block & (block - 1) == 0
    assert all(capacity % block == 0 for capacity in ladder)


def test_image_capacity_ladder_is_capped_by_the_translation_tile_budget():
    ladder = rp._cap_image_capacity_ladder(
        (32, 128, 512), n_fine_trans=100, n_recon_pixels=4324, max_tile_bytes=1_197_291_339
    )
    assert ladder and list(ladder) == sorted(ladder)
    assert max(ladder) <= 512
    # A budget that fits nothing still leaves the smallest class so a plan exists.
    assert rp._cap_image_capacity_ladder(
        (32, 128, 512), n_fine_trans=100, n_recon_pixels=4324, max_tile_bytes=1
    ) == (32,)


def test_driver_is_registered_behind_the_flag_only(monkeypatch):
    from recovar.em.helpers import oversampling

    monkeypatch.delenv(rp.RESIDENT_PASS2_ENV, raising=False)
    assert not rp.resident_pass2_requested()
    monkeypatch.setenv(rp.RESIDENT_PASS2_ENV, "1")
    assert rp.resident_pass2_requested()
    source = oversampling.compute_pass2_stats_sparse.__doc__ or ""
    del source
    import inspect

    dispatch = inspect.getsource(oversampling.compute_pass2_stats_sparse)
    assert "resident_pass2_requested()" in dispatch
    assert "compute_pass2_stats_resident" in dispatch


def test_signature_matches_the_compact_engine():
    import inspect

    from recovar.em.sparse_pass2.sparse_pass2_bucketed import (
        compute_pass2_stats_sparse_bucketed,
    )

    compact = inspect.signature(compute_pass2_stats_sparse_bucketed).parameters
    resident = inspect.signature(rp.compute_pass2_stats_resident).parameters
    assert list(compact) == list(resident)
    for name in compact:
        assert compact[name].default == resident[name].default, name
        assert compact[name].kind == resident[name].kind, name


# ---------------------------------------------------------------------------
# GPU: the whole driver against the compact engine
# ---------------------------------------------------------------------------


def _gpu_available():
    """Whether this process can run every CUDA-only resident stage."""

    if jax.default_backend() != "gpu":
        return False
    from recovar import cuda_backproject

    return bool(
        cuda_backproject.custom_cuda_requested()
        and cuda_backproject.sparse_pass2_segmented_supported()
        and cuda_backproject.relion_wavg_sequential_runtime_flat_rows_triplet_f32_supported()
        and cuda_backproject.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported()
    )


requires_resident_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason=(
        "the resident driver's scoring, segmented posterior, flat-row Wavg and "
        "x-half backprojection stages are all CUDA FFI targets"
    ),
)


def _z_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _driver_fixture_args(seed=20260918):
    """A small K=1 pass in the production configuration both engines accept.

    The fixture uses the 8x8 ``MockDataset`` of the bucketed parity tests with
    a current-size window (so the score window excludes the ``ky=-N/2`` Nyquist
    row), RELION's x-half M-step, the float32 fine posterior, the atomic Wavg
    triplet and one scale-correction group. It does not use RELION's exact
    BPref operands, which need a STAR-backed dataset, so the Wavg triplet takes
    the algebraic branch here; the production path takes the sequential CUDA
    branch, which the matched Slurm pair covers.
    """

    from helpers.em_arrays import _hermitian_volume
    from test_sparse_pass2_bucketed_parity import IMAGE_SHAPE, IMAGE_SIZE, VOLUME_SHAPE, MockDataset

    from recovar.em.sampling import rotation_grid_size

    nside_level = 1
    n_coarse_rot = rotation_grid_size(nside_level)
    n_images = 12
    children = 2
    rng = np.random.default_rng(seed)

    fine_rotations = np.stack(
        [_z_rotation(0.031 * k) for k in range(n_coarse_rot * children)]
    ).astype(np.float32)
    fine_parent = np.repeat(np.arange(n_coarse_rot, dtype=np.int32), children)
    translations = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    n_coarse_trans = translations.shape[0]
    fine_translations = np.concatenate([translations, translations + 0.5]).astype(np.float32)
    fine_translation_parent = np.concatenate(
        [np.arange(n_coarse_trans), np.arange(n_coarse_trans)]
    ).astype(np.int32)

    total = n_coarse_rot * n_coarse_trans
    samples = [None]
    for _ in range(1, n_images):
        count = int(rng.integers(1, total))
        samples.append(np.sort(rng.choice(total, size=count, replace=False).astype(np.int32)))

    n_shells = IMAGE_SHAPE[0] // 2 + 1
    return dict(
        experiment_dataset=MockDataset(n_images=n_images, seed=11),
        volume=_hermitian_volume(VOLUME_SHAPE, seed=17),
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32) * 0.8,
        translations=translations,
        significant_sample_indices=samples,
        nside_level=nside_level,
        disc_type="linear_interp",
        oversampling_order=0,
        current_size=6,
        translation_step=1.0,
        rotation_log_prior=rng.normal(scale=0.1, size=n_coarse_rot).astype(np.float32),
        score_with_masked_images=False,
        return_stats=True,
        translation_log_prior=rng.normal(
            scale=0.05, size=(n_images, n_coarse_trans)
        ).astype(np.float32),
        accumulate_noise=True,
        half_spectrum_scoring=True,
        projection_padding_factor=2,
        reconstruction_padding_factor=2,
        image_corrections=None,
        scale_corrections=None,
        image_pre_shifts=None,
        use_float64_scoring=False,
        random_perturbation=0.0,
        group_ids=np.zeros(n_images, dtype=np.int32),
        scale_correction_group_count=1,
        scale_correction_data_vs_prior=np.full(n_shells, 5.0, dtype=np.float64),
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
        relion_x_half_mstep=True,
        relion_fine_mstep_prune=True,
        relion_f32_fine_posterior=True,
        relion_exact_fine_gaussian=True,
        # Production K=1 runs the exact rectangular CUDA reduction (k_class.py
        # sets this whenever the custom library is available). Without it the
        # compact engine takes the XLA 256-lane emulation, which differs from
        # every CUDA scorer by a few ULP and would confound the comparison.
        relion_fine_diff2_fused_ffi=True,
        preserve_bpref_particle_order=True,
        source_faithful_spectrum_norm=False,
        return_score_log_z=True,
        # Production K=1 reaches the M-step call through
        # k_class.py::_run_sparse_k_class_adaptive_pass2, which always supplies
        # the other classes' log-Z. At K=1 there are no other classes, so the
        # vector is all -inf; carry it here so the fixture exercises the same
        # branch the matched Slurm pairs do.
        normalization_other_score_log_z=np.full(n_images, -np.inf, dtype=np.float64),
        normalization_score_mode="gaussian",
        adaptive_fraction=0.999,
    )


@pytest.fixture
def _resident_production_env(monkeypatch):
    monkeypatch.setenv("RECOVAR_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF", "1")
    monkeypatch.setenv("RECOVAR_RELION_WAVG_ATOMIC_SCALE_AA", "1")
    monkeypatch.setenv("RECOVAR_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES", "256,1024,4096")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES", "4,16,64")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS", "128")


@requires_resident_gpu
def test_resident_driver_matches_the_compact_engine(_resident_production_env):
    """Whole-driver comparison against ``compute_pass2_stats_sparse_bucketed``.

    Discrete state (pose, translation, rotation id) and every per-image score
    field must be bitwise identical: the scores come from the same CUDA body
    and T7's posterior is bitwise against the rectangular handler. The maps and
    the noise/scale accumulators change reduction order, which the user waived
    on 2026-09-18, so they are bounded by relative L2 at the values this
    fixture measured.
    """

    from recovar.em.sparse_pass2.sparse_pass2_bucketed import (
        compute_pass2_stats_sparse_bucketed,
    )

    args = _driver_fixture_args()
    compact = compute_pass2_stats_sparse_bucketed(**args)
    resident = rp.compute_pass2_stats_resident(**args)

    np.testing.assert_array_equal(compact.hard_assignment, resident.hard_assignment)
    np.testing.assert_array_equal(compact.best_rotation_indices, resident.best_rotation_indices)
    np.testing.assert_array_equal(compact.best_rotations, resident.best_rotations)
    np.testing.assert_array_equal(compact.best_translations, resident.best_translations)
    np.testing.assert_array_equal(
        np.asarray(compact.score_log_z), np.asarray(resident.score_log_z)
    )
    for field in (
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "rotation_posterior_sums",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(compact.relion_stats, field)),
            np.asarray(getattr(resident.relion_stats, field)),
            err_msg=field,
        )

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # Float32 BPref atomics and the blocked pixel-axis reductions; measured at
    # 1.2e-7 on this fixture, against a compact-vs-compact repeat band of 4e-8.
    assert rel_l2(compact.Ft_y, resident.Ft_y) < 1e-6
    assert rel_l2(compact.Ft_ctf, resident.Ft_ctf) < 1e-6
    # The Wavg diff2 residual cancels most of its magnitude, so the
    # shape-dependent float32 image-power contraction shows up here at 6.2e-6.
    assert rel_l2(
        compact.noise_stats.wsum_sigma2_noise, resident.noise_stats.wsum_sigma2_noise
    ) < 1e-4
    for field in (
        "wsum_img_power",
        "wsum_norm_correction",
        "wsum_scale_correction_xa",
        "wsum_scale_correction_aa",
    ):
        assert rel_l2(
            getattr(compact.noise_stats, field), getattr(resident.noise_stats, field)
        ) < 1e-6, field
    # No translation prior centers in this fixture, so the offset is exactly
    # zero on both paths.
    assert float(compact.noise_stats.wsum_sigma2_offset) == float(
        resident.noise_stats.wsum_sigma2_offset
    )
    # The support mass is a float64 sum over a reassociated float32 posterior
    # reduction; it came out bitwise on one A100 and 1.0e-8 relative on
    # another, so the bound is relative, not equality.
    assert abs(
        float(compact.noise_stats.sumw) - float(resident.noise_stats.sumw)
    ) <= 1e-6 * abs(float(compact.noise_stats.sumw))


@requires_resident_gpu
def test_resident_driver_repeats_itself(_resident_production_env):
    """The resident driver's own repeat band, the reference for the table above."""

    args = _driver_fixture_args()
    first = rp.compute_pass2_stats_resident(**args)
    second = rp.compute_pass2_stats_resident(**args)
    np.testing.assert_array_equal(first.hard_assignment, second.hard_assignment)
    # The statistics whose reductions are ordered are bit-reproducible.
    for field in ("wsum_sigma2_noise", "wsum_norm_correction"):
        np.testing.assert_array_equal(
            np.asarray(getattr(first.noise_stats, field)),
            np.asarray(getattr(second.noise_stats, field)),
            err_msg=field,
        )

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # Two reductions are not: the float32 BPref atomics, and the CUDA shell
    # binning behind the unweighted high image-power shell. Measured repeat
    # band on an A100: 1.5e-8 for the maps, 1.4e-8 for the image power.
    assert rel_l2(first.Ft_y, second.Ft_y) < 1e-7
    assert rel_l2(first.noise_stats.wsum_img_power, second.noise_stats.wsum_img_power) < 1e-7


@requires_resident_gpu
def test_glue_programs_match_the_loose_dispatch(_resident_production_env, monkeypatch):
    """P3-A: where the chunk loop's JIT boundary sits changes no output.

    ``RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT`` picks between one program per
    M-step block and the loose sequence of eager operations the per-stage path
    used before P3-A. The stage bodies are the same functions in both
    settings, so the discrete state and the ordered statistics must be
    bitwise; the two reductions that are not bit-reproducible even between two
    identical runs -- the float32 BPref atomics and the CUDA shell binning --
    are held to the repeat band
    :func:`test_resident_driver_repeats_itself` measures.
    """

    from recovar.em.helpers.deterministic_reduce import (
        deterministic_reductions_enabled,
    )

    args = _driver_fixture_args()
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT", "0")
    loose = rp.compute_pass2_stats_resident(**args)
    # The loose path's own repeat, in this process: the self-repeat band the
    # racing accumulators below are read against. It is measured, not assumed,
    # because the float32 BPref atomics and the CUDA shell binning do not
    # reproduce between two identical calls.
    loose_repeat = rp.compute_pass2_stats_resident(**args)
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT", "1")
    programs = rp.compute_pass2_stats_resident(**args)

    np.testing.assert_array_equal(loose.hard_assignment, programs.hard_assignment)
    np.testing.assert_array_equal(
        loose.best_rotation_indices, programs.best_rotation_indices
    )
    np.testing.assert_array_equal(loose.best_rotations, programs.best_rotations)
    np.testing.assert_array_equal(loose.best_translations, programs.best_translations)
    np.testing.assert_array_equal(
        np.asarray(loose.score_log_z), np.asarray(programs.score_log_z)
    )
    for field in (
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "rotation_posterior_sums",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(loose.relion_stats, field)),
            np.asarray(getattr(programs.relion_stats, field)),
            err_msg=field,
        )
    # ``wsum_sigma2_noise`` and ``wsum_norm_correction`` are shell sums fed by
    # the CUDA binning scatter, so they are bitwise only when the order is
    # pinned. Under ``RECOVAR_EM_DETERMINISTIC_REDUCTIONS=1`` that is the
    # assertion; otherwise each is held to the loose path's own repeat spread
    # measured just above, never to a fixed tolerance. Holding them to bitwise
    # without the opt-in is what made this test fail in long GPU sessions at
    # 3.5e-08 relative on one entry of twelve while passing in isolation.
    for field in ("wsum_sigma2_noise", "wsum_norm_correction"):
        candidate = np.asarray(getattr(programs.noise_stats, field), dtype=np.float64)
        reference = np.asarray(getattr(loose.noise_stats, field), dtype=np.float64)
        if deterministic_reductions_enabled():
            np.testing.assert_array_equal(candidate, reference, err_msg=field)
            continue
        repeat = np.asarray(getattr(loose_repeat.noise_stats, field), dtype=np.float64)
        band = np.abs(repeat - reference)
        # One repeat is one sample of a racing sum and can read exactly zero,
        # so the band is that sample or one float32 ulp of the value, whichever
        # is larger. Both are measurements of this reduction, not a tolerance
        # chosen to pass.
        ulp = np.spacing(np.abs(reference).astype(np.float32)).astype(np.float64)
        assert np.all(np.abs(candidate - reference) <= np.maximum(band, ulp)), (
            f"{field}: max deviation {np.abs(candidate - reference).max()} "
            f"exceeds the repeat band {band.max()} and one float32 ulp "
            f"{ulp.max()}"
        )

    def rel_l2(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # The maps ride the same racing BPref atomics; their band is the loose
    # path's own repeat, measured in this process.
    for name, candidate, reference, repeat in (
        ("Ft_y", programs.Ft_y, loose.Ft_y, loose_repeat.Ft_y),
        ("Ft_ctf", programs.Ft_ctf, loose.Ft_ctf, loose_repeat.Ft_ctf),
        (
            "wsum_img_power",
            programs.noise_stats.wsum_img_power,
            loose.noise_stats.wsum_img_power,
            loose_repeat.noise_stats.wsum_img_power,
        ),
    ):
        if deterministic_reductions_enabled():
            np.testing.assert_array_equal(
                np.asarray(candidate), np.asarray(reference), err_msg=name
            )
            continue
        band = rel_l2(reference, repeat)
        observed = rel_l2(reference, candidate)
        assert observed <= max(band, float(np.finfo(np.float32).eps)), (
            f"{name}: {observed} exceeds the repeat band {band}"
        )


@requires_resident_gpu
def test_degenerate_cross_class_normalizer_is_a_no_op_for_the_compact_engine(
    _resident_production_env,
):
    """The all -inf normalizer the K=1 route supplies changes no compact output.

    This is the premise the gate's relaxation rests on, so it is measured
    rather than argued: running the compact engine with and without the
    degenerate vector must give the same maps, poses and per-image statistics.
    """

    from recovar.em.sparse_pass2.sparse_pass2_bucketed import (
        compute_pass2_stats_sparse_bucketed,
    )

    with_norm = _driver_fixture_args()
    without_norm = _driver_fixture_args()
    without_norm.pop("normalization_other_score_log_z")
    without_norm.pop("normalization_score_mode")
    assert with_norm["normalization_other_score_log_z"] is not None

    a = compute_pass2_stats_sparse_bucketed(**with_norm)
    b = compute_pass2_stats_sparse_bucketed(**without_norm)
    np.testing.assert_array_equal(a.hard_assignment, b.hard_assignment)
    np.testing.assert_array_equal(a.best_rotation_indices, b.best_rotation_indices)
    np.testing.assert_array_equal(np.asarray(a.score_log_z), np.asarray(b.score_log_z))
    for field in (
        "log_evidence_per_image",
        "best_log_score_per_image",
        "max_posterior_per_image",
        "rotation_posterior_sums",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(a.relion_stats, field)),
            np.asarray(getattr(b.relion_stats, field)),
            err_msg=field,
        )


@requires_resident_gpu
def test_gate_refuses_a_finite_cross_class_normalizer(_resident_production_env):
    args = _driver_fixture_args()
    args["normalization_other_score_log_z"] = np.zeros(
        args["experiment_dataset"].n_units, dtype=np.float64
    )
    with pytest.raises(NotImplementedError, match="finite cross-class"):
        rp.compute_pass2_stats_resident(**args)


@requires_resident_gpu
def test_resident_driver_refuses_an_unsupported_pass(_resident_production_env):
    """A configuration outside the gate raises instead of silently falling back."""

    args = _driver_fixture_args()
    args["relion_x_half_mstep"] = False
    with pytest.raises(NotImplementedError, match="x-half M-step"):
        rp.compute_pass2_stats_resident(**args)


def test_driver_reads_the_compact_engine_behaviour_gates():
    """The driver must not turn on behaviour-changing knobs the engine gates.

    The 10k/256 matched pair diverged on 9987 of 10000 images because the
    driver narrowed ``Projector::data`` to complex64 unconditionally while the
    compact engine keeps that behind an opt-in flag, whose own docstring says
    narrowing changes float32 projection arithmetic. The fixture datasets have
    no RELION projector, so only a source check catches this class of bug.
    """

    import inspect

    from recovar.em.sparse_pass2 import sparse_pass2_bucketed

    driver = inspect.getsource(rp.compute_pass2_stats_resident)
    assert "_pass2_projector_complex64_enabled()" in driver
    narrowing = driver[driver.index("complex64 slab") if "complex64 slab" in driver else 0:]
    del narrowing
    # The cast must be guarded by the gate, not by dtype alone.
    cast = driver.index("astype(jnp.complex64)")
    guard = driver.rindex("_pass2_projector_complex64_enabled()", 0, cast)
    assert cast - guard < 400, "the projector cast is not inside the gated branch"
    # The gate itself still defaults off in the engine that owns it.
    assert "default=False" in inspect.getsource(
        sparse_pass2_bucketed._pass2_projector_complex64_enabled
    )


def test_chunk_operands_are_padded_on_the_host():
    """Per-chunk shapes must not depend on a chunk's occupancy.

    Every traced program is keyed on its operand shapes, so an operand sized
    by the chunk's valid image count compiles once per distinct occupancy. The
    early-state cold arm traced 9352 programs in one iteration for this
    reason. The batch handed to the preparation is padded on the host instead.
    """

    import inspect

    source = inspect.getsource(rp._prepare_chunk_reconstruction_operands)
    assert "_pad_batch_to_capacity(batch_data, image_capacity)" in source
    assert "_pad_batch_to_capacity(ctf_params, image_capacity)" in source
    assert "_zero_padded_images" in source
    # jnp.pad with an occupancy-dependent width is what this replaced.
    assert "jnp.pad" not in source
    driver = inspect.getsource(rp)
    assert "def _pad_image_axis" not in driver


def test_pad_batch_to_capacity_repeats_the_first_row():
    batch = np.arange(12, dtype=np.float32).reshape(3, 4)
    padded = rp._pad_batch_to_capacity(batch, 5)
    assert padded.shape == (5, 4)
    np.testing.assert_array_equal(padded[:3], batch)
    np.testing.assert_array_equal(padded[3], batch[0])
    np.testing.assert_array_equal(padded[4], batch[0])
    np.testing.assert_array_equal(rp._pad_batch_to_capacity(batch, 3), batch)
    with pytest.raises(ValueError, match="cannot pad"):
        rp._pad_batch_to_capacity(batch, 2)


def test_zero_padded_images_clears_only_the_padded_slots():
    values = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    valid = np.asarray([True, True, False, False])
    out = np.asarray(rp._zero_padded_images(jnp.asarray(values), jnp.asarray(valid)))
    np.testing.assert_array_equal(out[:2], values[:2])
    np.testing.assert_array_equal(out[2:], np.zeros_like(values[2:]))


def test_reorder_permutation_inverts_a_shuffled_fetch():
    requested = np.asarray([7, 3, 9, 1])
    fetched = np.asarray([1, 9, 7, 3])
    order = rp._reorder_permutation(fetched, requested, capacity=6)
    assert order.shape == (6,)
    # Position p of the table must read fetched slot order[p].
    np.testing.assert_array_equal(fetched[order[:4]], requested)
    with pytest.raises(ValueError, match="did not return every requested image"):
        rp._reorder_permutation(np.asarray([1, 9, 7, 7]), requested, capacity=6)


@pytest.mark.parametrize(
    ("mode", "winner", "expected"),
    [
        ("normalized_cc", False, "normalized-CC scoring"),
        ("gaussian", True, "winner-take-all"),
        ("normalized_cc", True, "normalized-CC scoring"),
    ],
)
def test_out_of_scope_scoring_modes_are_named_not_raised(mode, winner, expected):
    """The firstiter_cc route is out of scope, not a configuration mismatch.

    An end-to-end run at d6b3d3d57 stopped at iteration 1 because the driver
    raised on RELION's --firstiter_cc pass, which scores with normalized
    cross-correlation and takes the winner outright. That is a separate pass-2
    route the driver never covered, so the caller sends it to the compact
    engine; a mismatch inside the covered path still raises.
    """

    reason = rp.resident_pass2_out_of_scope_reason(
        relion_firstiter_score_mode=mode, relion_firstiter_winner_take_all=winner
    )
    assert reason is not None and expected in reason


def test_the_production_gaussian_pass_is_in_scope():
    assert (
        rp.resident_pass2_out_of_scope_reason(
            relion_firstiter_score_mode="gaussian", relion_firstiter_winner_take_all=False
        )
        is None
    )


def test_dispatcher_routes_out_of_scope_passes_to_the_compact_engine():
    """Selection, not the gate, decides which engine an out-of-scope pass uses."""

    import inspect

    from recovar.em.helpers import oversampling

    dispatch = inspect.getsource(oversampling.compute_pass2_stats_sparse)
    assert "resident_pass2_out_of_scope_reason(" in dispatch
    # The default must be the compact engine, with the resident driver chosen
    # only when the pass is both requested and in scope.
    assert "sparse_pass2_impl = compute_pass2_stats_sparse_bucketed" in dispatch
    assert "if out_of_scope is None:" in dispatch
    assert "does not cover %s" in dispatch


def test_gate_still_raises_for_in_scope_mismatches():
    """Everything that is not an out-of-scope scoring mode still raises."""

    with pytest.raises(NotImplementedError, match="x-half M-step"):
        rp.require_resident_production_configuration(
            **_production_gate_kwargs(relion_x_half_mstep=False)
        )
    with pytest.raises(NotImplementedError, match="float64 scoring"):
        rp.require_resident_production_configuration(
            **_production_gate_kwargs(use_float64_scoring=True)
        )
