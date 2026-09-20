"""The device-resident local fine pass 2 against the exact local engine (T12).

Ticket: ``em_parity_tickets_20260918/T12_resident_local_search.md``.

The two engines are compared through the production dispatch
(``local_search_iteration._run_local_search_iteration``) with the flag off and
on, so the test covers the wiring as well as the driver.

What is compared, and against what
----------------------------------
The resident stages implement the *compact* K=1 pass-2 arithmetic, which is a
different factorization of the same likelihood from the exact local engine's
(module docstring of ``resident_local_pass2`` lists the three differences).
Discrete state is therefore compared for agreement, and the continuous fields
are compared against the engine's own repeat band where one exists and against
explicit, measured bounds otherwise. Nothing here asserts bitwise equality
between the two engines; that would be false by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from helpers.em_arrays import _hermitian_volume
from test_sparse_pass2_bucketed_parity import IMAGE_SHAPE, VOLUME_SHAPE, MockDataset

import recovar.core.fourier_transform_utils as ftu
from recovar.em.local import local_search_iteration
from recovar.em.local.local_layout import (
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
)
from recovar.em.relion.relion_projector_setup import reference_to_relion_projector_half_maps
from recovar.em.sampling import build_local_search_grid_metadata
from recovar.em.sparse_pass2 import resident_local_pass2 as rlp
from recovar.em.sparse_pass2 import resident_pass2 as rp

pytestmark = pytest.mark.unit

PARENT_ORDER = 1
OVERSAMPLING = 1
N_IMAGES = 8
CURRENT_SIZE = 6


def _gpu_available():
    if jax.default_backend() != "gpu":
        return False
    from recovar import cuda_backproject

    return bool(
        cuda_backproject.custom_cuda_requested()
        and cuda_backproject.sparse_pass2_segmented_supported()
        and cuda_backproject.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported()
    )


requires_resident_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="every resident stage is a CUDA FFI target",
)


def _prior_eulers(n_images, seed):
    rng = np.random.default_rng(seed)
    eulers = np.zeros((n_images, 3), dtype=np.float64)
    eulers[:, 0] = rng.uniform(0.0, 360.0, size=n_images)
    eulers[:, 1] = rng.uniform(20.0, 160.0, size=n_images)
    eulers[:, 2] = rng.uniform(0.0, 360.0, size=n_images)
    return eulers


def _pass2_layout(seed=20260919, full_support=False):
    translations = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    parent = build_local_hypothesis_layout(
        _prior_eulers(N_IMAGES, seed),
        None,
        0.35,
        0.35,
        PARENT_ORDER,
        translations,
        np.zeros((N_IMAGES, 2), dtype=np.float32),
        3.0,
        None,
        1.0,
        grid_metadata=build_local_search_grid_metadata(PARENT_ORDER),
        translation_prior_reference_translations=translations,
        dtype=np.float32,
    )
    n_coarse_trans = int(translations.shape[0])
    rng = np.random.default_rng(seed + 1)
    samples = []
    for image in range(parent.n_images):
        if full_support:
            # ``None`` is how RELION's full-parent diagnostic spells "every
            # (rotation, translation) survives"; the layout then carries no
            # mask at all, and the driver must not materialize one.
            samples.append(None)
            continue
        start = int(parent.rotation_offsets[image])
        stop = int(parent.rotation_offsets[image + 1])
        parent_ids = np.asarray(parent.rotation_ids_flat[start:stop], dtype=np.int64)
        pairs = (parent_ids[:, None] * n_coarse_trans + np.arange(n_coarse_trans)).reshape(-1)
        keep = rng.choice(pairs, size=max(2, pairs.size // 3), replace=False)
        samples.append(np.sort(keep.astype(np.int64)))
    layout = build_local_adaptive_pass2_hypothesis_layout(
        parent,
        samples,
        PARENT_ORDER,
        oversampling_order=OVERSAMPLING,
        random_perturbation=0.0,
        dtype=np.float32,
    )
    return layout, translations


def _relion_projector(volume_real, current_size):
    halves, r_max = reference_to_relion_projector_half_maps(
        np.asarray(volume_real, dtype=np.float64)[None],
        current_size=int(current_size),
        padding_factor=1,
        interpolator=1,
        projector_setup_backend="jax",
    )
    return jnp.asarray(halves[0], dtype=jnp.complex64), int(r_max)


def _case(seed=20260919, full_support=False):
    """Dataset, volume, RELION projector and pass-2 layout for one half."""

    dataset = MockDataset(n_images=N_IMAGES, seed=seed % 2**31)
    volume_ft = _hermitian_volume(VOLUME_SHAPE, seed=17)
    volume_real = np.asarray(
        ftu.get_idft3(np.asarray(volume_ft).reshape(VOLUME_SHAPE)).real, dtype=np.float64
    )
    projector_half, r_max = _relion_projector(volume_real, IMAGE_SHAPE[0])
    layout, translations = _pass2_layout(seed, full_support=full_support)
    n_shells = IMAGE_SHAPE[0] // 2 + 1
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    return dict(
        dataset=dataset,
        volume=jnp.asarray(volume_ft),
        # High enough that the posterior spreads over several candidates: at a
        # low noise level every image has Pmax == 1 and the comparison would
        # only exercise the winner, not the posterior.
        noise_variance=jnp.ones(IMAGE_SHAPE[0] * IMAGE_SHAPE[1], dtype=jnp.float32) * 200.0,
        projector_half=projector_half,
        r_max=r_max,
        layout=layout,
        translations=translations,
        n_shells=n_shells,
        n_half=n_half,
    )


def _run(
    case,
    *,
    resident: bool,
    monkeypatch,
    current_size=CURRENT_SIZE,
    source_faithful_spectrum_norm=False,
    production_shapes=False,
    projector_dtype=None,
    resident_operands: bool | None = None,
):
    """``production_shapes`` mirrors what the refinement loop actually passes:
    a projector with a singleton class axis, per-image contrast and scale
    corrections, and translation-prior centres."""

    """One fine pass 2 through the production dispatch."""

    if resident:
        monkeypatch.setenv(rlp.RESIDENT_LOCAL_SEARCH_ENV, "1")
    else:
        monkeypatch.delenv(rlp.RESIDENT_LOCAL_SEARCH_ENV, raising=False)
    if resident_operands is None:
        monkeypatch.delenv(rp._RESIDENT_OPERANDS_ENV, raising=False)
    else:
        monkeypatch.setenv(rp._RESIDENT_OPERANDS_ENV, "1" if resident_operands else "0")
    layout = case["layout"]
    projector = case["projector_half"]
    if projector_dtype is not None:
        projector = projector.astype(projector_dtype)
    image_corrections = scale_corrections = trans_centers = None
    if production_shapes:
        projector = projector[None]  # the loop's singleton class axis
        rng = np.random.default_rng(4242)
        image_corrections = rng.uniform(0.9, 1.1, N_IMAGES).astype(np.float32)
        scale_corrections = rng.uniform(0.9, 1.1, N_IMAGES).astype(np.float32)
        trans_centers = rng.uniform(-0.5, 0.5, (N_IMAGES, 2)).astype(np.float32)
    return local_search_iteration._run_local_search_iteration(
        case["dataset"],
        case["volume"],
        case["noise_variance"],
        _prior_eulers(N_IMAGES, 20260919),
        None,
        PARENT_ORDER + OVERSAMPLING,
        0.35,
        0.35,
        case["translations"],
        np.zeros((N_IMAGES, 2), dtype=np.float32),
        3.0,
        "linear_interp",
        image_batch_size=4,
        rotation_block_size=64,
        current_size=current_size,
        reconstruction_current_size=current_size,
        accumulate_noise=True,
        projection_padding_factor=1,
        reconstruction_padding_factor=1,
        score_with_masked_images=False,
        half_spectrum_scoring=True,
        relion_exact_score_translation=True,
        projection_relion_texture_interp=None,
        relion_projector_half=projector,
        relion_projector_r_max=case["r_max"],
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        translation_prior_centers=trans_centers,
        do_gridding_correction=True,
        square_window=False,
        group_ids=np.zeros(N_IMAGES, dtype=np.int32),
        scale_correction_group_count=1,
        scale_correction_data_vs_prior=np.full(case["n_shells"], 5.0, dtype=np.float64),
        mstep_relion_x_half=True,
        reconstruct_significant_only=True,
        stats_use_reconstruction_probs=True,
        adaptive_fraction=0.999,
        max_significants=-1,
        return_best_pose_details=True,
        return_significant_counts=True,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        pass2_layout=layout,
    )


@pytest.fixture
def _resident_local_env(monkeypatch):
    monkeypatch.setenv("RECOVAR_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES", "64,256,1024")
    monkeypatch.setenv("RECOVAR_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES", "2,4,8")


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv(rlp.RESIDENT_LOCAL_SEARCH_ENV, raising=False)
    assert not rlp.resident_local_search_requested()
    monkeypatch.setenv(rlp.RESIDENT_LOCAL_SEARCH_ENV, "1")
    assert rlp.resident_local_search_requested()


def test_dispatch_routes_only_the_fine_pass():
    """The wiring: the fine pass routes, the parent probe and K-class do not."""

    import inspect

    source = inspect.getsource(local_search_iteration._run_local_search_iteration)
    assert "resident_local_search_requested()" in source
    assert "and not score_only" in source
    # and only below the full image box: RELION's final all-data shape is
    # RELION's radial support (P4-B step 1), but the resident driver still
    # disagrees with the exact engine at the packed Nyquist row there.
    assert "int(current_size) < int(experiment_dataset.image_shape[0])" in source
    assert "compute_local_search_resident" in source
    # K-class with the flag on refuses rather than running a K=1 driver.
    assert "K=1 only; K-class local search keeps the exact local engine" in source


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"score_only": True}, "parent probe"),
        ({"class_log_priors": np.zeros(2)}, "K-class"),
        ({"mstep_relion_x_half": False}, "x-half M-step"),
        ({"accumulate_noise": False}, "noise statistics"),
        ({"use_float64_scoring": True}, "float64"),
        ({"reconstruct_significant_only": False}, "pruned fine weights"),
        ({"relion_exact_score_translation": False}, "translation angles"),
        ({"relion_projector_half": None}, "PPref projector"),
        ({"group_ids": None}, "group scale terms"),
        ({"normalization_log_evidence": np.zeros(3)}, "externally supplied normalizer"),
        ({"return_reconstruction_sample_indices": True}, "significant-sample capture"),
        ({"use_window": False}, "allow_full_box_window"),
    ],
)
def test_gate_names_the_missing_piece(override, expected):
    kwargs = dict(
        class_log_priors=None,
        score_only=False,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        mstep_relion_x_half=True,
        accumulate_noise=True,
        reconstruct_significant_only=True,
        use_float64_scoring=False,
        use_float64_projections=False,
        relion_exact_score_translation=True,
        half_spectrum_scoring=True,
        relion_projector_half=object(),
        relion_projector_r_max=4,
        mstep_subtract_ctf_projection=False,
        normalization_log_z=None,
        normalization_log_evidence=None,
        return_reconstruction_sample_indices=False,
        group_ids=np.zeros(3, dtype=np.int32),
        use_window=True,
        relion_wavg_atomic_scale_aa=True,
        relion_wavg_atomic_direct_noise=True,
        relion_wavg_atomic_direct_norm=False,
    )
    kwargs.update(override)
    with pytest.raises(NotImplementedError, match=expected):
        rlp.require_resident_local_configuration(**kwargs)


def test_projector_call_bound_covers_the_shape_that_ran_out_of_memory():
    """The end-to-end on arm died at current size 52, order 4, asking 16.12 GiB.

    The shared compact projection-block helper returns full half-spectrum rows
    and windows them afterwards, so one call holds
    ``rows x n_half x itemsize(Projector::data)`` whatever the window is. At a
    256 box with a complex128 slab that is 528 KiB per row, and the 32768-row
    chunk the window-based budget allowed asked for 16.12 GiB. Pin the bound at
    that shape and at the state-C shape beside it.
    """

    n_half = 256 * (256 // 2 + 1)
    assert n_half == 33024
    budget = rlp._projection_call_transient_max_bytes()
    for window_px, slab_bytes in ((1104, 16), (3387, 16), (1022, 16), (3387, 8)):
        rows = max(1, budget // max(n_half * slab_bytes, 1))
        peak = rows * n_half * slab_bytes
        assert peak <= budget, (window_px, slab_bytes, peak)
        # The window must not enter the bound: the helper materializes n_half.
        assert rows == max(1, budget // (n_half * slab_bytes))
    # The allocation that failed, and what the bound permits in its place.
    failed_rows, slab_bytes = 32768, 16
    assert failed_rows * n_half * slab_bytes / 1024 ** 3 > 16.0
    bounded_rows = max(1, budget // (n_half * slab_bytes))
    assert bounded_rows == 8128
    assert bounded_rows * n_half * slab_bytes / 1024 ** 3 <= 4.0


def test_plan_log_line_formats_without_a_logging_error(caplog):
    """The plan line's placeholders and arguments must agree.

    A mismatch does not fail the run, because logging swallows it, but it
    replaces every plan line in a measured arm's log with a traceback, which is
    how the projector-call bound was nearly impossible to read back.
    """

    import inspect
    import re

    src = inspect.getsource(rlp.compute_local_search_resident)
    start = src.index('"Resident local pass-2 plan:')
    block = src[start : src.index("\n    )\n", start)]
    fmt = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', block))
    placeholders = len(re.findall(r"%[-0-9.]*[dsfgex]", fmt))
    arguments = len(
        [line for line in block.split("\n") if line.strip() and not line.strip().startswith('"')]
    )
    assert placeholders == arguments, (placeholders, arguments)


def test_row_capacity_ladder_is_capped_by_the_projection_budget():
    ladder = rlp._cap_row_capacity_ladder(
        (1024, 4096, 16384),
        n_score_pixels=3386,
        n_recon_pixels=4324,
        max_bytes=200 * 1024**2,
    )
    assert ladder and list(ladder) == sorted(ladder)
    # A budget that fits nothing still leaves the smallest class so a plan exists.
    assert rlp._cap_row_capacity_ladder(
        (1024, 4096), n_score_pixels=3386, n_recon_pixels=4324, max_bytes=1
    ) == (1024,)


@requires_resident_gpu
def test_final_all_data_shape_keeps_the_exact_local_engine(monkeypatch, _resident_local_env):
    """``current_size == image box`` is RELION's final all-data shape.

    P4-B step 1 settled the support question: RELION scores that iteration on
    the same radial support as every other one
    (``ml_optimiser.cpp:6955-6967`` for the labels, ``:8046-8053`` for the
    ``Minvsigma2`` support), and the whole-rectangle support the unwindowed
    exact engine uses adds only pixels outside the projector disk, whose
    reference is identically zero (``projector.cpp:642-646``). Measured inside
    the exact engine, rectangle versus radial support moves no winner, 3.9e-7
    of Pmax and 4e-7 of ``Ft_y``.

    The dispatch still keeps this iteration on the exact engine because the
    resident driver does not yet agree there: on this fixture it differs by
    0.17 Pmax with two winner flips, and removing the packed Nyquist row
    (RELION's ``+N/2``, recovar's ``ky = -N/2``) from both windows collapses
    that to 7e-7 with no flips. That row is the open item, not the support.
    """

    case = _case()
    full = IMAGE_SHAPE[0]
    with_flag = _run(case, resident=True, monkeypatch=monkeypatch, current_size=full)
    without = _run(case, resident=False, monkeypatch=monkeypatch, current_size=full)
    np.testing.assert_array_equal(
        np.asarray(with_flag.hard_assignment), np.asarray(without.hard_assignment)
    )
    a = np.asarray(with_flag.Ft_y, dtype=np.complex128)
    b = np.asarray(without.Ft_y, dtype=np.complex128)
    assert float(np.linalg.norm(a - b) / np.linalg.norm(b)) < 1e-6


@requires_resident_gpu
def test_resident_local_matches_the_exact_engine(monkeypatch, _resident_local_env):
    """Pose, translation and every statistic against the exact local engine.

    Bounds are the ones this fixture measured; they exist to catch a regression,
    not to certify parity. The two engines factor the likelihood differently
    (see the driver's module docstring), so the continuous fields agree to
    float32 association rather than bitwise.
    """

    case = _case()
    exact = _run(case, resident=False, monkeypatch=monkeypatch)
    resident = _run(case, resident=True, monkeypatch=monkeypatch)

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(a - b))

    # --- discrete state ----------------------------------------------------
    np.testing.assert_array_equal(
        np.asarray(exact.hard_assignment), np.asarray(resident.hard_assignment)
    )
    np.testing.assert_array_equal(
        np.asarray(exact.best_pose_rotation_ids), np.asarray(resident.best_pose_rotation_ids)
    )
    np.testing.assert_array_equal(
        np.asarray(exact.best_pose_translations), np.asarray(resident.best_pose_translations)
    )
    np.testing.assert_array_equal(
        np.asarray(exact.best_pose_rotations), np.asarray(resident.best_pose_rotations)
    )
    np.testing.assert_array_equal(
        np.asarray(exact.best_pose_eulers_deg), np.asarray(resident.best_pose_eulers_deg)
    )

    # --- maps --------------------------------------------------------------
    # Measured 5.1e-7 and 1.5e-7 on an A100; the resident driver's own repeat
    # band for Ft_y is 1e-7 (the float32 BPref atomics).
    assert rel_l2(exact.Ft_y, resident.Ft_y) < 1e-5
    assert rel_l2(exact.Ft_ctf, resident.Ft_ctf) < 1e-5

    # --- per-image statistics ----------------------------------------------
    # The posterior is non-degenerate here (Pmax between 0.12 and 0.87), so
    # these compare the whole posterior, not just its winner.
    exact_pmax = np.asarray(exact.relion_stats.max_posterior_per_image, dtype=np.float64)
    resident_pmax = np.asarray(resident.relion_stats.max_posterior_per_image, dtype=np.float64)
    assert exact_pmax.max() < 0.95, "the fixture must keep the posterior non-degenerate"
    np.testing.assert_allclose(exact_pmax, resident_pmax, rtol=0, atol=1e-5)
    assert rel_l2(
        exact.relion_stats.rotation_posterior_sums,
        resident.relion_stats.rotation_posterior_sums,
    ) < 1e-5

    # Absolute log evidence carries each engine's own additive per-image
    # constant: the exact engine subtracts its Hermitian-weighted full-spectrum
    # image power (``-0.5 * batch_norm``), this driver subtracts RELION's
    # common minimum over the current-size window plus the powerClass tail
    # (``-min_diff2``). The convention-free part is the distance from the
    # winner, which is what the posterior sees, so that is what is bounded.
    exact_span = np.asarray(
        exact.relion_stats.log_evidence_per_image, dtype=np.float64
    ) - np.asarray(exact.relion_stats.best_log_score_per_image, dtype=np.float64)
    resident_span = np.asarray(
        resident.relion_stats.log_evidence_per_image, dtype=np.float64
    ) - np.asarray(resident.relion_stats.best_log_score_per_image, dtype=np.float64)
    np.testing.assert_allclose(exact_span, resident_span, rtol=0, atol=1e-4)

    # --- noise statistics ---------------------------------------------------
    # wsum_sigma2_noise and wsum_img_power are one quantity split two ways, and
    # the two engines split it differently (RELION's direct low-shell residual
    # versus the algebraic A2-2XA). Compare the sum, which is the quantity the
    # sigma2 update consumes, and report the split separately.
    exact_total = np.asarray(exact.noise_stats.wsum_sigma2_noise) + np.asarray(
        exact.noise_stats.wsum_img_power
    )
    resident_total = np.asarray(resident.noise_stats.wsum_sigma2_noise) + np.asarray(
        resident.noise_stats.wsum_img_power
    )
    assert rel_l2(exact_total, resident_total) < 1e-4
    assert abs(float(exact.noise_stats.sumw) - float(resident.noise_stats.sumw)) <= 1e-5 * abs(
        float(exact.noise_stats.sumw)
    )
    assert rel_l2(
        exact.noise_stats.wsum_norm_correction, resident.noise_stats.wsum_norm_correction
    ) < 1e-4
    for field in ("wsum_scale_correction_xa", "wsum_scale_correction_aa"):
        assert rel_l2(
            getattr(exact.noise_stats, field), getattr(resident.noise_stats, field)
        ) < 1e-4, field
    assert float(exact.noise_stats.wsum_sigma2_offset) == pytest.approx(
        float(resident.noise_stats.wsum_sigma2_offset), abs=1e-9
    )


@requires_resident_gpu
def test_source_faithful_spectrum_norm_is_plumbed_not_refused(
    monkeypatch, _resident_local_env
):
    """The production local search sets this flag, so the driver must carry it.

    ``source_faithful_spectrum_norm`` selects RELION's powerClass shell
    spectrum for the image-power statistics and the deterministic float64 norm
    reduction. The exact local engine uses the caller's value directly, with no
    environment resolution of its own, and so does this driver.
    """

    case = _case()
    exact = _run(
        case, resident=False, monkeypatch=monkeypatch, source_faithful_spectrum_norm=True
    )
    resident = _run(
        case, resident=True, monkeypatch=monkeypatch, source_faithful_spectrum_norm=True
    )
    np.testing.assert_array_equal(
        np.asarray(exact.hard_assignment), np.asarray(resident.hard_assignment)
    )

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    assert rel_l2(exact.Ft_y, resident.Ft_y) < 1e-5
    total_exact = np.asarray(exact.noise_stats.wsum_sigma2_noise) + np.asarray(
        exact.noise_stats.wsum_img_power
    )
    total_resident = np.asarray(resident.noise_stats.wsum_sigma2_noise) + np.asarray(
        resident.noise_stats.wsum_img_power
    )
    assert rel_l2(total_exact, total_resident) < 1e-4


@requires_resident_gpu
def test_full_parent_support_layout_runs_without_a_mask(monkeypatch, _resident_local_env):
    """A layout whose ``sample_mask_bits`` is ``None`` drives the driver too.

    RELION's full-parent local pass 2 produces exactly that, and the adapter
    keeps the compact ``None`` spelling instead of materializing an all-ones
    mask. The whole pass must still agree with the exact local engine.
    """

    case = _case(full_support=True)
    assert case["layout"].sample_mask_bits is None
    exact = _run(case, resident=False, monkeypatch=monkeypatch)
    resident = _run(case, resident=True, monkeypatch=monkeypatch)
    np.testing.assert_array_equal(
        np.asarray(exact.hard_assignment), np.asarray(resident.hard_assignment)
    )
    np.testing.assert_allclose(
        np.asarray(exact.relion_stats.max_posterior_per_image, dtype=np.float64),
        np.asarray(resident.relion_stats.max_posterior_per_image, dtype=np.float64),
        rtol=0,
        atol=1e-5,
    )
    a = np.asarray(exact.Ft_y, dtype=np.complex128)
    b = np.asarray(resident.Ft_y, dtype=np.complex128)
    assert float(np.linalg.norm(a - b) / np.linalg.norm(a)) < 1e-5


def test_driver_does_not_narrow_the_projector_slab():
    """Narrowing Projector::data is a change of projection arithmetic.

    It also swaps the vmapped fallback for the texture projector, so a narrowed
    arm is both a different computation and a faster one than its control. The
    exact local engine preserves the slab's dtype and does not read the compact
    engine's opt-in gate, so neither does this driver.
    """

    import inspect

    source = inspect.getsource(rlp.compute_local_search_resident)
    assert "prepare_local_projector_slab" in source
    assert "astype(jnp.complex64)" not in source
    assert "_pass2_projector_complex64_enabled" not in source


@requires_resident_gpu
def test_complex128_projector_is_carried_through_unnarrowed(
    monkeypatch, _resident_local_env
):
    """A double Projector::data slab, which is what the refinement loop builds.

    The other fixtures build the slab through the JAX projector-setup backend,
    which already returns complex64, so they cannot see a narrowing. This one
    hands both engines the same complex128 slab; they must still agree, and the
    driver must not quietly cast it.
    """

    case = _case()
    exact = _run(
        case, resident=False, monkeypatch=monkeypatch, projector_dtype=jnp.complex128
    )
    resident = _run(
        case, resident=True, monkeypatch=monkeypatch, projector_dtype=jnp.complex128
    )
    np.testing.assert_array_equal(
        np.asarray(exact.hard_assignment), np.asarray(resident.hard_assignment)
    )
    np.testing.assert_allclose(
        np.asarray(exact.relion_stats.max_posterior_per_image, dtype=np.float64),
        np.asarray(resident.relion_stats.max_posterior_per_image, dtype=np.float64),
        rtol=0,
        atol=1e-5,
    )
    a = np.asarray(exact.Ft_y, dtype=np.complex128)
    b = np.asarray(resident.Ft_y, dtype=np.complex128)
    assert float(np.linalg.norm(a - b) / np.linalg.norm(a)) < 1e-5


@requires_resident_gpu
def test_production_shaped_inputs_are_accepted(monkeypatch, _resident_local_env):
    """The shapes the refinement loop actually hands local search.

    A projector with a singleton class axis (normalized by the same helper the
    exact local engine uses), per-image contrast and scale corrections, and
    translation-prior centres, which switch on the sigma2-offset accumulator.
    The 3-D fixture elsewhere in this file would not have caught the class axis.
    """

    case = _case()
    exact = _run(case, resident=False, monkeypatch=monkeypatch, production_shapes=True)
    resident = _run(case, resident=True, monkeypatch=monkeypatch, production_shapes=True)
    np.testing.assert_array_equal(
        np.asarray(exact.hard_assignment), np.asarray(resident.hard_assignment)
    )

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    assert rel_l2(exact.Ft_y, resident.Ft_y) < 1e-5
    # Translation-prior centres switch on the sigma2 offset on both paths.
    assert float(exact.noise_stats.wsum_sigma2_offset) != 0.0
    assert float(resident.noise_stats.wsum_sigma2_offset) == pytest.approx(
        float(exact.noise_stats.wsum_sigma2_offset), rel=1e-6
    )


@requires_resident_gpu
def test_resident_local_repeats_itself(monkeypatch, _resident_local_env):
    """The resident driver's own repeat band, the reference for the table above."""

    case = _case()
    first = _run(case, resident=True, monkeypatch=monkeypatch)
    second = _run(case, resident=True, monkeypatch=monkeypatch)
    np.testing.assert_array_equal(
        np.asarray(first.hard_assignment), np.asarray(second.hard_assignment)
    )
    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # Two unordered float32 reductions run per chunk: the x-half BPref atomics
    # and the flat-row Wavg rotation atomic that feeds the direct low-shell
    # noise residual. Measured repeat band on an A100: 1.0e-7 for the maps and
    # 2.5e-8 relative for the noise shells, so neither is asserted bitwise.
    assert rel_l2(first.Ft_y, second.Ft_y) < 1e-6
    assert rel_l2(
        first.noise_stats.wsum_sigma2_noise, second.noise_stats.wsum_sigma2_noise
    ) < 1e-6


def test_mstep_adapter_refuses_once_per_half_operands():
    """The local M-step entry point fails closed on T16's operand family.

    ``run_resident_mstep_blocks`` has no translate-and-sum kernel path, so a
    ``recon`` carrying the once-per-half per-image images instead of the
    per-chunk pre-shifted tiles must be refused by name rather than reach the
    XLA weighted sums with ``None`` operands.
    """

    resident_recon = {
        "shifted_recon": None,
        "shifted_noise": None,
        "recon_image": object(),
        "recon_weight": None,
        "noise_image": object(),
        "ctf2_over_nv_recon": object(),
        "direct_ctf_rfloat_recon": None,
        "raw_translated_wavg_rectangle": object(),
        "raw_translated_wavg_for_atomic": object(),
        "scale": object(),
    }
    with pytest.raises(ValueError, match="pre-shifted"):
        rp.run_resident_mstep_blocks(
            lambda start, stop: None,
            row_capacity=64,
            n_valid_rows=8,
            mstep_block_rows=64,
            image_capacity=2,
            row_image_local=None,
            kernel_row_image_ids=None,
            row_posterior=np.zeros((64, 1), dtype=np.float32),
            recon=resident_recon,
            n_rect=1,
            n_shells=2,
            n_recon_windowed=1,
            noise_variance_for_noise=None,
            shell_indices_noise=None,
            exact_positions_device=None,
            Ft_y_total=None,
            Ft_ctf_total=None,
            image_shape=IMAGE_SHAPE,
            recon_volume_shape=VOLUME_SHAPE,
            mstep_current_size=CURRENT_SIZE,
            relion_x_half_recon_indices=None,
            max_adjoint_block_bytes=1 << 20,
            cuda_backproject=None,
        )


@requires_resident_gpu
def test_local_chunk_runs_with_the_once_per_half_operand_flag(
    monkeypatch, _resident_local_env
):
    """One local-search chunk through the M-step adapter with T16's flag on.

    The T16 merge added ``recon_image``/``recon_weight``/``noise_image`` to
    ``_ChunkStageOperands`` and ``recon_pixel_indices`` to ``_ChunkStageTables``
    for the once-per-half kernel path. The local adapter still built the old
    field sets, so every resident local chunk raised ``TypeError``; the
    full-wave end-to-end (job 14168000) died at its first local search,
    iteration 16, while every global-order matched pair passed.

    The local pass prepares its own pre-shifted per-chunk tiles, so
    ``RECOVAR_SPARSE_PASS2_RESIDENT_OPERANDS`` must be inert here: both
    settings have to run the adapter and agree to the driver's own repeat band.
    """

    case = _case()
    calls = []
    real_mstep_blocks = rp.run_resident_mstep_blocks

    def counting_mstep_blocks(*args, **kwargs):
        calls.append(kwargs.get("image_capacity"))
        return real_mstep_blocks(*args, **kwargs)

    monkeypatch.setattr(rp, "run_resident_mstep_blocks", counting_mstep_blocks)

    on = _run(case, resident=True, monkeypatch=monkeypatch, resident_operands=True)
    assert calls, "the local pass must reach run_resident_mstep_blocks"
    off = _run(case, resident=True, monkeypatch=monkeypatch, resident_operands=False)

    np.testing.assert_array_equal(
        np.asarray(on.hard_assignment), np.asarray(off.hard_assignment)
    )
    np.testing.assert_array_equal(
        np.asarray(on.best_pose_rotation_ids), np.asarray(off.best_pose_rotation_ids)
    )
    np.testing.assert_array_equal(
        np.asarray(on.best_pose_translations), np.asarray(off.best_pose_translations)
    )

    def rel_l2(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        den = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b) / den) if den else 0.0

    # The flag selects nothing on this path, so the only spread is the same
    # float32 atomics the repeat test bounds at 1e-6.
    assert rel_l2(on.Ft_y, off.Ft_y) < 1e-6
    assert rel_l2(on.Ft_ctf, off.Ft_ctf) < 1e-6
    assert rel_l2(
        on.noise_stats.wsum_sigma2_noise, off.noise_stats.wsum_sigma2_noise
    ) < 1e-6
    np.testing.assert_allclose(
        np.asarray(on.relion_stats.max_posterior_per_image, dtype=np.float64),
        np.asarray(off.relion_stats.max_posterior_per_image, dtype=np.float64),
        rtol=0,
        atol=1e-6,
    )
