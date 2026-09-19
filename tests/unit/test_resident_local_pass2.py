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


def _pass2_layout(seed=20260919):
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


def _case(seed=20260919):
    """Dataset, volume, RELION projector and pass-2 layout for one half."""

    dataset = MockDataset(n_images=N_IMAGES, seed=seed % 2**31)
    volume_ft = _hermitian_volume(VOLUME_SHAPE, seed=17)
    volume_real = np.asarray(
        ftu.get_idft3(np.asarray(volume_ft).reshape(VOLUME_SHAPE)).real, dtype=np.float64
    )
    projector_half, r_max = _relion_projector(volume_real, IMAGE_SHAPE[0])
    layout, translations = _pass2_layout(seed)
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


def _run(case, *, resident: bool, monkeypatch, current_size=CURRENT_SIZE):
    """One fine pass 2 through the production dispatch."""

    if resident:
        monkeypatch.setenv(rlp.RESIDENT_LOCAL_SEARCH_ENV, "1")
    else:
        monkeypatch.delenv(rlp.RESIDENT_LOCAL_SEARCH_ENV, raising=False)
    layout = case["layout"]
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
        relion_projector_half=case["projector_half"],
        relion_projector_r_max=case["r_max"],
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
    # and only below the full image box (RELION's final all-data shape)
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
        ({"use_window": False}, "scientific decision"),
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
        source_faithful_spectrum_norm=False,
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

    The dispatch leaves that iteration on the exact local engine and says so,
    because the two engines disagree about the scoring support there, not about
    its layout: the exact engine scores the whole centred half including the
    FFTW rectangle's corners, while RELION's radial support (the one every
    windowed size uses, and the one the RELION Wavg rectangle requires) stops at
    ``|k| <= current_size/2``. Running the pass on the radial support was
    measured on this fixture to move the maps by 0.45 relative L2 and to flip a
    winner, so it is not a rounding-level difference.
    """

    case = _case()
    full = IMAGE_SHAPE[0]
    with_flag = _run(case, resident=True, monkeypatch=monkeypatch, current_size=full)
    without = _run(case, resident=False, monkeypatch=monkeypatch, current_size=full)
    np.testing.assert_array_equal(
        np.asarray(with_flag.hard_assignment), np.asarray(without.hard_assignment)
    )
    # Both arms ran the same engine, so they agree to that engine's own repeat
    # band (float32 backprojection atomics), not bitwise.
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
