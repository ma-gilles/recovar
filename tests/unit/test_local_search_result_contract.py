"""Optional local-search outputs retain their meanings for K-class callers."""

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.helpers.types import LocalEMResult
from recovar.em.local import local_search_iteration

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("return_best_pose_details", [False, True])
@pytest.mark.parametrize("accumulate_noise", [False, True])
@pytest.mark.parametrize("return_class_details", [False, True])
def test_kclass_optional_outputs_preserve_statistics(
    monkeypatch, n_classes, return_best_pose_details, accumulate_noise, return_class_details
):
    stats, noise = object(), object()
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    translations = np.zeros((2, 2), dtype=np.float32)
    assignments = np.array([0, 1], dtype=np.int32)
    class_sums = np.arange(1, n_classes + 1, dtype=np.float64)
    engine_result = SimpleNamespace(
        Ft_y=np.zeros((n_classes, 8), dtype=np.complex64),
        Ft_ctf=np.ones((n_classes, 8), dtype=np.float32),
        pose_assignments=assignments,
        best_pose_rotations=rotations if return_best_pose_details else None,
        best_pose_translations=translations if return_best_pose_details else None,
        best_pose_rotation_ids=assignments if return_best_pose_details else None,
        stats=stats,
        aggregate_noise_stats=noise if accumulate_noise else None,
        class_assignments=assignments,
        class_posterior_sums=class_sums,
    )

    def run_kclass(*args, **kwargs):
        assert kwargs["return_best_pose_details"] == return_best_pose_details
        assert kwargs["accumulate_noise"] == accumulate_noise
        return engine_result

    monkeypatch.setattr(local_search_iteration, "run_local_k_class_em", run_kclass)
    monkeypatch.setattr(
        local_search_iteration,
        "_estimate_relion_em_batch_sizes",
        lambda **kwargs: SimpleNamespace(
            image_batch_size=kwargs["requested_image_batch_size"],
            rotation_block_size=kwargs["requested_rotation_block_size"],
        ),
    )
    result = local_search_iteration._run_local_search_iteration(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2)),
        engine_result.Ft_y, None, rotations, rotations,
        healpix_order=0, sigma_rot=1.0, sigma_psi=1.0,
        translations=translations[:1], prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp", image_batch_size=2, rotation_block_size=1,
        current_size=2,
        pass2_layout=SimpleNamespace(rotation_counts=np.ones(2, dtype=np.int32), translation_grid=translations[:1]),
        class_log_priors=np.full(n_classes, -np.log(n_classes)),
        accumulate_noise=accumulate_noise,
        return_best_pose_details=return_best_pose_details,
        return_class_details=return_class_details,
    )

    assert result.relion_stats is stats
    assert result.noise_stats is (noise if accumulate_noise else None)
    assert result.Ft_y is engine_result.Ft_y
    assert result.Ft_ctf is engine_result.Ft_ctf
    np.testing.assert_array_equal(result.hard_assignment, assignments)
    assert result.best_pose_rotations is engine_result.best_pose_rotations
    assert result.best_pose_translations is engine_result.best_pose_translations
    assert result.best_pose_rotation_ids is engine_result.best_pose_rotation_ids
    assert result.profile_summary is None
    assert result.significant_counts is None
    if return_class_details:
        np.testing.assert_array_equal(result.class_assignments, assignments)
        np.testing.assert_array_equal(result.class_posterior_sums, class_sums)
        np.testing.assert_array_equal(result.class_full_posterior_sums, class_sums)
    else:
        assert result.class_assignments is None
        assert result.class_posterior_sums is None
        assert result.class_full_posterior_sums is None


@pytest.mark.parametrize("return_profile", [False, True])
@pytest.mark.parametrize("return_significant_counts", [False, True])
def test_local_sample_capture_does_not_shift_significant_counts(
    monkeypatch, return_profile, return_significant_counts
):
    """Sample capture enables an internal profile even if the caller hides it."""
    counts = np.array([3, 7], dtype=np.int32)
    profile = {"reconstruction_sample_indices_by_image": (np.array([1]), np.array([2]))}
    stats = object()

    def run_local(*args, **kwargs):
        assert kwargs["return_reconstruction_sample_indices"] is True
        assert kwargs["return_profile"] == return_profile
        assert kwargs["return_significant_counts"] == return_significant_counts
        return LocalEMResult(
            Ft_y=np.zeros(8, dtype=np.complex64),
            Ft_ctf=np.ones(8, dtype=np.float32),
            hard_assignments=np.array([0, 1], dtype=np.int32),
            stats=stats,
            profile=profile,
            significant_counts=counts if return_significant_counts else None,
        )

    monkeypatch.setattr(local_search_iteration, "run_local_em_exact", run_local)
    monkeypatch.setattr(
        local_search_iteration, "_estimate_relion_em_batch_sizes",
        lambda **kwargs: SimpleNamespace(
            image_batch_size=kwargs["requested_image_batch_size"],
            rotation_block_size=kwargs["requested_rotation_block_size"],
        ),
    )
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    translations = np.zeros((2, 2), dtype=np.float32)
    result = local_search_iteration._run_local_search_iteration(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2)),
        None, None, rotations, rotations,
        healpix_order=0, sigma_rot=1.0, sigma_psi=1.0,
        translations=translations[:1], prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp", image_batch_size=2, rotation_block_size=1,
        current_size=2,
        pass2_layout=SimpleNamespace(rotation_counts=np.ones(2, dtype=np.int32), translation_grid=translations[:1]),
        return_reconstruction_sample_indices=True,
        return_significant_counts=return_significant_counts,
        return_profile=return_profile,
    )
    assert result.relion_stats is stats
    assert result.significant_counts is (counts if return_significant_counts else None)
    if return_profile:
        assert result.profile_summary is not profile
        assert result.profile_summary["reconstruction_sample_indices_by_image"] is profile["reconstruction_sample_indices_by_image"]
    else:
        assert result.profile_summary is None
    assert set(profile) == {"reconstruction_sample_indices_by_image"}


def _run_wrapper_with_fake_engine(monkeypatch, *, score_only, captured, **overrides):
    """Drive the K=1 wrapper with a stubbed engine and capture its arguments."""

    def run_local(*args, **kwargs):
        captured.update(kwargs)
        captured["__args__"] = args
        return LocalEMResult(
            Ft_y=np.zeros(8, dtype=np.complex64),
            Ft_ctf=np.ones(8, dtype=np.float32),
            hard_assignments=np.array([0, 1], dtype=np.int32),
            stats=object(),
        )

    monkeypatch.setattr(local_search_iteration, "run_local_em_exact", run_local)
    monkeypatch.setattr(
        local_search_iteration,
        "_estimate_relion_em_batch_sizes",
        lambda **kwargs: SimpleNamespace(
            image_batch_size=kwargs["requested_image_batch_size"],
            rotation_block_size=kwargs["requested_rotation_block_size"],
        ),
    )
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    translations = np.zeros((2, 2), dtype=np.float32)
    kwargs = dict(
        healpix_order=0,
        sigma_rot=1.0,
        sigma_psi=1.0,
        translations=translations[:1],
        prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp",
        image_batch_size=2,
        rotation_block_size=1,
        current_size=2,
        pass2_layout=SimpleNamespace(
            rotation_counts=np.ones(2, dtype=np.int32), translation_grid=translations[:1]
        ),
        half_spectrum_scoring=True,
        relion_exact_score_translation=True,
        score_only=bool(score_only),
        disable_adjoint_y=bool(score_only),
        disable_adjoint_ctf=bool(score_only),
    )
    kwargs.update(overrides)
    return local_search_iteration._run_local_search_iteration(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2)),
        None,
        None,
        rotations,
        rotations,
        **kwargs,
    )


_SHAPE_STABLE_ALWAYS = {
    "relion_exact_bpref_operands": True,
    "relion_exact_fine_diff2": True,
    "_flat_local_rows_enabled": True,
    "_stable_flat_row_capacity_enabled": True,
    "_packed_local_projection_enabled": True,
    "unify_local_bucket_sizes": True,
}


@pytest.mark.parametrize("score_only", [True, False])
def test_shape_stable_flag_is_off_by_default(monkeypatch, score_only):
    """Without the opt-in the engine keeps its own defaults on both calls."""
    monkeypatch.delenv(local_search_iteration.LOCAL_SEARCH_SHAPE_STABLE_ENV, raising=False)
    captured = {}
    _run_wrapper_with_fake_engine(monkeypatch, score_only=score_only, captured=captured)
    for name in (*_SHAPE_STABLE_ALWAYS, "relion_wavg_sequential_cuda"):
        assert name not in captured


@pytest.mark.parametrize("score_only", [True, False])
def test_shape_stable_flag_forwards_to_both_local_search_calls(monkeypatch, score_only):
    """Pass-1 (score-only) and pass-2 both receive the shape-stable modes.

    ``relion_wavg_sequential_cuda`` is reconstruction-side, so only the
    pass-2 call requests it; the score-only probe leaves it unset.
    """
    monkeypatch.setenv(local_search_iteration.LOCAL_SEARCH_SHAPE_STABLE_ENV, "1")
    captured = {}
    layout = SimpleNamespace(
        rotation_counts=np.ones(2, dtype=np.int32),
        translation_grid=np.zeros((1, 2), dtype=np.float32),
    )
    _run_wrapper_with_fake_engine(
        monkeypatch, score_only=score_only, captured=captured, pass2_layout=layout
    )
    for name, value in _SHAPE_STABLE_ALWAYS.items():
        assert captured[name] is value
    if score_only:
        assert "relion_wavg_sequential_cuda" not in captured
    else:
        assert captured["relion_wavg_sequential_cuda"] is True
    # The modes that need the InitialModel residual M-step route stay off.
    assert "preserve_bpref_particle_order" not in captured
    assert "stable_fourier_window_shapes" not in captured
    # The flags must not touch the candidate set: the engine still receives
    # the caller's own pass-2 layout object.
    assert captured["__args__"][3] is layout


@pytest.mark.parametrize(
    "env_name,kwarg_name",
    [
        ("_LOCAL_SEARCH_STABLE_ROW_CAPACITY_ENV", "_stable_flat_row_capacity_enabled"),
        ("_LOCAL_SEARCH_STABLE_PACKED_PROJECTION_ENV", "_packed_local_projection_enabled"),
        ("_LOCAL_SEARCH_STABLE_UNIFY_BUCKETS_ENV", "unify_local_bucket_sizes"),
        ("_LOCAL_SEARCH_STABLE_WAVG_CUDA_ENV", "relion_wavg_sequential_cuda"),
    ],
)
def test_shape_stable_single_mode_can_be_dropped(monkeypatch, env_name, kwarg_name):
    """A rejected mode can be dropped without dropping the rest."""
    monkeypatch.setenv(local_search_iteration.LOCAL_SEARCH_SHAPE_STABLE_ENV, "1")
    monkeypatch.setenv(getattr(local_search_iteration, env_name), "0")
    captured = {}
    _run_wrapper_with_fake_engine(monkeypatch, score_only=False, captured=captured)
    assert captured[kwarg_name] is False
    assert captured["relion_exact_fine_diff2"] is True
    assert captured["_flat_local_rows_enabled"] is True


def test_shape_stable_flat_rows_off_drops_its_dependent_modes(monkeypatch):
    """Flat rows gate their dependents, matching the engine's preconditions."""
    monkeypatch.setenv(local_search_iteration.LOCAL_SEARCH_SHAPE_STABLE_ENV, "1")
    monkeypatch.setenv(local_search_iteration._LOCAL_SEARCH_STABLE_FLAT_ROWS_ENV, "0")
    captured = {}
    _run_wrapper_with_fake_engine(monkeypatch, score_only=False, captured=captured)
    assert captured["_flat_local_rows_enabled"] is False
    assert captured["_stable_flat_row_capacity_enabled"] is False
    assert captured["_packed_local_projection_enabled"] is False


def test_shape_stable_requires_exact_relion_score_translation(monkeypatch):
    """Exact fine diff2 has no translation angles without exact score translation."""
    monkeypatch.setenv(local_search_iteration.LOCAL_SEARCH_SHAPE_STABLE_ENV, "1")
    with pytest.raises(ValueError, match="exact RELION score translation"):
        _run_wrapper_with_fake_engine(
            monkeypatch,
            score_only=False,
            captured={},
            relion_exact_score_translation=False,
        )


def test_shape_stable_rejects_k_class_local_search(monkeypatch):
    """The modes are K=1-only; K-class fails closed instead of ignoring them."""
    monkeypatch.setenv(local_search_iteration.LOCAL_SEARCH_SHAPE_STABLE_ENV, "1")
    with pytest.raises(NotImplementedError, match="K=1-only"):
        _run_wrapper_with_fake_engine(
            monkeypatch,
            score_only=False,
            captured={},
            class_log_priors=np.full(2, -np.log(2.0)),
        )


def test_half_scoring_owns_no_shape_stable_kwargs_at_either_call_site():
    """Both local-search call sites delegate the modes to the wrapper."""
    import inspect

    from recovar.em.dense import half_scoring

    source = inspect.getsource(half_scoring._score_half_local)
    parent_call = source[
        source.index("parent_outputs = _run_local_search_iteration") : source.index(
            "parent_profile = parent_outputs.profile_summary"
        )
    ]
    fine_call = source[
        source.index("local_outputs = _run_local_search_iteration") : source.index(
            "Ft_y_k = local_outputs.Ft_y"
        )
    ]
    assert "score_only=True" in parent_call
    assert "score_only=diagnostic_score_only" in fine_call
    for call in (parent_call, fine_call):
        for name in (*_SHAPE_STABLE_ALWAYS, "relion_wavg_sequential_cuda"):
            assert name not in call
