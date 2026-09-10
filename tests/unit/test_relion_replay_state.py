"""Captured sampling, projector identity and replay override contracts."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import recovar.em.dense_single_volume.projector_preparation as projector_preparation
import recovar.em.dense_single_volume.relion_replay as relion_replay_module
import recovar.em.dense_single_volume.helpers.orientation_priors as orientation_priors_module

pytestmark = pytest.mark.unit
IMAGE_SIZE = 64


def _sealed_sampling_fixture():
    return {
        "consumer_relion_iteration": 2,
        "directions_ipix": np.asarray([7, 19, 503], dtype=np.int64),
        "rot_angles_deg": np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        "tilt_angles_deg": np.asarray([40.0, 50.0, 60.0], dtype=np.float64),
        "psi_angles_deg": np.asarray([0.0, 90.0], dtype=np.float64),
        "translations_x_angstrom": np.asarray([-2.0, 0.0, 2.0], dtype=np.float64),
        "translations_y_angstrom": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "translations_z_angstrom": np.empty(0, dtype=np.float64),
        "healpix_order_original": 3,
        "psi_step_deg": 90.0,
        "offset_range_angstrom": 2.0,
        "offset_step_angstrom": 1.0,
        "perturbation_factor": 0.5,
        "random_perturbation": 0.125,
        "sigma_rot_deg": 0.0,
        "sigma_psi_deg": 0.0,
        "is_3d": True,
        "is_3d_trans": False,
        "point_group": 202,
        "point_group_order": 1,
        "coarse_size": 56,
        "full_size": 256,
        "current_size": 56,
    }


def test_sealed_sampling_directly_materializes_restricted_eulers_and_translations():
    sampling = _sealed_sampling_fixture()

    _, eulers, translations = relion_replay_module._sealed_sampling_base_grids(
        sampling,
        voxel_size_angstrom=2.0,
    )

    np.testing.assert_array_equal(
        eulers,
        np.asarray(
            [
                [10.0, 40.0, 0.0],
                [20.0, 50.0, 0.0],
                [30.0, 60.0, 0.0],
                [10.0, 40.0, 90.0],
                [20.0, 50.0, 90.0],
                [30.0, 60.0, 90.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(translations),
        np.asarray([[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        relion_replay_module._sealed_sampling_rotation_ids(sampling),
        np.asarray([7, 19, 503, 775, 787, 1271], dtype=np.int64),
    )
    direction_prior = np.linspace(1.0, 2.0, 768, dtype=np.float32)
    expected_prior = np.log(
        np.tile(direction_prior[np.asarray([7, 19, 503])], 2)
    ).astype(np.float32)
    np.testing.assert_array_equal(
        orientation_priors_module._sealed_direction_log_prior(direction_prior, sampling),
        expected_prior,
    )


def test_sealed_sampling_override_never_reads_external_replay_files(monkeypatch, tmp_path):
    state = SimpleNamespace(
        healpix_order=3,
        max_healpix_order=3,
        auto_local_healpix_order=4,
        do_local_search=False,
        sigma_rot=0.0,
        sigma_psi=0.0,
        translation_range=99.0,
        translation_step=99.0,
    )
    monkeypatch.setattr(
        relion_replay_module,
        "read_relion_sampling_metadata",
        lambda path: pytest.fail(f"unexpected external sampling read: {path}"),
    )
    monkeypatch.setattr(
        relion_replay_module,
        "read_relion_model_metadata",
        lambda path: pytest.fail(f"unexpected external model read: {path}"),
    )

    result = relion_replay_module.apply_iter_replay_overrides(
        iter_replay_override=None,
        perturb_replay_relion_dir=str(tmp_path / "wrong_prefix_and_iteration"),
        perturb_replay_relion_prefix="wrong",
        init_relion_iteration=99,
        iteration=0,
        state=state,
        cs=8,
        cryo=SimpleNamespace(voxel_size=2.0),
        k_class_enabled=False,
        n_classes=1,
        relion_half_inputs=relion_replay_module._RelionHalfInputState.from_initial_values(
            previous_best_translations=None,
            previous_best_rotation_eulers=None,
            image_corrections=None,
            scale_corrections=None,
        ),
        previous_best_rotations=[None, None],
        noise_variance_per_half=[jnp.ones(IMAGE_SIZE), jnp.ones(IMAGE_SIZE)],
        noise_variance=jnp.ones(IMAGE_SIZE),
        previous_noise_radial_per_half=[None, None],
        previous_noise_radial=None,
        current_sigma_offset_angstrom=10.0,
        class_direction_prior_per_half=[None, None],
        class_direction_prior_order_per_half=[None, None],
        global_direction_prior_per_half=[None, None],
        global_direction_prior_order_per_half=[None, None],
        sealed_sampling_state=_sealed_sampling_fixture(),
    )

    assert result.cs == 56
    assert result.replay_meta["sealed_v3"] is True
    np.testing.assert_array_equal(
        np.asarray(result.prior_translations),
        np.asarray([[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0]], dtype=np.float32),
    )


def test_frozen_replay_explicitly_suppresses_external_direction_prior_reload(
    monkeypatch,
    tmp_path,
):
    state = SimpleNamespace(
        healpix_order=3,
        max_healpix_order=3,
        auto_local_healpix_order=4,
        do_local_search=False,
        sigma_rot=0.0,
        sigma_psi=0.0,
        translation_range=3.0,
        translation_step=1.0,
    )
    (tmp_path / "run_it002_half1_model.star").write_text("sealed control\n")
    for half in (1, 2):
        (tmp_path / f"run_it001_half{half}_model.star").write_text("must not read\n")
    monkeypatch.setattr(
        relion_replay_module,
        "read_relion_sampling_metadata",
        lambda path: {
            "random_perturbation": 0.0,
            "perturbation_factor": 0.0,
            "healpix_order": 3,
            "psi_step": 7.5,
            "offset_range": 3.0,
            "offset_step": 1.0,
        },
    )
    monkeypatch.setattr(
        relion_replay_module,
        "read_relion_model_metadata",
        lambda path: {"current_image_size": 8},
    )
    monkeypatch.setattr(
        relion_replay_module,
        "read_relion_direction_prior",
        lambda path: pytest.fail(f"unexpected direction-prior reload: {path}"),
    )
    priors = [
        np.full(768, 1.0 / 768.0, dtype=np.float32),
        np.full(768, 1.0 / 768.0, dtype=np.float32),
    ]

    relion_replay_module.apply_iter_replay_overrides(
        iter_replay_override={"relion_projector_state": None},
        perturb_replay_relion_dir=str(tmp_path),
        init_relion_iteration=0,
        iteration=1,
        state=state,
        cs=8,
        cryo=SimpleNamespace(voxel_size=1.0),
        k_class_enabled=False,
        n_classes=1,
        relion_half_inputs=relion_replay_module._RelionHalfInputState.from_initial_values(
            previous_best_translations=None,
            previous_best_rotation_eulers=None,
            image_corrections=None,
            scale_corrections=None,
        ),
        previous_best_rotations=[None, None],
        noise_variance_per_half=[jnp.ones(IMAGE_SIZE), jnp.ones(IMAGE_SIZE)],
        noise_variance=jnp.ones(IMAGE_SIZE),
        previous_noise_radial_per_half=[None, None],
        previous_noise_radial=None,
        current_sigma_offset_angstrom=10.0,
        class_direction_prior_per_half=[None, None],
        class_direction_prior_order_per_half=[None, None],
        global_direction_prior_per_half=priors,
        global_direction_prior_order_per_half=[3, 3],
        preserve_existing_direction_prior=True,
    )

    np.testing.assert_array_equal(priors[0], np.full(768, 1.0 / 768.0, dtype=np.float32))
    np.testing.assert_array_equal(priors[1], np.full(768, 1.0 / 768.0, dtype=np.float32))


def _captured_projector_override(projector):
    return {
        "projector_half_by_half": [projector, projector.copy()],
        "projector_r_max_by_half": [4, 4],
        "current_size": 8,
        "padding_factor": 2,
        "volume_shape": [8, 8, 8],
        "n_classes": 1,
        "source_manifest_sha256": "a" * 64,
    }


def test_captured_relion_projector_replay_state_is_atomic_and_copied():
    projector = np.zeros((1, 9, 9, 5), dtype=np.complex64)
    projector[0, 2, 3, 1] = np.complex64(1.25 - 0.5j)

    state = relion_replay_module._parse_relion_projector_replay_state(
        _captured_projector_override(projector),
        n_classes=1,
    )
    projector[...] = np.complex64(99.0 + 7.0j)

    assert state is not None
    assert state.source_manifest_sha256 == "a" * 64
    assert state.projector_r_max_by_half == (4, 4)
    assert state.projector_half_by_half[0][0, 2, 3, 1] == np.complex64(1.25 - 0.5j)
    assert state.projector_half_by_half[0].flags.writeable is False
    resolved, r_max = projector_preparation._validate_captured_relion_projector_for_iteration(
        state,
        current_size=8,
        volume_shape=(8, 8, 8),
        padding_factor=2,
        n_classes=1,
    )
    assert r_max == [4, 4]
    assert resolved[0] is state.projector_half_by_half[0]


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (lambda value: value.pop("source_manifest_sha256"), ValueError, "keys must match"),
        (
            lambda value: value.__setitem__("source_manifest_sha256", "not-a-sha"),
            ValueError,
            "64 lowercase hex digits",
        ),
        (
            lambda value: value.__setitem__("source_manifest_sha256", "A" * 64),
            ValueError,
            "64 lowercase hex digits",
        ),
        (
            lambda value: value["projector_half_by_half"].__setitem__(
                0, value["projector_half_by_half"][0].astype(np.complex128)
            ),
            TypeError,
            "must be complex64",
        ),
        (
            lambda value: value["projector_half_by_half"][0].__setitem__(
                (0, 0, 0, 0), np.complex64(np.nan + 0j)
            ),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_captured_relion_projector_replay_state_rejects_corruption(mutation, error_type, message):
    override = _captured_projector_override(np.zeros((1, 9, 9, 5), dtype=np.complex64))
    mutation(override)
    with pytest.raises(error_type, match=message):
        relion_replay_module._parse_relion_projector_replay_state(override, n_classes=1)


def test_captured_relion_projector_replay_state_rejects_live_geometry_mismatch():
    state = relion_replay_module._parse_relion_projector_replay_state(
        _captured_projector_override(np.zeros((1, 9, 9, 5), dtype=np.complex64)),
        n_classes=1,
    )
    with pytest.raises(ValueError, match="current_size captured=8 replay=10"):
        projector_preparation._validate_captured_relion_projector_for_iteration(
            state,
            current_size=10,
            volume_shape=(8, 8, 8),
            padding_factor=2,
            n_classes=1,
        )


@pytest.mark.parametrize("available", range(8))
def test_final_sampling_file_precedence(tmp_path, available):
    names = ["run_it021_sampling.star", "run_sampling.star", "run_it020_sampling.star"]
    labels = ["final-numbered", "final", "last-numbered"]
    for index, name in enumerate(names):
        if available & (1 << index):
            (tmp_path / name).touch()
    result = relion_replay_module.select_final_sampling_star(
        str(tmp_path), "run", final_iteration=21, previous_iteration=20,
        require_final_state=False,
    )
    selected = next((i for i in range(3) if available & (1 << i)), None)
    expected = (None, None) if selected is None else (str(tmp_path / names[selected]), labels[selected])
    assert result[:2] == expected
    assert result[2] == [(str(tmp_path / name), label) for name, label in zip(names, labels)]


@pytest.mark.parametrize("missing", ["sampling", "optimiser"])
@pytest.mark.parametrize("directory", [False, True])
def test_final_numbered_sampling_does_not_bypass_required_final_state(tmp_path, missing, directory):
    (tmp_path / "run_it021_sampling.star").touch()
    for suffix in ("sampling", "optimiser"):
        path = tmp_path / f"run_{suffix}.star"
        if suffix != missing:
            path.touch()
        elif directory:
            path.mkdir()
    with pytest.raises(RuntimeError, match=f"missing .*run_{missing}.star"):
        relion_replay_module.select_final_sampling_star(
            str(tmp_path), "run", final_iteration=21, previous_iteration=20,
            require_final_state=True,
        )


@pytest.mark.parametrize(
    "requested,diagnostic,history,expected_index,expected",
    [
        (5, {}, [None], 5, {}),
        (2, {"x": 1}, [{"x": 2}], 2, {"x": 1}),
        (0, None, [{"x": 2}], 0, {"x": 2}),
        (5, None, [{"x": 2}, {"x": 3}], 1, {"x": 3}),
        (1, None, [None, {}], 1, {}),
        (5, None, None, 5, None),
        (5, None, [], 5, None),
    ],
)
def test_final_replay_selection_preserves_explicit_state_and_history_identity(
    requested, diagnostic, history, expected_index, expected
):
    index, selected = relion_replay_module._select_final_replay_override(
        requested_index=requested,
        diagnostic_override=diagnostic,
        replay_overrides=history,
        has_overrides=history is not None and len(history) > 0,
        logger=relion_replay_module.logger,
    )
    assert index == expected_index
    assert selected == expected
    if diagnostic is not None:
        assert selected is diagnostic
    elif history:
        assert selected is history[expected_index]


@pytest.mark.parametrize("history", [[None], [{"x": 1}, None]])
def test_final_replay_missing_recorded_slot_fails_with_requested_index(history):
    with pytest.raises(RuntimeError, match="previous-state override at index 5"):
        relion_replay_module._select_final_replay_override(
            requested_index=5,
            diagnostic_override=None,
            replay_overrides=history,
            has_overrides=True,
            logger=relion_replay_module.logger,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_final_reference_substitution_preserves_each_half_dtype(dtype):
    means = [jnp.zeros(4, dtype=dtype), jnp.zeros(4, dtype=np.float64)]
    refs = [np.arange(4, dtype=np.float64) + 0.125, np.arange(4, dtype=np.float32)]
    result = relion_replay_module._prepare_final_replay_references(
        replay=SimpleNamespace(final_replay_reference_maps=refs, final_replay_source_iteration=2),
        diagnostic_override=None,
        numbered_iteration_count=2,
        means=means,
        final_join_means=means,
        k_class_enabled=False,
        logger=relion_replay_module.logger,
    )
    for index in range(2):
        assert result[index].dtype == means[index].dtype
        np.testing.assert_array_equal(result[index], refs[index].astype(means[index].dtype))


def test_absent_final_reference_substitution_preserves_list_identity():
    means = [jnp.zeros(4), jnp.ones(4)]
    assert (
        relion_replay_module._prepare_final_replay_references(
            replay=SimpleNamespace(final_replay_reference_maps=None, final_replay_source_iteration=99),
            diagnostic_override=None,
            numbered_iteration_count=2,
            means=means,
            final_join_means=means,
            k_class_enabled=True,
            logger=relion_replay_module.logger,
        )
        is means
    )


@pytest.mark.parametrize(
    "refs,source,diagnostic,kclass,error,match",
    [
        (None, 1, {}, False, RuntimeError, "source does not match"),
        ([np.zeros(4), np.zeros(4)], 1, None, True, RuntimeError, "source does not match"),
        ([np.zeros(4), np.zeros(4)], 2, None, True, RuntimeError, "K=1 only"),
        ([np.zeros(4)], 2, None, False, ValueError, "exactly two half maps"),
        ([np.zeros(4), np.zeros(3)], 2, None, False, ValueError, "shape mismatch"),
    ],
)
def test_final_reference_substitution_rejects_invalid_boundary_and_maps(refs, source, diagnostic, kclass, error, match):
    means = [jnp.zeros(4), jnp.zeros(4)]
    with pytest.raises(error, match=match):
        relion_replay_module._prepare_final_replay_references(
            replay=SimpleNamespace(final_replay_reference_maps=refs, final_replay_source_iteration=source),
            diagnostic_override=diagnostic,
            numbered_iteration_count=2,
            means=means,
            final_join_means=means,
            k_class_enabled=kclass,
            logger=relion_replay_module.logger,
        )
