from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("healpy")

import recovar.em.iterations as em_iterations
import recovar.em.sampling as em_sampling

pytestmark = pytest.mark.unit


def test_translations_to_indices_maps_centered_integer_offsets():
    image_shape = (8, 8)
    translations = np.array(
        [
            [0, 0],
            [1, -1],
            [-2, 3],
        ],
        dtype=np.int32,
    )

    out = np.asarray(em_sampling.translations_to_indices(translations, image_shape))
    expected = np.array([36, 29, 58], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)


def test_get_translation_grid_respects_radius_and_stride():
    grid = em_sampling.get_translation_grid(max_pixel=2, pixel_offset=2)
    grid_set = {tuple(v.tolist()) for v in grid}

    assert (0, 0) in grid_set
    assert (2, 0) in grid_set
    assert (-2, 0) in grid_set
    assert (0, 2) in grid_set
    assert (0, -2) in grid_set
    assert (2, 2) not in grid_set
    assert (-2, -2) not in grid_set
    assert all((x % 2 == 0 and y % 2 == 0) for x, y in grid_set)


def test_get_rotation_grid_shapes_with_and_without_matrices():
    euler = em_sampling.get_rotation_grid(nside_level=0, n_in_planes=4, matrices=False)
    mats = em_sampling.get_rotation_grid(nside_level=0, n_in_planes=4, matrices=True)

    assert euler.shape == (48, 3)
    assert mats.shape == (48, 3, 3)
    assert np.isfinite(euler).all()
    assert np.isfinite(mats).all()


def test_E_M_batches_2_small_memory_forces_single_image_batches():
    class _Dataset:
        n_units = 5

    class _State:
        name = "EM"
        mean = np.zeros((8,), dtype=np.complex64)
        noise_variance = np.ones((4,), dtype=np.float32)
        sgd_batchsize = 2

        def __init__(self):
            self.calls = []

        def E_step(self, _dataset, rotations, translations, _disc_type, big_image_batch):
            n = len(big_image_batch)
            probs = np.zeros((n, rotations.shape[0], translations.shape[0]), dtype=np.float32)
            probs[:, 0, 0] = 0.1
            probs[:, 1, 2] = 0.9
            return probs

        def M_step(self, _dataset, probabilities, _rotations, _translations, _disc_type, big_image_batch):
            self.calls.append((np.array(big_image_batch, dtype=np.int32), probabilities.shape))

    rotations = np.zeros((2, 3, 3), dtype=np.float32)
    translations = np.zeros((3, 2), dtype=np.float32)
    state = _State()

    out_state, hard = em_iterations.E_M_batches_2(
        _Dataset(),
        state,
        rotations,
        translations,
        disc_type="linear_interp",
        memory_to_use=1e-12,
    )

    assert out_state is state
    np.testing.assert_array_equal(hard, np.full((5,), 5, dtype=np.int64))
    seen = np.concatenate([idx for idx, _shape in state.calls])
    np.testing.assert_array_equal(np.sort(seen), np.arange(5, dtype=np.int32))
    assert all(shape[0] == 1 for _idx, shape in state.calls)


def test_E_M_batches_2_sgd_uses_explicit_sgd_batchsize():
    class _Dataset:
        n_units = 5

    class _State:
        name = "SGD"
        mean = np.zeros((4,), dtype=np.complex64)
        noise_variance = np.ones((4,), dtype=np.float32)
        sgd_batchsize = 2

        def __init__(self):
            self.calls = []

        def E_step(self, _dataset, rotations, translations, _disc_type, big_image_batch):
            probs = np.ones((len(big_image_batch), rotations.shape[0], translations.shape[0]), dtype=np.float32)
            return probs / np.sum(probs, axis=(1, 2), keepdims=True)

        def M_step(self, _dataset, probabilities, _rotations, _translations, _disc_type, big_image_batch):
            self.calls.append((np.array(big_image_batch), probabilities.shape))

    rotations = np.zeros((1, 3, 3), dtype=np.float32)
    translations = np.zeros((1, 2), dtype=np.float32)
    state = _State()
    _out_state, hard = em_iterations.E_M_batches_2(
        _Dataset(),
        state,
        rotations,
        translations,
        disc_type="linear_interp",
        memory_to_use=1e-12,
    )

    assert len(state.calls) == 3
    np.testing.assert_array_equal(hard, np.zeros((5,), dtype=np.int64))


def test_E_M_batches_2_sgd_float_batchsize_is_safely_cast_to_int():
    class _Dataset:
        n_units = 5

    class _State:
        name = "SGD"
        mean = np.zeros((4,), dtype=np.complex64)
        noise_variance = np.ones((4,), dtype=np.float32)
        sgd_batchsize = 2.7

        def __init__(self):
            self.calls = []

        def E_step(self, _dataset, rotations, translations, _disc_type, big_image_batch):
            probs = np.ones((len(big_image_batch), rotations.shape[0], translations.shape[0]), dtype=np.float32)
            return probs / np.sum(probs, axis=(1, 2), keepdims=True)

        def M_step(self, _dataset, probabilities, _rotations, _translations, _disc_type, big_image_batch):
            self.calls.append((np.array(big_image_batch), probabilities.shape))

    rotations = np.zeros((1, 3, 3), dtype=np.float32)
    translations = np.zeros((1, 2), dtype=np.float32)
    state = _State()

    _out_state, hard = em_iterations.E_M_batches_2(
        _Dataset(),
        state,
        rotations,
        translations,
        disc_type="linear_interp",
        memory_to_use=1e-12,
    )

    # 5 units with int(2.7)=2 => chunks [0,1], [2,3], [4]
    assert len(state.calls) == 3
    np.testing.assert_array_equal(hard, np.zeros((5,), dtype=np.int64))


def test_E_M_batches_2_rejects_invalid_hidden_or_sgd_batchsize():
    dataset = SimpleNamespace(n_units=3)
    state = SimpleNamespace(name="SGD", sgd_batchsize=0, E_step=lambda *_: None, M_step=lambda *_: None)

    with pytest.raises(ValueError, match="at least one rotation"):
        em_iterations.E_M_batches_2(
            dataset,
            state,
            rotations=np.zeros((0, 3, 3), dtype=np.float32),
            translations=np.zeros((1, 2), dtype=np.float32),
            disc_type="linear_interp",
        )

    with pytest.raises(ValueError, match="at least one translation"):
        em_iterations.E_M_batches_2(
            dataset,
            state,
            rotations=np.zeros((1, 3, 3), dtype=np.float32),
            translations=np.zeros((0, 2), dtype=np.float32),
            disc_type="linear_interp",
        )

    with pytest.raises(ValueError, match="batch size must be >= 1"):
        em_iterations.E_M_batches_2(
            dataset,
            state,
            rotations=np.zeros((1, 3, 3), dtype=np.float32),
            translations=np.zeros((1, 2), dtype=np.float32),
            disc_type="linear_interp",
        )


def test_split_E_M_v2_updates_state_means_noise_and_pose_assignments(monkeypatch):
    class _Dataset:
        def __init__(self, n_units, voxel_size):
            self.n_units = n_units
            self.voxel_size = voxel_size
            self.image_shape = (2, 2)
            self.volume_shape = (2, 2, 2)
            self.rotation_matrices = None
            self.translations = None

        def get_valid_frequency_indices(self, _cutoff):
            mask = np.zeros((8,), dtype=bool)
            mask[0] = True
            return mask

    class _State:
        def __init__(self, mean):
            self.name = "SGD"
            self.mean = mean.astype(np.complex64)
            self.noise_variance = np.ones((4,), dtype=np.float32)
            self.mean_variance = np.ones((8,), dtype=np.float32)
            self.finish_calls = 0

        def finish_up_M_step(self, _dataset, _disc_type):
            self.finish_calls += 1

    d0 = _Dataset(n_units=4, voxel_size=2.0)
    d1 = _Dataset(n_units=4, voxel_size=2.0)
    s0 = _State(mean=np.arange(8, dtype=np.float32))
    s1 = _State(mean=np.arange(8, dtype=np.float32) + 10.0)

    h0 = np.array([0, 1, 0, 1], dtype=np.int32)
    h1 = np.array([1, 0, 1, 0], dtype=np.int32)
    monkeypatch.setattr(
        em_iterations,
        "E_M_batches_2",
        lambda ds, st, _rots, _trs, _disc: (st, h0 if ds is d0 else h1),
    )
    monkeypatch.setattr(
        em_iterations,
        "hard_assignment_idx_to_pose",
        lambda hard, _rots, _trs: (
            np.tile(np.eye(3, dtype=np.float32)[None], (len(hard), 1, 1)),
            np.stack([hard.astype(np.float32), -hard.astype(np.float32)], axis=1),
        ),
    )

    from recovar import regularization, noise
    from recovar.heterogeneity import locres

    monkeypatch.setattr(regularization, "get_fsc_gpu", lambda *_args, **_kwargs: np.array([0.9, 0.7], dtype=np.float32))
    monkeypatch.setattr(regularization, "average_over_shells", lambda *_args, **_kwargs: np.array([1.0, 1.0], dtype=np.float32))
    monkeypatch.setattr(em_iterations.utils, "make_radial_image", lambda _ps, shape, extend_last_frequency=True: np.ones(int(np.prod(shape)), dtype=np.float32))
    monkeypatch.setattr(noise, "estimate_noise_level_no_masks", lambda *_args, **_kwargs: np.array([0.2, 0.3], dtype=np.float32))
    monkeypatch.setattr(noise, "make_radial_noise", lambda _n, _shape: np.ones((4,), dtype=np.float32) * 0.5)
    monkeypatch.setattr(locres, "find_fsc_resol", lambda _fsc, threshold=1 / 7: 4.0)

    rotations = np.zeros((2, 3, 3), dtype=np.float32)
    translations = np.zeros((2, 2), dtype=np.float32)
    out_states, pix_res, hard = em_iterations.split_E_M_v2(
        [d0, d1],
        [s0, s1],
        rotations,
        translations,
        disc_type="linear_interp",
        average_up_to_angstrom=5.0,
    )

    assert pix_res == 4.0
    assert len(hard) == 2
    np.testing.assert_array_equal(hard[0], h0)
    np.testing.assert_array_equal(hard[1], h1)
    assert s0.finish_calls == 1
    assert s1.finish_calls == 1
    np.testing.assert_allclose(out_states[0].noise_variance, np.ones((4,), dtype=np.float32) * 0.5)
    np.testing.assert_allclose(out_states[1].noise_variance, np.ones((4,), dtype=np.float32) * 0.5)
    np.testing.assert_allclose(out_states[0].mean_variance, np.ones((8,), dtype=np.float32) * 2.0 + 2e-6)
    np.testing.assert_allclose(out_states[1].mean_variance, np.ones((8,), dtype=np.float32) * 2.0 + 2e-6)
    # low-res averaging updates index 0 to shared value
    expected0 = (0.0 + 10.0) / 2.0
    assert np.isclose(out_states[0].mean[0].real, expected0)
    assert np.isclose(out_states[1].mean[0].real, expected0)
    assert d0.rotation_matrices.shape == (4, 3, 3)
    assert d1.rotation_matrices.shape == (4, 3, 3)
    assert d0.translations.shape == (4, 2)
    assert d1.translations.shape == (4, 2)


def test_split_E_M_v2_heterogeneous_branch_updates_covariance_prior_and_masks_u(monkeypatch):
    class _Dataset:
        def __init__(self):
            self.n_units = 3
            self.voxel_size = 1.5
            self.image_shape = (2, 2)
            self.volume_shape = (2, 2, 2)
            self.rotation_matrices = None
            self.translations = None

        def get_valid_frequency_indices(self, _cutoff):
            mask = np.zeros((8,), dtype=np.float32)
            mask[:3] = 1.0
            return mask

    class _State:
        def __init__(self, mean):
            self.name = "HeterogeneousEM"
            self.mean = mean.astype(np.complex64)
            self.noise_variance = np.ones((4,), dtype=np.float32)
            self.mean_variance = np.ones((8,), dtype=np.float32)
            self.finish_calls = 0
            self.Ft_CTF = np.ones((8,), dtype=np.float32)
            self.Ft_y = np.ones((8,), dtype=np.complex64)
            self.u = np.ones((2, 8), dtype=np.float32)
            self.subspace = np.ones((8, 2), dtype=np.float32)
            self.H = np.ones((8,), dtype=np.float32)
            self.B = np.ones((8,), dtype=np.complex64)
            self.covariance_prior = np.zeros((8,), dtype=np.float32)
            self.covariance_options = {
                "substract_shell_mean": False,
                "left_kernel": "triangular",
                "use_spherical_mask": False,
                "grid_correct": False,
                "prior_n_iterations": 1,
                "downsample_from_fsc": False,
            }

        def finish_up_M_step(self, _dataset, _disc_type):
            self.finish_calls += 1

    d0 = _Dataset()
    d1 = _Dataset()
    s0 = _State(mean=np.arange(8, dtype=np.float32))
    s1 = _State(mean=np.arange(8, dtype=np.float32) + 2.0)

    h0 = np.array([0, 1, 2], dtype=np.int32)
    h1 = np.array([2, 1, 0], dtype=np.int32)
    monkeypatch.setattr(
        em_iterations,
        "E_M_batches_2",
        lambda ds, st, _rots, _trs, _disc: (st, h0 if ds is d0 else h1),
    )
    monkeypatch.setattr(
        em_iterations,
        "hard_assignment_idx_to_pose",
        lambda hard, _rots, _trs: (
            np.tile(np.eye(3, dtype=np.float32)[None], (len(hard), 1, 1)),
            np.stack([hard.astype(np.float32), hard.astype(np.float32)], axis=1),
        ),
    )

    from recovar import regularization, relion_functions, noise
    from recovar.heterogeneity import locres

    monkeypatch.setattr(
        relion_functions,
        "post_process_from_filter",
        lambda _dataset, _ft_ctf, _ft_y, tau=None, disc_type="linear_interp": np.ones((8,), dtype=np.complex64) * 2.0,
    )
    monkeypatch.setattr(
        regularization,
        "compute_relion_prior",
        lambda _datasets, _noise_variance, _m0, _m1, _n: (
            np.ones((8,), dtype=np.float32) * 7.0,
            np.array([0.9, 0.8], dtype=np.float32),
            None,
        ),
    )
    monkeypatch.setattr(
        regularization,
        "prior_iteration_relion_style_batch",
        lambda *_args, **_kwargs: (None, np.ones((8,), dtype=np.float32) * 9.0, None),
    )
    monkeypatch.setattr(noise, "estimate_noise_level_no_masks", lambda *_args, **_kwargs: np.array([0.2, 0.3], dtype=np.float32))
    monkeypatch.setattr(noise, "make_radial_noise", lambda _n, _shape: np.ones((4,), dtype=np.float32) * 0.4)
    monkeypatch.setattr(locres, "find_fsc_resol", lambda _fsc, threshold=1 / 7: 3.0)

    rotations = np.zeros((3, 3, 3), dtype=np.float32)
    translations = np.zeros((2, 2), dtype=np.float32)
    out_states, pix_res, hard = em_iterations.split_E_M_v2(
        [d0, d1],
        [s0, s1],
        rotations,
        translations,
        disc_type="linear_interp",
        average_up_to_angstrom=None,
    )

    assert pix_res == 3.0
    np.testing.assert_array_equal(hard[0], h0)
    np.testing.assert_array_equal(hard[1], h1)
    assert s0.finish_calls == 1
    assert s1.finish_calls == 1
    np.testing.assert_allclose(out_states[0].noise_variance, np.ones((4,), dtype=np.float32) * 0.4)
    np.testing.assert_allclose(out_states[1].noise_variance, np.ones((4,), dtype=np.float32) * 0.4)
    np.testing.assert_allclose(out_states[0].mean_variance, np.ones((8,), dtype=np.float32) * 7.0)
    np.testing.assert_allclose(out_states[1].mean_variance, np.ones((8,), dtype=np.float32) * 7.0)
    np.testing.assert_allclose(out_states[0].covariance_prior, np.ones((8,), dtype=np.float32) * 9.0)
    np.testing.assert_allclose(out_states[1].covariance_prior, np.ones((8,), dtype=np.float32) * 9.0)

    # Only first 3 frequency bins remain due to valid-frequency masking.
    assert np.allclose(out_states[0].u[:, :3], 1.0)
    assert np.allclose(out_states[0].u[:, 3:], 0.0)
    assert np.allclose(out_states[1].u[:, :3], 1.0)
    assert np.allclose(out_states[1].u[:, 3:], 0.0)
