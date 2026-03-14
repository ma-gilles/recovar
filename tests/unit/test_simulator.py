import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy")

import recovar.simulation.simulator as simulator
from recovar import core

pytestmark = pytest.mark.unit


def test_set_constant_ctf_sets_expected_columns():
    params = np.arange(2 * 11, dtype=np.float32).reshape(2, 11)
    out = simulator.set_constant_ctf(params.copy())
    np.testing.assert_array_equal(out[:, 0], 0.0)
    np.testing.assert_array_equal(out[:, 1], 0.0)
    np.testing.assert_array_equal(out[:, 4], 0.0)
    np.testing.assert_array_equal(out[:, 5], -1.0)


def test_uniform_rotation_sampling_reproducible():
    r1 = simulator.uniform_rotation_sampling(8, grid_size=64, seed=123)
    r2 = simulator.uniform_rotation_sampling(8, grid_size=64, seed=123)
    np.testing.assert_allclose(r1, r2)


def test_nonuniform_rotation_sampling_reproducible():
    r1 = simulator.nonuniform_rotation_sampling(8, grid_size=64, seed=123)
    r2 = simulator.nonuniform_rotation_sampling(8, grid_size=64, seed=123)
    np.testing.assert_allclose(r1, r2)


def test_cryo_rotation_batch_properties_and_shapes():
    single = simulator.cryo_rotation_batch([0.0, 0.0, 2.0], np.pi / 3)
    assert single.shape == (3, 3)
    np.testing.assert_allclose(single @ single.T, np.eye(3), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.det(single), 1.0, atol=1e-6, rtol=1e-6)

    U = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    theta = np.array([0.2, -0.1])
    batch = simulator.cryo_rotation_batch(U, theta)
    assert batch.shape == (2, 3, 3)
    np.testing.assert_allclose(
        batch @ np.transpose(batch, (0, 2, 1)),
        np.broadcast_to(np.eye(3), batch.shape),
        atol=1e-6,
        rtol=1e-6,
    )


def test_cryo_rotation_batch_rejects_bad_inputs():
    with pytest.raises(ValueError):
        simulator.cryo_rotation_batch(np.zeros((2, 2)), 0.0)
    with pytest.raises(ValueError):
        simulator.cryo_rotation_batch([0.0, 0.0, 0.0], 0.0)


def test_generate_contrast_params_shape():
    contrast, noise_scale = simulator.generate_contrast_params(10, noise_scale_std=0.2, contrast_std=0.1)
    assert contrast.shape == (10,)
    assert noise_scale.shape == (10,)


def test_get_noise_model_white():
    out = simulator.get_noise_model("white", grid_size=16)
    np.testing.assert_array_equal(out, np.ones(16 // 2 - 1))


def test_get_pose_ctf_generator_dispatch():
    assert simulator.get_pose_ctf_generator("uniform") is simulator.random_sampling_scheme
    assert simulator.get_pose_ctf_generator("noctf") is simulator.noctf_random_sampling_scheme
    assert simulator.get_pose_ctf_generator("kent") is simulator.kent_sampling_scheme

    f_nonuniform = simulator.get_pose_ctf_generator("nonuniform")
    assert callable(f_nonuniform)

    f_kent_args = simulator.get_pose_ctf_generator([10, 5, [1, 0, 0], [0, 1, 0]])
    assert callable(f_kent_args)

    f_default = simulator.get_pose_ctf_generator("unknown-option")
    assert callable(f_default)


def _make_fake_param_generator(n_cols=9):
    def _generator(n_images, grid_size):
        ctf = np.zeros((n_images, n_cols), dtype=np.float32)
        ctf[:, core.CTFParamIndex.VOLT] = 300.0
        ctf[:, core.CTFParamIndex.CS] = 2.7
        ctf[:, core.CTFParamIndex.W] = 0.1
        ctf[:, core.CTFParamIndex.CONTRAST] = 1.0
        rots = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        trans = np.zeros((n_images, 2), dtype=np.float32)
        return ctf, rots, trans

    return _generator


def test_generate_simulated_dataset_tilt_branch_wires_ctf_and_metadata(monkeypatch):
    volumes = np.ones((2, 4**3), dtype=np.complex64)
    voxel_size = 1.5
    n_images = 6
    n_tilts = 3
    volume_distribution = np.array([0.6, 0.4], dtype=np.float32)
    noise_variance = np.ones(3, dtype=np.float32)

    created = []

    def _fake_cryo_dataset(image_stack, voxel_size, rots, trans, ctf_params, ctf_evaluator=None, dataset_indices=None, grid_size=None, **kwargs):
        obj = type("FakeDataset", (), {})()
        obj.n_images = rots.shape[0]
        obj.image_shape = (grid_size, grid_size)
        obj.CTF_params = ctf_params
        obj.ctf_evaluator = ctf_evaluator
        obj.grid_size = grid_size
        obj.voxel_size = voxel_size
        obj.volume_shape = (grid_size, grid_size, grid_size)
        obj.translations = trans
        obj.rotation_matrices = rots
        created.append(obj)
        return obj

    monkeypatch.setattr(simulator.dataset, "CryoEMDataset", _fake_cryo_dataset)
    monkeypatch.setattr(simulator.utils, "get_gpu_memory_total", lambda: 10)
    monkeypatch.setattr(simulator.utils, "get_image_batch_size", lambda grid_size, gpu_mem: 2)
    monkeypatch.setattr(
        simulator,
        "simulate_data",
        lambda experiment_dataset, volumes, noise_variance, batch_size, image_assignments, per_image_contrast, per_image_noise_scale, **kwargs: np.zeros(
            (experiment_dataset.n_images, experiment_dataset.image_shape[0], experiment_dataset.image_shape[1]), dtype=np.float32
        ),
    )

    out = simulator.generate_simulated_dataset(
        volumes=volumes,
        voxel_size=voxel_size,
        volume_distribution=volume_distribution,
        n_images=n_images,
        noise_variance=noise_variance,
        noise_scale_std=0.0,
        contrast_std=0.0,
        put_extra_particles=False,
        dataset_param_generator=_make_fake_param_generator(),
        n_tilts=n_tilts,
        dose_per_tilt=2.0,
        angle_per_tilt=5.0,
        image_offset_n_std=0.0,
    )

    main_image_stack, ctf_params, rots, trans, simulation_info, _, tilt_groups = out
    assert main_image_stack.shape == (n_images, 4, 4)
    np.testing.assert_array_equal(tilt_groups, np.array([0, 0, 0, 1, 1, 1]))
    assert ctf_params.shape[1] == 11  # base 9 + dose + angle
    np.testing.assert_array_equal(
        ctf_params[:, core.CTFParamIndex.BFACTOR],
        -4 * ((np.arange(n_images) % n_tilts) + 0.5) * 2.0,
    )
    assert simulation_info["n_tilts"] == n_tilts
    np.testing.assert_array_equal(simulation_info["tilt_groups"], tilt_groups)
    assert created[0].ctf_evaluator.mode == core.CTFMode.CRYO_ET


def test_generate_simulated_dataset_extra_particles_and_outliers(monkeypatch):
    np.random.seed(0)
    volumes = np.ones((1, 4**3), dtype=np.complex64)
    voxel_size = 1.0
    n_images = 5
    volume_distribution = np.array([1.0], dtype=np.float32)
    noise_variance = np.ones(3, dtype=np.float32)
    outlier_volume = np.ones((4**3,), dtype=np.complex64)

    class _FakeDatasetObj:
        def __init__(self, n_images, grid_size, ctf_eval, ctf_params, rots, trans):
            self.n_images = n_images
            self.image_shape = (grid_size, grid_size)
            self.ctf_evaluator = ctf_eval
            self.CTF_params = ctf_params
            self.rotation_matrices = rots
            self.translations = trans
            self.grid_size = grid_size
            self.voxel_size = voxel_size
            self.volume_shape = (grid_size, grid_size, grid_size)

    def _fake_cryo_dataset(image_stack, voxel_size, rots, trans, ctf_params, ctf_evaluator=None, dataset_indices=None, grid_size=None, **kwargs):
        return _FakeDatasetObj(rots.shape[0], grid_size, ctf_evaluator, ctf_params, rots, trans)

    monkeypatch.setattr(simulator.dataset, "CryoEMDataset", _fake_cryo_dataset)
    monkeypatch.setattr(simulator.utils, "get_gpu_memory_total", lambda: 10)
    monkeypatch.setattr(simulator.utils, "get_image_batch_size", lambda grid_size, gpu_mem: 2)

    call_values = [1.0, 10.0, 99.0]
    calls = {"n": 0}

    def _fake_simulate_data(experiment_dataset, volumes, noise_variance, batch_size, image_assignments, per_image_contrast, per_image_noise_scale, **kwargs):
        val = call_values[calls["n"]]
        calls["n"] += 1
        return np.full((experiment_dataset.n_images, experiment_dataset.image_shape[0], experiment_dataset.image_shape[1]), val, dtype=np.float32)

    monkeypatch.setattr(simulator, "simulate_data", _fake_simulate_data)

    out = simulator.generate_simulated_dataset(
        volumes=volumes,
        voxel_size=voxel_size,
        volume_distribution=volume_distribution,
        n_images=n_images,
        noise_variance=noise_variance,
        noise_scale_std=0.0,
        contrast_std=0.0,
        put_extra_particles=True,
        percent_outliers=0.4,
        outlier_volume=outlier_volume,
        dataset_param_generator=_make_fake_param_generator(),
        image_offset_n_std=0.0,
    )
    main_image_stack, _, _, _, simulation_info, _, _ = out
    assert calls["n"] == 3  # main + extra + outlier

    image_assignments = simulation_info["image_assignment"]
    n_outliers = int(np.round(0.4 * n_images))
    assert np.sum(image_assignments == -1) == n_outliers

    # Non-outliers should be main(1) + extra(10) = 11, outliers replaced by 99.
    means = main_image_stack.mean(axis=(1, 2))
    np.testing.assert_array_equal(np.sort(np.unique(means)), np.array([11.0, 99.0], dtype=np.float32))
