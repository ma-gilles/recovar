from types import SimpleNamespace

import healpy as hp
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


def test_rotation_indices_to_matrices_matches_full_grid_rows():
    order = 2
    indices = np.array([0, 7, 191, 192, 387, 1024], dtype=np.int64)
    expected = em_sampling.get_rotation_grid(order, matrices=True)[indices]
    actual = em_sampling.rotation_indices_to_matrices(indices, order)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_remap_rotation_indices_to_finer_order_preserves_pixel_centers_and_psi():
    src_order = 1
    dst_order = 2
    src_nside = 2 ** src_order
    src_npix = hp.nside2npix(src_nside)
    dst_nside = 2 ** dst_order
    dst_npix = hp.nside2npix(dst_nside)

    src_indices = np.array(
        [0, src_npix + 5, 3 * src_npix + 7, 11 * src_npix + 13],
        dtype=np.int64,
    )
    dst_indices = em_sampling.remap_rotation_indices_to_order(
        src_indices, src_order, dst_order,
    )

    theta, phi = hp.pix2ang(src_nside, src_indices % src_npix)
    expected_pixels = hp.ang2pix(dst_nside, theta, phi)
    expected_psi = (src_indices // src_npix) * 2

    np.testing.assert_array_equal(dst_indices % dst_npix, expected_pixels)
    np.testing.assert_array_equal(dst_indices // dst_npix, expected_psi)


def test_local_rotation_grid_fast_uses_exact_prior_rotation_angles():
    """Local search with a tiny sigma must select the grid pixel whose
    matrix is angularly closest to the prior matrix.

    Updated 2026-04-09 (Task #100): The original version of this test had
    an off-by-swap convention bug — it computed the expected pixel via
    `hp.ang2pix(nside, prior_euler[1], prior_euler[0])`, treating the
    second Euler angle as polar and the first as azimuthal. That convention
    gave a pixel ~70 degrees AWAY from the prior matrix's actual view
    direction; the test passed only because get_local_rotation_grid_fast
    used the same buggy formula internally. After the Task #100 fix, the
    function correctly identifies pixels whose MATRIX view direction
    matches the prior. The test now uses axis-angle distance to verify
    the selected pixels are actually close.
    """
    from recovar import utils
    from scipy.spatial.transform import Rotation as SciPyRot

    coarse_order = 2
    fine_order = 3
    child_rotations, _ = em_sampling.get_oversampled_rotation_grid_from_samples(
        np.array([0], dtype=np.int64),
        coarse_order,
        oversampling_order=1,
    )
    child_eulers = utils.R_to_relion(child_rotations, degrees=True)
    prior_idx = np.argmin(np.abs(child_eulers[:, 2] - 3.75))
    prior_rotation = child_rotations[prior_idx : prior_idx + 1]

    selected_indices, log_prior = em_sampling.get_local_rotation_grid_fast(
        prior_rotation,
        sigma_rot=np.deg2rad(0.2),
        sigma_psi=np.deg2rad(2.0),
        healpix_order=fine_order,
        sigma_cutoff=2.0,
    )

    # Selected matrices must be angularly close to the prior matrix.
    n_pixels = hp.nside2npix(2 ** fine_order)
    M_prior = prior_rotation[0]

    selected_matrices = em_sampling.rotation_indices_to_matrices(
        selected_indices, fine_order,
    )
    # Axis-angle distance: ||log(M_prior^T @ M_sel)||
    R_diffs = np.einsum("ij,kjl->kil", M_prior.T, selected_matrices)
    traces = np.trace(R_diffs, axis1=1, axis2=2)
    angles_deg = np.rad2deg(np.arccos(np.clip((traces - 1) / 2, -1, 1)))

    # All selected matrices must be within ~5 deg of the prior (small
    # sigma=0.2 deg cone, but the order-3 grid has 7.5 deg pixel spacing,
    # so the closest grid pixel can be up to ~3.75 deg away in direction
    # plus a small psi offset).
    assert np.all(angles_deg < 6.0), (
        f"Some selected matrices are too far from prior: max={angles_deg.max():.2f} deg"
    )

    # The selected pixels should be the unique pixel index closest to the
    # prior, with at least one selected psi value.
    same_pixel_indices = selected_indices % n_pixels
    unique_pixels = np.unique(same_pixel_indices)
    assert len(unique_pixels) == 1, (
        f"Expected 1 closest pixel, got {len(unique_pixels)}: {unique_pixels}"
    )

    # The log_prior values for the selected indices should be finite (not
    # the -1e30 mask sentinel) for at least one psi.
    finite_lp = log_prior[np.isfinite(log_prior)]
    assert finite_lp.size > 0
    assert np.all(finite_lp > -1e25)


def test_local_rotation_grid_fast_respects_provided_perturbed_grid():
    from recovar import utils

    order = 2
    prior_index = 137
    random_perturbation = 0.3
    sigma_rot = np.deg2rad(0.2)
    sigma_psi = np.deg2rad(2.0)

    base_rotations = np.asarray(em_sampling.get_rotation_grid(order, matrices=True), dtype=np.float32)
    perturbed_rotations = em_sampling.apply_relion_rotation_perturbation(
        base_rotations,
        random_perturbation,
        em_sampling.relion_angular_sampling_deg(order),
    ).astype(np.float32)
    perturbed_eulers = utils.R_to_relion(perturbed_rotations, degrees=True).astype(np.float32)
    grid_metadata = em_sampling.build_local_search_grid_metadata(order, perturbed_eulers)

    selected_indices, log_prior = em_sampling.get_local_rotation_grid_fast(
        perturbed_eulers[prior_index : prior_index + 1],
        sigma_rot=sigma_rot,
        sigma_psi=sigma_psi,
        healpix_order=order,
        sigma_cutoff=2.0,
        per_image=True,
        grid_metadata=grid_metadata,
    )
    best_idx = int(selected_indices[int(np.argmax(log_prior[0]))])

    assert best_idx == prior_index
    np.testing.assert_allclose(
        perturbed_rotations[best_idx],
        perturbed_rotations[prior_index],
        rtol=1e-6,
        atol=1e-6,
    )


def test_perturbed_rotation_grid_metadata_is_not_factorized():
    from recovar import utils

    order = 2
    base_metadata = em_sampling.build_local_search_grid_metadata(order)
    assert base_metadata["mode"] == "factorized"

    base_rotations = np.asarray(em_sampling.get_rotation_grid(order, matrices=True), dtype=np.float32)
    perturbed_rotations = em_sampling.apply_relion_rotation_perturbation(
        base_rotations,
        random_perturbation=0.3,
        angular_sampling_deg=em_sampling.relion_angular_sampling_deg(order),
    ).astype(np.float32)
    perturbed_eulers = utils.R_to_relion(perturbed_rotations, degrees=True).astype(np.float32)
    perturbed_metadata = em_sampling.build_local_search_grid_metadata(order, perturbed_eulers)

    assert perturbed_metadata["mode"] == "full"


def test_factorized_local_search_metadata_matches_relion_grid_view_directions():
    order = 3
    metadata = em_sampling.build_local_search_grid_metadata(order)
    n_pixels = hp.nside2npix(2**order)
    n_psi = em_sampling.rotation_grid_n_in_planes(order)
    relion_grid = np.asarray(em_sampling.get_relion_rotation_grid(order), dtype=np.float64).reshape(
        n_psi,
        n_pixels,
        3,
        3,
    )
    psi0_view_dirs = relion_grid[0, :, :, 2]
    psi0_view_dirs /= np.linalg.norm(psi0_view_dirs, axis=1, keepdims=True)

    dots = np.sum(np.asarray(metadata["dir_vecs"], dtype=np.float64) * psi0_view_dirs, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang_deg = np.rad2deg(np.arccos(dots))
    assert float(np.max(ang_deg)) < 1e-4


def test_local_rotation_grid_fast_factorized_support_stays_in_cone():
    from recovar import utils

    order = 4
    prior_index = 12345
    sigma_deg = 7.5
    sigma_cutoff = 3.0

    relion_grid = np.asarray(em_sampling.get_relion_rotation_grid(order), dtype=np.float64)
    relion_eulers = np.asarray(em_sampling.get_relion_rotation_grid_eulers(order), dtype=np.float64)
    prior_euler = relion_eulers[prior_index : prior_index + 1]
    prior_matrix = utils.R_from_relion(prior_euler, degrees=True)[0]

    selected_indices, log_prior = em_sampling.get_local_rotation_grid_fast(
        prior_euler,
        sigma_rot=np.deg2rad(sigma_deg),
        sigma_psi=np.deg2rad(sigma_deg),
        healpix_order=order,
        sigma_cutoff=sigma_cutoff,
        per_image=True,
    )

    finite_mask = np.asarray(log_prior[0]) > -1e20
    selected_matrices = relion_grid[selected_indices[finite_mask]]
    r_diffs = np.einsum("ij,kjl->kil", prior_matrix.T, selected_matrices)
    traces = np.trace(r_diffs, axis1=1, axis2=2)
    ang_deg = np.rad2deg(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))

    assert ang_deg.size > 0
    assert float(np.max(ang_deg)) < sigma_cutoff * sigma_deg + 7.5


def test_local_rotation_grid_fast_uses_max_of_sigma_rot_and_sigma_psi_for_direction_cone():
    from recovar import utils

    order = 4
    prior_index = 12345
    sigma_rot_deg = 0.2
    sigma_psi_deg = 7.5
    sigma_cutoff = 3.0

    relion_eulers = np.asarray(em_sampling.get_relion_rotation_grid_eulers(order), dtype=np.float64)
    prior_euler = relion_eulers[prior_index : prior_index + 1]
    prior_matrix = utils.R_from_relion(prior_euler, degrees=True)[0]
    prior_view = prior_matrix[:, 2] / np.linalg.norm(prior_matrix[:, 2])

    selected_indices, _ = em_sampling.get_local_rotation_grid_fast(
        prior_euler,
        sigma_rot=np.deg2rad(sigma_rot_deg),
        sigma_psi=np.deg2rad(sigma_psi_deg),
        healpix_order=order,
        sigma_cutoff=sigma_cutoff,
        per_image=True,
    )

    n_pixels = hp.nside2npix(2**order)
    unique_pixels = np.unique(selected_indices % n_pixels)
    assert unique_pixels.size > 1

    relion_grid = np.asarray(em_sampling.get_relion_rotation_grid(order), dtype=np.float64).reshape(-1, 3, 3)
    selected_views = relion_grid[unique_pixels, :, 2]
    selected_views /= np.linalg.norm(selected_views, axis=1, keepdims=True)
    diff_deg = np.rad2deg(np.arccos(np.clip(np.sum(selected_views * prior_view[None, :], axis=1), -1.0, 1.0)))

    assert float(np.max(diff_deg)) < sigma_cutoff * max(sigma_rot_deg, sigma_psi_deg) + 1e-4


def test_local_rotation_grid_fast_per_image_priors_prefer_each_image_peak():
    fine_order = 3
    n_pixels = hp.nside2npix(2 ** fine_order)
    prior_indices = np.array([0, n_pixels], dtype=np.int64)
    prior_rotations = em_sampling.rotation_indices_to_matrices(prior_indices, fine_order)

    selected_indices, log_prior = em_sampling.get_local_rotation_grid_fast(
        prior_rotations,
        sigma_rot=np.deg2rad(0.2),
        sigma_psi=np.deg2rad(2.0),
        healpix_order=fine_order,
        sigma_cutoff=2.0,
        per_image=True,
    )

    assert log_prior.shape == (2, selected_indices.shape[0])
    per_image_best = selected_indices[np.argmax(log_prior, axis=1)]

    reference_best = []
    for i in range(2):
        ref_indices, ref_log_prior = em_sampling.get_local_rotation_grid_fast(
            prior_rotations[i : i + 1],
            sigma_rot=np.deg2rad(0.2),
            sigma_psi=np.deg2rad(2.0),
            healpix_order=fine_order,
            sigma_cutoff=2.0,
        )
        reference_best.append(ref_indices[np.argmax(ref_log_prior)])

    np.testing.assert_array_equal(
        per_image_best,
        np.array(reference_best, dtype=np.int64),
    )


def test_rotation_grid_size_matches_grid_shape():
    for order in range(4):
        grid = em_sampling.get_rotation_grid(order, matrices=True)
        assert em_sampling.rotation_grid_size(order) == grid.shape[0]


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

        def update_poses(self, rots, trans):
            self.rotation_matrices = rots
            self.translations = trans

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

    from recovar.reconstruction import regularization, noise
    from recovar.heterogeneity import locres

    monkeypatch.setattr(regularization, "get_fsc_gpu", lambda *_args, **_kwargs: np.array([0.9, 0.7], dtype=np.float32))
    monkeypatch.setattr(
        regularization, "average_over_shells", lambda *_args, **_kwargs: np.array([1.0, 1.0], dtype=np.float32)
    )
    monkeypatch.setattr(
        em_iterations.utils,
        "make_radial_image",
        lambda _ps, shape, extend_last_frequency=True: np.ones(int(np.prod(shape)), dtype=np.float32),
    )
    monkeypatch.setattr(
        noise, "estimate_noise_level_no_masks", lambda *_args, **_kwargs: np.array([0.2, 0.3], dtype=np.float32)
    )
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

        def update_poses(self, rots, trans):
            self.rotation_matrices = rots
            self.translations = trans

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

    from recovar.reconstruction import regularization, relion_functions, noise
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
    monkeypatch.setattr(
        noise, "estimate_noise_level_no_masks", lambda *_args, **_kwargs: np.array([0.2, 0.3], dtype=np.float32)
    )
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
