from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("healpy")

import recovar.em.iterations as em_iterations
import recovar.em.sampling as em_sampling

pytestmark = pytest.mark.unit


def test_read_relion_sampling_metadata_includes_psi_step(tmp_path):
    sampling_star = tmp_path / "run_it010_sampling.star"
    sampling_star.write_text(
        "\n".join(
            [
                "_rlnSamplingPerturbInstance 0.47674",
                "_rlnSamplingPerturbFactor 0.5",
                "_rlnHealpixOrder 6",
                "_rlnPsiStep 0.9375",
                "_rlnOffsetRange 1.853",
                "_rlnOffsetStep 1.237",
                "",
            ]
        )
    )

    meta = em_sampling.read_relion_sampling_metadata(sampling_star)

    assert meta["random_perturbation"] == pytest.approx(0.47674)
    assert meta["perturbation_factor"] == pytest.approx(0.5)
    assert meta["healpix_order"] == 6
    assert meta["psi_step"] == pytest.approx(0.9375)
    assert meta["offset_range"] == pytest.approx(1.853)
    assert meta["offset_step"] == pytest.approx(1.237)


def test_read_relion_model_metadata_includes_local_prior_sigmas(tmp_path):
    model_star = tmp_path / "run_it010_half1_model.star"
    model_star.write_text(
        "\n".join(
            [
                "_rlnCurrentImageSize 128",
                "_rlnCurrentResolution 4.250000",
                "_rlnOrientationalPriorMode 1",
                "_rlnSigmaPriorRotAngle 3.750000",
                "_rlnSigmaPriorTiltAngle 3.750000",
                "_rlnSigmaPriorPsiAngle 1.875000",
                "",
            ]
        )
    )

    meta = em_sampling.read_relion_model_metadata(model_star)

    assert meta["current_image_size"] == 128
    assert meta["current_resolution"] == pytest.approx(4.25)
    assert meta["orientational_prior_mode"] == 1
    assert meta["sigma_prior_rot_angle"] == pytest.approx(3.75)
    assert meta["sigma_prior_tilt_angle"] == pytest.approx(3.75)
    assert meta["sigma_prior_psi_angle"] == pytest.approx(1.875)


def test_read_relion_optimiser_metadata_reads_replay_accuracies(tmp_path):
    optimiser_star = tmp_path / "run_it010_optimiser.star"
    optimiser_star.write_text(
        "\n".join(
            [
                "_rlnOverallAccuracyRotations 1.030",
                "_rlnOverallAccuracyTranslationsAngst 1.649",
                "_rlnHasConverged 0",
                "_rlnNumberOfIterWithoutResolutionGain 1",
                "_rlnChangesOptimalOrientations 0.25",
                "_rlnChangesOptimalOffsets 0.33",
                "_rlnSmallestChangesOrientations 0.5405",
                "_rlnSmallestChangesOffsets 0.4216",
                "",
            ]
        )
    )

    meta = em_sampling.read_relion_optimiser_metadata(optimiser_star)

    assert meta["overall_accuracy_rotations"] == pytest.approx(1.030)
    assert meta["overall_accuracy_translations_angst"] == pytest.approx(1.649)
    assert meta["has_converged"] == 0
    assert meta["number_iter_without_resolution_gain"] == 1
    assert meta["changes_optimal_orientations"] == pytest.approx(0.25)
    assert meta["changes_optimal_offsets"] == pytest.approx(0.33)
    assert meta["smallest_changes_orientations"] == pytest.approx(0.5405)
    assert meta["smallest_changes_offsets"] == pytest.approx(0.4216)


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
    n_pixels = hp.nside2npix(2**fine_order)
    M_prior = prior_rotation[0]

    full_grid = em_sampling.get_relion_rotation_grid(fine_order)
    selected_matrices = full_grid[selected_indices]
    # Axis-angle distance: ||log(M_prior^T @ M_sel)||
    R_diffs = np.einsum("ij,kjl->kil", M_prior.T, selected_matrices)
    traces = np.trace(R_diffs, axis1=1, axis2=2)
    angles_deg = np.rad2deg(np.arccos(np.clip((traces - 1) / 2, -1, 1)))

    # All selected matrices must be within ~5 deg of the prior (small
    # sigma=0.2 deg cone, but the order-3 grid has 7.5 deg pixel spacing,
    # so the closest grid pixel can be up to ~3.75 deg away in direction
    # plus a small psi offset).
    assert np.all(angles_deg < 6.0), f"Some selected matrices are too far from prior: max={angles_deg.max():.2f} deg"

    # The selected pixels should be the unique pixel index closest to the
    # prior, with at least one selected psi value.
    same_pixel_indices = selected_indices % n_pixels
    unique_pixels = np.unique(same_pixel_indices)
    assert len(unique_pixels) == 1, f"Expected 1 closest pixel, got {len(unique_pixels)}: {unique_pixels}"

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


def test_perturbed_rotation_grid_metadata_reuses_precomputed_rotations():
    from recovar import utils

    order = 2
    base_rotations = np.asarray(em_sampling.get_rotation_grid(order, matrices=True), dtype=np.float32)
    perturbed_rotations = em_sampling.apply_relion_rotation_perturbation(
        base_rotations,
        random_perturbation=0.3,
        angular_sampling_deg=em_sampling.relion_angular_sampling_deg(order),
    ).astype(np.float32)
    perturbed_eulers = utils.R_to_relion(perturbed_rotations, degrees=True).astype(np.float32)

    metadata_from_eulers = em_sampling.build_local_search_grid_metadata(order, perturbed_eulers)
    metadata_from_rotations = em_sampling.build_local_search_grid_metadata(
        order,
        perturbed_eulers,
        grid_rotations=perturbed_rotations,
    )

    assert metadata_from_eulers["mode"] == metadata_from_rotations["mode"] == "full"
    np.testing.assert_allclose(
        metadata_from_rotations["dir_vecs_full"],
        metadata_from_eulers["dir_vecs_full"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        metadata_from_rotations["psi_deg_full"],
        metadata_from_eulers["psi_deg_full"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_relion_psi_from_rotation_matrices_matches_full_euler_conversion():
    from recovar import utils

    order = 3
    base_rotations = np.asarray(em_sampling.get_rotation_grid(order, matrices=True), dtype=np.float32)
    perturbed_rotations = em_sampling.apply_relion_rotation_perturbation(
        base_rotations,
        random_perturbation=0.3,
        angular_sampling_deg=em_sampling.relion_angular_sampling_deg(order),
    ).astype(np.float32)
    sample = perturbed_rotations[::97]

    psi_fast = em_sampling.relion_psi_from_rotation_matrices(sample)
    psi_ref = utils.R_to_relion(sample, degrees=True)[:, 2].astype(np.float32)
    np.testing.assert_allclose(psi_fast, psi_ref, rtol=1e-5, atol=1e-5)


def test_local_rotation_grid_fast_full_mode_matches_reference_loop():
    from recovar import utils

    def _reference_full_mode(prior_rotations, sigma_rot, sigma_psi, metadata):
        prior_rotations = np.asarray(prior_rotations, dtype=np.float64).reshape(-1, 3, 3)
        prior_eulers = utils.R_to_relion(prior_rotations, degrees=True)
        prior_dir_vecs = np.asarray(prior_rotations[:, 2, :], dtype=np.float64)
        prior_dir_vecs /= np.linalg.norm(prior_dir_vecs, axis=1, keepdims=True)

        dir_vecs_full = np.asarray(metadata["dir_vecs_full"], dtype=np.float64)
        psi_deg_full = np.asarray(metadata["psi_deg_full"], dtype=np.float64)
        sigma_rot_deg = float(np.rad2deg(sigma_rot))
        sigma_psi_deg = float(np.rad2deg(sigma_psi))
        biggest_sigma_deg = sigma_rot_deg
        sigma_rot_scale = max(biggest_sigma_deg, np.finfo(np.float64).tiny)
        sigma_psi_scale = max(sigma_psi_deg, np.finfo(np.float64).tiny)

        selected_union = set()
        prior_entries = []
        for i in range(prior_rotations.shape[0]):
            if sigma_rot_deg > 0.0:
                dots = np.clip(dir_vecs_full @ prior_dir_vecs[i], -1.0, 1.0)
                diffang = np.rad2deg(np.arccos(dots))
                dir_mask = diffang < 3.0 * biggest_sigma_deg
            else:
                diffang = np.zeros(dir_vecs_full.shape[0], dtype=np.float64)
                dir_mask = np.ones(dir_vecs_full.shape[0], dtype=bool)

            if sigma_psi_deg > 0.0:
                diffpsi = em_sampling._wrapped_abs_diff_deg(
                    psi_deg_full,
                    float(np.mod(prior_eulers[i, 2], 360.0)),
                )
                psi_mask = diffpsi < 3.0 * sigma_psi_deg
            else:
                diffpsi = np.zeros(dir_vecs_full.shape[0], dtype=np.float64)
                psi_mask = np.ones(dir_vecs_full.shape[0], dtype=bool)

            joint_mask = dir_mask & psi_mask
            flat_indices = np.flatnonzero(joint_mask).astype(np.int64)
            if flat_indices.size == 0:
                joint_cost = np.zeros(dir_vecs_full.shape[0], dtype=np.float64)
                if sigma_rot_deg > 0.0:
                    joint_cost += (diffang / sigma_rot_scale) ** 2
                if sigma_psi_deg > 0.0:
                    joint_cost += (diffpsi / sigma_psi_scale) ** 2
                flat_indices = np.array([int(np.argmin(joint_cost))], dtype=np.int64)
                flat_log_prior = np.zeros(1, dtype=np.float32)
            else:
                joint_logw = np.zeros(flat_indices.shape[0], dtype=np.float64)
                if sigma_rot_deg > 0.0:
                    joint_logw += -0.5 * (diffang[flat_indices] / sigma_rot_scale) ** 2
                if sigma_psi_deg > 0.0:
                    joint_logw += -0.5 * (diffpsi[flat_indices] / sigma_psi_scale) ** 2
                max_logw = float(np.max(joint_logw))
                flat_log_prior = (
                    joint_logw - (max_logw + np.log(np.sum(np.exp(joint_logw - max_logw))))
                ).astype(np.float32)
            prior_entries.append((flat_indices, flat_log_prior))
            selected_union.update(flat_indices.tolist())

        selected_indices = np.array(sorted(selected_union), dtype=np.int64)
        log_prior = np.full((prior_rotations.shape[0], selected_indices.shape[0]), -1e30, dtype=np.float32)
        index_to_pos = {int(idx): pos for pos, idx in enumerate(selected_indices.tolist())}
        for i, (flat_indices, flat_log_prior) in enumerate(prior_entries):
            positions = np.array([index_to_pos[int(idx)] for idx in flat_indices], dtype=np.int64)
            log_prior[i, positions] = flat_log_prior
        return selected_indices, log_prior

    order = 2
    sigma_rot = np.deg2rad(2.5)
    sigma_psi = np.deg2rad(4.0)
    base_rotations = np.asarray(em_sampling.get_rotation_grid(order, matrices=True), dtype=np.float32)
    perturbed_rotations = em_sampling.apply_relion_rotation_perturbation(
        base_rotations,
        random_perturbation=0.3,
        angular_sampling_deg=em_sampling.relion_angular_sampling_deg(order),
    ).astype(np.float32)
    perturbed_eulers = utils.R_to_relion(perturbed_rotations, degrees=True).astype(np.float32)
    metadata = em_sampling.build_local_search_grid_metadata(
        order,
        perturbed_eulers,
        grid_rotations=perturbed_rotations,
    )
    priors = perturbed_rotations[[5, 137, 911]]

    selected_indices, log_prior = em_sampling.get_local_rotation_grid_fast(
        priors,
        sigma_rot=sigma_rot,
        sigma_psi=sigma_psi,
        healpix_order=order,
        sigma_cutoff=3.0,
        per_image=True,
        grid_metadata=metadata,
    )
    ref_indices, ref_log_prior = _reference_full_mode(priors, sigma_rot, sigma_psi, metadata)

    np.testing.assert_array_equal(selected_indices, ref_indices)
    np.testing.assert_allclose(log_prior, ref_log_prior, rtol=1e-6, atol=1e-6)


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
    psi0_view_dirs = relion_grid[0, :, 2, :]
    psi0_view_dirs /= np.linalg.norm(psi0_view_dirs, axis=1, keepdims=True)

    dots = np.sum(np.asarray(metadata["dir_vecs"], dtype=np.float64) * psi0_view_dirs, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang_deg = np.rad2deg(np.arccos(dots))
    assert float(np.max(ang_deg)) < 2e-2


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
    prior_view = prior_matrix[2, :] / np.linalg.norm(prior_matrix[2, :])
    selected_views = relion_grid[selected_indices[finite_mask], 2, :]
    selected_views /= np.linalg.norm(selected_views, axis=1, keepdims=True)
    ang_deg = np.rad2deg(
        np.arccos(
            np.clip(
                np.sum(selected_views * prior_view[None, :], axis=1),
                -1.0,
                1.0,
            )
        )
    )

    assert ang_deg.size > 0
    assert float(np.max(ang_deg)) < sigma_cutoff * sigma_deg + 7.5


def test_local_rotation_grid_fast_uses_sigma_rot_not_sigma_psi_for_direction_cone():
    from recovar import utils

    order = 4
    prior_index = 12345
    sigma_rot_deg = 0.2
    sigma_psi_deg = 7.5
    sigma_cutoff = 3.0

    relion_eulers = np.asarray(em_sampling.get_relion_rotation_grid_eulers(order), dtype=np.float64)
    prior_euler = relion_eulers[prior_index : prior_index + 1]
    prior_matrix = utils.R_from_relion(prior_euler, degrees=True)[0]
    prior_view = prior_matrix[2, :] / np.linalg.norm(prior_matrix[2, :])

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
    assert unique_pixels.size == 1

    relion_grid = np.asarray(em_sampling.get_relion_rotation_grid(order), dtype=np.float64).reshape(-1, 3, 3)
    selected_views = relion_grid[unique_pixels, 2, :]
    selected_views /= np.linalg.norm(selected_views, axis=1, keepdims=True)
    diff_deg = np.rad2deg(np.arccos(np.clip(np.sum(selected_views * prior_view[None, :], axis=1), -1.0, 1.0)))

    assert float(np.max(diff_deg)) < sigma_cutoff * sigma_rot_deg + 1e-4


def test_local_rotation_grid_fast_per_image_priors_prefer_each_image_peak():
    fine_order = 3
    n_pixels = hp.nside2npix(2**fine_order)
    prior_indices = np.array([0, n_pixels], dtype=np.int64)
    full_grid = em_sampling.get_rotation_grid(fine_order, matrices=True)
    prior_rotations = full_grid[prior_indices]

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

    from recovar.heterogeneity import locres
    from recovar.reconstruction import noise, regularization

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

    from recovar.heterogeneity import locres
    from recovar.reconstruction import noise, regularization, relion_functions

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
