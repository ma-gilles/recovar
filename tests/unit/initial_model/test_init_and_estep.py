"""Phase 3 tests: denovo state init + E-step adapter helpers.

The full E-step (wiring into dense kernels) is exercised against the
RELION fixture in Phase 4. Here we validate:

  - Denovo state has the right shapes, dtypes, and constants.
  - The 0.07·ori_size rule + current_size derivation match the handoff's
    observed `current_size=28` on the 64-px / 8.5-Å fixture.
  - `Igrad2` is initialised to MOM2_INIT_CONSTANT + 1j*MOM2_INIT_CONSTANT.
  - `pdf_class` / `pdf_direction` are uniform.
  - Pseudo-halfset slot packing matches RELION convention (K-th class's
    halfset 0 at slot k, halfset 1 at slot K+k).
  - `seed_noise_from_mavg` transfers spectra correctly.
  - `minvsigma2_with_dc_zero` zeroes ires=0 and inverts the rest.
  - `hermitian_weights_relion` produces all-ones half-complex map.
  - `fourier_crop_half` keeps low-|k| rows from the top and high-|k| rows
    from the tail.
  - `build_posterior_summary` extracts Pmax / argmax correctly on a
    synthetic posterior tensor.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.initial_model import (
    MOM2_INIT_CONSTANT,
    build_posterior_summary,
    compute_current_size_for_denovo,
    compute_ini_high_angstrom,
    compute_ini_high_shell,
    fourier_crop_half,
    half_slot_count,
    half_slot_index,
    hermitian_weights_relion,
    initialise_data_vs_prior_from_references,
    initialise_denovo_state,
    minvsigma2_with_dc_zero,
    seed_noise_from_mavg,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Denovo init
# ---------------------------------------------------------------------------


class TestIniHighAndCurrentSize:
    def test_fixture_64px_8p5A_matches_handoff(self):
        """Handoff note asserts on the 64-px / 8.5-Å fixture:
        current_resolution_shell = 4
        current_resolution_angstrom = 136.0
        current_size = 28
        """
        ori = 64
        px = 8.5
        shell = compute_ini_high_shell(ori)
        assert shell == 4
        ini_high_A = compute_ini_high_angstrom(ori, px)
        # ori * pixel_size / shell = 64*8.5/4 = 136.0
        assert abs(ini_high_A - 136.0) < 1e-9
        current_size = compute_current_size_for_denovo(ori)
        # 2 * (4 + 10) = 28, clamped below 64
        assert current_size == 28

    def test_large_box_clamped_to_ori_size(self):
        # For ori_size=16, shell=1, current=2*(1+10)=22 > 16 -> clamp
        assert compute_current_size_for_denovo(16) == 16


class TestInitialiseDenovoState:
    def test_gui_defaults(self):
        state = initialise_denovo_state(
            ori_size=64,
            pixel_size=8.5,
            K=1,
            nr_iter=200,
            n_directions=192,
            nr_optics_groups=1,
            pseudo_halfsets=True,
            padding_factor=1,
        )
        assert state.iter == 0
        assert state.nr_iter == 200
        assert state.K == 1
        assert state.ori_size == 64
        assert state.pseudo_halfsets is True
        assert state.current_resolution_shell == 4
        assert state.current_size == 28
        assert abs(state.ini_high - 136.0) < 1e-9
        assert state.sigma2_offset == pytest.approx(100.0)

    def test_iref_is_zero(self):
        state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=2, nr_iter=50, n_directions=48)
        assert state.Iref.shape == (2, 16, 16, 16)
        assert state.Iref.dtype == np.float64
        np.testing.assert_array_equal(state.Iref, 0.0)

    def test_igrad1_zero_slots(self):
        K = 3
        state = initialise_denovo_state(
            ori_size=16,
            pixel_size=1.0,
            K=K,
            nr_iter=50,
            n_directions=48,
            pseudo_halfsets=True,
        )
        # H=2 with pseudo_halfsets, pad=1 -> (2K, 16, 16, 9)
        assert state.Igrad1.shape == (2 * K, 16, 16, 9)
        assert state.Igrad1.dtype == np.complex128
        np.testing.assert_array_equal(state.Igrad1, 0.0)

    def test_igrad1_without_pseudo_halfsets_k_slots(self):
        K = 3
        state = initialise_denovo_state(
            ori_size=16,
            pixel_size=1.0,
            K=K,
            nr_iter=50,
            n_directions=48,
            pseudo_halfsets=False,
        )
        assert state.Igrad1.shape == (K, 16, 16, 9)

    def test_igrad2_constant(self):
        state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=2, nr_iter=50, n_directions=48)
        assert state.Igrad2.shape == (2, 16, 16, 9)
        assert state.Igrad2.dtype == np.complex128
        expected = MOM2_INIT_CONSTANT + 1j * MOM2_INIT_CONSTANT
        assert np.all(state.Igrad2 == expected)

    def test_pdf_class_uniform(self):
        for K in [1, 2, 5]:
            state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=K, nr_iter=50, n_directions=48)
            assert state.pdf_class.shape == (K,)
            np.testing.assert_allclose(state.pdf_class, 1.0 / K)

    def test_pdf_direction_uniform(self):
        K = 3
        n_dir = 10
        state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=K, nr_iter=50, n_directions=n_dir)
        assert state.pdf_direction.shape == (K, n_dir)
        np.testing.assert_allclose(state.pdf_direction, 1.0 / (K * n_dir))

    def test_padding_factor_2(self):
        state = initialise_denovo_state(
            ori_size=8,
            pixel_size=1.0,
            K=1,
            nr_iter=10,
            n_directions=12,
            padding_factor=2,
        )
        # Default pseudo_halfsets=True with K=1 -> 2 slots; pf=2 -> (16, 16, 9)
        assert state.Igrad1.shape == (2, 16, 16, 9)

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            initialise_denovo_state(ori_size=16, pixel_size=1.0, K=0, nr_iter=10, n_directions=12)


# ---------------------------------------------------------------------------
# Slot index / count helpers
# ---------------------------------------------------------------------------


class TestHalfSlotIndex:
    def test_counts(self):
        assert half_slot_count(3, pseudo_halfsets=True) == 6
        assert half_slot_count(3, pseudo_halfsets=False) == 3

    def test_index_layout_pseudo(self):
        # K=3, pseudo_halfsets=True -> slots = [h0_k0, h0_k1, h0_k2,
        #                                       h1_k0, h1_k1, h1_k2]
        K = 3
        assert half_slot_index(0, 0, K, True) == 0
        assert half_slot_index(1, 0, K, True) == 1
        assert half_slot_index(2, 0, K, True) == 2
        assert half_slot_index(0, 1, K, True) == 3
        assert half_slot_index(1, 1, K, True) == 4
        assert half_slot_index(2, 1, K, True) == 5

    def test_index_no_pseudo_rejects_halfset_1(self):
        with pytest.raises(ValueError):
            half_slot_index(0, 1, K=2, pseudo_halfsets=False)


# ---------------------------------------------------------------------------
# seed_noise_from_mavg
# ---------------------------------------------------------------------------


class TestSeedNoiseFromMavg:
    def test_roundtrip(self):
        state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=1, nr_iter=10, n_directions=12)
        sigma = np.arange(9, dtype=np.float64).reshape(1, 9) * 0.1
        new_state = seed_noise_from_mavg(state, sigma)
        np.testing.assert_array_equal(new_state.sigma2_noise, sigma)
        # Original unchanged
        np.testing.assert_array_equal(state.sigma2_noise, 0.0)

    def test_shape_mismatch_raises(self):
        state = initialise_denovo_state(
            ori_size=16,
            pixel_size=1.0,
            K=1,
            nr_iter=10,
            n_directions=12,
            nr_optics_groups=1,
        )
        with pytest.raises(ValueError):
            seed_noise_from_mavg(state, np.zeros((2, 9)))


class TestInitialiseDataVsPrior:
    def test_seeds_nonzero_tau2_and_relion_data_vs_prior_formula(self):
        state = initialise_denovo_state(
            ori_size=16,
            pixel_size=2.0,
            K=2,
            nr_iter=10,
            n_directions=12,
            nr_optics_groups=2,
        )
        state.Iref[0, 8, 8, 8] = 1.0
        state.Iref[1, 8, 8, 8] = 2.0
        state.sigma2_noise[:] = np.asarray(
            [
                np.linspace(2.0, 3.0, 9),
                np.linspace(4.0, 5.0, 9),
            ],
            dtype=np.float64,
        )

        out = initialise_data_vs_prior_from_references(state, nr_particles=200)

        assert np.all(out.tau2_class > 0.0)
        assert np.all(out.data_vs_prior_class > 0.0)
        # RELION: spectrum = average(|FFT_norm(Iref)|^2), tau2 = spectrum*N^2/2.
        # A single centered delta has constant FFT_norm amplitude 1/N^3.
        n = 16.0
        expected_tau2_c0 = 1.0 / (2.0 * n**4)
        expected_tau2_c1 = 4.0 / (2.0 * n**4)
        np.testing.assert_allclose(out.tau2_class[0], expected_tau2_c0, rtol=1e-12)
        np.testing.assert_allclose(out.tau2_class[1], expected_tau2_c1, rtol=1e-12)
        avg_sigma2_shell1 = (state.sigma2_noise[0, 1] + state.sigma2_noise[1, 1]) / 2.0
        expected_dvp_shell1 = 200.0 * 0.5 / avg_sigma2_shell1 / 2.0 * expected_tau2_c0
        assert out.data_vs_prior_class[0, 1] == pytest.approx(expected_dvp_shell1, rel=1e-12)
        np.testing.assert_array_equal(state.tau2_class, 0.0)
        np.testing.assert_array_equal(state.data_vs_prior_class, 0.0)

    def test_rejects_missing_noise(self):
        state = initialise_denovo_state(ori_size=16, pixel_size=1.0, K=1, nr_iter=10, n_directions=12)
        state.Iref[0, 8, 8, 8] = 1.0

        with pytest.raises(ValueError, match="sigma2_noise"):
            initialise_data_vs_prior_from_references(state, nr_particles=10)


# ---------------------------------------------------------------------------
# E-step helpers
# ---------------------------------------------------------------------------


class TestMinvsigma2DcZero:
    def test_1d_input(self):
        sigma = np.array([1.0, 2.0, 4.0, 8.0])
        out = minvsigma2_with_dc_zero(sigma)
        np.testing.assert_array_equal(out, np.array([0.0, 0.5, 0.25, 0.125]))

    def test_2d_input(self):
        sigma = np.array([[1.0, 2.0, 4.0], [1.0, 1.0, 0.5]])
        out = minvsigma2_with_dc_zero(sigma)
        expected = np.array([[0.0, 0.5, 0.25], [0.0, 1.0, 2.0]])
        np.testing.assert_array_equal(out, expected)

    def test_zero_sigma_stays_zero(self):
        sigma = np.array([1.0, 0.0, 0.5])
        out = minvsigma2_with_dc_zero(sigma)
        np.testing.assert_array_equal(out, np.array([0.0, 0.0, 2.0]))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            minvsigma2_with_dc_zero(np.zeros((2, 3, 4)))


class TestHermitianWeightsRelion:
    def test_all_ones(self):
        w = hermitian_weights_relion(8)
        assert w.shape == (8, 5)
        np.testing.assert_array_equal(w, 1.0)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            hermitian_weights_relion(1)


class TestFourierCropHalf:
    def test_noop_when_current_equals_ori(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((8, 5)) + 1j * rng.standard_normal((8, 5))
        out = fourier_crop_half(img, current_size=8)
        np.testing.assert_array_equal(out, img)

    def test_crop_to_half(self):
        img = np.arange(8 * 5).reshape(8, 5).astype(np.complex128)
        out = fourier_crop_half(img, current_size=4)
        assert out.shape == (4, 3)
        # First half: rows [0, 1]  (half_cs = 2, so rows [0:2])
        np.testing.assert_array_equal(out[:2, :3], img[:2, :3])
        # Second half: rows [6, 7] (out_y - half_cs = 2, ori_size - 2 = 6)
        np.testing.assert_array_equal(out[2:, :3], img[6:, :3])

    def test_invalid_current_size(self):
        img = np.zeros((8, 5), dtype=np.complex128)
        with pytest.raises(ValueError):
            fourier_crop_half(img, current_size=10)
        with pytest.raises(ValueError):
            fourier_crop_half(img, current_size=3)
        with pytest.raises(ValueError):
            fourier_crop_half(img, current_size=0)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            fourier_crop_half(np.zeros(5, dtype=np.complex128), current_size=4)
        with pytest.raises(ValueError):
            fourier_crop_half(np.zeros((8, 6), dtype=np.complex128), current_size=4)


# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------


class TestBuildPosteriorSummary:
    def test_basic_extraction(self):
        N, K, n_rot, n_trans = 3, 2, 4, 5
        # Force argmax at (img=0, k=1, r=2, t=3) for image 0
        weights = np.zeros((N, K, n_rot, n_trans))
        weights[0, 1, 2, 3] = 0.7
        weights[0, 0, 0, 0] = 0.3
        weights[1, 0, 1, 1] = 1.0
        weights[2, 1, 3, 4] = 0.4  # Not unique across image 2
        weights[2, 0, 0, 0] = 0.6  # argmax here for image 2

        post = build_posterior_summary(weights, significance_threshold=0.1)
        np.testing.assert_allclose(post.pmax, [0.7, 1.0, 0.6])

        # Image 0: best is (k=1, r=2, t=3)
        assert post.best_class[0] == 1
        assert post.best_euler[0, 0] == 2  # rot_idx
        assert post.best_trans[0, 0] == 3  # trans_idx

        # Image 1: best is (k=0, r=1, t=1)
        assert post.best_class[1] == 0
        assert post.best_euler[1, 0] == 1
        assert post.best_trans[1, 0] == 1

        # Image 2: best is (k=0, r=0, t=0)
        assert post.best_class[2] == 0
        assert post.best_euler[2, 0] == 0
        assert post.best_trans[2, 0] == 0

        # Significance counts
        # image 0: 2 entries > 0.1 (0.7, 0.3)
        # image 1: 1 entry > 0.1 (1.0)
        # image 2: 2 entries > 0.1 (0.6, 0.4)
        np.testing.assert_array_equal(post.nr_significant, [2, 1, 2])

    def test_bad_shape(self):
        with pytest.raises(ValueError):
            build_posterior_summary(np.zeros((3, 4, 5)))
