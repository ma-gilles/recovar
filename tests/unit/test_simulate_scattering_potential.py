import numpy as np
import pytest

pytest.importorskip("finufft")

from recovar.simulation import simulate_scattering_potential as ssp

pytestmark = pytest.mark.unit


def test_get_exponent_and_constant_of_gaussian_ft_scales_with_sigma():
    sigma_small = 0.7
    sigma_large = 1.4

    tau_small, c_small = ssp.get_exponent_and_constant_of_gaussian_FT(sigma_small, dim=3)
    tau_large, c_large = ssp.get_exponent_and_constant_of_gaussian_FT(sigma_large, dim=3)

    assert tau_small > 0
    assert c_small > 0
    assert tau_large > tau_small
    np.testing.assert_allclose(
        tau_large / tau_small,
        (sigma_large / sigma_small) ** 2,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(c_large, c_small, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_small, 1.0, rtol=1e-12, atol=1e-12)


def test_gaussian_fn_on_k_matches_gaussian_atom_shape_fn():
    sigma = 1.2
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, -0.5, 0.1],
            [0.7, 0.1, -0.2],
        ],
        dtype=np.float64,
    )

    fn = ssp.get_gaussian_fn_on_k(sigma)
    expected = ssp.gaussian_atom_shape_fn(coords, sigma)
    np.testing.assert_allclose(fn(coords), expected, rtol=1e-12, atol=1e-12)


def test_choose_number_of_atoms_is_monotonic_in_radius():
    assert ssp.choose_number_of_atoms(8.0) > ssp.choose_number_of_atoms(4.0)


def test_get_random_points_in_unit_ball_are_within_radius_one():
    pts = ssp.get_random_points_in_unit_ball(512)
    assert pts.shape == (512, 3)
    radii = np.linalg.norm(pts, axis=1)
    assert np.all(radii <= 1.0 + 1e-12)


def test_get_center_coord_offset_returns_box_midpoint():
    coords = np.array(
        [
            [-3.0, 2.0, 1.0],
            [5.0, -4.0, 7.0],
            [1.0, 9.0, -2.0],
        ],
        dtype=np.float64,
    )
    offset = ssp.get_center_coord_offset(coords)
    expected = (coords.min(axis=0) + coords.max(axis=0)) / 2.0
    np.testing.assert_allclose(offset, expected, rtol=0, atol=0)


def test_voltage_to_wavelength_decreases_with_voltage():
    wl_100kv = ssp.voltage_to_wavelength(100.0)
    wl_300kv = ssp.voltage_to_wavelength(300.0)
    assert wl_300kv < wl_100kv
    assert wl_100kv > 0


def test_generate_volume_from_atom_positions_and_types_at_freq_coords_smoke():
    atom_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, -0.25, 0.2],
        ],
        dtype=np.float64,
    )
    atom_types = np.array(["C", "C"], dtype="<U1")
    freq_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [-0.05, 0.08, -0.02],
        ],
        dtype=np.float64,
    )

    density = ssp.generate_volume_from_atom_positions_and_types(
        atom_coords,
        atom_types,
        voxel_size=1.0,
        freq_coords=freq_coords,
    )

    assert density.shape == (freq_coords.shape[0],)
    assert np.isfinite(density.real).all()
    assert np.isfinite(density.imag).all()
    assert np.abs(density[0]) > 0


def test_generate_volume_from_atom_positions_and_types_accepts_lowercase_atom_types():
    atom_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.1, -0.2],
        ],
        dtype=np.float64,
    )
    freq_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.05, 0.02],
        ],
        dtype=np.float64,
    )
    upper = ssp.generate_volume_from_atom_positions_and_types(
        atom_coords=atom_coords,
        atom_types=np.array(["C", "C"], dtype="<U1"),
        voxel_size=1.0,
        freq_coords=freq_coords,
    )
    lower = ssp.generate_volume_from_atom_positions_and_types(
        atom_coords=atom_coords,
        atom_types=np.array(["c", "c"], dtype="<U1"),
        voxel_size=1.0,
        freq_coords=freq_coords,
    )
    np.testing.assert_allclose(lower, upper, rtol=1e-12, atol=1e-12)


def test_generate_volume_from_atom_positions_and_types_rejects_length_mismatch():
    atom_coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=np.float64)
    with pytest.raises(ValueError, match="length mismatch"):
        ssp.generate_volume_from_atom_positions_and_types(
            atom_coords=atom_coords,
            atom_types=np.array(["C"], dtype="<U1"),
            voxel_size=1.0,
            freq_coords=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        )


def test_generate_volume_from_atom_positions_and_types_rejects_unknown_atom_types():
    atom_coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Unknown atom types"):
        ssp.generate_volume_from_atom_positions_and_types(
            atom_coords=atom_coords,
            atom_types=np.array(["XX"], dtype="<U2"),
            voxel_size=1.0,
            freq_coords=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        )


def test_generate_volume_from_atom_positions_and_types_uses_complex64_for_float32_inputs():
    atom_coords = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]], dtype=np.float32)
    freq_coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, -0.1]], dtype=np.float32)
    density = ssp.generate_volume_from_atom_positions_and_types(
        atom_coords=atom_coords,
        atom_types=np.array(["C", "C"], dtype="<U1"),
        voxel_size=1.0,
        freq_coords=freq_coords,
    )
    assert density.dtype == np.complex64


def test_generate_volume_from_atom_positions_and_types_uses_complex128_for_float64_inputs():
    atom_coords = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]], dtype=np.float64)
    freq_coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, -0.1]], dtype=np.float64)
    density = ssp.generate_volume_from_atom_positions_and_types(
        atom_coords=atom_coords,
        atom_types=np.array(["C", "C"], dtype="<U1"),
        voxel_size=1.0,
        freq_coords=freq_coords,
    )
    assert density.dtype == np.complex128
