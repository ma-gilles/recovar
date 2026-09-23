"""EM-only regularization tests (split from test_regularization.py in the relax split, P1)."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.em.reconstruction import regularization_relion
from recovar.reconstruction import regularization

pytestmark = pytest.mark.unit


def test_compute_relion_fsc_from_backprojector_uses_relion_rounding_and_half_layout():
    shape = (4, 4, 4)
    padded_shape = (8, 8, 8)

    data0 = np.zeros(padded_shape, dtype=np.complex128)
    data1 = np.zeros(padded_shape, dtype=np.complex128)
    weight0 = np.zeros(padded_shape, dtype=np.float64)
    weight1 = np.zeros(padded_shape, dtype=np.float64)

    # Valid RELION half-layout sample: logical RELION (z=0, y=2, x=0)
    # downsamples to shell 1 for padding_factor=2. In RECOVAR's centered
    # full-array layout, RELION x is axis 0 for this helper.
    valid_idx = (4, 4, 6)
    data0[valid_idx] = 1.0
    data1[valid_idx] = 1.0
    weight0[valid_idx] = 1.0
    weight1[valid_idx] = 1.0

    # Logical RELION x=-1 would banker-round to x=0, but RELION's ROUND(-0.5)
    # returns -1 and the compact half layout excludes it.
    invalid_negative_x_idx = (3, 4, 6)
    data0[invalid_negative_x_idx] = 100.0
    data1[invalid_negative_x_idx] = -100.0
    weight0[invalid_negative_x_idx] = 1.0
    weight1[invalid_negative_x_idx] = 1.0

    fsc = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            data0.reshape(-1),
            data1.reshape(-1),
            weight0.reshape(-1),
            weight1.reshape(-1),
            shape,
            padding_factor=2,
            r_max=1,
        )
    )

    assert fsc[0] == 1.0
    np.testing.assert_allclose(fsc[1], 1.0, atol=1e-7, rtol=1e-7)

    fsc_f64 = regularization_relion.compute_relion_fsc_from_backprojector(
        data0.reshape(-1),
        data1.reshape(-1),
        weight0.reshape(-1),
        weight1.reshape(-1),
        shape,
        padding_factor=2,
        r_max=1,
        output_dtype=np.float64,
    )
    assert np.asarray(fsc_f64).dtype == np.float64


def test_compute_relion_fsc_from_backprojector_accepts_packed_half_accumulators(
    monkeypatch,
):
    monkeypatch.setattr(
        regularization_relion,
        "_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS",
        0,
    )
    shape = (4, 4, 4)
    padding_factor = 2
    padded_shape = tuple(s * padding_factor for s in shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(padded_shape)
    rng = np.random.default_rng(0)

    data0_half = (
        rng.normal(size=half_shape).astype(np.float32)
        + 1j * rng.normal(size=half_shape).astype(np.float32)
    )
    data1_half = (
        rng.normal(size=half_shape).astype(np.float32)
        + 1j * rng.normal(size=half_shape).astype(np.float32)
    )
    weight0_half = (0.25 + rng.random(size=half_shape)).astype(np.float32)
    weight1_half = (0.25 + rng.random(size=half_shape)).astype(np.float32)

    data0_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(data0_half, padded_shape))
    data1_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(data1_half, padded_shape))
    weight0_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(weight0_half, padded_shape))
    weight1_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(weight1_half, padded_shape))

    fsc_from_full = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            data0_full.reshape(-1),
            data1_full.reshape(-1),
            weight0_full.reshape(-1),
            weight1_full.reshape(-1),
            shape,
            padding_factor=padding_factor,
        )
    )
    fsc_from_half = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            data0_half.reshape(-1),
            data1_half.reshape(-1),
            weight0_half.reshape(-1),
            weight1_half.reshape(-1),
            shape,
            padding_factor=padding_factor,
        )
    )

    np.testing.assert_allclose(fsc_from_half, fsc_from_full, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(fsc_from_half, fsc_from_full)


def test_compute_relion_fsc_from_backprojector_accepts_odd_packed_half_accumulators(
    monkeypatch,
):
    monkeypatch.setattr(
        regularization_relion,
        "_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS",
        0,
    )
    shape = (8, 8, 8)
    padding_factor = 2
    accumulator_shape = (19, 19, 19)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape)
    rng = np.random.default_rng(19)

    data0_half = (
        rng.normal(size=half_shape).astype(np.float32)
        + 1j * rng.normal(size=half_shape).astype(np.float32)
    )
    data1_half = (
        rng.normal(size=half_shape).astype(np.float32)
        + 1j * rng.normal(size=half_shape).astype(np.float32)
    )
    weight0_half = (0.25 + rng.random(size=half_shape)).astype(np.float32)
    weight1_half = (0.25 + rng.random(size=half_shape)).astype(np.float32)

    data0_full = np.asarray(
        fourier_transform_utils.half_volume_to_full_volume(data0_half, accumulator_shape)
    )
    data1_full = np.asarray(
        fourier_transform_utils.half_volume_to_full_volume(data1_half, accumulator_shape)
    )
    weight0_full = np.asarray(
        fourier_transform_utils.half_volume_to_full_volume(weight0_half, accumulator_shape)
    )
    weight1_full = np.asarray(
        fourier_transform_utils.half_volume_to_full_volume(weight1_half, accumulator_shape)
    )

    fsc_from_full = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            data0_full.reshape(-1),
            data1_full.reshape(-1),
            weight0_full.reshape(-1),
            weight1_full.reshape(-1),
            shape,
            padding_factor=padding_factor,
            r_max=shape[0] // 2,
            accumulator_volume_shape=accumulator_shape,
        )
    )
    fsc_from_half = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            data0_half.reshape(-1),
            data1_half.reshape(-1),
            weight0_half.reshape(-1),
            weight1_half.reshape(-1),
            shape,
            padding_factor=padding_factor,
            r_max=shape[0] // 2,
            accumulator_volume_shape=accumulator_shape,
        )
    )

    np.testing.assert_allclose(fsc_from_half, fsc_from_full, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(fsc_from_half, fsc_from_full)


def test_compute_relion_fsc_from_backprojector_applies_exact_rmax_before_shell_binning():
    shape = (4, 4, 4)
    data0 = np.zeros(shape, dtype=np.complex128)
    data1 = np.zeros(shape, dtype=np.complex128)
    weight0 = np.ones(shape, dtype=np.float64)
    weight1 = np.ones(shape, dtype=np.float64)

    # Shifted-grid coordinate (z=1, y=0, x=0): exact R=1, shell 1.
    inside = (3, 2, 2)
    data0[inside] = 1.0
    data1[inside] = 1.0

    # Shifted-grid coordinate (z=1, y=1, x=0): exact R=sqrt(2) > r_max,
    # but ROUND(R) == 1. RELION excludes it before shell binning.
    rounded_shell_but_outside_rmax = (3, 3, 2)
    data0[rounded_shell_but_outside_rmax] = 1.0
    data1[rounded_shell_but_outside_rmax] = -1.0

    fsc = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            data0.reshape(-1),
            data1.reshape(-1),
            weight0.reshape(-1),
            weight1.reshape(-1),
            shape,
            padding_factor=1,
            r_max=1,
        )
    )

    np.testing.assert_allclose(fsc[1], 1.0, atol=1e-7, rtol=1e-7)


def test_join_halves_at_low_resolution_uses_explicit_padding_for_current_size_accumulator():
    native_shape = (16, 16, 16)
    accumulator_shape = (23, 23, 23)
    padding_factor = 2

    data0 = np.zeros(accumulator_shape, dtype=np.complex64)
    data1 = np.zeros(accumulator_shape, dtype=np.complex64)
    weight0 = np.zeros(accumulator_shape, dtype=np.float32)
    weight1 = np.zeros(accumulator_shape, dtype=np.float32)

    # Native join radius is ceil(16 / 8 A) = 2 shells. RELION applies the
    # backprojector padding factor before comparing accumulator coordinates,
    # so this radius-3 accumulator voxel must be joined when padding_factor=2.
    joined_only_with_explicit_padding = (14, 11, 11)
    data0[joined_only_with_explicit_padding] = 1.0 + 0.0j
    data1[joined_only_with_explicit_padding] = 0.0 + 1.0j
    weight0[joined_only_with_explicit_padding] = 1.0
    weight1[joined_only_with_explicit_padding] = 1.0

    legacy_join = regularization_relion.join_halves_at_low_resolution(
        data0.reshape(-1),
        data1.reshape(-1),
        weight0.reshape(-1),
        weight1.reshape(-1),
        accumulator_shape,
        1.0,
        native_shape[0],
        8.0,
    )
    np.testing.assert_allclose(
        np.asarray(legacy_join[0]).reshape(accumulator_shape)[joined_only_with_explicit_padding],
        1.0 + 0.0j,
    )

    joined = regularization_relion.join_halves_at_low_resolution(
        data0.reshape(-1),
        data1.reshape(-1),
        weight0.reshape(-1),
        weight1.reshape(-1),
        accumulator_shape,
        1.0,
        native_shape[0],
        8.0,
        padding_factor=padding_factor,
    )
    expected = np.complex64(0.5 + 0.5j)
    np.testing.assert_allclose(
        np.asarray(joined[0]).reshape(accumulator_shape)[joined_only_with_explicit_padding],
        expected,
    )
    np.testing.assert_allclose(
        np.asarray(joined[1]).reshape(accumulator_shape)[joined_only_with_explicit_padding],
        expected,
    )

    fsc_before_join = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            data0.reshape(-1),
            data1.reshape(-1),
            weight0.reshape(-1),
            weight1.reshape(-1),
            native_shape,
            padding_factor=padding_factor,
            r_max=2,
            accumulator_volume_shape=accumulator_shape,
        )
    )
    fsc_after_join = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            joined[0],
            joined[1],
            joined[2],
            joined[3],
            native_shape,
            padding_factor=padding_factor,
            r_max=2,
            accumulator_volume_shape=accumulator_shape,
        )
    )

    np.testing.assert_allclose(fsc_before_join[2], 0.0, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(fsc_after_join[2], 1.0, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
    ("padding_factor", "weight_value", "expected_sigma2"),
    [
        (1, 2.0, 1.0 / 2.0),
        (2, 2.0, 1.0 / (2.0 * 8.0)),
    ],
)
def test_compute_relion_tau2_from_weights_constant_weight_details(padding_factor, weight_value, expected_sigma2):
    shape = (8, 8, 8)
    n_shells = shape[0] // 2 + 1
    padded_shape = tuple(s * padding_factor for s in shape)
    weight = np.full(np.prod(padded_shape), weight_value, dtype=np.float32)
    fsc = np.full(n_shells, 0.5, dtype=np.float32)  # SSNR = 1

    prior, fsc_out, details = regularization_relion.compute_relion_tau2_from_weights(
        weight,
        weight,
        fsc,
        shape,
        padding_factor=padding_factor,
        return_details=True,
    )

    np.testing.assert_allclose(np.asarray(fsc_out), 0.5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(details["avg_weight_shells"]), weight_value, atol=1e-6)
    np.testing.assert_allclose(np.asarray(details["sigma2_shells"]), expected_sigma2, atol=1e-6)
    np.testing.assert_allclose(np.asarray(details["prior_shells"]), expected_sigma2, atol=1e-6)
    np.testing.assert_allclose(np.asarray(prior), expected_sigma2, atol=1e-6)
    assert np.asarray(details["shell_count"]).shape == (n_shells,)
    assert np.all(np.asarray(details["shell_count"]) > 0)


def test_compute_relion_tau2_from_weights_preserves_requested_float64_output():
    shape = (8, 8, 8)
    weight = np.full(np.prod(shape), np.nextafter(2.0, 3.0), dtype=np.float64)
    fsc = np.full(shape[0] // 2 + 1, np.nextafter(0.5, 1.0), dtype=np.float64)

    prior, fsc_out, details = regularization_relion.compute_relion_tau2_from_weights(
        weight,
        weight,
        fsc,
        shape,
        return_details=True,
        output_dtype=np.float64,
    )

    assert np.asarray(prior).dtype == np.float64
    assert np.asarray(fsc_out).dtype == np.float64
    assert np.asarray(details["prior_shells"]).dtype == np.float64
    assert np.asarray(details["sigma2_shells"]).dtype == np.float64
    assert np.asarray(details["avg_weight_shells"]).dtype == np.float64


def test_compute_relion_tau2_from_weights_respects_relion_rmax_support():
    shape = (8, 8, 8)
    padding_factor = 2
    n_shells = shape[0] // 2 + 1
    padded_shape = tuple(s * padding_factor for s in shape)
    weight = np.ones(np.prod(padded_shape), dtype=np.float32)
    fsc = np.full(n_shells, 0.5, dtype=np.float32)

    prior, _, details = regularization_relion.compute_relion_tau2_from_weights(
        weight,
        weight,
        fsc,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        return_details=True,
    )

    shell_count = np.asarray(details["shell_count"])
    prior_shells = np.asarray(details["prior_shells"])
    assert shell_count[4] == 0.0
    assert prior_shells[4] <= 1e-12

    radii = np.asarray(
        regularization.fourier_transform_utils.get_grid_of_radial_distances(
            shape,
            scaled=False,
            frequency_shift=0,
        )
        .astype(int)
        .reshape(-1)
    )
    assert np.all(np.asarray(prior)[radii == 4] <= 1e-12)


def test_compute_relion_tau2_from_weights_rejects_wrong_grid_size():
    shape = (8, 8, 8)
    fsc = np.full(shape[0] // 2 + 1, 0.5, dtype=np.float32)
    bad_weight = np.ones(np.prod(shape) - 1, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected full or half Fourier weight"):
        regularization_relion.compute_relion_tau2_from_weights(
            bad_weight,
            bad_weight,
            fsc,
            shape,
            padding_factor=1,
        )


def test_relion_weight_shell_stats_floor_bins_reconstruct_support():
    shape = (8, 8, 8)
    padding_factor = 2
    current_size = 6
    r_max = current_size // 2
    full_shape = tuple(s * padding_factor for s in shape)
    weight = np.ones(np.prod(full_shape), dtype=np.float32)

    stats_floor = regularization_relion._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        r_max=r_max,
        shell_rounding="floor",
    )
    stats_round = regularization_relion._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        r_max=r_max,
        shell_rounding="round",
    )

    coords = np.arange(-(full_shape[0] // 2), full_shape[0] // 2)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy + zz * zz)
    mask = (xx >= 0) & (radius < padding_factor * r_max)
    shell = np.floor(radius / padding_factor).astype(int)
    expected = np.bincount(shell[mask].ravel(), minlength=shape[0] // 2 + 1)

    np.testing.assert_array_equal(np.asarray(stats_floor["shell_count"])[: expected.shape[0]], expected)
    assert not np.array_equal(
        np.asarray(stats_floor["shell_count"])[: expected.shape[0]],
        np.asarray(stats_round["shell_count"])[: expected.shape[0]],
    )


def test_relion_weight_shell_stats_rounds_half_integer_radii_up():
    """RELION ``ROUND`` maps positive 0.5/2.5 radii to shells 1/3."""
    shape = (8, 8, 8)
    padding_factor = 2
    full_shape = tuple(s * padding_factor for s in shape)
    weight = np.zeros(full_shape, dtype=np.float32)
    center = tuple(s // 2 for s in full_shape)
    weight[center[0], center[1], center[2] + 1] = 2.0  # radius / padding = 0.5
    weight[center[0], center[1], center[2] + 5] = 3.0  # radius / padding = 2.5

    stats = regularization_relion._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        shell_rounding="round",
    )

    shell_sum = np.asarray(stats["shell_sum"])
    assert shell_sum[0] == pytest.approx(0.0)
    assert shell_sum[1] == pytest.approx(2.0)
    assert shell_sum[2] == pytest.approx(0.0)
    assert shell_sum[3] == pytest.approx(3.0)


def test_relion_weight_shell_stats_large_grid_cpu_path_matches_device_path(monkeypatch):
    shape = (8, 8, 8)
    padding_factor = 2
    full_shape = tuple(s * padding_factor for s in shape)
    rng = np.random.default_rng(10)
    weight = (0.25 + rng.random(np.prod(full_shape))).astype(np.float32)

    device_stats = regularization_relion._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        shell_rounding="round",
    )
    monkeypatch.setattr(regularization_relion, "_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS", 1)
    cpu_stats = regularization_relion._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        shell_rounding="round",
    )

    for key in ("shell_sum", "shell_count", "avg_weight_shells"):
        np.testing.assert_allclose(np.asarray(cpu_stats[key]), np.asarray(device_stats[key]), rtol=1e-6, atol=1e-6)


def test_compute_relion_tau2_from_weights_large_grid_cpu_path_matches_device_path(monkeypatch):
    shape = (8, 8, 8)
    padding_factor = 2
    full_shape = tuple(s * padding_factor for s in shape)
    n_shells = shape[0] // 2 + 1
    rng = np.random.default_rng(11)
    weight0 = (0.25 + rng.random(np.prod(full_shape))).astype(np.float32)
    weight1 = (0.25 + rng.random(np.prod(full_shape))).astype(np.float32)
    fsc = np.linspace(0.95, 0.25, n_shells, dtype=np.float32)

    prior_device, _, details_device = regularization_relion.compute_relion_tau2_from_weights(
        weight0,
        weight1,
        fsc,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        return_details=True,
    )
    monkeypatch.setattr(regularization_relion, "_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS", 1)
    prior_cpu, _, details_cpu = regularization_relion.compute_relion_tau2_from_weights(
        weight0,
        weight1,
        fsc,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        return_details=True,
    )

    np.testing.assert_allclose(np.asarray(prior_cpu), np.asarray(prior_device), rtol=1e-6, atol=1e-6)
    for key in ("shell_sum", "shell_count", "avg_weight_shells", "prior_shells"):
        np.testing.assert_allclose(
            np.asarray(details_cpu[key]),
            np.asarray(details_device[key]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_compute_relion_tau2_from_iref_power_spectrum_matches_relion_binding_scaling():
    from pathlib import Path

    from recovar.utils.helpers import load_relion_volume

    relion_dir = Path(
        "/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k4_5k_128/relion_pdb_k4_os0_ref"
    )
    volume_path = relion_dir / "run_it000_class001.mrc"
    model_path = relion_dir / "run_it001_model.star"

    vol_recovar = np.asarray(load_relion_volume(str(volume_path)), dtype=np.float64)
    ft_recovar = np.asarray(fourier_transform_utils.get_dft3(jnp.asarray(vol_recovar)).reshape(-1))

    tau2, details = regularization_relion.compute_relion_tau2_from_iref_power_spectrum(
        ft_recovar,
        vol_recovar.shape,
        padding_factor=2,
        current_size=56,
        return_details=True,
    )

    expected_tau2 = []
    in_class = False
    in_loop = False
    for line in model_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "data_model_class_1":
            in_class = True
            continue
        if in_class and stripped == "loop_":
            in_loop = True
            continue
        if in_loop and stripped.startswith("data_model_"):
            break
        if in_loop and stripped and stripped[0].isdigit():
            expected_tau2.append(float(stripped.split()[7]))
            if len(expected_tau2) == 2:
                break

    assert tau2.shape == (np.prod(vol_recovar.shape),)
    assert details["tau2_shells"].shape[0] >= len(expected_tau2)
    np.testing.assert_allclose(np.asarray(details["tau2_shells"][: len(expected_tau2)]), expected_tau2, rtol=2e-2, atol=2e-6)


def test_streamed_packed_half_backprojector_fsc_avoids_padded_full_allocation(
    monkeypatch,
):
    shape = (6, 6, 6)
    accumulator_shape = (13, 13, 13)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(
        accumulator_shape
    )
    rng = np.random.default_rng(13)
    data0 = (
        rng.normal(size=half_shape).astype(np.float32)
        + 1j * rng.normal(size=half_shape).astype(np.float32)
    )
    data1 = (
        rng.normal(size=half_shape).astype(np.float32)
        + 1j * rng.normal(size=half_shape).astype(np.float32)
    )
    weight0 = (0.25 + rng.random(size=half_shape)).astype(np.float32)
    weight1 = (0.25 + rng.random(size=half_shape)).astype(np.float32)

    monkeypatch.setattr(
        regularization_relion,
        "_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS",
        0,
    )
    monkeypatch.setenv("RECOVAR_MSTEP_FSC_DUMP_AVG", "1")
    monkeypatch.delenv("RECOVAR_MSTEP_FSC_DUMP_DIR", raising=False)
    original_empty = regularization.np.empty
    original_zeros = regularization.np.zeros

    def _reject_full_shape(allocator):
        def checked(shape_arg, *args, **kwargs):
            if tuple(np.atleast_1d(shape_arg)) == accumulator_shape:
                raise AssertionError("streamed FSC allocated a padded full cube")
            return allocator(shape_arg, *args, **kwargs)

        return checked

    monkeypatch.setattr(regularization.np, "empty", _reject_full_shape(original_empty))
    monkeypatch.setattr(regularization.np, "zeros", _reject_full_shape(original_zeros))
    fsc = np.asarray(
        regularization_relion.compute_relion_fsc_from_backprojector(
            data0.reshape(-1),
            data1.reshape(-1),
            weight0.reshape(-1),
            weight1.reshape(-1),
            shape,
            padding_factor=2,
            r_max=shape[0] // 2,
            accumulator_volume_shape=accumulator_shape,
        )
    )

    assert fsc.shape == (shape[0] // 2 + 1,)
    assert np.all(np.isfinite(fsc))


@pytest.mark.parametrize("output_dtype", [jnp.float32, jnp.float64])
def test_streamed_fsc_preserves_requested_output_precision(monkeypatch, output_dtype):
    shape = (6, 6, 6)
    accumulator_shape = (13, 13, 13)
    half_shape = (13, 13, 7)
    rng = np.random.default_rng(1775735620)
    data = [(rng.normal(size=half_shape) + 1j * rng.normal(size=half_shape)).astype(np.complex64) for _ in range(2)]
    weights = [(0.25 + rng.random(size=half_shape)).astype(np.float32) for _ in range(2)]
    kwargs = dict(padding_factor=2, r_max=3, accumulator_volume_shape=accumulator_shape, output_dtype=output_dtype)
    monkeypatch.setenv("RECOVAR_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS", "1000000")
    expected = regularization_relion.compute_relion_fsc_from_backprojector(*data, *weights, shape, **kwargs)
    monkeypatch.setenv("RECOVAR_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS", "0")
    actual = regularization_relion.compute_relion_fsc_from_backprojector(*data, *weights, shape, **kwargs)
    assert actual.dtype == expected.dtype == output_dtype
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
