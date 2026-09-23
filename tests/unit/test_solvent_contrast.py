"""Atomic-volume transform (solvent contrast plus B-factor): filter, metadata and GT loading."""

import sys

import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.commands import make_test_dataset
from recovar.commands import run_test_all_metrics as rtam
from recovar.simulation import simulator, solvent_contrast, synthetic_dataset

pytestmark = pytest.mark.unit


def _reference_filter(grid_size, voxel_size, a, B, atomic_bfactor):
    """Independent H * exp(-B_atomic q^2 / 4) on the centered grid (frequencies -N/2..N/2-1)."""
    k = np.arange(-(grid_size // 2), grid_size - grid_size // 2) / (grid_size * voxel_size)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    q2 = kx**2 + ky**2 + kz**2
    return ((1.0 - a * np.exp(-B * q2 / 4.0)) * np.exp(-atomic_bfactor * q2 / 4.0)).reshape(-1)


def _random_real_fourier_volumes(n_volumes, grid_size, seed):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n_volumes, grid_size, grid_size, grid_size))
    return np.stack([np.asarray(ftu.get_dft3(v)).reshape(-1) for v in real])


def _enabled_simulation_info(grid_size, voxel_size, scale_vol=1.0, **record_kwargs):
    return {
        "volumes_path_root": "/nonexistent/vol",
        "grid_size": grid_size,
        "trailing_zero_format_in_vol_name": True,
        "scale_vol": scale_vol,
        "image_assignment": np.array([0, 1, 1, 2, 2, 2], dtype=np.int32),
        "per_image_contrast": np.ones(6, dtype=np.float32),
        solvent_contrast.METADATA_KEY: solvent_contrast.make_record(
            True, voxel_size=voxel_size, grid_size=grid_size, **record_kwargs
        ),
    }


# 1. Disabled option and legacy metadata


@pytest.mark.parametrize("record", [None, {"enabled": False}])
def test_legacy_and_disabled_metadata_load_uncorrected_truth(monkeypatch, record):
    raw = _random_real_fourier_volumes(3, 4, seed=0)
    monkeypatch.setattr(simulator, "load_volumes_from_folder", lambda *a, **k: raw.copy())
    info = _enabled_simulation_info(4, 2.0, scale_vol=3.0)
    if record is None:
        del info[solvent_contrast.METADATA_KEY]
    else:
        info[solvent_contrast.METADATA_KEY] = record
    assert solvent_contrast.record_from_simulation_info(info) is None
    np.testing.assert_array_equal(synthetic_dataset.load_ground_truth_volumes(info), raw * 3.0)


def test_disabled_record_is_minimal_and_overrides_require_enable():
    assert solvent_contrast.make_record(False) == {"enabled": False}
    with pytest.raises(ValueError, match="require atomic_solvent_correction"):
        simulator.generate_synthetic_dataset("/nonexistent", 1.0, "/nonexistent/vol", 4, solvent_contrast_a=0.5)
    with pytest.raises(ValueError, match="require atomic_solvent_correction"):
        simulator.generate_synthetic_dataset("/nonexistent", 1.0, "/nonexistent/vol", 4, atomic_bfactor=0.0)


# 2. Attenuation of known Fourier modes


def test_filter_defaults_dc_and_known_mode():
    grid_size, voxel_size = 8, 2.5
    filt = solvent_contrast.solvent_contrast_filter((grid_size,) * 3, voxel_size)
    assert (solvent_contrast.DEFAULT_A, solvent_contrast.DEFAULT_B) == (0.8, 2000.0)
    assert solvent_contrast.DEFAULT_ATOMIC_BFACTOR == 100.0
    dc = np.ravel_multi_index((grid_size // 2,) * 3, (grid_size,) * 3)
    assert filt[dc] == pytest.approx(0.2, abs=1e-15)
    reference = _reference_filter(grid_size, voxel_size, 0.8, 2000.0, 100.0)
    np.testing.assert_allclose(filt, reference, rtol=0, atol=1e-15)
    # Index (+2, 0, -1): |q|^2 = 5 / (N * voxel_size)^2 in cycles^2 / angstrom^2.
    idx = np.ravel_multi_index((grid_size // 2 + 2, grid_size // 2, grid_size // 2 - 1), (grid_size,) * 3)
    q2 = 5.0 / (grid_size * voxel_size) ** 2
    expected = (1 - 0.8 * np.exp(-2000.0 * q2 / 4)) * np.exp(-100.0 * q2 / 4)
    assert filt[idx] == pytest.approx(expected, abs=1e-15)


def test_zero_atomic_bfactor_is_solvent_contrast_only_and_bfactor_only_is_relion_convention():
    grid_size, voxel_size = 8, 1.5
    h_only = solvent_contrast.solvent_contrast_filter((grid_size,) * 3, voxel_size, atomic_bfactor=0.0)
    np.testing.assert_allclose(h_only, _reference_filter(grid_size, voxel_size, 0.8, 2000.0, 0.0), atol=1e-15)
    # a = 0 leaves exp(-B s^2 / 4), the simulator's existing B-factor convention.
    b_only = solvent_contrast.solvent_contrast_filter((grid_size,) * 3, voxel_size, a=0.0, atomic_bfactor=100.0)
    np.testing.assert_allclose(
        b_only, simulator.get_B_factor_scaling((grid_size,) * 3, voxel_size, 100.0), rtol=1e-14, atol=0
    )


def test_filter_depends_on_physical_frequency_not_pixel_index():
    # q = k / (N * voxel_size): index 1 on 8^3 at 1 A and index 4 on 16^3 at 2 A
    # are both 1/8 cycles/A, while index 1 on 8^3 at 0.5 A is 1/4 cycles/A.
    small = solvent_contrast.solvent_contrast_filter((8,) * 3, 1.0, atomic_bfactor=0.0).reshape((8,) * 3)
    large = solvent_contrast.solvent_contrast_filter((16,) * 3, 2.0, atomic_bfactor=0.0).reshape((16,) * 3)
    np.testing.assert_allclose(small[4 + 1, 4, 4], large[8 + 4, 8, 8], rtol=0, atol=1e-15)
    fine = solvent_contrast.solvent_contrast_filter((8,) * 3, 0.5, atomic_bfactor=0.0).reshape((8,) * 3)
    assert fine[4 + 1, 4, 4] > small[4 + 1, 4, 4]
    np.testing.assert_allclose(fine[4 + 1, 4, 4], 1 - 0.8 * np.exp(-2000.0 * (1 / 4.0) ** 2 / 4), atol=1e-15)


def test_real_space_plane_wave_is_scaled_by_h_at_its_frequency():
    grid_size, voxel_size, a, B, b_atomic = 16, 3.0, 0.7, 1500.0, 80.0
    k = np.array([1, 2, 3])
    x = np.arange(grid_size)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    wave = np.cos(2 * np.pi * (k[0] * xx + k[1] * yy + k[2] * zz) / grid_size)
    wave_ft = np.asarray(ftu.get_dft3(wave)).reshape(-1)
    out = solvent_contrast.apply_solvent_contrast(wave_ft, (grid_size,) * 3, voxel_size, a, B, b_atomic)
    out_real = np.asarray(ftu.get_idft3(out.reshape((grid_size,) * 3)))
    q2 = np.sum(k**2) / (grid_size * voxel_size) ** 2
    expected = (1 - a * np.exp(-B * q2 / 4)) * np.exp(-b_atomic * q2 / 4) * wave
    np.testing.assert_allclose(out_real.real, expected, atol=1e-12)
    np.testing.assert_allclose(out_real.imag, 0, atol=1e-12)


def test_apply_keeps_single_precision_and_does_not_renormalize():
    vols = _random_real_fourier_volumes(2, 4, seed=1).astype(np.complex64)
    out = solvent_contrast.apply_solvent_contrast(vols, (4,) * 3, 5.0)
    assert out.dtype == np.complex64
    assert np.all(np.linalg.norm(out, axis=-1) < np.linalg.norm(vols, axis=-1))


@pytest.mark.parametrize(
    "a,B,b_atomic",
    [
        (-0.1, 2000.0, 100.0),
        (1.5, 2000.0, 100.0),
        (0.8, -1.0, 100.0),
        (np.nan, 2000.0, 100.0),
        (0.8, np.inf, 100.0),
        (True, 2000.0, 100.0),
        (0.8, 2000.0, -5.0),
        (0.8, 2000.0, np.nan),
    ],
)
def test_invalid_parameters_are_rejected(a, B, b_atomic):
    with pytest.raises(ValueError):
        solvent_contrast.make_record(True, voxel_size=1.0, grid_size=4, a=a, B=B, atomic_bfactor=b_atomic)


# 3. Metadata round trip, overrides and exactly-once application


def test_record_round_trips_and_repeated_loading_is_not_cumulative(monkeypatch, tmp_path):
    raw = _random_real_fourier_volumes(3, 8, seed=2)
    monkeypatch.setattr(simulator, "load_volumes_from_folder", lambda *a, **k: raw.copy())
    info = _enabled_simulation_info(8, 1.7, scale_vol=0.5, a=0.5, B=1000.0, atomic_bfactor=40.0)
    path = tmp_path / "simulation_info.pkl"
    utils.pickle_dump(info, str(path))
    loaded = utils.pickle_load(str(path))
    record = loaded[solvent_contrast.METADATA_KEY]
    assert record == info[solvent_contrast.METADATA_KEY]
    assert (record["a"], record["B"], record["B_atomic"]) == (0.5, 1000.0, 40.0)
    assert (record["voxel_size"], record["grid_size"]) == (1.7, 8)
    assert record["model"] == "henderson_mcmullan_2013" and record["model_version"] == 2
    assert record["units"]["B"] == record["units"]["B_atomic"] == "angstrom^2"
    assert record["units"]["q"] == "cycles/angstrom"
    assert record["ground_truth_representation"] == solvent_contrast.UNCORRECTED_INPUTS

    expected = raw * 0.5 * _reference_filter(8, 1.7, 0.5, 1000.0, 40.0)
    first = synthetic_dataset.load_ground_truth_volumes(loaded)
    second = synthetic_dataset.load_ground_truth_volumes(loaded)
    hvd = synthetic_dataset.load_heterogeneous_reconstruction(str(path))
    np.testing.assert_allclose(first, expected, rtol=1e-13, atol=0)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(hvd.volumes, expected * hvd.valid_indices[None], rtol=1e-13, atol=0)


def test_version_1_record_loads_without_bfactor(monkeypatch):
    raw = _random_real_fourier_volumes(2, 8, seed=5)
    monkeypatch.setattr(simulator, "load_volumes_from_folder", lambda *a, **k: raw.copy())
    info = _enabled_simulation_info(8, 2.0, scale_vol=1.5)
    record = info[solvent_contrast.METADATA_KEY]
    record["model_version"] = 1
    del record["B_atomic"]
    expected = raw * 1.5 * _reference_filter(8, 2.0, 0.8, 2000.0, 0.0)
    np.testing.assert_allclose(synthetic_dataset.load_ground_truth_volumes(info), expected, rtol=1e-13, atol=0)


def test_corrected_effective_representation_is_not_filtered_again(monkeypatch):
    raw = _random_real_fourier_volumes(2, 4, seed=3)
    monkeypatch.setattr(simulator, "load_volumes_from_folder", lambda *a, **k: raw.copy())
    info = _enabled_simulation_info(4, 2.0, scale_vol=2.0)
    info[solvent_contrast.METADATA_KEY]["ground_truth_representation"] = solvent_contrast.CORRECTED_EFFECTIVE
    np.testing.assert_array_equal(synthetic_dataset.load_ground_truth_volumes(info), raw * 2.0)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r.update(model_version=3), "unsupported"),
        (lambda r: r.pop("B_atomic"), "incomplete"),
        (lambda r: r.update(B_atomic=-1.0), "atomic_bfactor"),
        (lambda r: r.update(model="other_model"), "unsupported"),
        (lambda r: r.pop("B"), "incomplete"),
        (lambda r: r.pop("voxel_size"), "incomplete"),
        (lambda r: r.update(ground_truth_representation="mystery"), "ground_truth_representation"),
        (lambda r: r.update(grid_size=16), "grid_size"),
        (lambda r: r.update(a=2.0), "amplitude"),
    ],
)
def test_bad_enabled_records_fail_clearly(monkeypatch, mutate, match):
    monkeypatch.setattr(simulator, "load_volumes_from_folder", lambda *a, **k: np.ones((2, 64), np.complex64))
    info = _enabled_simulation_info(4, 2.0)
    mutate(info[solvent_contrast.METADATA_KEY])
    with pytest.raises(ValueError, match=match):
        synthetic_dataset.load_ground_truth_volumes(info)


def test_enabled_record_without_scale_vol_fails(monkeypatch):
    monkeypatch.setattr(simulator, "load_volumes_from_folder", lambda *a, **k: np.ones((2, 64), np.complex64))
    info = _enabled_simulation_info(4, 2.0)
    del info["scale_vol"]
    with pytest.raises(ValueError, match="scale_vol"):
        synthetic_dataset.load_ground_truth_volumes(info)


def test_cli_flags_forward_to_simulator(monkeypatch, tmp_path):
    calls = {}

    def fake_generate(*args, **kwargs):
        calls.update(kwargs)
        return object(), {}

    monkeypatch.setattr(make_test_dataset.simulator, "generate_synthetic_dataset", fake_generate)
    argv = ["make_test_dataset", str(tmp_path), "--atomic-solvent-correction", "--solvent-contrast-a", "0.6"]
    monkeypatch.setattr(sys, "argv", argv + ["--solvent-contrast-b", "1200", "--atomic-bfactor", "0"])
    make_test_dataset.main()
    assert calls["atomic_solvent_correction"] is True
    assert (calls["solvent_contrast_a"], calls["solvent_contrast_B"], calls["atomic_bfactor"]) == (0.6, 1200.0, 0.0)

    monkeypatch.setattr(sys, "argv", ["make_test_dataset", str(tmp_path)])
    make_test_dataset.main()
    assert calls["atomic_solvent_correction"] is False
    assert calls["solvent_contrast_a"] is None and calls["solvent_contrast_B"] is None
    assert calls["atomic_bfactor"] is None


def test_em_development_preset_is_one_switch_with_bfactor_100():
    preset = solvent_contrast.EM_DEVELOPMENT_PRESET
    assert preset == {
        "atomic_solvent_correction": True,
        "solvent_contrast_a": 0.8,
        "solvent_contrast_B": 2000.0,
        "atomic_bfactor": 100.0,
    }
    import argparse

    parser = argparse.ArgumentParser()
    solvent_contrast.add_cli_arguments(parser)
    assert solvent_contrast.kwargs_from_cli_args(parser.parse_args(["--atomic-solvent-correction"])) == {
        "atomic_solvent_correction": True,
        "solvent_contrast_a": None,
        "solvent_contrast_B": None,
        "atomic_bfactor": None,
    }


def test_run_test_all_metrics_forwards_atomic_volume_options(monkeypatch, tmp_path):
    for i in range(3):
        (tmp_path / f"vol{i:04d}.mrc").write_bytes(b"")
    calls = {}

    def fake_generate(*args, **kwargs):
        calls.update(kwargs)
        return object(), {}

    monkeypatch.setattr(rtam.simulator, "generate_synthetic_dataset", fake_generate)
    rtam.make_big_test_dataset(
        str(tmp_path / "vol"),
        str(tmp_path / "out"),
        grid_size=16,
        n_images=10,
        **solvent_contrast.EM_DEVELOPMENT_PRESET,
    )
    assert all(calls[key] == value for key, value in solvent_contrast.EM_DEVELOPMENT_PRESET.items())
    calls.clear()
    rtam.make_big_test_dataset(str(tmp_path / "vol"), str(tmp_path / "out2"), grid_size=16, n_images=10)
    assert "atomic_solvent_correction" not in calls


# 5. Ground-truth mean, covariance and PCA of the corrected ensemble


def test_ground_truth_statistics_match_direct_corrected_ensemble(monkeypatch):
    grid_size, voxel_size = 8, 2.0
    raw = _random_real_fourier_volumes(4, grid_size, seed=4)
    monkeypatch.setattr(simulator, "load_volumes_from_folder", lambda *a, **k: raw.copy())
    info = _enabled_simulation_info(grid_size, voxel_size)
    info["image_assignment"] = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, -1], dtype=np.int32)
    info["per_image_contrast"] = np.ones(info["image_assignment"].size, dtype=np.float32)
    hvd = synthetic_dataset.load_heterogeneous_reconstruction(info)

    weights = hvd.get_probs_of_state()  # existing ensemble weights, float32
    np.testing.assert_allclose(weights, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
    op = _reference_filter(grid_size, voxel_size, 0.8, 2000.0, 100.0) * hvd.valid_indices
    raw_mean = weights @ raw
    raw_centered = (raw - raw_mean) * np.sqrt(weights)[:, None]
    raw_cov = raw_centered.T @ raw_centered.conj()

    # mean_true = T(mean) and covariance_true = T C T* (T is diagonal in Fourier).
    np.testing.assert_allclose(hvd.get_mean(), op * raw_mean, rtol=1e-12, atol=1e-12)
    cov_true = op[:, None] * raw_cov * op[None, :]
    u, s = hvd.get_u(), hvd.get_s()
    np.testing.assert_allclose((u * s) @ u.conj().T, cov_true, rtol=0, atol=1e-10 * np.abs(cov_true).max())

    # Low-rank route from the uncorrected PCs: T(U) sqrt(L) = U_new S W*.
    u_raw, s_raw, _ = np.linalg.svd(raw_centered.T, full_matrices=False)
    u_new, s_new, _ = np.linalg.svd(op[:, None] * u_raw * s_raw[None, :], full_matrices=False)
    rank = 3  # four states, one centering constraint
    np.testing.assert_allclose(s[:rank], s_new[:rank] ** 2, rtol=1e-10)
    assert np.all(s[rank:] < 1e-12 * s[0])
    proj_hvd = u[:, :rank] @ u[:, :rank].conj().T
    proj_new = u_new[:, :rank] @ u_new[:, :rank].conj().T
    np.testing.assert_allclose(proj_hvd, proj_new, atol=1e-10)


# 4. End to end: simulator input volumes equal the loaded effective truth


def _write_blob_volumes(prefix, grid_size, voxel_size, n_volumes=2):
    x = np.linspace(-1.0, 1.0, grid_size)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    for i in range(n_volumes):
        vol = np.exp(-((xx - 0.3 * i) ** 2 + yy**2 + zz**2) / (2 * 0.2**2)).astype(np.float32)
        utils.write_mrc(f"{prefix}{i:04d}.mrc", vol, voxel_size=voxel_size)


def _simulate(tmp_path, name, monkeypatch, **kwargs):
    captured = []
    original = simulator.simulate_data

    def capturing_simulate_data(dataset, volumes, *args, **kw):
        captured.append(np.array(volumes))
        return original(dataset, volumes, *args, **kw)

    monkeypatch.setattr(simulator, "simulate_data", capturing_simulate_data)
    np.random.seed(0)
    out = tmp_path / name
    _, info = simulator.generate_synthetic_dataset(
        str(out),
        4.0,
        str(tmp_path / "vol"),
        16,
        grid_size=16,
        volume_distribution=np.array([0.5, 0.5]),
        dataset_params_option="uniform",
        noise_level=1.0,
        noise_model="white",
        put_extra_particles=False,
        percent_outliers=0.0,
        noise_scale_std=0.0,
        contrast_std=0.0,
        noise_rng_batch_size=16,
        **kwargs,
    )
    monkeypatch.setattr(simulator, "simulate_data", original)
    return info, captured[-1], str(out / "simulation_info.pkl")


def test_end_to_end_simulator_and_loader_agree(tmp_path, monkeypatch):
    _write_blob_volumes(str(tmp_path / "vol"), 16, 4.0)
    vol_bytes = (tmp_path / "vol0000.mrc").read_bytes()

    info_off, projected_off, _ = _simulate(tmp_path, "off", monkeypatch)
    info_on, projected_on, sim_info_path = _simulate(
        tmp_path, "on", monkeypatch, **{**solvent_contrast.EM_DEVELOPMENT_PRESET, "solvent_contrast_B": 1500.0}
    )

    assert info_off[solvent_contrast.METADATA_KEY] == {"enabled": False}
    record = utils.pickle_load(sim_info_path)[solvent_contrast.METADATA_KEY]
    assert record["enabled"] and (record["a"], record["B"], record["B_atomic"]) == (0.8, 1500.0, 100.0)
    assert record["voxel_size"] == 4.0 and record["grid_size"] == 16
    assert (tmp_path / "vol0000.mrc").read_bytes() == vol_bytes

    # The loaded effective truth is exactly the array handed to the projector.
    gt_on = synthetic_dataset.load_ground_truth_volumes(utils.pickle_load(sim_info_path))
    np.testing.assert_array_equal(gt_on, projected_on)
    hvd = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    np.testing.assert_array_equal(hvd.volumes, projected_on * hvd.valid_indices[None])
    gt_off = synthetic_dataset.load_ground_truth_volumes(info_off)

    # Fixed noise model: the attenuation survives the image-power normalization,
    # which rescales signal and noise together (SNR set before the correction).
    # Tolerances are float32 storage (MRC volumes, stored noise); the float64
    # filter and scaling arithmetic is checked in the round-trip test.
    filt = _reference_filter(16, 4.0, 0.8, 1500.0, 100.0)
    ratio = info_on["scale_vol"] / info_off["scale_vol"]
    np.testing.assert_allclose(gt_on, gt_off * ratio * filt, rtol=2e-5, atol=2e-6 * np.abs(gt_on).max())
    np.testing.assert_allclose(info_on["noise_variance"] / info_off["noise_variance"], ratio**2, rtol=1e-5)
    signal_to_noise = [
        np.sum(np.abs(g) ** 2) / i["noise_variance"][0] for g, i in ((gt_off, info_off), (gt_on, info_on))
    ]
    assert signal_to_noise[1] < 0.5 * signal_to_noise[0]


def test_cli_helper_enabled_by_default_has_an_opt_out():
    import argparse

    parser = argparse.ArgumentParser()
    solvent_contrast.add_cli_arguments(parser, enabled_by_default=True)
    assert solvent_contrast.kwargs_from_cli_args(parser.parse_args([]))["atomic_solvent_correction"] is True
    off = solvent_contrast.kwargs_from_cli_args(parser.parse_args(["--no-atomic-solvent-correction"]))
    assert off == {
        "atomic_solvent_correction": False,
        "solvent_contrast_a": None,
        "solvent_contrast_B": None,
        "atomic_bfactor": None,
    }
