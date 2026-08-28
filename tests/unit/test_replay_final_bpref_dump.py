from __future__ import annotations

import inspect
import struct

import numpy as np

from scripts import replay_final_bpref_dump


def test_replay_final_bpref_dump_defaults_old_dumps_to_native_half_axis():
    assert replay_final_bpref_dump.resolve_tau2_full_half_axis({}) == -1


def test_replay_final_bpref_dump_reads_recorded_relion_x_half_axis():
    dump = {"mstep_full_half_axis": np.asarray(0, dtype=np.int32)}

    assert replay_final_bpref_dump.resolve_tau2_full_half_axis(dump) == 0


def test_replay_final_bpref_dump_axis_override_wins():
    dump = {"mstep_full_half_axis": np.asarray(0, dtype=np.int32)}

    assert replay_final_bpref_dump.resolve_tau2_full_half_axis(dump, override=-1) == -1


def test_replay_final_bpref_dump_resolves_recorded_accumulator_shape():
    dump = {"mstep_accumulator_shape": np.asarray([259, 259, 259], dtype=np.int32)}

    assert replay_final_bpref_dump.resolve_mstep_accumulator_shape(dump, (128, 128, 128), 2) == (
        259,
        259,
        259,
    )


def test_replay_final_bpref_dump_old_shape_defaults_to_even_padding():
    assert replay_final_bpref_dump.resolve_mstep_accumulator_shape({}, (128, 128, 128), 2) == (
        256,
        256,
        256,
    )


def test_replay_final_bpref_dump_uses_joined_half_weight_sum():
    source = inspect.getsource(replay_final_bpref_dump.main)

    assert 'weight_combination="sum"' in source
    assert '"tau2_weight_combination": "sum"' in source


def test_replay_fsc_uses_canonical_non_nyquist_shell_range():
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((16, 16, 16))

    fsc = replay_final_bpref_dump.shell_fsc(volume, volume)

    assert fsc.shape == (16 // 2 - 1,)
    np.testing.assert_allclose(fsc, 1.0, atol=1e-12)


def test_read_relion_bpref_array_reads_shape_header(tmp_path):
    path = tmp_path / "bpref.bin"
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    with path.open("wb") as stream:
        stream.write(struct.pack("qqq", *values.shape))
        values.tofile(stream)

    loaded = replay_final_bpref_dump.read_relion_bpref_array(path, dtype=np.dtype(np.float64))

    np.testing.assert_array_equal(loaded, values)


def test_read_relion_spectrum_reads_length_header(tmp_path):
    path = tmp_path / "tau2.bin"
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    with path.open("wb") as stream:
        stream.write(struct.pack("q", values.size))
        values.tofile(stream)

    loaded = replay_final_bpref_dump.read_relion_spectrum(path)

    np.testing.assert_array_equal(loaded, values)


def test_relion_bpref_numerator_conversion_applies_global_sign_and_fft_scale():
    relion_layout_converted = np.asarray([4.0 + 8.0j, -12.0 + 16.0j])

    actual = replay_final_bpref_dump.relion_bpref_numerator_to_recovar_units(
        relion_layout_converted,
        grid_size=2,
    )

    np.testing.assert_array_equal(actual, np.asarray([-1.0 - 2.0j, 3.0 - 4.0j]))


def test_relion_bpref_numerator_conversion_rejects_nonpositive_grid_size():
    with np.testing.assert_raises_regex(ValueError, "grid_size must be positive"):
        replay_final_bpref_dump.relion_bpref_numerator_to_recovar_units(
            np.ones(1),
            grid_size=0,
        )


def test_streaming_field_metrics_reports_exact_equality_across_chunks():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)

    metrics = replay_final_bpref_dump.streaming_field_metrics(
        values,
        values.copy(),
        chunk_size=5,
    )

    assert metrics == {
        "element_count": 12,
        "exact_equal": True,
        "mismatch_count": 0,
        "first_mismatch_flat_index": None,
        "maximum_absolute_residual": 0.0,
        "relative_l2": 0.0,
        "source_to_target_least_squares_scale": 1.0,
        "relative_l2_after_scale": 0.0,
    }


def test_streaming_field_metrics_separates_scale_from_structural_residual():
    source = np.asarray([1.0 + 2.0j, -3.0 + 1.0j, 2.0 - 4.0j])
    target = 2.5 * source

    metrics = replay_final_bpref_dump.streaming_field_metrics(
        source,
        target,
        chunk_size=2,
    )

    assert metrics["exact_equal"] is False
    assert metrics["mismatch_count"] == source.size
    assert metrics["first_mismatch_flat_index"] == 0
    np.testing.assert_allclose(metrics["source_to_target_least_squares_scale"], 2.5)
    np.testing.assert_allclose(metrics["relative_l2"], 0.6)
    np.testing.assert_allclose(metrics["relative_l2_after_scale"], 0.0, atol=1e-15)


def test_compare_relion_boundary_spectra_applies_native_frame_scale(tmp_path):
    prefix = tmp_path / "mstep"
    grid_size = 4
    n4 = float(grid_size**4)
    native = {
        "tau2": np.asarray([1.0, 2.0, 3.0]),
        "sigma2": np.asarray([4.0, 5.0, 6.0]),
        "data_vs_prior": np.asarray([7.0, 8.0, 9.0]),
        "fsc": np.asarray([0.9, 0.8, 0.7]),
        "fourier_coverage": np.asarray([0.1, 0.2, 0.3]),
    }
    for name, values in native.items():
        with (tmp_path / f"mstep_{name}.bin").open("wb") as stream:
            stream.write(struct.pack("q", values.size))
            values.tofile(stream)
    dump = {
        "tau2_prior_shells": native["tau2"] * n4,
        "tau2_sigma2_shells": native["sigma2"] * n4,
        "tau2_ssnr_shells": native["data_vs_prior"],
        "fsc_shells": native["fsc"],
    }

    report = replay_final_bpref_dump.compare_relion_boundary_spectra(
        dump,
        prefix,
        grid_size=grid_size,
    )

    assert report["native_to_recovar_tau2_sigma2_scale"] == n4
    assert all(
        row["exact_equal"]
        for row in report["comparisons"].values()
    )
    assert report["native_fourier_coverage"]["element_count"] == 3
