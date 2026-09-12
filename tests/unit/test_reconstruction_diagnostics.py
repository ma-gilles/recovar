"""Saved reconstruction captures retain their independent file contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.diagnostics import reconstruction as dumps

pytestmark = pytest.mark.unit


@pytest.fixture
def capture_inputs(tmp_path):
    spectrum = np.array([1, 2, 4], dtype=np.float64)
    complex_rows = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
    stats = {key: spectrum for key in ("avg_weight_shells", "shell_sum", "shell_count")}
    return dict(
        output_dir=tmp_path,
        iteration=2,
        class_idx=1,
        current_size=8,
        grid_size=16,
        PADDING_FACTOR=2,
        PROJECTION_PADDING_FACTOR=3,
        voxel_size=1.6375,
        computed_cs=10,
        prev_cs=6,
        raw_cs=9,
        res_shell=3,
        per_class_res_shell=[2, 3],
        relion_incr_size=2,
        relion_has_high_fsc_at_limit=True,
        state=SimpleNamespace(ave_Pmax=0.75, current_resolution=0.2, previous_resolution=0.1),
        data_vs_prior_prev_raw=spectrum,
        data_vs_prior_prev=spectrum,
        previous_means=[complex_rows, -complex_rows],
        Ft_y_combined=complex_rows,
        Ft_ctf_combined=complex_rows,
        Ft_ctf_0=complex_rows,
        Ft_ctf_1=-complex_rows,
        tau2_fudge=4.0,
        kclass_tau2_frame_scale=2.0,
        kclass_tau2_source="iref",
        mstep_accumulator_shape=(4, 4, 3),
        mstep_full_half_axis=0,
        tau2_shells_recovar_frame_k=spectrum,
        tau2_shells_relion_frame_k=spectrum * 2,
        shell_stats_k=stats,
        reconstruct_floor_stats_k=stats,
        data_vs_prior_k=spectrum,
        pixel_res=3.0,
        dvp_iter=spectrum,
        fsc=spectrum / 4,
        tau2_update_details={"prior_shells": spectrum, "fsc_shells": None},
        tau2_update_details_per_half=[None, {"ssnr_shells": spectrum}],
        perturb_replay_relion_dir=None,
        perturb_replay_relion_prefix="run",
        sealed_sampling_state=None,
        _replay_meta={"source": "test"},
        final_current_size=16,
        volume_shape=(16, 16, 16),
        final_mstep_accumulator_shape=(4, 4, 3),
        final_mstep_full_half_axis=0,
        k_class_enabled=True,
        final_grid_correct=False,
        final_Ft_y_0=complex_rows,
        final_Ft_y_1=-complex_rows,
        final_Ft_ctf_0=complex_rows,
        final_Ft_ctf_1=-complex_rows,
        final_unfiltered_Ft_y_0=complex_rows * 2,
        final_unfiltered_Ft_y_1=-complex_rows * 2,
        final_unfiltered_Ft_ctf_0=complex_rows * 2,
        final_unfiltered_Ft_ctf_1=-complex_rows * 2,
        final_ft_y=complex_rows * 3,
        final_ft_ctf=complex_rows * 3,
        final_iter_fsc=spectrum / 4,
        final_tau2_update_details={"prior_shells": spectrum},
        logger=SimpleNamespace(info=lambda *args: None),
    )


def invoke(writer, values):
    # Fixture contains inputs for all four capture boundaries.
    import inspect

    writer(**{name: values[name] for name in inspect.signature(writer).parameters})


def test_current_size_schema_and_casts(capture_inputs):
    invoke(dumps.write_kclass_current_size, capture_inputs)
    path = capture_inputs["output_dir"] / "recovar_kclass_current_size_it003.npz"
    with np.load(path) as saved:
        assert set(saved.files) == {
            "iteration",
            "previous_current_size",
            "grid_size",
            "resolution_shell",
            "per_class_resolution_shells",
            "ave_Pmax",
            "state_current_resolution",
            "state_previous_resolution",
            "relion_incr_size",
            "relion_has_high_fsc_at_limit",
            "data_vs_prior_prev_raw",
            "data_vs_prior_prev",
            "raw_current_size",
            "quantized_current_size",
        }
        assert saved["iteration"].item() == 3
        assert saved["quantized_current_size"].item() == 10
        assert saved["iteration"].dtype == np.int32
        assert saved["ave_Pmax"].dtype == np.float64
        assert saved["data_vs_prior_prev"].dtype == np.float32
        np.testing.assert_array_equal(saved["per_class_resolution_shells"], [2, 3])


@pytest.mark.parametrize("token,preserve", [("", False), (" OFF ", False), ("1", True), ("unrecognized", True)])
@pytest.mark.parametrize("missing_half", [False, True])
def test_mstep_class_selection_and_dtype(capture_inputs, monkeypatch, token, preserve, missing_half):
    monkeypatch.setenv("RECOVAR_KCLASS_DUMP_PRESERVE_DTYPE", token)
    if missing_half:
        capture_inputs["Ft_ctf_1"] = None
    invoke(dumps.write_kclass_mstep, capture_inputs)
    with np.load(capture_inputs["output_dir"] / "recovar_kclass_mstep_it003_c02.npz") as saved:
        expected_dtype = np.complex128 if preserve else np.complex64
        assert saved["Ft_y_combined"].dtype == expected_dtype
        np.testing.assert_array_equal(saved["Ft_y_combined"], capture_inputs["Ft_y_combined"][1])
        assert saved["previous_mean_half1"].dtype == np.complex64
        np.testing.assert_array_equal(saved["previous_mean_half1"], capture_inputs["previous_means"][1][1])
        if missing_half:
            assert saved["Ft_ctf_1"].size == 0
            assert saved["Ft_ctf_1"].dtype == np.complex64
        assert saved["dump_preserve_dtype"].item() == preserve
        assert saved["tau2_shells"].dtype == np.float64


@pytest.mark.parametrize("include_fsc", [False, True])
def test_tau2_optional_fields_and_replay_path(capture_inputs, include_fsc):
    capture_inputs["fsc"] = np.array([0.8, 0.4]) if include_fsc else None
    capture_inputs["perturb_replay_relion_dir"] = capture_inputs["output_dir"]
    invoke(dumps.write_tau2_update, capture_inputs)
    with np.load(capture_inputs["output_dir"] / "recovar_tau2_debug_it003.npz") as saved:
        assert ("current_iter_fsc" in saved) == include_fsc
        assert "tau2_fsc_shells" not in saved
        assert "half1_ssnr_shells" not in saved
        assert saved["half2_ssnr_shells"].dtype == np.float64
        assert saved["voxel_size"].item() == 1.6375
        assert not saved["relion_model_exists"].item()
        assert saved["relion_model_path"].item().endswith("run_it003_half1_model.star")
        assert saved["replay_meta_source"].item() == "test"


@pytest.mark.parametrize("k_class", [False, True])
def test_final_bpref_preserves_complex_data_and_real_weights(capture_inputs, k_class):
    capture_inputs["k_class_enabled"] = k_class
    invoke(dumps.write_final_bpref_accumulators, capture_inputs)
    with np.load(capture_inputs["output_dir"] / "recovar_final_bpref_accum.npz") as saved:
        assert saved["Ft_y"].dtype == np.complex128
        assert saved["Ft_ctf"].dtype == np.float64
        np.testing.assert_array_equal(saved["Ft_y"], capture_inputs["final_ft_y"])
        np.testing.assert_array_equal(saved["Ft_ctf"], capture_inputs["final_ft_ctf"].real)
        assert saved["voxel_size"].dtype == np.float32
        assert saved["tau2_weight_combination"].item() == ("class_iref" if k_class else "sum")
        assert "tau2_fsc_shells" not in saved
