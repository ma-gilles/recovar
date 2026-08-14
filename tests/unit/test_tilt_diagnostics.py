from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from recovar.data_io.cryoem_dataset import CryoEMDataset
from recovar.data_io.image_metadata import ImageMetadata
from recovar.output import plot_utils
from recovar.output.tilt_diagnostics import (
    _neutralized_group_dataset,
    _plot_reconstruction_power_comparisons,
    _plot_six_view_rows,
    _remove_scale_dose_from_power,
    group_preexposures,
)
from recovar.reconstruction import noise

pytestmark = pytest.mark.unit


class _PipelineOutput:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        if key not in self.values:
            raise KeyError(key)
        return self.values[key]


def test_group_preexposures_sorts_and_clusters_roundoff():
    groups = group_preexposures([3.0, 0.0, 1.000000004, 3.000000003, 1.0, 0.0])

    np.testing.assert_allclose([group.pre_exposure for group in groups], [0.0, 1.000000002, 3.0000000015])
    np.testing.assert_array_equal(groups[0].image_indices, [1, 5])
    np.testing.assert_array_equal(groups[1].image_indices, [2, 4])


def test_plot_covariance_column_fscs_writes_compact_figure(tmp_path):
    po = _PipelineOutput(
        {
            "column_fscs": np.linspace(-0.1, 1.0, 48, dtype=np.float32).reshape(4, 12),
            "volume_shape": (32, 32, 32),
            "voxel_size": 2.0,
        }
    )
    destination = tmp_path / "covariance_column_fscs.png"

    figure, axes = plot_utils.plot_covariance_column_fscs(po, destination)

    assert destination.is_file()
    assert np.asarray(axes).size == 2
    plt.close(figure)


def test_plot_noise_group_summary_matches_each_group_power_spectrum(tmp_path):
    n_groups, n_shells = 3, 12
    po = _PipelineOutput(
        {
            "noise_var_used": np.arange(n_groups * n_shells, dtype=np.float32).reshape(n_groups, n_shells) + 1,
            "noise_group_image_PS": np.arange(n_groups * n_shells, dtype=np.float32).reshape(n_groups, n_shells) + 3,
            "noise_group_metadata": [
                {
                    "group_index": index,
                    "pre_exposure": float(index * 3),
                    "median_tilt_angle_deg": float(index * 10),
                    "tilt_angle_source": "ctf_tilt_angle",
                }
                for index in range(n_groups)
            ],
            "volume_shape": (32, 32, 32),
            "voxel_size": 2.0,
            "input_args": SimpleNamespace(ignore_zero_frequency=False),
        }
    )
    destination = tmp_path / "noise_power_by_tilt.png"

    figure, axes = plot_utils.plot_noise_group_summary(po, destination)

    assert destination.is_file()
    assert np.asarray(axes).shape == (2, 2)
    assert len(axes[1, 0].lines) == 2 * n_groups
    assert len(axes[1, 1].lines) == n_groups
    detail_dir = tmp_path / "noise_power_by_tilt_individual"
    assert sorted(path.name for path in detail_dir.glob("*.png")) == [
        "tilt_000_vs_000.png",
        "tilt_001_vs_000.png",
        "tilt_002_vs_000.png",
    ]
    plt.close(figure)


def test_plot_volume_variance_prior_uses_saved_pipeline_arrays_without_rescaling(tmp_path, monkeypatch):
    volume_shape = (4, 4, 4)
    n_voxels = int(np.prod(volume_shape))
    variance_est = {
        "corrected0": np.full(n_voxels, 2.0, dtype=np.float32),
        "corrected1": np.full(n_voxels, 4.0, dtype=np.float32),
        "combined": np.full(n_voxels, 3.0, dtype=np.float32),
        "prior": np.full(n_voxels, 7.0, dtype=np.float32),
    }
    po = _PipelineOutput(
        {
            "volume_shape": volume_shape,
            "voxel_size": 2.0,
            "mean": np.full(n_voxels, 2.0 + 0.0j, dtype=np.complex64),
            "variance_est": variance_est,
            "covariance_options": {"column_sampling_scheme": "high_snr_from_var_est"},
        }
    )
    monkeypatch.setattr(
        plot_utils.regularization,
        "average_over_shells",
        lambda values, _shape: np.asarray(values).reshape(-1)[:3],
    )
    destination = tmp_path / "volume_variance_prior.png"

    figure, axes = plot_utils.plot_volume_variance_prior_summary(po, destination)

    assert destination.is_file()
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), 4.0)
    np.testing.assert_allclose(axes[0].lines[1].get_ydata(), 3.0)
    np.testing.assert_allclose(axes[1].lines[-1].get_ydata(), 7.0)
    plt.close(figure)


def test_tilt_reconstruction_plots_write_six_views_and_rank_zero_comparisons(tmp_path):
    rng = np.random.default_rng(4)
    volumes = [rng.standard_normal((8, 8, 8)).astype(np.float32) for _ in range(3)]
    records = [
        {
            "group_index": index,
            "pre_exposure": float(index * 3),
            "n_images": 20 - index,
            "tilt_angle_deg": float(index * 10),
            "tilt_angle_inferred": False,
        }
        for index in range(3)
    ]
    six_view = tmp_path / "six_view.png"
    _plot_six_view_rows(volumes, records, six_view)
    assert six_view.is_file()

    frequencies = np.arange(6, dtype=np.float64) / 16.0
    power = np.arange(18, dtype=np.float64).reshape(3, 6) + 1
    predicted_amplitudes = np.asarray([[1.0] * 6, [0.9] * 6, [0.8] * 6])
    corrected_power = _remove_scale_dose_from_power(power, predicted_amplitudes)
    overview = tmp_path / "reconstruction_power.png"
    details = tmp_path / "reconstruction_power_individual"
    _plot_reconstruction_power_comparisons(
        power,
        corrected_power,
        records,
        frequencies,
        overview,
        details,
    )
    assert overview.is_file()
    assert sorted(path.name for path in details.glob("*.png")) == [
        "tilt_000_vs_000.png",
        "tilt_001_vs_000.png",
        "tilt_002_vs_000.png",
    ]


def test_remove_scale_dose_from_power_collapses_forward_attenuated_power():
    reference = np.asarray([4.0, 9.0, 16.0])
    envelope = np.asarray([[1.0, 1.0, 1.0], [0.8, 0.6, 0.5]])
    observed = reference[None, :] * envelope**2

    corrected = _remove_scale_dose_from_power(observed, envelope)

    np.testing.assert_allclose(corrected, np.broadcast_to(reference, corrected.shape))


def test_neutralized_group_dataset_removes_contrast_dose_and_tilt_envelope():
    from recovar import core

    n_images = 4
    metadata = ImageMetadata(
        np.repeat(np.eye(3, dtype=np.float32)[None], n_images, axis=0),
        np.zeros((n_images, 2), dtype=np.float32),
        np.asarray(
            [
                [10000, 11000, 0, 300, 2.7, 0.1, 0, 0, 0.8, 0, 0],
                [10000, 11000, 0, 300, 2.7, 0.1, 0, 0, 0.7, 3, 20],
                [10000, 11000, 0, 300, 2.7, 0.1, 0, 0, 0.6, 6, 30],
                [10000, 11000, 0, 300, 2.7, 0.1, 0, 0, 0.5, 9, 40],
            ],
            dtype=np.float32,
        ),
    )
    dataset = CryoEMDataset(
        None,
        2.0,
        metadata,
        grid_size=8,
        tilt_series_flag=True,
        ctf_evaluator=core.CTFEvaluator(mode=core.CTFMode.CRYO_ET),
    )
    source = dataset.get_ctf_params_copy()
    group = group_preexposures(source[:, core.CTFParamIndex.DOSE])[1]

    neutral, unit_noise = _neutralized_group_dataset(dataset, group, source)

    np.testing.assert_array_equal(neutral.CTF_params[:, core.CTFParamIndex.CONTRAST], 1.0)
    np.testing.assert_array_equal(neutral.CTF_params[:, core.CTFParamIndex.BFACTOR], 0.0)
    np.testing.assert_array_equal(neutral.CTF_params[:, core.CTFParamIndex.DOSE], 0.0)
    np.testing.assert_array_equal(neutral.CTF_params[:, core.CTFParamIndex.TILT_ANGLE], 0.0)
    assert neutral.ctf_evaluator.mode == core.CTFMode.SPA
    np.testing.assert_array_equal(unit_noise, 1.0)


def test_variable_noise_rows_are_independent_when_broadcast_from_one_profile():
    profile = np.arange(4, dtype=np.float32)
    model = noise.VariableRadialNoiseModel(
        np.tile(profile, (3, 1)),
        np.asarray([0, 1, 2], dtype=np.int32),
        image_shape=(8, 8),
    )

    model.noise_variance_radials[0, 0] = 99

    assert model.noise_variance_radials[1, 0] == 0
    assert model.noise_variance_radials[2, 0] == 0
