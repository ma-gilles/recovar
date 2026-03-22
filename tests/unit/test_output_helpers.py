import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("jax")
from recovar import utils
from recovar.output import output

pytestmark = pytest.mark.unit


def test_sum_over_other_matches_manual_reduction():
    x = np.arange(24).reshape(2, 3, 4)
    out = output.sum_over_other(x, use_axis=[1])
    expected = np.sum(x, axis=(0, 2))
    np.testing.assert_array_equal(out, expected)


def test_half_slice_other_kept_axes():
    density = np.arange(27).reshape(3, 3, 3)
    out = output.half_slice_other(density, axes=[0, 2])
    np.testing.assert_array_equal(out, density[:, 1, :])


def test_slice_at_point_kept_axes():
    density = np.arange(27).reshape(3, 3, 3)
    point = np.array([2, 0, 1])
    out = output.slice_at_point(density, axes=[0, 2], point=point)
    np.testing.assert_array_equal(out, density[:, 0, :])


def test_resample_trajectory_uses_interpolated_indices(monkeypatch):
    monkeypatch.setattr(output, "get_resampled_distances", lambda _: np.array([0.0, 2.0, 4.0]))
    gt_vols = np.zeros((3, 8))
    indices = output.resample_trajectory(gt_vols, n_vols_along_path=5)
    np.testing.assert_array_equal(indices, np.array([0, 0, 1, 2, 2]))


def test_pipeline_output_get_embedding_component_uses_particle_halfsets(tmp_path):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    model_dir.mkdir(parents=True)

    params = {
        "version": "0.2",
        "input_args": SimpleNamespace(tilt_series=False, shared_contrast_across_tilts=True),
    }
    utils.pickle_dump(params, str(model_dir / "params.pkl"))
    utils.pickle_dump(
        {
            "latent_coords": {2: np.array([[0, 10], [1, 11], [2, 12], [3, 13]], dtype=np.float32)},
            "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
            "contrasts": {2: np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
        },
        str(model_dir / "embeddings.pkl"),
    )
    # image halfsets != particle halfsets to make selection explicit.
    utils.pickle_dump([np.array([1, 3], dtype=np.int32), np.array([0, 2], dtype=np.int32)], str(model_dir / "halfsets.pkl"))
    utils.pickle_dump(
        [np.array([2, 0], dtype=np.int32), np.array([3, 1], dtype=np.int32)],
        str(model_dir / "particles_halfsets.pkl"),
    )

    po = output.PipelineOutput(str(result_path))
    zs_sel = po.get_embedding_component("latent_coords", 2)
    contrasts_sel = po.get_embedding_component("contrasts", 2)
    np.testing.assert_array_equal(zs_sel, np.array([[2, 12], [0, 10], [3, 13], [1, 11]], dtype=np.float32))
    np.testing.assert_array_equal(contrasts_sel, np.array([2.0, 0.0, 3.0, 1.0], dtype=np.float32))


def test_pipeline_output_get_embedding_component_uses_image_halfsets_for_unshared_tilt_contrast(tmp_path):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    model_dir.mkdir(parents=True)

    params = {
        "version": "0.2",
        "input_args": SimpleNamespace(tilt_series=True, shared_contrast_across_tilts=False),
    }
    utils.pickle_dump(params, str(model_dir / "params.pkl"))
    utils.pickle_dump(
        {
            "latent_coords": {2: np.array([[0, 10], [1, 11], [2, 12], [3, 13]], dtype=np.float32)},
            "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
            "contrasts": {2: np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
        },
        str(model_dir / "embeddings.pkl"),
    )
    utils.pickle_dump([np.array([1, 3], dtype=np.int32), np.array([0, 2], dtype=np.int32)], str(model_dir / "halfsets.pkl"))
    utils.pickle_dump(
        [np.array([2, 0], dtype=np.int32), np.array([3, 1], dtype=np.int32)],
        str(model_dir / "particles_halfsets.pkl"),
    )

    po = output.PipelineOutput(str(result_path))
    contrasts_sel = po.get_embedding_component("contrasts", 2)
    # For tilt_series with unshared contrast, halfsets should be image halfsets.
    np.testing.assert_array_equal(contrasts_sel, np.array([1.0, 3.0, 0.0, 2.0], dtype=np.float32))


def test_pipeline_output_get_embedding_component_missing_shared_flag_defaults_to_particle_halfsets(tmp_path):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    model_dir.mkdir(parents=True)

    # shared_contrast_across_tilts missing: should default to shared=True (particle-halfset path).
    params = {"version": "0.2", "input_args": SimpleNamespace(tilt_series=True)}
    utils.pickle_dump(params, str(model_dir / "params.pkl"))
    utils.pickle_dump(
        {
            "latent_coords": {2: np.zeros((4, 2), dtype=np.float32)},
            "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
            "contrasts": {2: np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
        },
        str(model_dir / "embeddings.pkl"),
    )
    utils.pickle_dump([np.array([1, 3], dtype=np.int32), np.array([0, 2], dtype=np.int32)], str(model_dir / "halfsets.pkl"))
    utils.pickle_dump(
        [np.array([2, 0], dtype=np.int32), np.array([3, 1], dtype=np.int32)],
        str(model_dir / "particles_halfsets.pkl"),
    )

    po = output.PipelineOutput(str(result_path))
    contrasts_sel = po.get_embedding_component("contrasts", 2)
    np.testing.assert_array_equal(contrasts_sel, np.array([2.0, 0.0, 3.0, 1.0], dtype=np.float32))


def test_pipeline_output_get_embedding_component_dict_input_args_unshared_tilt_uses_image_halfsets(tmp_path):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    model_dir.mkdir(parents=True)

    params = {
        "version": "0.2",
        "input_args": {"tilt_series": True, "shared_contrast_across_tilts": False},
    }
    utils.pickle_dump(params, str(model_dir / "params.pkl"))
    utils.pickle_dump(
        {
            "latent_coords": {2: np.zeros((4, 2), dtype=np.float32)},
            "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
            "contrasts": {2: np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
        },
        str(model_dir / "embeddings.pkl"),
    )
    utils.pickle_dump([np.array([1, 3], dtype=np.int32), np.array([0, 2], dtype=np.int32)], str(model_dir / "halfsets.pkl"))
    utils.pickle_dump(
        [np.array([2, 0], dtype=np.int32), np.array([3, 1], dtype=np.int32)],
        str(model_dir / "particles_halfsets.pkl"),
    )

    po = output.PipelineOutput(str(result_path))
    contrasts_sel = po.get_embedding_component("contrasts", 2)
    np.testing.assert_array_equal(contrasts_sel, np.array([1.0, 3.0, 0.0, 2.0], dtype=np.float32))


def test_pipeline_output_get_unsorted_embedding_component_migrates_legacy_keys(tmp_path):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    model_dir.mkdir(parents=True)

    params = {"version": "0.2", "input_args": SimpleNamespace(tilt_series=False)}
    utils.pickle_dump(params, str(model_dir / "params.pkl"))
    utils.pickle_dump(
        {
            "zs": {4: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
            "cov_zs": {4: np.zeros((2, 2, 2), dtype=np.float32)},
            "contrasts": {4: np.array([0.1, 0.9], dtype=np.float32)},
        },
        str(model_dir / "embeddings.pkl"),
    )

    po = output.PipelineOutput(str(result_path))
    zs = po.get_unsorted_embedding_component("latent_coords", 4)
    np.testing.assert_array_equal(zs, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_pipeline_output_get_u_real_uses_available_saved_eigenvectors(tmp_path, monkeypatch):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    vols_dir = result_path / "output" / "volumes"
    model_dir.mkdir(parents=True)
    vols_dir.mkdir(parents=True)

    utils.pickle_dump({"version": "0", "volume_shape": (2, 2, 2)}, str(model_dir / "params.pkl"))
    for i in range(3):
        (vols_dir / f"eigen_pos{i:04d}.mrc").touch()

    calls = []

    def _fake_load_mrc(path):
        calls.append(path)
        idx = int(path.split("eigen_pos")[-1].split(".mrc")[0])
        return np.full((2, 2, 2), idx, dtype=np.float32)

    monkeypatch.setattr(output.utils, "load_mrc", _fake_load_mrc)

    po = output.PipelineOutput(str(result_path))
    u_real = po.get("u_real")
    assert u_real.shape == (3, 2, 2, 2)
    assert len(calls) == 3
    np.testing.assert_array_equal(u_real[0], np.zeros((2, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(u_real[2], np.full((2, 2, 2), 2, dtype=np.float32))


def test_pipeline_output_get_u_real_method_respects_requested_n_pcs(tmp_path, monkeypatch):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    vols_dir = result_path / "output" / "volumes"
    model_dir.mkdir(parents=True)
    vols_dir.mkdir(parents=True)

    utils.pickle_dump({"version": "0", "volume_shape": (2, 2, 2)}, str(model_dir / "params.pkl"))
    for i in range(4):
        (vols_dir / f"eigen_pos{i:04d}.mrc").touch()

    calls = []

    def _fake_load_mrc(path):
        calls.append(path)
        return np.ones((2, 2, 2), dtype=np.float32)

    monkeypatch.setattr(output.utils, "load_mrc", _fake_load_mrc)

    po = output.PipelineOutput(str(result_path))
    u_real = po.get_u_real(2)
    assert u_real.shape == (2, 2, 2, 2)
    assert len(calls) == 2


def test_pipeline_output_get_u_real_caps_to_50_saved_eigenvectors(tmp_path, monkeypatch):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    vols_dir = result_path / "output" / "volumes"
    model_dir.mkdir(parents=True)
    vols_dir.mkdir(parents=True)

    utils.pickle_dump({"version": "0", "volume_shape": (2, 2, 2)}, str(model_dir / "params.pkl"))
    for i in range(55):
        (vols_dir / f"eigen_pos{i:04d}.mrc").touch()

    calls = []
    monkeypatch.setattr(
        output.utils,
        "load_mrc",
        lambda path: calls.append(path) or np.zeros((2, 2, 2), dtype=np.float32),
    )

    po = output.PipelineOutput(str(result_path))
    u_real = po.get("u_real")
    assert u_real.shape == (50, 2, 2, 2)
    assert len(calls) == 50


def test_pipeline_output_get_u_real_raises_when_no_eigenvectors_saved(tmp_path):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    vols_dir = result_path / "output" / "volumes"
    model_dir.mkdir(parents=True)
    vols_dir.mkdir(parents=True)

    utils.pickle_dump({"version": "0", "volume_shape": (2, 2, 2)}, str(model_dir / "params.pkl"))

    po = output.PipelineOutput(str(result_path))
    with pytest.raises(ValueError, match="No eigenvector volumes found"):
        po.get("u_real")


def test_pipeline_output_get_u_real_handles_sparse_saved_indices(tmp_path, monkeypatch):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    vols_dir = result_path / "output" / "volumes"
    model_dir.mkdir(parents=True)
    vols_dir.mkdir(parents=True)

    utils.pickle_dump({"version": "0", "volume_shape": (2, 2, 2)}, str(model_dir / "params.pkl"))
    # Missing 0001 intentionally.
    (vols_dir / "eigen_pos0000.mrc").touch()
    (vols_dir / "eigen_pos0002.mrc").touch()

    calls = []

    def _fake_load_mrc(path):
        calls.append(path)
        idx = int(path.split("eigen_pos")[-1].split(".mrc")[0])
        return np.full((2, 2, 2), idx, dtype=np.float32)

    monkeypatch.setattr(output.utils, "load_mrc", _fake_load_mrc)

    po = output.PipelineOutput(str(result_path))
    u_real = po.get("u_real")
    assert u_real.shape == (2, 2, 2, 2)
    assert len(calls) == 2
    assert calls[0].endswith("eigen_pos0000.mrc")
    assert calls[1].endswith("eigen_pos0002.mrc")
    np.testing.assert_array_equal(u_real[0], np.zeros((2, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(u_real[1], np.full((2, 2, 2), 2, dtype=np.float32))


def test_pipeline_output_get_u_frequency_space_shape_for_sparse_indices(tmp_path, monkeypatch):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    vols_dir = result_path / "output" / "volumes"
    model_dir.mkdir(parents=True)
    vols_dir.mkdir(parents=True)

    utils.pickle_dump({"version": "0", "volume_shape": (2, 2, 2)}, str(model_dir / "params.pkl"))
    (vols_dir / "eigen_pos0001.mrc").touch()
    (vols_dir / "eigen_pos0003.mrc").touch()

    monkeypatch.setattr(output.utils, "load_mrc", lambda _path: np.ones((2, 2, 2), dtype=np.float32))
    seen_shapes = []

    def _fake_get_dft3(u):
        seen_shapes.append(np.asarray(u).shape)
        return np.asarray(u, dtype=np.float32).astype(np.complex64) * (1 + 1j)

    monkeypatch.setattr(output.fourier_transform_utils, "get_dft3", _fake_get_dft3)

    po = output.PipelineOutput(str(result_path))
    u = po.get("u")
    assert u.shape == (2, 8)
    assert np.iscomplexobj(u)
    # Ensure transform is executed per volume (streaming path), not on a stacked tensor.
    assert seen_shapes == [(2, 2, 2), (2, 2, 2)]


def test_pipeline_output_get_u_streaming_matches_expected_per_volume_content(tmp_path, monkeypatch):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    vols_dir = result_path / "output" / "volumes"
    model_dir.mkdir(parents=True)
    vols_dir.mkdir(parents=True)

    utils.pickle_dump({"version": "0", "volume_shape": (2, 2, 2)}, str(model_dir / "params.pkl"))
    (vols_dir / "eigen_pos0000.mrc").touch()
    (vols_dir / "eigen_pos0002.mrc").touch()

    def _fake_load_mrc(path):
        idx = int(path.split("eigen_pos")[-1].split(".mrc")[0])
        return np.full((2, 2, 2), idx, dtype=np.float32)

    monkeypatch.setattr(output.utils, "load_mrc", _fake_load_mrc)
    monkeypatch.setattr(
        output.fourier_transform_utils,
        "get_dft3",
        lambda u: np.asarray(u, dtype=np.float32).astype(np.complex64),
    )

    po = output.PipelineOutput(str(result_path))
    u = po.get("u")
    assert u.shape == (2, 8)
    np.testing.assert_array_equal(u[0], np.zeros(8, dtype=np.complex64))
    np.testing.assert_array_equal(u[1], np.full(8, 2 + 0j, dtype=np.complex64))


def test_pipeline_output_get_u_real_rejects_nonpositive_n_pcs(tmp_path):
    result_path = tmp_path / "pipeline_output"
    model_dir = result_path / "model"
    vols_dir = result_path / "output" / "volumes"
    model_dir.mkdir(parents=True)
    vols_dir.mkdir(parents=True)
    utils.pickle_dump({"version": "0", "volume_shape": (2, 2, 2)}, str(model_dir / "params.pkl"))
    (vols_dir / "eigen_pos0000.mrc").touch()
    po = output.PipelineOutput(str(result_path))
    with pytest.raises(ValueError, match="n_pcs must be positive"):
        po.get_u_real(0)
