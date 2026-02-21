import os
from types import SimpleNamespace

import numpy as np
import pytest
import matplotlib.pyplot as plt

from recovar.commands import analyze as analyze_cmd
from recovar.commands import compute_state as compute_state_cmd

pytestmark = pytest.mark.unit


def _fake_pipeline_output(payload):
    class FakePipelineOutput:
        def __init__(self, _path):
            self.params = {}

        def get(self, key):
            return payload[key]

    return FakePipelineOutput


def test_pick_pairs_returns_requested_number_and_valid_indices():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    pairs = analyze_cmd.pick_pairs(centers, n_pairs=4)
    assert len(pairs) == 4
    for i, j in pairs:
        assert 0 <= i < centers.shape[0]
        assert 0 <= j < centers.shape[0]
        assert i != j


def test_analyze_runs_centers_and_trajectories_with_density(monkeypatch, tmp_path):
    n_images = 6
    zs3 = np.arange(n_images * 3, dtype=np.float32).reshape(n_images, 3)
    cov3 = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], n_images, axis=0)
    payload = {
        "zs": {3: zs3},
        "cov_zs": {3: cov3},
        "dataset": ["cryo0"],
        "lazy_dataset": ["lazy_cryo0"],
        "contrasts": {3: np.ones(n_images, dtype=np.float32)},
        "particles_halfsets": np.zeros(n_images, dtype=np.int32),
        "noise_var_used": np.ones(8, dtype=np.float32),
        "volume_mask": np.ones((8, 8, 8), dtype=np.float32),
    }
    monkeypatch.setattr(analyze_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))
    monkeypatch.setattr(analyze_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analyze_cmd.dataset, "reorder_to_original_indexing_from_halfsets", lambda arr, _h: arr)
    monkeypatch.setattr(analyze_cmd.utils, "basic_config_logger", lambda *_args, **_kwargs: None)

    calls = {"reweighted": [], "traj": []}

    centers = np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    monkeypatch.setattr(analyze_cmd.o, "kmeans_analysis", lambda *_args, **_kwargs: (centers, labels))
    monkeypatch.setattr(analyze_cmd.o, "mkdir_safe", lambda path: os.makedirs(path, exist_ok=True))
    monkeypatch.setattr(analyze_cmd.o, "plot_over_density", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        analyze_cmd.o,
        "make_trajectory_plots_from_results",
        lambda *_args, **_kwargs: (
            np.zeros((5, 2), dtype=np.float32),
            np.linspace(0.0, 1.0, 4, dtype=np.float32)[:, None].repeat(2, axis=1),
        ),
    )

    monkeypatch.setattr(
        analyze_cmd.o,
        "compute_and_save_reweighted",
        lambda cryos, target_zs, zs, cov_zs, output_folder, *_args, **_kwargs: calls["reweighted"].append(
            (len(cryos), target_zs.shape, zs.shape, cov_zs.shape, output_folder)
        ),
    )
    monkeypatch.setattr(
        analyze_cmd.latent_density,
        "get_grid_z_mappings",
        lambda *_args, **_kwargs: (None, lambda x: x),
    )
    monkeypatch.setattr(
        analyze_cmd.utils,
        "pickle_load",
        lambda _p: {"density": np.ones((2, 2), dtype=np.float32), "latent_space_bounds": {"x": [-1, 1]}},
    )

    analyze_cmd.analyze(
        recovar_result_dir=str(tmp_path / "pipeline_out"),
        output_folder=str(tmp_path / "analysis"),
        zdim=3,
        n_clusters=3,
        n_paths=2,
        skip_umap=True,
        n_vols_along_path=4,
        skip_centers=False,
        normalize_kmeans=False,
        density_path=str(tmp_path / "density.pkl"),
        no_z_reg=False,
        lazy=False,
        n_min_particles=1,
        maskrad_fraction=0.5,
        apply_global_filtering=True,
        fsc_mask_radius=2.0,
        fsc_mask_edgewidth=1.0,
        args=SimpleNamespace(copy_to_folder=None, no_cleanup=False),
    )

    # One for kmeans centers + one per selected trajectory pair (n_paths=2 -> 2 pairs).
    assert len(calls["reweighted"]) == 3
    # Density input is 2D, so zs/cov_zs were truncated from 3D to 2D.
    assert calls["reweighted"][0][2] == (n_images, 2)
    assert calls["reweighted"][0][3] == (n_images, 2, 2)


def test_compute_state_reads_txt_and_reweights(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "zs": {2: np.zeros((5, 2), dtype=np.float32)},
        "cov_zs": {2: np.zeros((5, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(5, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda cryos, target_zs, zs, cov_zs, output_folder, *_args, **kwargs: captured.setdefault(
            "call", (cryos, target_zs, zs, cov_zs, output_folder, kwargs)
        ),
    )
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda out, n: captured.setdefault("move", (out, n)))

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=True,
        apply_global_filtering=True,
        fsc_mask_radius=2.0,
        fsc_mask_edgewidth=1.0,
    )

    compute_state_cmd.compute_state(args)

    cryos, target_zs, zs, cov_zs, output_folder, kwargs = captured["call"]
    assert cryos == ["d0"]
    assert target_zs.shape == (2, 2)
    assert zs.shape == (5, 2)
    assert cov_zs.shape == (5, 2, 2)
    assert output_folder.endswith("/state_out/")
    assert kwargs["apply_global_filtering"] is True
    assert kwargs["fsc_mask"] is not None
    assert captured["move"][1] == 2


def test_compute_state_reads_pkl_latent_points(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.pkl"
    import pickle
    with open(latent_path, "wb") as f:
        pickle.dump(latent_points, f)

    payload = {
        "zs": {2: np.zeros((5, 2), dtype=np.float32)},
        "cov_zs": {2: np.zeros((5, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(5, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda _cryos, target_zs, *_args, **_kwargs: captured.setdefault("target_zs", target_zs.copy()),
    )
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    compute_state_cmd.compute_state(args)
    np.testing.assert_array_equal(captured["target_zs"], latent_points)


def test_compute_state_uses_noreg_key_when_requested(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "zs": {2: np.zeros((5, 2), dtype=np.float32), "2_noreg": np.ones((5, 2), dtype=np.float32)},
        "cov_zs": {2: np.zeros((5, 2, 2), dtype=np.float32), "2_noreg": np.ones((5, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(5, dtype=np.float32), "2_noreg": np.full(5, 2.0, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda _cryos, _target_zs, zs, cov_zs, *_args, **_kwargs: captured.setdefault(
            "vals", (zs.copy(), cov_zs.copy())
        ),
    )
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=True,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    compute_state_cmd.compute_state(args)
    zs_used, cov_used = captured["vals"]
    np.testing.assert_array_equal(zs_used, payload["zs"]["2_noreg"])
    np.testing.assert_array_equal(cov_used, payload["cov_zs"]["2_noreg"])


def test_compute_state_rejects_unknown_latent_extension(monkeypatch, tmp_path):
    payload = {
        "zs": {1: np.zeros((4, 1), dtype=np.float32)},
        "cov_zs": {1: np.zeros((4, 1, 1), dtype=np.float32)},
        "contrasts": {1: np.ones(4, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(tmp_path / "latent.csv"),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=True,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )

    with pytest.raises(ValueError, match="Target zs should be a .txt or .pkl file"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_missing_zdim_with_clear_error(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)  # zdim=2
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "zs": {1: np.zeros((4, 1), dtype=np.float32)},
        "cov_zs": {1: np.zeros((4, 1, 1), dtype=np.float32)},
        "contrasts": {1: np.ones(4, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "input_args": SimpleNamespace(particles="p", datadir=None, strip_prefix=None),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=True,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    with pytest.raises(ValueError, match="zdim 2 .* not found"):
        compute_state_cmd.compute_state(args)


def test_compute_state_missing_input_args_ignores_overrides(monkeypatch, tmp_path):
    latent_points = np.array([[0.0], [1.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO:
        def __init__(self, _path):
            self.params = {}  # missing input_args
            self._payload = {
                "zs": {1: np.zeros((3, 1), dtype=np.float32)},
                "cov_zs": {1: np.zeros((3, 1, 1), dtype=np.float32)},
                "contrasts": {1: np.ones(3, dtype=np.float32)},
                "dataset": ["d0"],
                "lazy_dataset": ["ld0"],
                "noise_var_used": np.ones(4, dtype=np.float32),
                "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
            }

        def get(self, key):
            return self._payload[key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "compute_and_save_reweighted", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles="override_particles",
        datadir="override_datadir",
        strip_prefix="override_prefix",
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=True,
        no_z_regularization=False,
        lazy=True,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    # Should run without KeyError even though params['input_args'] is missing.
    compute_state_cmd.compute_state(args)


def test_compute_state_apply_global_filtering_without_volume_mask(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO:
        def __init__(self, _path):
            self.params = {}
            self._payload = {
                "zs": {2: np.zeros((3, 2), dtype=np.float32)},
                "cov_zs": {2: np.zeros((3, 2, 2), dtype=np.float32)},
                "contrasts": {2: np.ones(3, dtype=np.float32)},
                "dataset": ["d0"],
                "lazy_dataset": ["ld0"],
                "noise_var_used": np.ones(4, dtype=np.float32),
            }

        def get(self, key):
            if key == "volume_mask":
                raise KeyError("missing")
            return self._payload[key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)
    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda *_args, **kwargs: captured.setdefault("kwargs", kwargs),
    )
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=True,
        fsc_mask_radius=2.0,
        fsc_mask_edgewidth=1.0,
    )
    compute_state_cmd.compute_state(args)
    assert captured["kwargs"]["apply_global_filtering"] is True
    assert captured["kwargs"]["fsc_mask"] is None


def test_compute_state_main_dispatches(monkeypatch, tmp_path):
    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        outdir=str(tmp_path / "state_out"),
        latent_points=str(tmp_path / "latent.txt"),
    )
    fake_parser = SimpleNamespace(parse_args=lambda: args)
    monkeypatch.setattr(compute_state_cmd, "add_args", lambda _parser: fake_parser)

    called = {}
    monkeypatch.setattr(compute_state_cmd, "compute_state", lambda a: called.setdefault("args", a))

    compute_state_cmd.main()
    assert called["args"] is args


def test_compute_state_updates_input_args_from_cli_overrides(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    base_input_args = SimpleNamespace(
        particles="old_particles",
        datadir="old_datadir",
        strip_prefix="old_prefix",
    )
    payload = {
        "zs": {2: np.zeros((4, 2), dtype=np.float32)},
        "cov_zs": {2: np.zeros((4, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(4, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
        "input_args": base_input_args,
    }
    class _PO:
        def __init__(self, _path):
            self.params = {"input_args": base_input_args}

        def get(self, key):
            return payload[key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "compute_and_save_reweighted", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles="new_particles",
        datadir="new_datadir",
        strip_prefix="new_prefix",
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    compute_state_cmd.compute_state(args)

    assert base_input_args.particles == "new_particles"
    assert base_input_args.datadir == "new_datadir"
    assert base_input_args.strip_prefix == "new_prefix"


def test_compute_state_copy_to_folder_triggers_cleanup_unless_no_cleanup(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "zs": {2: np.zeros((3, 2), dtype=np.float32)},
        "cov_zs": {2: np.zeros((3, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(3, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
        "input_args": SimpleNamespace(particles="p", datadir=None, strip_prefix=None),
    }
    class _PO:
        def __init__(self, _path):
            self.params = {"input_args": payload["input_args"]}

        def get(self, key):
            return payload[key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "compute_and_save_reweighted", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd, "copy_data_from_pipeline_output", lambda *_args, **_kwargs: {"temp_folder": "/tmp/fake"})
    cleaned = {"count": 0}
    monkeypatch.setattr(compute_state_cmd, "cleanup_temp_files", lambda _m: cleaned.__setitem__("count", cleaned["count"] + 1))

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=str(tmp_path / "copytmp"),
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=True,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    compute_state_cmd.compute_state(args)
    assert cleaned["count"] == 1

    args.no_cleanup = True
    compute_state_cmd.compute_state(args)
    assert cleaned["count"] == 1


def test_compute_state_zdim1_handles_scalar_txt_latent_point(monkeypatch, tmp_path):
    latent_path = tmp_path / "latent_scalar.txt"
    latent_path.write_text("0.75\n")

    payload = {
        "zs": {1: np.zeros((3, 1), dtype=np.float32)},
        "cov_zs": {1: np.zeros((3, 1, 1), dtype=np.float32)},
        "contrasts": {1: np.ones(3, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda cryos, target_zs, *_args, **_kwargs: captured.setdefault("shape", target_zs.shape),
    )
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=True,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    compute_state_cmd.compute_state(args)
    assert captured["shape"] == (1, 1)


def test_compute_state_rejects_missing_contrasts_or_cov_zs_key(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "zs": {2: np.zeros((3, 2), dtype=np.float32)},
        "cov_zs": {2: np.zeros((3, 2, 2), dtype=np.float32)},
        # Missing 2_noreg key on purpose.
        "contrasts": {2: np.ones(3, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=True,  # requests 2_noreg
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    with pytest.raises(ValueError, match="Requested embedding key 2_noreg is missing"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_missing_cov_zs_key(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "zs": {2: np.zeros((3, 2), dtype=np.float32)},
        # Missing cov_zs[2] key.
        "cov_zs": {},
        "contrasts": {2: np.ones(3, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    with pytest.raises(ValueError, match="Requested embedding key 2 is missing"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_scalar_without_zdim1(monkeypatch, tmp_path):
    latent_path = tmp_path / "latent_scalar.txt"
    latent_path.write_text("0.75\n")

    payload = {
        "zs": {1: np.zeros((3, 1), dtype=np.float32)},
        "cov_zs": {1: np.zeros((3, 1, 1), dtype=np.float32)},
        "contrasts": {1: np.ones(3, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    with pytest.raises(ValueError, match="Scalar latent point requires --zdim1"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_bad_shape_when_zdim1_true(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "zs": {1: np.zeros((3, 1), dtype=np.float32)},
        "cov_zs": {1: np.zeros((3, 1, 1), dtype=np.float32)},
        "contrasts": {1: np.ones(3, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=True,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    with pytest.raises(ValueError, match="--zdim1 expects scalar/1D latent points or Nx1 arrays"):
        compute_state_cmd.compute_state(args)


def test_analyze_main_dispatches(monkeypatch, tmp_path):
    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        outdir=str(tmp_path / "analysis"),
        zdim=4,
        n_clusters=3,
        n_trajectories=1,
        skip_umap=True,
        Bfactor=0.0,
        n_bins=20,
        n_vols_along_path=4,
        skip_centers=False,
        normalize_kmeans=False,
        density=None,
        no_z_regularization=False,
        lazy=False,
        n_min_particles=1,
        maskrad_fraction=0.5,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    fake_parser = SimpleNamespace(parse_args=lambda: args)
    monkeypatch.setattr(analyze_cmd, "add_args", lambda _parser: fake_parser)

    called = {}
    monkeypatch.setattr(analyze_cmd, "analyze", lambda *a, **k: called.setdefault("call", (a, k)))

    analyze_cmd.main()
    call_args, kwargs = called["call"]
    assert call_args[0] == args.result_dir
    assert kwargs["output_folder"] == args.outdir
    assert kwargs["zdim"] == 4


def test_compute_state_add_args_requires_outdir_and_latent_points():
    import argparse

    parser = compute_state_cmd.add_args(argparse.ArgumentParser())
    parsed = parser.parse_args(
        [
            "/tmp/result_dir",
            "-o",
            "/tmp/outdir",
            "--latent-points",
            "/tmp/latent.txt",
        ]
    )
    assert parsed.result_dir == "/tmp/result_dir"
    assert parsed.outdir == "/tmp/outdir"
    assert parsed.latent_points == "/tmp/latent.txt"

    with pytest.raises(SystemExit):
        parser.parse_args(["/tmp/result_dir", "--latent-points", "/tmp/latent.txt"])


def test_compute_state_1d_latent_warns_and_reshapes_to_single_point(monkeypatch, tmp_path, caplog):
    latent_points = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # 1D: interpreted as one zdim=3 point
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "zs": {3: np.zeros((5, 3), dtype=np.float32)},
        "cov_zs": {3: np.zeros((5, 3, 3), dtype=np.float32)},
        "contrasts": {3: np.ones(5, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }
    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda _cryos, target_zs, *_args, **_kwargs: captured.setdefault("target_zs", target_zs.copy()),
    )
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )
    with caplog.at_level("WARNING"):
        compute_state_cmd.compute_state(args)
    assert "Did you mean to use --zdim1?" in caplog.text
    np.testing.assert_array_equal(captured["target_zs"], latent_points[None, :])


def test_compute_state_uses_lazy_dataset_when_requested(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO:
        def __init__(self, _path):
            self.params = {
                "input_args": SimpleNamespace(particles="p", datadir=None, strip_prefix=None),
            }
            self._payload = {
                "zs": {2: np.zeros((4, 2), dtype=np.float32)},
                "cov_zs": {2: np.zeros((4, 2, 2), dtype=np.float32)},
                "contrasts": {2: np.ones(4, dtype=np.float32)},
                "dataset": ["dataset_obj"],
                "lazy_dataset": ["lazy_dataset_obj"],
                "noise_var_used": np.ones(4, dtype=np.float32),
            }

        def get(self, key):
            if key == "volume_mask":
                raise KeyError("volume_mask not used here")
            return self._payload[key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda cryos, *_args, **_kwargs: captured.setdefault("cryos", cryos),
    )
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=True,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )

    compute_state_cmd.compute_state(args)
    assert captured["cryos"] == ["lazy_dataset_obj"]


def test_compute_state_uses_nonlazy_dataset_when_lazy_false(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO:
        def __init__(self, _path):
            self.params = {
                "input_args": SimpleNamespace(particles="p", datadir=None, strip_prefix=None),
            }
            self._payload = {
                "zs": {2: np.zeros((4, 2), dtype=np.float32)},
                "cov_zs": {2: np.zeros((4, 2, 2), dtype=np.float32)},
                "contrasts": {2: np.ones(4, dtype=np.float32)},
                "dataset": ["dataset_obj"],
                "lazy_dataset": ["lazy_dataset_obj"],
                "noise_var_used": np.ones(4, dtype=np.float32),
            }

        def get(self, key):
            if key == "volume_mask":
                raise KeyError("volume_mask not used here")
            return self._payload[key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda cryos, *_args, **_kwargs: captured.setdefault("cryos", cryos),
    )
    monkeypatch.setattr(compute_state_cmd.o, "move_to_one_folder", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        copy_to_folder=None,
        no_cleanup=False,
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=False,
        lazy=False,
        n_bins=20,
        Bfactor=0.0,
        maskrad_fraction=0.5,
        n_min_particles=1,
        save_all_estimates=False,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
    )

    compute_state_cmd.compute_state(args)
    assert captured["cryos"] == ["dataset_obj"]
