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

        def get_embedding_keys(self, entry):
            return list(payload[entry].keys())

        def get_embedding_component(self, entry, key):
            return payload[entry][key]

    return FakePipelineOutput


class _PayloadEmbeddingAccessMixin:
    def get_embedding_keys(self, entry):
        return list(self._payload[entry].keys())

    def get_embedding_component(self, entry, key):
        return self._payload[entry][key]


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


def _reference_pick_pairs(centers, n_pairs):
    pairs = []
    xmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)

    for _ in range(n_pairs // 2):
        i_idx, j_idx = np.unravel_index(np.argmax(xmat), xmat.shape)
        xmat[i_idx, :] = 0
        xmat[:, i_idx] = 0
        xmat[j_idx, :] = 0
        xmat[:, j_idx] = 0
        pairs.append([int(i_idx), int(j_idx)])

    zdim = centers.shape[-1]
    max_k = np.min([(n_pairs - n_pairs // 2), zdim])
    for k in range(max_k):
        i_idx = np.argmax(centers[:, k])
        j_idx = np.argmin(centers[:, k])
        pairs.append([int(i_idx), int(j_idx)])
    return pairs


@pytest.mark.parametrize(
    "shape,n_pairs",
    [
        ((8, 2), 4),
        ((11, 3), 7),
        ((16, 5), 10),
    ],
)
def test_pick_pairs_matches_reference_selection(shape, n_pairs):
    rng = np.random.default_rng(123)
    centers = rng.standard_normal(shape).astype(np.float32)

    expected = _reference_pick_pairs(centers.copy(), n_pairs=n_pairs)
    got = analyze_cmd.pick_pairs(centers.copy(), n_pairs=n_pairs)
    assert got == expected


def test_analyze_reads_particles_halfsets_once(monkeypatch, tmp_path):
    n_images = 8
    zs2 = np.arange(n_images * 2, dtype=np.float32).reshape(n_images, 2)
    cov2 = np.repeat(np.eye(2, dtype=np.float32)[None, :, :], n_images, axis=0)

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {}
            self._halfset_calls = 0
            self._payload = {
                "latent_coords": {2: zs2},
                "latent_precision": {2: cov2},
                "dataset": ["cryo0"],
                "contrasts": {2: np.ones(n_images, dtype=np.float32)},
                "particles_halfsets": [np.array([0, 2, 4, 6], dtype=np.int32), np.array([1, 3, 5, 7], dtype=np.int32)],
            }

        def get(self, key):
            if key == "particles_halfsets":
                self._halfset_calls += 1
            return self._payload[key]

    po_instance = None

    def _make_po(path):
        nonlocal po_instance
        po_instance = _PO(path)
        return po_instance

    monkeypatch.setattr(analyze_cmd.o, "PipelineOutput", _make_po)
    monkeypatch.setattr(analyze_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analyze_cmd.utils, "basic_config_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        analyze_cmd.latent_density,
        "compute_latent_space_density",
        lambda *_args, **_kwargs: (np.ones((4, 4), dtype=np.float32), {"x": [-1, 1]}),
    )
    monkeypatch.setattr(
        analyze_cmd.o,
        "kmeans_analysis",
        lambda *_args, **_kwargs: (
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(analyze_cmd.o, "mkdir_safe", lambda path: os.makedirs(path, exist_ok=True))
    monkeypatch.setattr(analyze_cmd.o, "compute_and_save_reweighted", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        analyze_cmd.o,
        "umap_latent_space",
        lambda *_args, **_kwargs: SimpleNamespace(embedding_=np.zeros((n_images, 2), dtype=np.float32)),
    )
    monkeypatch.setattr(analyze_cmd.o, "plot_umap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    analyze_cmd.analyze(
        recovar_result_dir=str(tmp_path / "pipeline_out"),
        output_folder=str(tmp_path / "analysis"),
        zdim=2,
        n_clusters=2,
        n_paths=0,
        skip_umap=False,
        skip_centers=True,
        density_path=None,
        no_z_reg=False,
        lazy=False,
        args=SimpleNamespace(),
    )

    assert po_instance is not None
    assert po_instance._halfset_calls == 1


def test_analyze_runs_centers_and_trajectories_with_density(monkeypatch, tmp_path):
    n_images = 6
    zs3 = np.arange(n_images * 3, dtype=np.float32).reshape(n_images, 3)
    cov3 = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], n_images, axis=0)
    payload = {
        "latent_coords": {3: zs3},
        "latent_precision": {3: cov3},
        "dataset": ["cryo0"],
        "lazy_dataset": ["lazy_cryo0"],
        "contrasts": {3: np.ones(n_images, dtype=np.float32)},
        "particles_halfsets": [np.array([0, 2, 4], dtype=np.int32), np.array([1, 3, 5], dtype=np.int32)],
        "noise_var_used": np.ones(8, dtype=np.float32),
        "volume_mask": np.ones((8, 8, 8), dtype=np.float32),
    }
    monkeypatch.setattr(analyze_cmd.o, "PipelineOutput", _fake_pipeline_output(payload))
    monkeypatch.setattr(analyze_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
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
        args=SimpleNamespace(),
    )

    # One for kmeans centers + one per selected trajectory pair (n_paths=2 -> 2 pairs).
    assert len(calls["reweighted"]) == 3
    # Density input is 2D, so zs/cov_zs were truncated from 3D to 2D.
    assert calls["reweighted"][0][2] == (n_images, 2)
    assert calls["reweighted"][0][3] == (n_images, 2, 2)


def test_analyze_uses_embedding_component_api_when_available(monkeypatch, tmp_path):
    component_calls = []
    captured = {}

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {}

        def get_embedding_keys(self, _entry):
            return [2]

        def get_embedding_component(self, entry, key):
            assert key == 2
            component_calls.append(entry)
            if entry == "latent_coords":
                return np.zeros((6, 2), dtype=np.float64)
            if entry == "latent_precision":
                return np.repeat(np.eye(2, dtype=np.float64)[None, :, :], 6, axis=0)
            if entry == "contrasts":
                return np.ones(6, dtype=np.float64)
            raise KeyError(entry)

        def get(self, key):
            if key in {"latent_coords", "latent_precision", "contrasts"}:
                raise AssertionError(f"analyze should not call get('{key}') when component API exists")
            if key == "dataset":
                return ["d0"]
            if key == "particles_halfsets":
                return [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4, 5], dtype=np.int32)]
            if key == "volume_mask":
                return np.ones((4, 4, 4), dtype=np.float32)
            raise KeyError(key)

    monkeypatch.setattr(analyze_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(
        analyze_cmd.embedding,
        "set_contrasts_in_cryos",
        lambda _cryos, contrasts: captured.setdefault("contrast_dtype", np.asarray(contrasts).dtype),
    )
    monkeypatch.setattr(analyze_cmd.utils, "basic_config_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        analyze_cmd.latent_density,
        "compute_latent_space_density",
        lambda *_args, **_kwargs: (np.ones((4, 4), dtype=np.float32), {"x": [-1, 1]}),
    )
    monkeypatch.setattr(
        analyze_cmd.o,
        "kmeans_analysis",
        lambda *_args, **_kwargs: (
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(analyze_cmd.o, "mkdir_safe", lambda path: os.makedirs(path, exist_ok=True))
    monkeypatch.setattr(
        analyze_cmd.o,
        "compute_and_save_reweighted",
        lambda *_args, **_kwargs: captured.__setitem__("reweighted_calls", captured.get("reweighted_calls", 0) + 1),
    )
    monkeypatch.setattr(plt, "savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    analyze_cmd.analyze(
        recovar_result_dir=str(tmp_path / "pipeline_out"),
        output_folder=str(tmp_path / "analysis"),
        zdim=2,
        n_clusters=2,
        n_paths=0,
        skip_umap=True,
        skip_centers=False,
        density_path=None,
        no_z_reg=False,
        lazy=False,
        args=SimpleNamespace(),
    )

    assert component_calls.count("latent_coords") == 1
    assert component_calls.count("latent_precision") == 1
    assert component_calls.count("contrasts") == 1
    assert captured["contrast_dtype"] == np.float32
    assert captured["reweighted_calls"] == 1


def test_compute_state_reads_txt_and_reweights(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "latent_coords": {2: np.zeros((5, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((5, 2, 2), dtype=np.float32)},
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
    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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
    assert output_folder.rstrip("/").endswith("/state_out")
    assert kwargs["apply_global_filtering"] is True
    assert kwargs["fsc_mask"] is not None


def test_compute_state_accepts_pathlike_latent_points(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "latent_coords": {2: np.zeros((5, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((5, 2, 2), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
        latent_points=latent_path,  # Path object, not string.
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


def test_compute_state_accepts_pathlike_result_and_out_dirs(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "latent_coords": {2: np.zeros((5, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((5, 2, 2), dtype=np.float32)},
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
        lambda _cryos, _target_zs, _zs, _cov_zs, output_folder, *_args, **_kwargs: captured.setdefault(
            "output_folder", output_folder
        ),
    )

    args = SimpleNamespace(
        result_dir=tmp_path / "pipeline_out",  # Path object
        particles=None,
        datadir=None,
        strip_prefix=None,
        latent_points=latent_path,
        outdir=tmp_path / "state_out",  # Path object
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
    assert captured["output_folder"].rstrip("/").endswith("state_out")


def test_compute_state_reads_pkl_latent_points(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.pkl"
    import pickle

    with open(latent_path, "wb") as f:
        pickle.dump(latent_points, f)

    payload = {
        "latent_coords": {2: np.zeros((5, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((5, 2, 2), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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
        "latent_coords": {2: np.zeros((5, 2), dtype=np.float32)},
        "latent_coords_noreg": {2: np.ones((5, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((5, 2, 2), dtype=np.float32)},
        "latent_precision_noreg": {2: np.ones((5, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(5, dtype=np.float32)},
        "contrasts_noreg": {2: np.full(5, 2.0, dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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
    np.testing.assert_array_equal(zs_used, payload["latent_coords_noreg"][2])
    np.testing.assert_array_equal(cov_used, payload["latent_precision_noreg"][2])


def test_compute_state_uses_embedding_component_api_when_available(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {}
            self._forbidden_gets = {"latent_coords", "latent_precision", "contrasts"}
            self._payload = {
                "dataset": ["d0"],
                "lazy_dataset": ["ld0"],
                "noise_var_used": np.ones(4, dtype=np.float32),
                "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
            }
            self._emb = {
                "latent_coords": {2: np.full((5, 2), 7.0, dtype=np.float32)},
                "latent_precision": {2: np.full((5, 2, 2), 3.0, dtype=np.float32)},
                "contrasts": {2: np.full(5, 2.0, dtype=np.float32)},
            }

        def get(self, key):
            if key in self._forbidden_gets:
                raise AssertionError(f"compute_state should not call get('{key}') when key-level API exists")
            return self._payload[key]

        def get_embedding_keys(self, entry):
            return list(self._emb[entry].keys())

        def get_embedding_component(self, entry, key):
            return self._emb[entry][key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.embedding,
        "set_contrasts_in_cryos",
        lambda _cryos, contrasts: captured.setdefault("contrasts", contrasts.copy()),
    )
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda _cryos, _target_zs, zs, cov_zs, *_args, **_kwargs: captured.setdefault(
            "vals", (zs.copy(), cov_zs.copy())
        ),
    )

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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
    zs_used, cov_used = captured["vals"]
    np.testing.assert_array_equal(captured["contrasts"], np.full(5, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(zs_used, np.full((5, 2), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(cov_used, np.full((5, 2, 2), 3.0, dtype=np.float32))


def test_compute_state_casts_embedding_arrays_to_float32(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {}
            self._payload = {
                "dataset": ["d0"],
                "lazy_dataset": ["ld0"],
                "noise_var_used": np.ones(4, dtype=np.float32),
                "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
            }
            self._emb = {
                "latent_coords": {2: np.full((5, 2), 7.0, dtype=np.float64)},
                "latent_precision": {2: np.full((5, 2, 2), 3.0, dtype=np.float64)},
                "contrasts": {2: np.full(5, 2.0, dtype=np.float64)},
            }

        def get(self, key):
            return self._payload[key]

        def get_embedding_keys(self, entry):
            return list(self._emb[entry].keys())

        def get_embedding_component(self, entry, key):
            return self._emb[entry][key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.embedding,
        "set_contrasts_in_cryos",
        lambda _cryos, contrasts: captured.setdefault("contrast_dtype", contrasts.dtype),
    )
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda _cryos, _target_zs, zs, cov_zs, *_args, **_kwargs: captured.setdefault(
            "dtypes", (zs.dtype, cov_zs.dtype)
        ),
    )

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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
    assert captured["contrast_dtype"] == np.float32
    assert captured["dtypes"] == (np.float32, np.float32)


def test_compute_state_rejects_unknown_latent_extension(monkeypatch, tmp_path):
    payload = {
        "latent_coords": {1: np.zeros((4, 1), dtype=np.float32)},
        "latent_precision": {1: np.zeros((4, 1, 1), dtype=np.float32)},
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


def test_compute_state_rejects_empty_latent_points_file(monkeypatch, tmp_path):
    latent_path = tmp_path / "latent.txt"
    latent_path.write_text("")

    payload = {
        "latent_coords": {1: np.zeros((4, 1), dtype=np.float32)},
        "latent_precision": {1: np.zeros((4, 1, 1), dtype=np.float32)},
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
    with pytest.raises(ValueError, match="Target zs file is empty"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_nonfinite_latent_points(monkeypatch, tmp_path):
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, np.array([[0.0, np.nan]], dtype=np.float32))

    payload = {
        "latent_coords": {2: np.zeros((4, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(4, dtype=np.float32)},
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
    with pytest.raises(ValueError, match="non-finite values"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_nonnumeric_latent_points(monkeypatch, tmp_path):
    latent_path = tmp_path / "latent.pkl"
    import pickle

    with open(latent_path, "wb") as f:
        pickle.dump([["a", "b"]], f)

    payload = {
        "latent_coords": {2: np.zeros((4, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(4, dtype=np.float32)},
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
    with pytest.raises(ValueError, match="Target zs must be numeric"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_missing_latent_points_file(monkeypatch, tmp_path):
    payload = {
        "latent_coords": {1: np.zeros((4, 1), dtype=np.float32)},
        "latent_precision": {1: np.zeros((4, 1, 1), dtype=np.float32)},
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
        latent_points=str(tmp_path / "does_not_exist.txt"),
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
    with pytest.raises(FileNotFoundError, match="Latent points file not found"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_missing_zdim_with_clear_error(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)  # zdim=2
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "latent_coords": {1: np.zeros((4, 1), dtype=np.float32)},
        "latent_precision": {1: np.zeros((4, 1, 1), dtype=np.float32)},
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

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {}  # missing input_args
            self._payload = {
                "latent_coords": {1: np.zeros((3, 1), dtype=np.float32)},
                "latent_precision": {1: np.zeros((3, 1, 1), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles="override_particles",
        datadir="override_datadir",
        strip_prefix="override_prefix",
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


def test_compute_state_params_none_ignores_overrides(monkeypatch, tmp_path):
    latent_points = np.array([[0.0], [1.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = None
            self._payload = {
                "latent_coords": {1: np.zeros((3, 1), dtype=np.float32)},
                "latent_precision": {1: np.zeros((3, 1, 1), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles="override_particles",
        datadir="override_datadir",
        strip_prefix="override_prefix",
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

    compute_state_cmd.compute_state(args)


def test_compute_state_allows_missing_override_attrs_on_args(monkeypatch, tmp_path):
    latent_points = np.array([[0.0], [1.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = None
            self._payload = {
                "latent_coords": {1: np.zeros((3, 1), dtype=np.float32)},
                "latent_precision": {1: np.zeros((3, 1, 1), dtype=np.float32)},
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

    # Intentionally omit particles/datadir/strip_prefix attrs.
    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
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
    compute_state_cmd.compute_state(args)


def test_compute_state_uses_defaults_when_optional_attrs_missing(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "latent_coords": {2: np.zeros((3, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((3, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(3, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
    }

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = None
            self._payload = payload

        def get(self, key):
            return self._payload[key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)

    captured = {}
    monkeypatch.setattr(
        compute_state_cmd.o,
        "compute_and_save_reweighted",
        lambda _cryos, _target_zs, _zs, _cov_zs, _out, _bf, **kwargs: captured.setdefault("kwargs", kwargs),
    )

    # Intentionally omit many optional attributes; code should use parser defaults.
    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
    )
    compute_state_cmd.compute_state(args)

    assert captured["kwargs"]["n_bins"] == 50
    assert captured["kwargs"]["maskrad_fraction"] == 20
    assert captured["kwargs"]["n_min_particles"] is None
    assert captured["kwargs"]["save_all_estimates"] is False
    assert captured["kwargs"]["apply_global_filtering"] is False
    assert captured["kwargs"]["fsc_mask_radius"] is None
    assert captured["kwargs"]["fsc_mask_edgewidth"] is None


def test_compute_state_apply_global_filtering_without_volume_mask(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {}
            self._payload = {
                "latent_coords": {2: np.zeros((3, 2), dtype=np.float32)},
                "latent_precision": {2: np.zeros((3, 2, 2), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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
        "latent_coords": {2: np.zeros((4, 2), dtype=np.float32)},
        "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(4, dtype=np.float32)},
        "dataset": ["d0"],
        "lazy_dataset": ["ld0"],
        "noise_var_used": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones((4, 4, 4), dtype=np.float32),
        "input_args": base_input_args,
    }

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {"input_args": base_input_args}
            self._payload = payload

        def get(self, key):
            return payload[key]

    monkeypatch.setattr(compute_state_cmd.o, "PipelineOutput", _PO)
    monkeypatch.setattr(compute_state_cmd.embedding, "set_contrasts_in_cryos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "mkdir_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute_state_cmd.o, "compute_and_save_reweighted", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles="new_particles",
        datadir="new_datadir",
        strip_prefix="new_prefix",
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


def test_compute_state_zdim1_handles_scalar_txt_latent_point(monkeypatch, tmp_path):
    latent_path = tmp_path / "latent_scalar.txt"
    latent_path.write_text("0.75\n")

    payload = {
        "latent_coords": {1: np.zeros((3, 1), dtype=np.float32)},
        "latent_precision": {1: np.zeros((3, 1, 1), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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
        "latent_coords": {2: np.zeros((3, 2), dtype=np.float32)},
        "latent_coords_noreg": {},  # Empty: zdim=2 not present in noreg.
        "latent_precision": {2: np.zeros((3, 2, 2), dtype=np.float32)},
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
        latent_points=str(latent_path),
        outdir=str(tmp_path / "state_out"),
        zdim1=False,
        no_z_regularization=True,  # requests noreg entry
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
    with pytest.raises(ValueError, match="zdim 2 from provided latent points is not found"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_missing_cov_zs_key(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "latent_coords": {2: np.zeros((3, 2), dtype=np.float32)},
        # Missing cov_zs[2] key.
        "latent_precision": {},
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
    with pytest.raises(KeyError):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_missing_zs_key(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        # Missing latent_coords_noreg[2] while base embedding exists.
        "latent_coords": {2: np.zeros((3, 2), dtype=np.float32)},
        "latent_coords_noreg": {},  # Empty: zdim=2 not present.
        "latent_precision": {2: np.zeros((3, 2, 2), dtype=np.float32)},
        "latent_precision_noreg": {2: np.zeros((3, 2, 2), dtype=np.float32)},
        "contrasts": {2: np.ones(3, dtype=np.float32)},
        "contrasts_noreg": {2: np.ones(3, dtype=np.float32)},
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
    with pytest.raises(ValueError, match="zdim 2 from provided latent points is not found"):
        compute_state_cmd.compute_state(args)


def test_compute_state_rejects_scalar_without_zdim1(monkeypatch, tmp_path):
    latent_path = tmp_path / "latent_scalar.txt"
    latent_path.write_text("0.75\n")

    payload = {
        "latent_coords": {1: np.zeros((3, 1), dtype=np.float32)},
        "latent_precision": {1: np.zeros((3, 1, 1), dtype=np.float32)},
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
        "latent_coords": {1: np.zeros((3, 1), dtype=np.float32)},
        "latent_precision": {1: np.zeros((3, 1, 1), dtype=np.float32)},
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


def test_compute_state_add_args_requires_latent_points():
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

    # --outdir is optional (auto-numbering fills in the default)
    parsed2 = parser.parse_args(["/tmp/result_dir", "--latent-points", "/tmp/latent.txt"])
    assert parsed2.outdir is None

    # --latent-points is still required
    with pytest.raises(SystemExit):
        parser.parse_args(["/tmp/result_dir"])


def test_compute_state_1d_latent_warns_and_reshapes_to_single_point(monkeypatch, tmp_path, caplog):
    latent_points = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # 1D: interpreted as one zdim=3 point
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    payload = {
        "latent_coords": {3: np.zeros((5, 3), dtype=np.float32)},
        "latent_precision": {3: np.zeros((5, 3, 3), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {
                "input_args": SimpleNamespace(particles="p", datadir=None, strip_prefix=None),
            }
            self._payload = {
                "latent_coords": {2: np.zeros((4, 2), dtype=np.float32)},
                "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {
                "input_args": SimpleNamespace(particles="p", datadir=None, strip_prefix=None),
            }
            self._payload = {
                "latent_coords": {2: np.zeros((4, 2), dtype=np.float32)},
                "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
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

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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


def test_compute_state_builds_explicit_halfsets_for_reweighting(monkeypatch, tmp_path):
    latent_points = np.array([[0.0, 1.0]], dtype=np.float32)
    latent_path = tmp_path / "latent.txt"
    np.savetxt(latent_path, latent_points)

    class _Dataset:
        def __init__(self):
            self.halfset_indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
            self.materialize_calls = []

        def can_reload_from_original_images(self):
            return True

        def materialize_halfset_datasets(self, *, independent=None, lazy=None):
            self.materialize_calls.append((independent, lazy))
            return ("half0", "half1")

    dataset_obj = _Dataset()

    class _PO(_PayloadEmbeddingAccessMixin):
        def __init__(self, _path):
            self.params = {
                "input_args": SimpleNamespace(particles="p", datadir=None, strip_prefix=None),
            }
            self._payload = {
                "latent_coords": {2: np.zeros((4, 2), dtype=np.float32)},
                "latent_precision": {2: np.zeros((4, 2, 2), dtype=np.float32)},
                "contrasts": {2: np.ones(4, dtype=np.float32)},
                "dataset": dataset_obj,
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

    def _capture(dataset, *_args, **kwargs):
        captured["dataset"] = dataset

    monkeypatch.setattr(compute_state_cmd.o, "compute_and_save_reweighted", _capture)

    args = SimpleNamespace(
        result_dir=str(tmp_path / "pipeline_out"),
        particles=None,
        datadir=None,
        strip_prefix=None,
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

    # compute_state passes the dataset directly; halfsets are obtained
    # lazily via dataset.get_halfset(k) downstream.
    assert captured["dataset"] is dataset_obj
