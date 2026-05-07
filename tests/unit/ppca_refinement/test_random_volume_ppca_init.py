import importlib.util
from pathlib import Path

import numpy as np
import pytest

from recovar.utils import helpers


pytestmark = pytest.mark.unit


def _load_script_module():
    repo = Path(__file__).resolve().parents[3]
    script_path = repo / "scripts" / "prepare_random_volume_ppca_init.py"
    spec = importlib.util.spec_from_file_location("prepare_random_volume_ppca_init", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_random_volume_ppca_init_downsamples_and_preserves_covariance_trace(tmp_path):
    module = _load_script_module()
    rng = np.random.default_rng(0)
    volume_dir = tmp_path / "vols"
    volume_dir.mkdir()
    volume_paths = []
    for idx in range(5):
        vol = rng.normal(size=(8, 8, 8)).astype(np.float32)
        path = volume_dir / f"vol{idx:04d}.mrc"
        helpers.write_mrc(path, vol, voxel_size=1.5)
        volume_paths.append(path)

    summary = module.prepare_random_volume_ppca_init(
        volume_paths,
        output_dir=tmp_path / "init",
        k=4,
        q=3,
        target_grid_size=4,
        seed=11,
        frame="recovar",
    )

    assert summary["passed"]
    assert summary["k"] == 4
    assert summary["q"] == 3
    assert summary["stats"]["mu_shape"] == [4, 4, 4]
    assert summary["stats"]["W_shape"] == [3, 4, 4, 4]
    assert summary["output_voxel_size"] == pytest.approx(3.0)
    assert summary["stats"]["trace_relative_error"] < 1e-6
    assert summary["stats"]["retained_covariance_fraction"] == pytest.approx(1.0)
    assert not summary["stats"]["rank_truncated"]
    assert len(summary["written_volume_paths"]) == 4

    init = np.load(tmp_path / "init" / "ppca_init.npz")
    assert init["mu"].shape == (4, 4, 4)
    assert init["W"].shape == (3, 4, 4, 4)
    assert init["aligned_volumes"].shape == (4, 4, 4, 4)
    assert (tmp_path / "init" / "summary.json").exists()


def test_random_volume_ppca_init_can_truncate_pc_count(tmp_path):
    module = _load_script_module()
    rng = np.random.default_rng(1)
    volume_dir = tmp_path / "vols"
    volume_dir.mkdir()
    volume_paths = []
    for idx in range(6):
        vol = rng.normal(size=(4, 4, 4)).astype(np.float32)
        path = volume_dir / f"vol{idx:04d}.mrc"
        helpers.write_mrc(path, vol)
        volume_paths.append(path)

    summary = module.prepare_random_volume_ppca_init(
        volume_paths,
        output_dir=tmp_path / "init_q2",
        k=6,
        q=2,
        target_grid_size=4,
        seed=5,
        frame="recovar",
        write_maps=False,
    )

    assert summary["q"] == 2
    assert summary["stats"]["W_shape"] == [2, 4, 4, 4]
    assert summary["stats"]["rank_truncated"]
    assert 0.0 < summary["stats"]["retained_covariance_fraction"] < 1.0


def test_random_volume_ppca_init_rejects_upsampling(tmp_path):
    module = _load_script_module()
    with pytest.raises(ValueError, match="only downsamples"):
        module._downsample_recovar_volume(np.zeros((4, 4, 4), dtype=np.float32), 8)
