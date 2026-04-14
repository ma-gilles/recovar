import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

pytestmark = pytest.mark.unit


class _FakeAtoms:
    def __init__(self):
        self._coords = np.zeros((4, 3), dtype=np.float32)

    def getCoords(self):
        return self._coords.copy()

    def setCoords(self, coords):
        self._coords = np.asarray(coords, dtype=np.float32)


def test_make_spike_main_uses_utils_pickle_dump(monkeypatch, tmp_path):
    from recovar.commands import make_spike_datasets as msd

    calls = {"pickle_paths": [], "assignment_disc_type": None, "assignment_wrapper": None, "dataset_lazy": None}

    monkeypatch.setattr(msd.ssp, "_parsePDB", lambda _path: _FakeAtoms())
    monkeypatch.setattr(msd.ssp, "get_center_coord_offset", lambda _coords: np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(
        msd.ssp,
        "generate_molecule_spectrum_from_pdb_id",
        lambda *_args, **_kwargs: np.ones((8,), dtype=np.complex64),
    )
    monkeypatch.setattr(msd.simulator, "Bfactorize_vol", lambda vol, *_args, **_kwargs: np.asarray(vol))
    monkeypatch.setattr(msd.output, "mkdir_safe", lambda path: None)
    monkeypatch.setattr(msd.output, "save_volumes", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        msd.simulator,
        "generate_synthetic_dataset",
        lambda *_args, **_kwargs: (
            None,
            {
                "volumes_path_root": str(tmp_path / "vols" / "vol"),
                "grid_size": 8,
                "trailing_zero_format_in_vol_name": True,
                "scale_vol": 1.0,
                "noise_variance": np.array([1.0, 1.0], dtype=np.float32),
                "image_assignment": np.array([0, 1], dtype=np.int32),
            },
        ),
    )
    monkeypatch.setattr(
        msd.simulator,
        "load_volumes_from_folder",
        lambda *_args, **_kwargs: np.ones((2, 8), dtype=np.float32),
    )

    def _fake_load_dataset(**kwargs):
        calls["dataset_lazy"] = kwargs.get("lazy", True)
        return SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2))

    monkeypatch.setattr(msd.cryoem_dataset, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(
        msd.noise,
        "make_radial_noise",
        lambda *_args, **_kwargs: np.ones((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        msd.core,
        "to_cubic",
        lambda vols, _volume_shape, **_kwargs: msd.core.CubicVolume.from_coeffs(np.asarray(vols)),
    )

    def _fake_compute_image_assignment(_cryo, _vols, _noise_cov, _batch_size):
        calls["assignment_disc_type"] = getattr(_vols, "disc_type", None)
        calls["assignment_wrapper"] = type(_vols).__name__
        return np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)

    monkeypatch.setattr(msd.image_assignment, "compute_image_assignment", _fake_compute_image_assignment)
    monkeypatch.setattr(msd.image_assignment, "estimate_false_positive_rate", lambda *_args, **_kwargs: 0.25)
    monkeypatch.setattr(msd.utils, "pickle_dump", lambda _obj, path: calls["pickle_paths"].append(path))
    monkeypatch.setattr(msd.plt, "show", lambda: None)

    msd.main(
        output_folder=str(tmp_path),
        pdb_folder=str(tmp_path),
        n_images=2,
        grid_size=8,
        noise_level_tests=np.array([10.0], dtype=np.float32),
    )

    assert calls["dataset_lazy"] is False
    assert calls["assignment_disc_type"] == "cubic"
    assert calls["assignment_wrapper"] == "CubicVolume"
    assert any(path.endswith("dataset0/result.pkl") for path in calls["pickle_paths"])
    assert any(path.endswith("curve.pkl") for path in calls["pickle_paths"])
