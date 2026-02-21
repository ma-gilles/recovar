import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import cryodrgn_load as cload
from recovar import utils


def test_load_ctf_for_training_updates_apix_and_strips_D(tmp_path):
    ctf = np.array(
        [
            [128, 1.5, 10000, 10000, 0, 300, 2.7, 0.1, 0],
            [128, 1.5, 11000, 11000, 0, 300, 2.7, 0.1, 0],
        ],
        dtype=np.float32,
    )
    ctf_path = tmp_path / "ctf.pkl"
    utils.pickle_dump(ctf, str(ctf_path))

    out = cload.load_ctf_for_training(D=64, ctf_params_pkl=str(ctf_path))
    assert out.shape == (2, 8)
    assert np.allclose(out[:, 0], np.array([3.0, 3.0], dtype=np.float32))


def test_load_poses_with_translation_scales_by_D(tmp_path):
    nimg = 4
    D = 32
    rots = np.tile(np.eye(3, dtype=np.float32), (nimg, 1, 1))
    trans = np.array([[0.1, -0.2], [0.0, 0.0], [0.5, -0.5], [0.25, 0.25]], dtype=np.float32)
    r_path = tmp_path / "rots.pkl"
    t_path = tmp_path / "trans.pkl"
    utils.pickle_dump(rots, str(r_path))
    utils.pickle_dump(trans, str(t_path))

    out_rots, out_trans, _ = cload.load_poses([str(r_path), str(t_path)], Nimg=nimg, D=D)
    assert out_rots.shape == (nimg, 3, 3)
    assert np.allclose(out_trans, trans * D)


def test_load_poses_without_translation_returns_none(tmp_path):
    nimg = 3
    rots = np.tile(np.eye(3, dtype=np.float32), (nimg, 1, 1))
    path = tmp_path / "rots_only.pkl"
    utils.pickle_dump(rots, str(path))

    out_rots, out_trans, _ = cload.load_poses(str(path), Nimg=nimg, D=16)
    assert out_rots.shape == (nimg, 3, 3)
    assert out_trans is None

