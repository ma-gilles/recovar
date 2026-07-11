import numpy as np
import jax.numpy as jnp

from recovar.core import fourier_transform_utils
from recovar.em.dense_single_volume.iteration_loop import _maybe_debug_replay_relion_references
from recovar.utils.helpers import write_relion_mrc


def _real_from_ft(flat, shape):
    return np.real(np.asarray(fourier_transform_utils.get_idft3(np.asarray(flat).reshape(shape))))


def test_debug_relion_reference_replay_loads_requested_iteration(monkeypatch, tmp_path):
    shape = (4, 4, 4)
    half1 = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100.0).astype(np.float32)
    half2 = (half1 + np.float32(2.0)).astype(np.float32)
    write_relion_mrc(tmp_path / "run_it010_half1_class001.mrc", half1, voxel_size=1.0)
    write_relion_mrc(tmp_path / "run_it010_half2_class001.mrc", half2, voxel_size=1.0)

    original = [
        jnp.zeros(np.prod(shape), dtype=jnp.complex64),
        jnp.ones(np.prod(shape), dtype=jnp.complex64),
    ]
    monkeypatch.setenv("RECOVAR_DEBUG_REPLAY_RELION_REFERENCES", "1")
    monkeypatch.setenv("RECOVAR_DEBUG_REPLAY_RELION_REFERENCES_ITERATION", "11")

    unchanged = _maybe_debug_replay_relion_references(
        means=original,
        perturb_replay_relion_dir=tmp_path,
        init_relion_iteration=0,
        iteration=9,
        volume_shape=shape,
        n_classes=1,
    )
    assert unchanged is original

    replayed = _maybe_debug_replay_relion_references(
        means=original,
        perturb_replay_relion_dir=tmp_path,
        init_relion_iteration=0,
        iteration=10,
        volume_shape=shape,
        n_classes=1,
    )

    assert replayed is not original
    np.testing.assert_allclose(_real_from_ft(replayed[0], shape), half1, rtol=0, atol=5e-6)
    np.testing.assert_allclose(_real_from_ft(replayed[1], shape), half2, rtol=0, atol=5e-6)
