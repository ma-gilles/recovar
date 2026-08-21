import jax.numpy as jnp
import numpy as np

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


def test_debug_relion_reference_replay_loads_shared_kclass_maps(monkeypatch, tmp_path):
    shape = (4, 4, 4)
    class_maps = []
    for class_number in range(1, 5):
        volume = np.full(shape, np.float32(class_number) / np.float32(10.0), dtype=np.float32)
        class_maps.append(volume)
        write_relion_mrc(
            tmp_path / f"run_it002_class{class_number:03d}.mrc",
            volume,
            voxel_size=1.0,
        )

    original = [
        jnp.zeros((4, np.prod(shape)), dtype=jnp.complex64),
        jnp.ones((4, np.prod(shape)), dtype=jnp.complex64),
    ]
    monkeypatch.setenv("RECOVAR_DEBUG_REPLAY_RELION_REFERENCES", "1")
    monkeypatch.setenv("RECOVAR_DEBUG_REPLAY_RELION_REFERENCES_ITERATION", "3")

    replayed = _maybe_debug_replay_relion_references(
        means=original,
        perturb_replay_relion_dir=tmp_path,
        init_relion_iteration=0,
        iteration=2,
        volume_shape=shape,
        n_classes=4,
    )

    assert replayed is not original
    assert replayed[0].shape == (4, np.prod(shape))
    assert replayed[1].shape == (4, np.prod(shape))
    for half_idx in range(2):
        for class_idx, expected in enumerate(class_maps):
            np.testing.assert_allclose(
                _real_from_ft(replayed[half_idx][class_idx], shape),
                expected,
                rtol=0,
                atol=5e-6,
            )


def test_state_swap_force_replays_target_references_without_environment(tmp_path):
    shape = (4, 4, 4)
    half1 = np.full(shape, np.float32(0.25), dtype=np.float32)
    half2 = np.full(shape, np.float32(0.75), dtype=np.float32)
    write_relion_mrc(tmp_path / "run_it004_half1_class001.mrc", half1, voxel_size=1.0)
    write_relion_mrc(tmp_path / "run_it004_half2_class001.mrc", half2, voxel_size=1.0)
    original = [
        jnp.zeros(np.prod(shape), dtype=jnp.complex64),
        jnp.ones(np.prod(shape), dtype=jnp.complex64),
    ]

    replayed = _maybe_debug_replay_relion_references(
        means=original,
        perturb_replay_relion_dir=tmp_path,
        init_relion_iteration=0,
        iteration=4,
        volume_shape=shape,
        n_classes=1,
        force=True,
    )

    np.testing.assert_allclose(_real_from_ft(replayed[0], shape), half1, rtol=0, atol=5e-6)
    np.testing.assert_allclose(_real_from_ft(replayed[1], shape), half2, rtol=0, atol=5e-6)
