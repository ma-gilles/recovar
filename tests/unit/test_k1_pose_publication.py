"""Exact publication and dispatch checks for sole-class pose selection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import k_class
from recovar.em.dense_single_volume.helpers.types import make_relion_stats

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_selector(monkeypatch, token, expected):
    monkeypatch.delenv(k_class._K1_POSE_PUBLISH_DIRECT_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(k_class._K1_POSE_PUBLISH_DIRECT_ENV, token)
    assert k_class._k1_pose_publish_direct_requested() is expected


@pytest.mark.parametrize("token", ["", "true", "false", "2", "-1", "typo"])
def test_invalid_selector(monkeypatch, token):
    monkeypatch.setenv(k_class._K1_POSE_PUBLISH_DIRECT_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        k_class._k1_pose_publish_direct_requested()


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((3, 3), np.float32),
        ((3, 3), np.float64),
        ((2,), np.float32),
        ((2,), np.float64),
        ((), np.int32),
        ((), np.int64),
    ],
)
@pytest.mark.parametrize("n_images", [0, 200, 208, 1000])
@pytest.mark.parametrize("device", [False, True])
def test_single_class_publication_is_bitwise_and_device_identity(shape, dtype, n_images, device):
    shape = (n_images, *shape)
    value = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    if value.size and np.issubdtype(dtype, np.floating):
        value.flat[:4] = [0.0, -0.0, np.nan, -np.inf]
    if device:
        value = jnp.asarray(value)
    assignments = np.zeros(n_images, dtype=np.int32)
    expected = k_class._selected_by_class([value], assignments)
    actual = k_class._selected_by_class([value], assignments, direct_single_class=True)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    if device:
        assert actual is value


@pytest.mark.parametrize("classes", [1, 4])
def test_actual_result_assembly_preserves_every_field(monkeypatch, classes):
    n_images = 8
    values = [jnp.arange(n_images * 9, dtype=jnp.float32).reshape(n_images, 3, 3) + k for k in range(classes)]
    stats = tuple(
        make_relion_stats(
            log_evidence_per_image=np.zeros(n_images, dtype=np.float64),
            best_log_score_per_image=np.where(np.arange(n_images) % classes == k, -0.1, -2.0),
            max_posterior_per_image=np.full(n_images, 0.75, dtype=np.float32),
            rotation_posterior_sums=jnp.arange(3, dtype=jnp.float32),
        )
        for k in range(classes)
    )
    kwargs = dict(
        class_log_evidence=np.zeros((classes, n_images), dtype=np.float64),
        new_means=None,
        Ft_y=[jnp.arange(4, dtype=jnp.complex64)] * classes,
        Ft_ctf=[jnp.arange(4, dtype=jnp.float32)] * classes,
        per_class_hard_assignments=np.arange(classes * n_images).reshape(classes, n_images),
        per_class_stats=stats,
        noise_stats=None,
        per_class_best_pose_rotations=values,
        per_class_best_pose_translations=[v[:, 0, :2] for v in values],
        per_class_best_pose_rotation_ids=[jnp.arange(n_images, dtype=jnp.int32) + k for k in range(classes)],
    )
    monkeypatch.setenv(k_class._K1_POSE_PUBLISH_DIRECT_ENV, "0")
    expected = k_class._assemble_result(**kwargs)
    monkeypatch.setenv(k_class._K1_POSE_PUBLISH_DIRECT_ENV, "1")
    actual = k_class._assemble_result(**kwargs)
    first, spec = jax.tree_util.tree_flatten(expected)
    second, other_spec = jax.tree_util.tree_flatten(actual)
    assert spec == other_spec
    for want, got in zip(first, second, strict=True):
        want, got = np.asarray(want), np.asarray(got)
        assert want.shape == got.shape and want.dtype == got.dtype and want.tobytes() == got.tobytes()
    if classes == 1:
        assert actual.best_pose_rotations is values[0]


@pytest.mark.parametrize("mode", ["k4", "device_assignments", "different_image_count", "nonzero_class", "missing"])
def test_unhandled_inputs_keep_original_gather_path(monkeypatch, mode):
    value = jnp.arange(24, dtype=jnp.float32).reshape(8, 3)
    values = [value]
    assignments = np.zeros(8, dtype=np.int32)
    if mode == "k4":
        values = [value + k for k in range(4)]
        assignments = np.arange(8, dtype=np.int32) % 4
    elif mode == "device_assignments":
        assignments = jnp.asarray(assignments)
    elif mode == "different_image_count":
        assignments = assignments[:4]
    elif mode == "nonzero_class":
        assignments[-1] = -1
    elif mode == "missing":
        values = None
    expected = k_class._selected_by_class(values, assignments)
    calls = []
    original = k_class._stack_or_none

    def record(values):
        calls.append(values)
        return original(values)

    monkeypatch.setattr(k_class, "_stack_or_none", record)
    actual = k_class._selected_by_class(values, assignments, direct_single_class=True)
    assert len(calls) == 1
    if expected is None:
        assert actual is None
    else:
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
