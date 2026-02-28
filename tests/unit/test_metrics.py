import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import metrics

pytestmark = pytest.mark.unit


def test_variance_of_zs_zero_with_constant_labels():
    z = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [4.0, 5.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    label_vars, ratio = metrics.variance_of_zs(z, labels)
    # This function uses np.var over all elements in each label block.
    np.testing.assert_allclose(label_vars, np.array([0.25, 0.25], dtype=np.float32))
    assert ratio == pytest.approx(0.0)


def test_embedding_from_median_and_projection_helpers():
    zs = np.array(
        [
            [1.0, 10.0],
            [3.0, 30.0],
            [2.0, 20.0],
            [8.0, 80.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    emb = metrics.get_embedding_from_median(zs, labels)
    np.testing.assert_allclose(emb[0], np.array([2.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(emb[1], np.array([5.0, 50.0], dtype=np.float32))

    gt_vols = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    u = np.eye(2, dtype=np.float32)
    mean = np.array([0.5, 0.5], dtype=np.float32)
    proj = metrics.get_gt_embedding_from_projection(gt_vols, u, mean)
    np.testing.assert_allclose(proj, gt_vols - mean, atol=1e-7, rtol=1e-7)


def test_fro_norm_diff_low_rank_matches_dense():
    u = np.eye(2, dtype=np.float32)
    v = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    s = np.array([3.0, 1.0], dtype=np.float32)
    d = np.array([2.0, 0.5], dtype=np.float32)

    out = float(metrics.fro_norm_diff_low_rank(u, s, v, d))
    a = u @ np.diag(s) @ u.T
    b = v @ np.diag(d) @ v.T
    expected = np.linalg.norm(a - b, ord="fro")
    assert out == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_compute_volume_error_metrics_from_gt_aggregates(monkeypatch):
    monkeypatch.setattr(metrics, "local_fsc_metric", lambda *args, **kwargs: (1.0, 2.0, 3.0, 4.0))
    monkeypatch.setattr(metrics, "local_error_metric", lambda *args, **kwargs: (5.0, 6.0))

    gt = np.ones((2, 2, 2), dtype=np.float32)
    est = np.ones((2, 2, 2), dtype=np.float32)
    mask = np.array(
        [
            [[True, False], [True, False]],
            [[True, True], [False, False]],
        ]
    )
    partial_mask = ~mask

    out = metrics.compute_volume_error_metrics_from_gt(
        gt_map=gt,
        estimate_map=est,
        voxel_size=1.5,
        mask=mask,
        partial_mask=partial_mask,
        normalize_by_map1=True,
    )

    assert out["median_locres"] == 1.0
    assert out["ninety_pc_locres"] == 2.0
    assert out["median_auc"] == 3.0
    assert out["ten_pc_auc"] == 4.0
    assert out["median_error"] == 5.0
    assert out["ninety_pc_error"] == 6.0
    assert out["partial_median_locres"] == 1.0
    assert out["partial_ninety_pc_error"] == 6.0
    assert out["mask"] is mask
    assert out["partial_mask"] is partial_mask


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_fro_norm_diff_low_rank_gpu(gpu_device):
    u = np.eye(2, dtype=np.float32)
    v = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    s = np.array([3.0, 1.0], dtype=np.float32)
    d = np.array([2.0, 0.5], dtype=np.float32)

    cpu_out = float(metrics.fro_norm_diff_low_rank(u, s, v, d))

    with jax.default_device(gpu_device):
        u_g = jax.device_put(jnp.array(u), gpu_device)
        v_g = jax.device_put(jnp.array(v), gpu_device)
        s_g = jax.device_put(jnp.array(s), gpu_device)
        d_g = jax.device_put(jnp.array(d), gpu_device)
        gpu_out = float(metrics.fro_norm_diff_low_rank(u_g, s_g, v_g, d_g))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)
