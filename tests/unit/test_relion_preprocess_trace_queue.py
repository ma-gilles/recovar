"""A traced preprocessing status must never escape into a Python queue."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from recovar import cuda_backproject as cuda
from recovar.em.cuda import kernels as em_cuda_kernels

@pytest.mark.parametrize("traced", [False, True])
def test_deferred_check_only_queues_concrete_results(monkeypatch, traced):
    monkeypatch.setattr(em_cuda_kernels, "_RELION_PREPROCESS_PENDING_CHECKS", [])
    seen = []
    def native(images, *args):
        seen.append(args[-1])
        return images, images, jnp.zeros((1,), dtype=jnp.int32)
    monkeypatch.setattr(em_cuda_kernels, "_relion_preprocess_real_f32_jit", native)
    def call(images):
        return em_cuda_kernels.relion_preprocess_real_f32(images, None, None, 1., 1., deferred_finite_check=True)[1]
    images=jnp.ones((1,2,2),dtype=jnp.float32)
    actual=(jax.jit(call) if traced else call)(images)
    np.testing.assert_array_equal(actual, images)
    assert seen == [traced]
    assert em_cuda_kernels.pending_relion_preprocess_checks() == (0 if traced else 1)
    assert em_cuda_kernels.drain_relion_preprocess_checks() == (0 if traced else 1)


def test_deferred_invalid_status_fails_and_drains(monkeypatch):
    monkeypatch.setattr(em_cuda_kernels, "_RELION_PREPROCESS_PENDING_CHECKS", [])
    em_cuda_kernels._queue_relion_preprocess_check(jnp.asarray([2],dtype=jnp.int32))
    with pytest.raises(RuntimeError, match="2 image"):
        em_cuda_kernels.drain_relion_preprocess_checks()
    assert em_cuda_kernels.pending_relion_preprocess_checks() == 0
