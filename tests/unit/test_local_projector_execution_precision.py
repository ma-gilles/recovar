"""Projection precision is resolved before host slab upload."""
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.relion.relion_projector_setup import cast_relion_projector_for_execution


@pytest.mark.parametrize("classes", [1, 4])
@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("host", [False, True])
def test_local_projector_execution_precision(classes, double, host):
    setup = (np.arange(classes * 7 * 7 * 4).reshape(classes, 7, 7, 4)
             + 1j / 7).astype(np.complex128)
    before = setup.copy()
    source = setup if host else jnp.asarray(setup)
    output = cast_relion_projector_for_execution(source, use_float64_projections=double)
    dtype = np.complex128 if double else np.complex64
    assert output.dtype == dtype
    assert output.shape == setup.shape
    assert isinstance(output, np.ndarray) == host
    np.testing.assert_array_equal(np.asarray(output), setup.astype(dtype))
    np.testing.assert_array_equal(setup, before)


def test_no_local_relion_projector():
    assert cast_relion_projector_for_execution(None) is None


@pytest.mark.parametrize("double", [False, True])
def test_fused_k_class_resolves_projector_consumer_precision(monkeypatch, double):
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed

    class PrecisionResolved(Exception):
        pass

    setup = np.ones((4, 7, 7, 4), dtype=np.complex128)

    def resolve(projector, *, use_float64_projections):
        assert projector is setup
        assert use_float64_projections is double
        result = cast_relion_projector_for_execution(
            projector, use_float64_projections=use_float64_projections,
        )
        assert result.dtype == (np.complex128 if double else np.complex64)
        raise PrecisionResolved

    monkeypatch.setattr(bucketed, "cast_relion_projector_for_execution", resolve)
    with pytest.raises(PrecisionResolved):
        bucketed.compute_k_class_pass2_stats_sparse_fused(
            None, np.zeros((4, 8), dtype=np.complex64), np.ones(8),
            np.zeros((1, 2)), [None] * 4,
            rotation_log_priors_by_class=[None] * 4,
            nside_level=1, disc_type="linear_interp", oversampling_order=1,
            current_size=6, relion_projector_half=setup,
            relion_projector_r_max=3, use_float64_scoring=double,
        )
