"""
Unit tests for recovar.em.states.

Covers construction and initial attribute values for:
  EMState, SGDState, HeterogeneousEMState

No E-step or M-step is invoked – only attribute inspection.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.em.states import EMState, SGDState, HeterogeneousEMState

pytestmark = pytest.mark.unit

# grid_size=4 → volume_size = 4**3 = 64
VOLUME_SIZE = 64


def _mean():
    return np.ones(VOLUME_SIZE, dtype=np.complex64)


def _mean_variance():
    return np.ones(VOLUME_SIZE, dtype=np.float32) * 1e3


def _noise_variance():
    return np.ones(VOLUME_SIZE, dtype=np.float32) * 0.1


# ---------------------------------------------------------------------------
# EMState
# ---------------------------------------------------------------------------

def test_EMState_stores_mean_and_variance():
    mean = _mean()
    mean_var = _mean_variance()
    noise_var = _noise_variance()
    state = EMState(mean, mean_var, noise_var)
    np.testing.assert_array_equal(state.mean, mean)
    np.testing.assert_array_equal(state.mean_variance, mean_var)
    np.testing.assert_array_equal(state.noise_variance, noise_var)


def test_EMState_initial_accumulators_are_zero():
    state = EMState(_mean(), _mean_variance(), _noise_variance())
    assert state.Ft_y == 0
    assert state.Ft_CTF == 0


def test_EMState_name_is_EM():
    state = EMState(_mean(), _mean_variance(), _noise_variance())
    assert state.name == "EM"


# ---------------------------------------------------------------------------
# SGDState
# ---------------------------------------------------------------------------

def test_SGDState_stores_attributes():
    mean = _mean()
    mean_var = _mean_variance()
    noise_var = _noise_variance()
    state = SGDState(mean, mean_var, noise_var)
    np.testing.assert_array_equal(state.mean, mean)
    np.testing.assert_array_equal(state.mean_variance, mean_var)
    np.testing.assert_array_equal(state.noise_variance, noise_var)


def test_SGDState_update_is_zero_at_init():
    state = SGDState(_mean(), _mean_variance(), _noise_variance())
    assert state.update == 0


def test_SGDState_name_is_SGD():
    state = SGDState(_mean(), _mean_variance(), _noise_variance())
    assert state.name == "SGD"


def test_SGDState_has_sgd_batchsize_attribute():
    state = SGDState(_mean(), _mean_variance(), _noise_variance())
    assert hasattr(state, "sgd_batchsize")
    assert state.sgd_batchsize > 0


# ---------------------------------------------------------------------------
# HeterogeneousEMState
# ---------------------------------------------------------------------------

def test_HeterogeneousEMState_stores_attributes():
    mean = _mean()
    mean_var = _mean_variance()
    noise_var = _noise_variance()
    state = HeterogeneousEMState(mean, mean_var, noise_var)
    np.testing.assert_array_equal(state.mean, mean)
    np.testing.assert_array_equal(state.mean_variance, mean_var)
    np.testing.assert_array_equal(state.noise_variance, noise_var)


def test_HeterogeneousEMState_name():
    state = HeterogeneousEMState(_mean(), _mean_variance(), _noise_variance())
    assert state.name == "HeterogeneousEM"


def test_HeterogeneousEMState_H_B_accumulators_are_zero():
    state = HeterogeneousEMState(_mean(), _mean_variance(), _noise_variance())
    assert state.H == 0
    assert state.B == 0


def test_HeterogeneousEMState_u_subspace_cov_cols_are_none():
    state = HeterogeneousEMState(_mean(), _mean_variance(), _noise_variance())
    assert state.u is None
    assert state.subspace is None
    assert state.cov_cols is None


def test_HeterogeneousEMState_volume_mask_is_created():
    """__init__ creates a volume_mask via raised_cosine_mask (non-None)."""
    state = HeterogeneousEMState(_mean(), _mean_variance(), _noise_variance())
    assert state.volume_mask is not None
    # volume_mask must have size equal to VOLUME_SIZE
    import numpy as np
    assert np.asarray(state.volume_mask).size == VOLUME_SIZE


def test_HeterogeneousEMState_projected_cov_accumulators_are_zero():
    state = HeterogeneousEMState(_mean(), _mean_variance(), _noise_variance())
    assert state.projected_cov_lhs == 0
    assert state.projected_cov_rhs == 0
