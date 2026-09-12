"""External normalization retains image axes and caller-specific domains."""

import numpy as np
import pytest

from recovar.em.helpers.normalization_inputs import optional_normalization_vector, prepare_local_normalization_inputs

pytestmark = pytest.mark.unit


def test_missing_inputs_remain_none():
    result = prepare_local_normalization_inputs(n_images=2)
    assert result.log_z is None and result.log_evidence is None
    assert result.max_posterior is None and result.reconstruction_threshold is None


def test_existing_f64_views_are_retained():
    original = np.asarray([-0.0, 2.0, -np.inf, 4.0], dtype=np.float64)
    result = prepare_local_normalization_inputs(n_images=2, normalization_log_z=original[::2])
    assert np.shares_memory(result.log_z, original)
    assert result.log_z.strides == original[::2].strides
    assert result.log_z.tobytes() == original[::2].tobytes()


@pytest.mark.parametrize("name", ["normalization_log_z", "normalization_log_evidence"])
def test_log_domain_is_not_silently_tightened(name):
    result = prepare_local_normalization_inputs(n_images=2, **{name: [np.nan, -np.inf]})
    value = result.log_z if name.endswith("log_z") else result.log_evidence
    np.testing.assert_array_equal(value, [np.nan, -np.inf])


@pytest.mark.parametrize("shape", [(), (1,), (2, 1), (1, 2)])
def test_optional_vector_does_not_flatten_or_broadcast(shape):
    with pytest.raises(ValueError, match="normalization_log_z must have shape"):
        optional_normalization_vector(np.zeros(shape), name="normalization_log_z", n_images=2)


def test_f32_inputs_are_converted_without_mutation():
    source = np.asarray([0.1, 0.9], dtype=np.float32)
    before = source.tobytes()
    result = prepare_local_normalization_inputs(n_images=2, normalization_max_posterior=source)
    assert result.max_posterior.dtype == np.float64
    np.testing.assert_array_equal(result.max_posterior, source.astype(np.float64))
    assert source.tobytes() == before


@pytest.mark.parametrize("value", [0.0, -1.0, 1.01, np.inf, np.nan])
def test_local_pmax_requires_finite_probability(value):
    with pytest.raises(ValueError, match=r"finite probabilities in \(0, 1\]"):
        prepare_local_normalization_inputs(n_images=1, normalization_max_posterior=[value])


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
def test_reconstruction_threshold_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="reconstruction_probability_threshold must"):
        prepare_local_normalization_inputs(n_images=1, reconstruction_probability_threshold=[value])


def test_empty_arrays_and_zero_reconstruction_threshold_are_valid():
    empty = prepare_local_normalization_inputs(n_images=0, normalization_max_posterior=[])
    assert empty.max_posterior.shape == (0,)
    result = prepare_local_normalization_inputs(
        n_images=1,
        normalization_max_posterior=[1.0],
        reconstruction_probability_threshold=[0.0],
    )
    np.testing.assert_array_equal(result.max_posterior, [1.0])
    np.testing.assert_array_equal(result.reconstruction_threshold, [0.0])


def test_conflicting_logs_fail_before_pmax_conversion():
    with pytest.raises(ValueError, match="Provide only one of"):
        prepare_local_normalization_inputs(
            n_images=1,
            normalization_log_z=[0.0],
            normalization_log_evidence=[0.0],
            normalization_max_posterior=["not a number"],
        )


@pytest.mark.parametrize("log_name", ["normalization_log_z", "normalization_log_evidence"])
def test_pmax_is_exclusive_with_either_log_mode(log_name):
    with pytest.raises(ValueError, match="mutually exclusive with external log normalization"):
        prepare_local_normalization_inputs(
            n_images=1,
            normalization_max_posterior=[0.5],
            **{log_name: [0.0]},
        )
