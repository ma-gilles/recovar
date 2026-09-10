"""Host history formatting preserves shell layout and pre-append validation."""

from dataclasses import fields

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.iteration_history import RefinementHistory

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k_class_enabled", [False, True])
def test_noise_history_formats_host_arrays_without_changing_alias_contract(dtype, k_class_enabled):
    shells = np.asarray([1.25, 2.5, 3.75], dtype=dtype)
    keys = ("prior_shells", "sigma2_shells", "avg_weight_shells", "shell_sum", "shell_count", "ssnr_shells")
    details = dict.fromkeys(keys, shells)
    if not k_class_enabled:
        details["fsc_shells"] = shells
    history = RefinementHistory()
    history.record_noise_and_tau2(shells, [shells, shells], details, k_class_enabled=k_class_enabled)

    for stored in [history.noise_radial_trajectory[0], history.tau2_radial_trajectory[0]]:
        assert stored.dtype == np.float64
        np.testing.assert_array_equal(stored, shells)
        assert np.shares_memory(stored, shells) == (dtype == np.float64)
    per_half = history.noise_radial_per_half_trajectory[0]
    assert per_half.shape == (2, 3) and per_half.dtype == np.float64
    np.testing.assert_array_equal(per_half, [shells, shells])
    assert not np.shares_memory(per_half, shells)
    if k_class_enabled:
        assert history.tau2_fsc_used_trajectory == [None]
    else:
        np.testing.assert_array_equal(history.tau2_fsc_used_trajectory[0], shells)
    assert all(value is shells for value in details.values())


def test_noise_history_without_tau2_records_none_for_every_tau2_series():
    history = RefinementHistory()
    history.record_noise_and_tau2([1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]], None, k_class_enabled=False)
    assert history.noise_radial_trajectory[0].dtype == np.float64
    for field in fields(history):
        if field.name.startswith("tau2_"):
            assert getattr(history, field.name) == [None]


@pytest.mark.parametrize("details", [{}, {"prior_shells": [1.0, 2.0]}])
def test_malformed_tau2_does_not_partially_append_history(details):
    history = RefinementHistory()
    before = {field.name: list(getattr(history, field.name)) for field in fields(history)}
    with pytest.raises(KeyError):
        history.record_noise_and_tau2([1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]], details, k_class_enabled=False)
    assert {field.name: getattr(history, field.name) for field in fields(history)} == before
