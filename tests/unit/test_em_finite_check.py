"""Unit tests for the P4-D pass-2 finite and bound checks.

The checks exist because job 14178118's compact control overflowed its BPref
accumulators from operands that are provably bounded, and because a
large-but-finite corruption raises nothing until the accumulated value reaches
infinity. Each test below pins one of the properties that makes the checks
useful: they are silent unless asked for, they name the first offending field
and index, they fire on a bound that arithmetic cannot break, and they can be
asked to record every occurrence instead of dying on the first.
"""

import logging

import numpy as np
import pytest

from recovar.em.diagnostics import finite_check


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(finite_check.FINITE_CHECK_ENV, "1")
    monkeypatch.delenv(finite_check.FINITE_CHECK_WARN_ENV, raising=False)
    finite_check.report_tracked("test reset")
    yield
    finite_check.report_tracked("test reset")


@pytest.fixture
def warn_only(monkeypatch):
    monkeypatch.setenv(finite_check.FINITE_CHECK_ENV, "1")
    monkeypatch.setenv(finite_check.FINITE_CHECK_WARN_ENV, "1")
    finite_check.report_tracked("test reset")
    yield
    finite_check.report_tracked("test reset")


def test_every_check_is_silent_unless_the_flag_is_set(monkeypatch):
    """Production must pay nothing, so an unset flag returns before any work."""

    monkeypatch.delenv(finite_check.FINITE_CHECK_ENV, raising=False)
    poisoned = np.array([1.0, np.inf])
    assert finite_check.check_arrays("s", {"a": poisoned}) is None
    assert finite_check.check_per_image("s", {"a": poisoned}) is None
    assert finite_check.check_posterior_bounds("s", np.array([[5.0]])) is None
    assert finite_check.check_bundle("s", {"a": poisoned}, posterior=np.array([[5.0]])) is None
    assert finite_check.track_max("a", poisoned) is None
    assert finite_check.report_tracked("nothing") is None


def test_check_arrays_names_the_first_offender_and_its_operands(enabled):
    clean = np.array([3.0, 4.0])
    poisoned = np.array([1.0, 2.0, np.nan, np.inf])
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_arrays(
            "mstep-sums",
            {"clean": clean, "poisoned": poisoned},
            context="half=2 bucket=17",
            operands={"operand": clean},
        )
    message = str(excinfo.value)
    assert "mstep-sums" in message and "half=2 bucket=17" in message
    assert "first offender: poisoned at flat index 2" in message
    assert "nonfinite=2/4" in message
    assert "operand operand" in message


def test_check_per_image_names_the_offending_image_ids(enabled):
    statistic = np.array([1.0, np.nan, 3.0, np.inf])
    ids = np.array([100, 101, 102, 103])
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_per_image(
            "norm", {"wsum_norm_correction": statistic}, image_ids=ids,
        )
    message = str(excinfo.value)
    assert "2/4 non-finite" in message
    assert "[101, 103]" in message


def test_posterior_bound_passes_a_pruned_posterior_and_fails_a_broken_one(enabled):
    # RELION prunes to the adaptive fraction, so a healthy image total sits
    # just under one and must not be reported.
    pruned = np.array([[0.5, 0.3, 0.199], [0.999, 0.0, 0.0]], dtype=np.float32)
    assert finite_check.check_posterior_bounds("p", pruned, image_ids=np.array([7, 8])) is None

    # A stale or uninitialised padded row shows up here: the total is taken
    # over every row and translation of the full array.
    broken = pruned.copy()
    broken[1, 1] = 0.5
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_posterior_bounds("p", broken, image_ids=np.array([7, 8]))
    assert "image ids [8]" in str(excinfo.value)


def test_reduction_bound_fires_on_a_sum_larger_than_its_operand(enabled):
    operand = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    within = {"summed": operand * 0.9, "shifted": operand}
    bound = (("summed", "shifted", 1e-3),)
    assert finite_check.check_bundle("m", within, dominated_by=bound) is None

    beyond = {"summed": operand * 1.5, "shifted": operand}
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_bundle("m", beyond, dominated_by=bound)
    assert "max |summed| = 6" in str(excinfo.value)


def test_operand_bound_uses_the_product_of_several_factors(enabled):
    ctf = np.array([[0.5, -0.8], [0.9, 0.2]], dtype=np.float32)
    inverse_noise = np.array([1e-4, 2e-4], dtype=np.float32)
    weighted = ctf * inverse_noise[None, :]
    bound = (("weighted", ("ctf", "inverse_noise"), 1e-3),)
    arrays = {"ctf": ctf, "inverse_noise": inverse_noise, "weighted": weighted}
    assert finite_check.check_bundle("io", arrays, dominated_by=bound) is None

    arrays["weighted"] = weighted.copy()
    arrays["weighted"][1, 1] = np.float32(1e3)
    with pytest.raises(finite_check.FiniteCheckError) as excinfo:
        finite_check.check_bundle("io", arrays, dominated_by=bound)
    message = str(excinfo.value)
    assert "max |ctf| * max |inverse_noise|" in message


def test_warn_mode_records_every_occurrence_instead_of_dying(warn_only, caplog):
    poisoned = np.array([1.0, np.inf])
    with caplog.at_level(logging.ERROR, logger=finite_check.__name__):
        first = finite_check.check_arrays("s", {"a": poisoned})
        second = finite_check.check_posterior_bounds("p", np.array([[2.0]], dtype=np.float32))
    assert first is not None and second is not None
    assert len(caplog.records) == 2


def test_tracked_maxima_survive_a_clean_pass_and_record_infinity(enabled):
    finite_check.track_max("ctf_probs", np.array([1.0, -7.5]))
    finite_check.track_max("ctf_probs", np.array([3.0]))
    finite_check.track_max("summed", np.array([np.inf, 1.0]))
    message = finite_check.report_tracked("iteration=15 half=2")
    assert "iteration=15 half=2" in message
    assert "ctf_probs=7.5" in message
    assert "summed=inf" in message
    # Reporting clears them, so the next pass starts from its own maxima.
    assert finite_check.report_tracked("again") is None
