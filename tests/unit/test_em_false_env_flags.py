"""Diagnostic flags accept known booleans without silently enabling typos."""

import logging

import pytest

from recovar.em.helpers.env_flags import (
    parse_env_binary_flag,
    parse_env_capacity_ladder,
    parse_env_flag,
    parse_env_flag_or_false,
    parse_env_true_flag,
)

pytestmark = pytest.mark.unit
NAME = "RECOVAR_TEST_FALSE_FLAG"
LOG = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("", False), (" \t\n", False)]
    + [(v, False) for v in ("0", "false", "no", "off", " FaLsE ")]
    + [(v, True) for v in ("1", "true", "yes", "on", " On\n")],
)
def test_known_tokens_and_absence_are_silent(monkeypatch, caplog, value, expected):
    if value is None:
        monkeypatch.delenv(NAME, raising=False)
    else:
        monkeypatch.setenv(NAME, value)
    assert parse_env_true_flag(NAME) is expected
    assert parse_env_flag_or_false(NAME, logger=LOG) is expected
    assert not caplog.records


@pytest.mark.parametrize("value", ["unexpected", "2", "false true", "none"])
def test_invalid_flag_preserves_caller_warning(monkeypatch, caplog, value):
    monkeypatch.setenv(NAME, value)
    assert parse_env_true_flag(NAME) is False
    assert not caplog.records
    assert parse_env_flag_or_false(NAME, logger=LOG) is False
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == LOG.name
    assert record.msg == "Ignoring invalid %s=%r; using default false"
    assert record.args == (NAME, value)


def test_parsing_is_live_and_other_flag_contracts_remain_distinct(monkeypatch):
    monkeypatch.setenv(NAME, "1")
    assert parse_env_flag_or_false(NAME, logger=LOG) is True
    monkeypatch.setenv(NAME, "unexpected")
    assert parse_env_flag_or_false(NAME, logger=LOG) is False
    assert parse_env_flag(NAME) is True
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(NAME)


@pytest.mark.parametrize("raw, expected", [
    (None, (2, 4)), ("", (2, 4)), ("  ", (2, 4)),
    ("1, 2  4", (1, 2, 4)), ("2,2,4", (2, 2, 4)),
])
def test_capacity_ladder_preserves_defaults_and_equal_adjacent_values(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(NAME, raising=False)
    else:
        monkeypatch.setenv(NAME, raw)
    assert parse_env_capacity_ladder(NAME, ("2", 4)) == expected


@pytest.mark.parametrize("raw", [",", "0,2", "-1,2", "4,2", "two", "1;2"])
def test_capacity_ladder_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv(NAME, raw)
    with pytest.raises(ValueError):
        parse_env_capacity_ladder(NAME, (2, 4))
