"""Unit tests for the canonical RECOVAR_DISABLE_CUDA / typo aliasing."""

from __future__ import annotations

import logging

import pytest

pytestmark = [pytest.mark.unit]


def _reset_warn_state():
    from recovar.utils import cuda_env

    cuda_env._warned_typo = False


def test_neither_set_means_custom_cuda(monkeypatch):
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.delenv("RECOVAR_CUDA_DISABLE", raising=False)
    _reset_warn_state()

    from recovar.utils import cuda_env

    disabled, warns = cuda_env.custom_cuda_disabled_from_env()
    assert disabled is False
    assert warns == []


def test_canonical_only_no_warning(monkeypatch, caplog):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.delenv("RECOVAR_CUDA_DISABLE", raising=False)
    _reset_warn_state()

    from recovar.utils import cuda_env

    caplog.set_level(logging.WARNING, logger=cuda_env.logger.name)
    disabled, warns = cuda_env.custom_cuda_disabled_from_env()
    assert disabled is True
    assert warns == []
    assert "alias" not in caplog.text.lower()


def test_typo_only_aliases_with_warning(monkeypatch, caplog):
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setenv("RECOVAR_CUDA_DISABLE", "1")
    _reset_warn_state()

    from recovar.utils import cuda_env

    caplog.set_level(logging.WARNING, logger=cuda_env.logger.name)
    disabled, warns = cuda_env.custom_cuda_disabled_from_env()
    assert disabled is True
    assert warns and "RECOVAR_CUDA_DISABLE" in warns[0]
    assert "RECOVAR_DISABLE_CUDA" in warns[0]
    assert any("alias" in r.message.lower() for r in caplog.records)


def test_typo_zero_means_enabled(monkeypatch):
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setenv("RECOVAR_CUDA_DISABLE", "0")
    _reset_warn_state()

    from recovar.utils import cuda_env

    disabled, _ = cuda_env.custom_cuda_disabled_from_env()
    assert disabled is False


def test_both_set_canonical_wins(monkeypatch, caplog):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "0")  # canonical says enabled
    monkeypatch.setenv("RECOVAR_CUDA_DISABLE", "1")  # typo says disabled
    _reset_warn_state()

    from recovar.utils import cuda_env

    caplog.set_level(logging.WARNING, logger=cuda_env.logger.name)
    disabled, warns = cuda_env.custom_cuda_disabled_from_env()
    # Canonical was "0" (falsy) → custom CUDA is enabled
    assert disabled is False
    assert warns and "Both" in warns[0]


def test_typo_warning_fires_only_once(monkeypatch, caplog):
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setenv("RECOVAR_CUDA_DISABLE", "1")
    _reset_warn_state()

    from recovar.utils import cuda_env

    caplog.set_level(logging.WARNING, logger=cuda_env.logger.name)
    cuda_env.custom_cuda_disabled_from_env()
    cuda_env.custom_cuda_disabled_from_env()
    cuda_env.custom_cuda_disabled_from_env()
    typo_warnings = [r for r in caplog.records if "alias" in r.message.lower()]
    assert len(typo_warnings) == 1
