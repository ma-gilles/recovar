"""Unit tests for the canonical RECOVAR_DISABLE_CUDA env var."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]


def test_neither_set_means_custom_cuda(monkeypatch):
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)

    from recovar.utils import cuda_env

    disabled, warns = cuda_env.custom_cuda_disabled_from_env()
    assert disabled is False
    assert warns == []


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_disable_cuda_truthy_values_disable_custom_cuda(monkeypatch, value):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", value)

    from recovar.utils import cuda_env

    disabled, warns = cuda_env.custom_cuda_disabled_from_env()
    assert disabled is True
    assert warns == []


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_disable_cuda_falsy_values_keep_custom_cuda(monkeypatch, value):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", value)

    from recovar.utils import cuda_env

    disabled, warns = cuda_env.custom_cuda_disabled_from_env()
    assert disabled is False
    assert warns == []
