"""Admission for the diagnostic native residual route in the real engine."""

import itertools

import pytest

from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_binary_flag

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True)])
def test_native_noise_selector(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_NOISE_NATIVE_RESIDUAL_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_NATIVE_RESIDUAL_ENV, token)
    assert parse_env_binary_flag(engine.EXACT_LOCAL_NOISE_NATIVE_RESIDUAL_ENV) is expected


@pytest.mark.parametrize("token", ["", "true", "2", "-1"])
def test_native_noise_invalid_selector(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_NATIVE_RESIDUAL_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(engine.EXACT_LOCAL_NOISE_NATIVE_RESIDUAL_ENV)


@pytest.mark.parametrize(
    "stable,deferred,packed,noise",
    list(itertools.product((False, True), repeat=4))[:-1],
)
def test_native_noise_rejects_unsupported_engine_before_data_access(
    monkeypatch, stable, deferred, packed, noise
):
    monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_NATIVE_RESIDUAL_ENV, "1")
    with pytest.raises(ValueError, match="native noise residual requires"):
        engine.run_local_em_exact(
            None, None, None, None, None, "linear_interp",
            image_batch_size=1, rotation_block_size=1, current_size=8,
            stable_fourier_window_shapes=stable,
            _defer_packed_vdam_enabled=deferred,
            _packed_final_noise_enabled=packed,
            accumulate_noise=noise,
        )
