"""Admission and publication of the private norm carry capacity experiment."""

import ast
import inspect
import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_binary_flag

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_selector(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV, token)
    assert parse_env_binary_flag(engine.EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV) is expected


@pytest.mark.parametrize("token", ["", "true", "false", "2", "-1", "typo"])
def test_invalid_selector(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(engine.EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV)


@pytest.mark.parametrize("stable,deferred,packed,noise", list(itertools.product((False, True), repeat=4))[:-1])
def test_unsupported_mode_fails_before_dataset_access(monkeypatch, stable, deferred, packed, noise):
    monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV, "1")
    with pytest.raises(ValueError, match="noise norm capacity requires"):
        engine.run_local_em_exact(
            None, None, None, None, None, "linear_interp",
            image_batch_size=1, rotation_block_size=1, current_size=8,
            stable_fourier_window_shapes=stable,
            _defer_packed_vdam_enabled=deferred,
            _packed_final_noise_enabled=packed,
            accumulate_noise=noise,
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("n_images", [0, 200, 1000, 1024, 1025])
def test_actual_allocation_and_publication_expressions(enabled, n_images):
    tree = ast.parse(inspect.getsource(engine.run_local_em_exact))
    allocations = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "noise_norm_correction" for t in n.targets)
        and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "jnp.zeros"
    ]
    publications = [
        k.value for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "make_noise_stats"
        for k in n.keywords if k.arg == "wsum_norm_correction"
    ]
    assert len(allocations) == len(publications) == 1
    env = dict(jnp=jnp, n_images=n_images, noise_norm_capacity_enabled=enabled,
               source_faithful_spectrum_norm=True, _noise_norm_capacity=engine._noise_norm_capacity)
    carry = eval(compile(ast.Expression(allocations[0]), "<actual allocation>", "eval"), env)
    assert carry.shape == (engine._noise_norm_capacity(n_images, enabled=enabled),)
    carry = carry.at[:n_images].set(jnp.arange(n_images, dtype=carry.dtype))
    env["noise_norm_correction"] = carry
    published = eval(compile(ast.Expression(publications[0]), "<actual publication>", "eval"), env)
    assert published.shape == (n_images,)
    np.testing.assert_array_equal(published, np.arange(n_images))
    if not enabled:
        assert published is carry
