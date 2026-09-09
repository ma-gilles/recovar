"""Exact local host results without compiling per-tail device bookkeeping."""

from collections import defaultdict
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_binary_flag

pytestmark = pytest.mark.unit
DEVICE_FIELDS = (
    "best_argmax",
    "batch_norm",
    "log_Z",
    "best_log_score",
    "max_posterior",
    "probs_sum_t",
    "n_significant_samples",
    "reconstruction_sample_mask",
)


def make_case(physical, logical, dtype, optional=True, profile=True):
    rng = np.random.default_rng(3109)
    rows, translations, global_count = 3, 4, 256
    image_ids = np.arange(logical - 1, -1, -1, dtype=np.int32) * 2
    rotation_ids = np.tile(np.array([5, 11, 19], dtype=np.int32), (logical, 1))
    mask = np.tile(np.array([True, True, False]), (logical, 1))
    buffers = engine._LocalPostprocessBuffers(
        hard_assignment=np.full(global_count, -1, dtype=np.int32),
        log_evidence_per_image=np.zeros(global_count, dtype=np.float32),
        best_log_score_per_image=np.zeros(global_count, dtype=np.float32),
        max_posterior_per_image=np.zeros(global_count, dtype=np.float32),
        rotation_posterior_sums=np.linspace(0, 1, 32, dtype=np.float64),
        transfer_profile=defaultdict(float),
        chunk_nonzero_posterior_rows=[],
        chunk_significant_samples=[],
        chunk_reconstruction_rows=[],
        seen_global_rotations=np.zeros(32 if optional else 0, dtype=bool),
        seen_nonzero_global_rotations=np.zeros(32 if optional else 0, dtype=bool),
        seen_reconstruction_global_rotations=np.zeros(32 if optional else 0, dtype=bool),
        significant_counts=np.zeros(global_count, dtype=np.int32) if optional else None,
        best_pose_rotations=np.zeros((global_count, 3, 3), dtype=np.float32) if optional else None,
        best_pose_translations=np.zeros((global_count, 2), dtype=np.float32) if optional else None,
        best_pose_rotation_ids=np.full(global_count, -1, dtype=np.int32) if optional else None,
        reconstruction_sample_indices_by_image=[None] * global_count if optional else None,
    )
    values = dict(
        image_indices=image_ids,
        local_rotation_ids=rotation_ids,
        local_rotation_mask=mask,
        local_rotations=rng.normal(size=(logical, rows, 3, 3)).astype(np.float32),
        local_rotation_posterior_ids=rotation_ids + 1,
        translation_grid=rng.normal(size=(translations, 2)).astype(np.float32),
        n_trans=translations,
        best_argmax=np.arange(physical, dtype=np.int64) % (2 * translations),
        batch_norm=rng.normal(size=(physical, 1)).astype(dtype),
        log_Z=rng.normal(size=physical).astype(dtype),
        best_log_score=rng.normal(size=physical).astype(dtype),
        max_posterior=rng.uniform(size=physical).astype(dtype),
        probs_sum_t=rng.uniform(size=(physical, rows)).astype(dtype),
        n_significant_samples=np.arange(physical, dtype=np.int32),
        reconstruction_sample_mask=rng.uniform(size=(physical, rows, translations)) > 0.5,
        collect_profile_stats=profile,
        reconstruction_row_count=logical * 2,
        reconstruction_take_indices=np.tile(np.array([0, 1]), (logical, 1)),
        reconstruction_pack_mask=np.ones((logical, 2), dtype=bool),
    )
    # Poison unused physical rows: the host path must discard them before math.
    for key in DEVICE_FIELDS:
        value = values[key]
        if value.dtype.kind == "f":
            value[logical:] = np.nan
        elif value.dtype.kind in "iu":
            value[logical:] = -1000
    return values, buffers


def assert_buffers_equal(left, right):
    assert vars(left).keys() == vars(right).keys()
    for key, a in vars(left).items():
        b = getattr(right, key)
        if key == "transfer_profile":
            assert a.keys() == b.keys()
        elif isinstance(a, np.ndarray):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert a.tobytes() == b.tobytes(), key
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                if isinstance(x, np.ndarray):
                    assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), key
                else:
                    assert x == y, key
        else:
            assert a == b, key


@pytest.mark.parametrize("physical,logical", [(42, 32), (42, 42), (75, 73), (96, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("optional,profile", [(False, False), (False, True), (True, False), (True, True)])
def test_all_published_host_buffers_bitwise(physical, logical, dtype, optional, profile):
    values, expected = make_case(physical, logical, dtype, optional, profile)
    actual = copy.deepcopy(expected)
    device = {k: jnp.asarray(v) if k in DEVICE_FIELDS else v for k, v in values.items()}
    legacy = {k: v[:logical] if k in DEVICE_FIELDS else v for k, v in device.items()}
    left = engine._postprocess_local_bucket(**legacy, buffers=expected)
    right = engine._postprocess_local_bucket(**device, buffers=actual, host_prefix=True)
    assert left == right
    assert_buffers_equal(actual, expected)


def test_host_publication_dispatches_no_executables(monkeypatch):
    from jax._src import compiler

    values, buffers = make_case(42, 32, np.float32)
    device = {k: jnp.asarray(v) if k in DEVICE_FIELDS else v for k, v in values.items()}
    jax.block_until_ready(tuple(device[k] for k in DEVICE_FIELDS))
    jax.clear_caches()

    def forbidden(*args, **kwargs):
        raise AssertionError("Host-only publication attempted device compilation")

    monkeypatch.setattr(compiler, "compile_or_get_cached", forbidden)
    engine._postprocess_local_bucket(**device, buffers=buffers, host_prefix=True)


def test_unused_sample_mask_has_no_host_transfer():
    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Unused reconstruction mask transferred")

    values, buffers = make_case(42, 32, np.float32, optional=False, profile=False)
    values["reconstruction_sample_mask"] = Unreadable()
    engine._postprocess_local_bucket(**values, buffers=buffers, host_prefix=True)


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_selector(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_HOST_PUBLICATION_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PUBLICATION_ENV, token)
    assert parse_env_binary_flag(engine.EXACT_LOCAL_HOST_PUBLICATION_ENV) is expected


@pytest.mark.parametrize("token", ["", "true", "-1", "2", "typo"])
def test_invalid_selector(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PUBLICATION_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(engine.EXACT_LOCAL_HOST_PUBLICATION_ENV)


def test_unsupported_mode_rejected_before_dataset_access(monkeypatch):
    monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PUBLICATION_ENV, "1")
    with pytest.raises(ValueError, match="host publication requires"):
        engine.run_local_em_exact(
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=8,
        )
