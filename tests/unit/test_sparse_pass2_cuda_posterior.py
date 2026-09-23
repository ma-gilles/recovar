"""Fused CUDA sparse pass-2 posterior: wrapper contracts, reference and kernel parity."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.em.cuda import kernels as em_cuda_kernels
from recovar.em.sparse_pass2 import sparse_pass2_posterior as posterior

pytestmark = pytest.mark.unit

_ENV = posterior._SPARSE_PASS2_CUDA_POSTERIOR_ENV


def reference_fused_posterior(scores, log_z, *, adaptive_fraction, keep_all=False, external_sum_weight=None):
    """Independent NumPy statement of the fused semantics (sequential float32 scan).

    Order-free outputs (max, argmax, masks, counts, pass-through log-Z) are
    exact; the float32 cumulative sum is sequential, so threshold-adjacent
    quantities may differ from a CUB scan at the last float32 bit.
    """

    scores = np.asarray(scores, dtype=np.float32)
    rows = scores.shape[0]
    flat = scores.reshape(rows, -1)
    n = flat.shape[1]
    finite = np.isfinite(flat)
    masked = np.where(finite, flat, -np.inf).astype(np.float32)
    best = masked.max(axis=1)
    argmax = masked.argmax(axis=1)
    has_finite = np.isfinite(best)
    log_z = np.asarray(log_z, dtype=np.float64)
    has_finite_norm = has_finite & np.isfinite(log_z)
    safe_log_z = np.where(has_finite_norm, log_z, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        probs = np.exp(masked.astype(np.float64) - safe_log_z[:, None])
    probs = np.where(has_finite_norm[:, None] & np.isfinite(probs), probs, 0.0)
    safe_best = np.where(has_finite, best, np.float32(0.0)).astype(np.float32)
    exponent_add = (np.float32(50.0) - safe_best).astype(np.float32)
    exponent = (masked + exponent_add[:, None]).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.where(exponent < np.float32(-88.0), np.float32(0.0), np.exp(exponent)).astype(np.float32)
    raw = np.where(finite & np.isfinite(raw), raw, np.float32(0.0)).astype(np.float32)
    sorted_raw = np.sort(raw, axis=1)
    cumulative = np.empty_like(sorted_raw)
    for r in range(rows):
        acc = np.float32(0.0)
        for i in range(n):
            acc = np.float32(acc + sorted_raw[r, i])
            cumulative[r, i] = acc
    fine_sum = cumulative[:, -1]
    sum_weight = fine_sum if external_sum_weight is None else np.asarray(external_sum_weight, dtype=np.float32)
    has_mass = has_finite & np.isfinite(sum_weight) & (sum_weight > 0)
    if keep_all:
        threshold = np.zeros(rows, dtype=np.float32)
        mask = has_mass[:, None] & finite & (raw > 0)
    else:
        parsed = np.float32(adaptive_fraction)
        target = ((1.0 - np.float64(parsed)) * fine_sum.astype(np.float64)).astype(np.float32)
        idx = np.array([np.searchsorted(cumulative[r], target[r], side="right") for r in range(rows)])
        idx = np.minimum(idx, n - 1)
        threshold = sorted_raw[np.arange(rows), idx]
        mask = has_mass[:, None] & finite & (raw >= threshold[:, None])
    safe_sum = np.where(has_mass, sum_weight, np.float32(1.0)).astype(np.float32)
    normalized = (raw / safe_sum[:, None]).astype(np.float32)
    recon = np.where(mask, normalized, np.float32(0.0)).astype(np.float32)
    shape = scores.shape
    return dict(
        log_z=safe_log_z,
        probs=probs.reshape(shape),
        best_log_score=np.where(has_finite_norm, best, -np.inf).astype(np.float32),
        best_argmax=np.where(has_finite_norm, argmax, 0).astype(np.int64),
        max_posterior=recon.max(axis=1).astype(np.float32),
        normalized_weights=normalized.reshape(shape),
        reconstruction_probs=recon.reshape(shape),
        mask=mask.reshape(shape),
        n_significant=mask.sum(axis=1).astype(np.int32),
        sum_weight=sum_weight.astype(np.float32),
        threshold=threshold.astype(np.float32),
    )


def reference_log_z(scores):
    scores = np.asarray(scores, dtype=np.float32).reshape(np.shape(scores)[0], -1)
    masked = np.where(np.isfinite(scores), scores, -np.inf).astype(np.float32)
    best = masked.max(axis=1)
    has_finite = np.isfinite(best)
    safe_best = np.where(has_finite, best, np.float32(0.0)).astype(np.float32)
    shifted = np.where(has_finite[:, None], (masked - safe_best[:, None]).astype(np.float32), -np.inf)
    total = np.exp(shifted.astype(np.float64)).sum(axis=1)
    has_mass = has_finite & (total > 0) & np.isfinite(total)
    with np.errstate(divide="ignore"):
        return np.where(has_mass, safe_best.astype(np.float64) + np.log(np.where(has_mass, total, 1.0)), -np.inf)


def make_scores(shape, seed, *, all_inf_row=False, nan=False):
    rng = np.random.default_rng(seed)
    scores = (rng.normal(size=shape) * 30 - 200).astype(np.float32)
    pad = rng.random(shape) < 0.3
    scores[pad] = -np.inf
    if nan:
        scores[tuple(0 for _ in shape)] = np.nan
    if all_inf_row and shape[0] > 1:
        scores[1] = -np.inf
    # Guarantee a finite winner somewhere in every other row.
    for r in range(shape[0]):
        if all_inf_row and r == 1:
            continue
        flat = scores[r].reshape(-1)
        if not np.isfinite(flat).any():
            flat[0] = -150.0
    return scores


def test_env_gate_default_off(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert posterior.sparse_pass2_cuda_posterior_enabled() is False
    monkeypatch.setenv(_ENV, "1")
    assert posterior.sparse_pass2_cuda_posterior_enabled() is True
    monkeypatch.setenv(_ENV, "off")
    assert posterior.sparse_pass2_cuda_posterior_enabled() is False


def test_row_state_bytes_match_header():
    header = os.path.join(os.path.dirname(cb.__file__), "em", "cuda", "sparse_pass2_posterior.cuh")
    body = open(header).read().split("struct RowState", 1)[1].split("};", 1)[0]
    sizes = {"float": 4, "double": 8, "int": 4}
    fields = [line.split()[0] for line in body.splitlines() if line.strip() and line.strip()[0] not in "{/"]
    total = 0
    for kind in fields:
        size = sizes[kind]
        total = (total + size - 1) // size * size + size
    total = (total + 7) // 8 * 8
    assert total == em_cuda_kernels._SPARSE_PASS2_ROW_STATE_BYTES


@pytest.mark.parametrize(
    "case",
    ["dtype", "rank", "empty", "log_z_dtype", "log_z_shape", "external_dtype", "static_bool"],
)
def test_wrapper_rejects_bad_operands(case):
    scores = jnp.zeros((2, 3, 4), jnp.float32)
    log_z = jnp.zeros((2,), jnp.float64)
    external = jnp.ones((2,), jnp.float32)
    kwargs = dict(adaptive_fraction=0.999, keep_all=False, use_external_sum_weight=False)
    if case == "dtype":
        scores = scores.astype(jnp.float64)
    if case == "rank":
        scores = jnp.zeros((5,), jnp.float32)
    if case == "empty":
        scores = jnp.zeros((2, 0, 4), jnp.float32)
    if case == "log_z_dtype":
        log_z = log_z.astype(jnp.float32)
    if case == "log_z_shape":
        log_z = jnp.zeros((3,), jnp.float64)
    if case == "external_dtype":
        external = external.astype(jnp.float64)
    if case == "static_bool":
        kwargs["keep_all"] = 1
    with pytest.raises((TypeError, ValueError)):
        em_cuda_kernels.sparse_pass2_posterior_f32(scores, log_z, external, **kwargs)


def test_wrapper_requires_gpu_backend():
    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only contract")
    scores = jnp.zeros((2, 3, 4), jnp.float32)
    with pytest.raises(RuntimeError, match="GPU backend"):
        em_cuda_kernels.sparse_pass2_log_z_f64(scores)
    with pytest.raises(RuntimeError, match="GPU backend"):
        posterior.cuda_fused_pass2_posterior(scores, jnp.zeros((2,), jnp.float64), adaptive_fraction=0.999)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (4, 33, 12)])
@pytest.mark.parametrize("variant", ["plain", "nan_and_inf_row", "external", "keep_all"])
def test_reference_matches_xla_semantics(shape, variant):
    """The NumPy reference agrees with the XLA glue it restates (any backend)."""

    scores = make_scores(shape, 11, all_inf_row=variant == "nan_and_inf_row", nan=variant == "nan_and_inf_row")
    log_z = np.array(posterior._logsumexp_pass2_bucket_score_only(jnp.asarray(scores)), dtype=np.float64)
    if variant == "nan_and_inf_row" and shape[0] > 2:
        log_z[2] = np.inf
    external = None if variant != "external" else (np.abs(np.random.default_rng(3).normal(size=shape[0])) + 0.5).astype(np.float32)
    keep_all = variant == "keep_all"
    ref = reference_fused_posterior(scores, log_z, adaptive_fraction=0.999, keep_all=keep_all, external_sum_weight=external)
    x_log_z, x_probs, x_best, x_argmax, _ = posterior._normalize_pass2_bucket_with_log_z(jnp.asarray(scores), jnp.asarray(log_z))
    np.testing.assert_array_equal(np.asarray(x_log_z), ref["log_z"])
    np.testing.assert_array_equal(np.asarray(x_best), ref["best_log_score"])
    np.testing.assert_array_equal(np.asarray(x_argmax), ref["best_argmax"])
    np.testing.assert_allclose(np.asarray(x_probs), ref["probs"], rtol=1e-14, atol=0)
    full = posterior._relion_f32_fine_posterior(
        jnp.asarray(scores),
        adaptive_fraction=0.999,
        normalization_sum_weight=None if external is None else jnp.asarray(external),
        keep_all=keep_all,
    )
    normalized, recon, mask, n_sig, sum_weight, threshold = (np.asarray(v) for v in full)
    # XLA CPU flushes float32 denormals; allow that below the normal range.
    np.testing.assert_allclose(normalized, ref["normalized_weights"], rtol=2e-6, atol=1.2e-38)
    np.testing.assert_allclose(sum_weight, ref["sum_weight"], rtol=2e-6, atol=0)
    # Discrete outputs agree except at a scan-rounding boundary candidate.
    assert np.abs(n_sig.astype(np.int64) - ref["n_significant"].astype(np.int64)).max() <= 1
    disagreements = int((mask != ref["mask"]).sum())
    assert disagreements <= shape[0]
    xla_log_z = np.asarray(posterior._logsumexp_pass2_bucket_score_only(jnp.asarray(scores)))
    np.testing.assert_array_equal(np.isfinite(reference_log_z(scores)), np.isfinite(xla_log_z))
    finite = np.isfinite(xla_log_z)
    np.testing.assert_allclose(reference_log_z(scores)[finite], xla_log_z[finite], rtol=1e-14, atol=0)


def _gpu_library_has(symbol):
    cb._ensure_ffi()
    return hasattr(cb._get_lib(), symbol)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (4, 257, 84), (2, 1000, 3), (6, 4096, 21)])
@pytest.mark.parametrize("variant", ["plain", "nan_and_inf_row", "external", "keep_all", "ties"])
def test_kernel_matches_xla_gpu_path(shape, variant):
    """Fused kernel versus the XLA + CUDA-primitive path, bitwise on every output but log-Z."""

    assert jax.default_backend() == "gpu"
    if not _gpu_library_has("SparsePass2PosteriorF32"):
        pytest.skip("loaded CUDA library lacks SparsePass2PosteriorF32")
    assert cb.custom_cuda_requested()
    scores = make_scores(shape, 5, all_inf_row=variant == "nan_and_inf_row", nan=variant == "nan_and_inf_row")
    if variant == "ties":
        scores[:, 0] = scores.reshape(shape[0], -1).max(axis=1)[:, None]
    log_z = np.array(posterior._logsumexp_pass2_bucket_score_only(jnp.asarray(scores)), dtype=np.float64)
    if variant == "nan_and_inf_row" and shape[0] > 2:
        log_z[2] = np.inf
    external = None if variant != "external" else (np.abs(np.random.default_rng(3).normal(size=shape[0])) + 0.5).astype(np.float32)
    keep_all = variant == "keep_all"

    fused = posterior.cuda_fused_pass2_posterior(
        jnp.asarray(scores),
        jnp.asarray(log_z),
        adaptive_fraction=0.999,
        normalization_sum_weight=None if external is None else jnp.asarray(external),
        keep_all=keep_all,
    )
    fused = {k: np.asarray(v) for k, v in fused._asdict().items()}
    x_log_z, x_probs, x_best, x_argmax, _ = posterior._normalize_pass2_bucket_with_log_z(jnp.asarray(scores), jnp.asarray(log_z))
    normalized, recon, mask, n_sig, sum_weight, threshold = posterior._relion_f32_fine_posterior(
        jnp.asarray(scores),
        adaptive_fraction=0.999,
        normalization_sum_weight=None if external is None else jnp.asarray(external),
        keep_all=keep_all,
    )
    expected = dict(
        log_z=x_log_z,
        probs=x_probs,
        best_log_score=x_best,
        best_argmax=x_argmax,
        max_posterior=jnp.max(recon.reshape(recon.shape[0], -1), axis=1),
        normalized_weights=normalized,
        reconstruction_probs=recon,
        mask=mask,
        n_significant=n_sig,
        sum_weight=sum_weight,
        threshold=threshold,
    )
    for key, value in expected.items():
        value = np.asarray(value)
        assert fused[key].dtype == value.dtype, key
        assert fused[key].shape == value.shape, key
        np.testing.assert_array_equal(fused[key], value, err_msg=key)
    ref = reference_fused_posterior(scores, log_z, adaptive_fraction=0.999, keep_all=keep_all, external_sum_weight=external)
    for key in ("best_log_score", "best_argmax", "log_z"):
        np.testing.assert_array_equal(fused[key], ref[key], err_msg=key)

    kernel_log_z = np.asarray(posterior.cuda_logsumexp_pass2_bucket_score_only(jnp.asarray(scores)))
    xla_log_z = np.asarray(posterior._logsumexp_pass2_bucket_score_only(jnp.asarray(scores)))
    assert kernel_log_z.dtype == np.float64 and kernel_log_z.shape == xla_log_z.shape
    np.testing.assert_array_equal(np.isfinite(kernel_log_z), np.isfinite(xla_log_z))
    finite = np.isfinite(xla_log_z)
    np.testing.assert_allclose(kernel_log_z[finite], xla_log_z[finite], rtol=1e-14, atol=0)
