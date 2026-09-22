"""``RECOVAR_EM_OPERAND_PRECISION_CHECK``: catch operands carried wider than the precision policy.

Production EM precision is float32 (``recovar/em/CLAUDE.md``). The ``DensePrecisionPolicy.cast_*``
helpers narrow the score operands unconditionally but narrow the reconstruction operands only when
float64 scoring is on, so a single float64 factor upstream -- dividing by the binary64 sigma2
spectrum, as the generic K-class path did until 2a63ae96c -- silently promoted the reconstruction and
M-step rows to complex128 and every consumer of them, with no visible effect on results.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.helpers import dtype_policy
from recovar.em.helpers.dtype_policy import (
    DensePrecisionPolicy,
    audit_operand_precision,
    operand_precision_check_mode,
)

F32 = DensePrecisionPolicy(use_float64_scoring=False)
F64 = DensePrecisionPolicy(use_float64_scoring=True)


def _clean_f32():
    return {
        "shifted_score_half": jnp.zeros((2, 4), jnp.complex64),
        "ctf2_over_nv_half": jnp.zeros((2, 4), jnp.float32),
        "shifted_recon_half": jnp.zeros((2, 4), jnp.complex64),
    }


def test_mode_parsing(monkeypatch):
    monkeypatch.delenv(dtype_policy.OPERAND_PRECISION_CHECK_ENV, raising=False)
    assert operand_precision_check_mode() == "warn"
    for mode in ("warn", "raise", "off", "RAISE"):
        monkeypatch.setenv(dtype_policy.OPERAND_PRECISION_CHECK_ENV, mode)
        assert operand_precision_check_mode() == mode.lower()
    monkeypatch.setenv(dtype_policy.OPERAND_PRECISION_CHECK_ENV, "yes")
    with pytest.raises(ValueError):
        operand_precision_check_mode()


def test_float32_policy_accepts_narrow_operands():
    assert audit_operand_precision(F32, _clean_f32(), where="t", mode="raise") == ()


def test_float32_policy_flags_the_reconstruction_leak():
    # exactly the shape of the bug: score operands narrow, reconstruction row promoted
    operands = _clean_f32() | {"shifted_recon_half": jnp.zeros((2, 4), jnp.complex128)}
    offenders = audit_operand_precision(F32, operands, where="t", mode="warn")
    assert offenders == ("shifted_recon_half=complex128",)
    with pytest.raises(ValueError, match="wider than the float32 precision policy"):
        audit_operand_precision(F32, operands, where="t", mode="raise")


def test_float64_reals_are_flagged_too():
    operands = _clean_f32() | {"ctf2_over_nv_half": jnp.zeros((2, 4), jnp.float64)}
    assert audit_operand_precision(F32, operands, where="t", mode="warn") == ("ctf2_over_nv_half=float64",)


def test_float64_policy_accepts_wide_operands():
    operands = {
        "shifted_score_half": jnp.zeros((2, 4), jnp.complex128),
        "ctf2_over_nv_half": jnp.zeros((2, 4), jnp.float64),
    }
    assert audit_operand_precision(F64, operands, where="t", mode="raise") == ()


def test_non_floating_and_missing_operands_are_ignored():
    operands = {
        "indices": jnp.zeros((2, 4), jnp.int64),
        "mask": jnp.zeros((2, 4), bool),
        "absent": None,
        "scalar": 3,
        "host": np.zeros((2, 4), np.float32),
    }
    assert audit_operand_precision(F32, operands, where="t", mode="raise") == ()


def test_off_mode_reports_nothing():
    operands = _clean_f32() | {"shifted_recon_half": jnp.zeros((2, 4), jnp.complex128)}
    assert audit_operand_precision(F32, operands, where="t", mode="off") == ()


def test_warn_mode_reports_each_violation_once(caplog):
    dtype_policy._REPORTED_PRECISION_VIOLATIONS.clear()
    operands = _clean_f32() | {"shifted_recon_half": jnp.zeros((2, 4), jnp.complex128)}
    with caplog.at_level("WARNING", logger=dtype_policy.logger.name):
        audit_operand_precision(F32, operands, where="w", mode="warn")
        audit_operand_precision(F32, operands, where="w", mode="warn")
    assert sum("EM operand precision" in r.message for r in caplog.records) == 1


def test_audit_catches_the_generic_k_class_noise_promotion():
    """The exact arithmetic that leaked: float32 CTF divided by a binary64 sigma2 spectrum.

    RELION's model STAR supplies sigma2 in binary64, so the pre-2a63ae96c generic K-class path
    (``ctf_half ** 2 / noise_variance_half`` and ``processed * ctf_half / noise_variance_half``)
    produced float64 / complex128 rows under a float32 policy. The fix forms RELION's own XFLOAT
    reciprocal first, which keeps the accumulation dtype.
    """
    from recovar.em.sparse_pass2.sparse_pass2_bucket_io import (
        _ctf_over_noise_weighted_pair,
        _ctf2_over_noise_and_ctf2,
        _weighted_ctf_pair,
    )

    rng = np.random.default_rng(0)
    n_images, n_pixels = 3, 16
    ctf = jnp.asarray(rng.random((n_images, n_pixels)), jnp.float32)
    noise_f64 = jnp.asarray(rng.random(n_pixels) + 0.5, jnp.float64)  # RELION model STAR precision
    images = jnp.asarray(
        rng.normal(size=(n_images, n_pixels)) + 1j * rng.normal(size=(n_images, n_pixels)), jnp.complex64
    )

    # pre-fix: divide by the binary64 spectrum
    leaked_ratio, _ = _ctf2_over_noise_and_ctf2(ctf, noise_f64)
    leaked_score, leaked_recon = _ctf_over_noise_weighted_pair(images, images, ctf, noise_f64)
    assert leaked_ratio.dtype == jnp.float64 and leaked_score.dtype == jnp.complex128
    offenders = audit_operand_precision(
        F32,
        {"ctf2_over_nv_half": leaked_ratio, "shifted_recon_half": leaked_recon},
        where="t",
        mode="warn",
    )
    assert set(offenders) == {"ctf2_over_nv_half=float64", "shifted_recon_half=complex128"}

    # post-fix: RELION's XFLOAT reciprocal, then multiply
    inverse_noise = jnp.reciprocal(noise_f64).astype(jnp.float32)
    weighted_ctf = ctf * inverse_noise[None, :]
    fixed_ratio = weighted_ctf * ctf
    fixed_score, fixed_recon = _weighted_ctf_pair(images, images, weighted_ctf)
    assert fixed_ratio.dtype == jnp.float32 and fixed_score.dtype == jnp.complex64
    assert audit_operand_precision(
        F32,
        {"ctf2_over_nv_half": fixed_ratio, "shifted_recon_half": fixed_recon},
        where="t",
        mode="raise",
    ) == ()

    # and the two agree to float32 tolerance, so the narrowing is the only change
    np.testing.assert_allclose(np.asarray(fixed_ratio), np.asarray(leaked_ratio), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fixed_recon), np.asarray(leaked_recon), rtol=1e-6)
