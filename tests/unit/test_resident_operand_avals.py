"""P4-J: the half's operand avals, without preparing them.

`resident_half_operand_avals` is what lets a pass-2 program be lowered and
compiled before the preparation that fills its operands has run. It is only
useful if it agrees with the real preparation exactly, so the contract is
tested two ways:

* on CPU, that the shapes are the ones the dataclass itself declares and that
  they agree with the byte estimate the driver's admission check already uses;
* on GPU, field for field against `prepare_resident_half_operands` on the same
  inputs, which is the check that would catch a drift the day one of these
  containers gains a field.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.sparse_pass2.resident_operands import (
    ResidentHalfOperands,
    resident_half_operand_avals,
    resident_half_operand_bytes,
)

pytestmark = pytest.mark.unit

SHAPE_ARGS = dict(
    n_images=17,
    n_score_pixels=11,
    n_recon_pixels=7,
    n_half_pixels=29,
    n_fine_trans=5,
)
DTYPES = dict(
    score_complex_dtype=jnp.complex64,
    score_real_dtype=jnp.float32,
    acc_real_dtype=jnp.float32,
    # float64 to match what the admission check assumes for the exact RELION
    # CTF; see the constructor's docstring for why that is not yet settled.
    rfloat_ctf_dtype=jnp.float64,
)

ARRAY_FIELDS = (
    "score_input", "corr_img_score", "highres_xi2_half", "translation_prior",
    "recon_image", "recon_weight", "noise_image", "ctf2_over_nv_recon",
    "direct_ctf_rfloat_recon", "processed_image_half", "relion_norm_high_shell",
    "scale", "group_ids",
)


def _avals(**overrides):
    kwargs = dict(SHAPE_ARGS, **DTYPES, has_recon_weight=True, has_direct_ctf_rfloat=True)
    kwargs.update(overrides)
    return resident_half_operand_avals(**kwargs)


def test_it_returns_avals_and_allocates_nothing():
    operands = _avals()
    assert isinstance(operands, ResidentHalfOperands)
    for name in ARRAY_FIELDS:
        value = getattr(operands, name)
        assert value is None or isinstance(value, jax.ShapeDtypeStruct), name


def test_the_shapes_are_the_ones_the_dataclass_declares():
    """``ResidentHalfOperands.__post_init__`` is the contract; it must accept these."""

    operands = _avals()  # __post_init__ runs on construction and validates
    n = SHAPE_ARGS["n_images"]
    assert operands.score_input.shape == (n, SHAPE_ARGS["n_score_pixels"])
    assert operands.corr_img_score.shape == (n, SHAPE_ARGS["n_score_pixels"])
    assert operands.recon_image.shape == (n, SHAPE_ARGS["n_recon_pixels"])
    assert operands.noise_image.shape == (n, SHAPE_ARGS["n_recon_pixels"])
    assert operands.ctf2_over_nv_recon.shape == (n, SHAPE_ARGS["n_recon_pixels"])
    assert operands.processed_image_half.shape == (n, SHAPE_ARGS["n_half_pixels"])
    assert operands.translation_prior.shape == (n, SHAPE_ARGS["n_fine_trans"])
    for name in ("scale", "group_ids", "highres_xi2_half", "relion_norm_high_shell"):
        assert getattr(operands, name).shape == (n,), name


@pytest.mark.parametrize("field,flag", [
    ("recon_weight", "has_recon_weight"),
    ("direct_ctf_rfloat_recon", "has_direct_ctf_rfloat"),
    ("highres_xi2_half", "has_highres_xi2"),
    ("relion_norm_high_shell", "has_relion_norm_high_shell"),
])
def test_the_optional_operands_follow_their_flag(field, flag):
    assert getattr(_avals(**{flag: True}), field) is not None
    assert getattr(_avals(**{flag: False}), field) is None


def test_the_dtypes_are_the_callers():
    """The precision policy owns them; a second copy would be a second mistake."""

    operands = _avals(score_complex_dtype=jnp.complex128,
                      score_real_dtype=jnp.float64,
                      acc_real_dtype=jnp.float64)
    assert operands.score_input.dtype == jnp.dtype(jnp.complex128)
    assert operands.corr_img_score.dtype == jnp.dtype(jnp.float64)
    assert operands.ctf2_over_nv_recon.dtype == jnp.dtype(jnp.float64)
    # these two are not policy-dependent
    assert operands.scale.dtype == jnp.dtype(jnp.float32)
    assert operands.group_ids.dtype == jnp.dtype(jnp.int32)


def test_the_avals_agree_with_the_byte_estimate():
    """The admission check and the avals must describe the same arrays.

    This is the cross-check that matters: the check must never admit a half
    whose operands are larger than it thought. It caught a real disagreement
    when this function was first written -- 476 bytes on this case -- which is
    the exact-RELION-CTF dtype the constructor's docstring now records.
    """

    operands = _avals()
    counted = 0
    for name in ARRAY_FIELDS:
        value = getattr(operands, name)
        if value is None:
            continue
        counted += int(np.prod(value.shape)) * jnp.dtype(value.dtype).itemsize
    estimate = resident_half_operand_bytes(**SHAPE_ARGS)
    assert counted == estimate, (
        f"the avals sum to {counted} bytes and the admission check estimates "
        f"{estimate}; one of the two is wrong about the operand set"
    )
