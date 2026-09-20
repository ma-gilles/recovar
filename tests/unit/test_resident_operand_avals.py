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


# --------------------------------------------------- against the real thing ---


@pytest.mark.gpu
def test_the_avals_match_the_real_preparation(monkeypatch, custom_cuda_lib, gpu_device):
    """Field for field against `prepare_resident_half_operands` on one case.

    This is the check that decides the exact-RELION-CTF dtype the CPU tests
    could only hold to a consistent story, and the one that would catch a drift
    the day `ResidentHalfOperands` gains a field.
    """

    from test_resident_operands import _case, _gpu_case, _resident_operands

    _gpu_case(monkeypatch, custom_cuda_lib)
    case = _case(relion_angles=True)
    real = _resident_operands(case)

    predicted = resident_half_operand_avals(
        n_images=real.n_images,
        n_score_pixels=real.n_score_pixels,
        n_recon_pixels=real.n_recon_pixels,
        n_half_pixels=real.n_half_pixels,
        n_fine_trans=real.n_fine_trans,
        # The dtypes a real caller has: the float32 precision policy this case
        # runs under, plus the float64 the admission check assumes for the
        # exact RELION CTF. Taking them from `real` instead would make the
        # dtype half of this test circular, which the first version of it was.
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        acc_real_dtype=jnp.float32,
        rfloat_ctf_dtype=jnp.float64,
        has_recon_weight=real.recon_weight is not None,
        has_direct_ctf_rfloat=real.direct_ctf_rfloat_recon is not None,
        has_highres_xi2=real.highres_xi2_half is not None,
        has_relion_norm_high_shell=real.relion_norm_high_shell is not None,
    )

    mismatches = []
    observed = {
        name: (None if getattr(real, name) is None
               else (tuple(int(d) for d in getattr(real, name).shape),
                     jnp.dtype(getattr(real, name).dtype).name))
        for name in ARRAY_FIELDS
    }
    print("real operands:", observed)
    for name in ARRAY_FIELDS:
        got, want = getattr(predicted, name), getattr(real, name)
        if (got is None) != (want is None):
            mismatches.append(f"{name}: predicted {got!r}, real {want!r}")
            continue
        if want is None:
            continue
        if tuple(int(d) for d in got.shape) != tuple(int(d) for d in want.shape):
            mismatches.append(f"{name}: shape {got.shape} vs {want.shape}")
        if jnp.dtype(got.dtype) != jnp.dtype(want.dtype):
            mismatches.append(f"{name}: dtype {got.dtype} vs {want.dtype}")
    assert not mismatches, "the avals disagree with the preparation:\n  " + "\n  ".join(mismatches)

    # every array field of the dataclass is covered by ARRAY_FIELDS, so a new
    # field cannot slip past this test unnoticed
    import dataclasses

    declared = {
        f.name for f in dataclasses.fields(ResidentHalfOperands)
        if not f.name.startswith("n_")
    }
    assert declared == set(ARRAY_FIELDS), (
        f"ResidentHalfOperands has fields this test does not check: "
        f"{sorted(declared - set(ARRAY_FIELDS))}"
    )


@pytest.mark.gpu
def test_the_optional_operands_dtypes_against_the_real_preparation(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """The exact-BPref configuration, where the four optional operands exist.

    The case the test above uses produces none of them, so that test covers the
    shape derivation and the required fields only. This one turns on the exact
    RELION BPref operands and the noise accumulation so `recon_weight`,
    `direct_ctf_rfloat_recon`, `highres_xi2_half` and `relion_norm_high_shell`
    are really built, which is what settles their dtypes -- in particular
    whether the exact RELION CTF is the float64 the admission check assumes.
    """

    from test_resident_operands import N_FINE_TRANS, N_IMAGES, _case, _gpu_case
    from recovar.em.sparse_pass2.resident_operands import (
        ResidentOperandsUnsupported,
        prepare_resident_half_operands,
    )

    _gpu_case(monkeypatch, custom_cuda_lib)
    case = _case(relion_angles=True)
    kwargs = dict(case["bucket_io_kwargs"])
    kwargs["relion_exact_bpref_operands"] = True
    try:
        real = prepare_resident_half_operands(
            case["dataset"],
            np.arange(N_IMAGES),
            bucket_io_kwargs=kwargs,
            window_indices=case["window_indices"],
            recon_window_indices=case["window_indices"],
            image_shape=case["image_shape"],
            current_size=case["current_size"],
            n_fine_trans=N_FINE_TRANS,
            use_exact_relion_gaussian=False,
            accumulate_noise=True,
            source_faithful_spectrum_norm=False,
            image_batch_size=4,
        )
    except (ResidentOperandsUnsupported, NotImplementedError, ValueError) as exc:
        # The mock dataset is not STAR-backed, and the exact RELION CTF is read
        # from the source STAR, so this fixture cannot reach the configuration
        # that builds the optional operands. Recorded rather than worked around:
        # their dtypes, and with them whether the exact RELION CTF really is the
        # float64 the admission check assumes, are settled by a STAR-backed
        # fixture and not by anything in this file.
        pytest.skip(f"this fixture cannot build the exact-BPref operands: {exc}")

    present = {
        name: jnp.dtype(getattr(real, name).dtype).name
        for name in ("recon_weight", "direct_ctf_rfloat_recon",
                     "highres_xi2_half", "relion_norm_high_shell")
        if getattr(real, name) is not None
    }
    print("optional operands built by this case:", present)
    if not present:
        pytest.skip("this configuration still produces none of the optional operands")

    predicted = resident_half_operand_avals(
        n_images=real.n_images,
        n_score_pixels=real.n_score_pixels,
        n_recon_pixels=real.n_recon_pixels,
        n_half_pixels=real.n_half_pixels,
        n_fine_trans=real.n_fine_trans,
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        acc_real_dtype=jnp.float32,
        rfloat_ctf_dtype=jnp.float64,
        has_recon_weight=real.recon_weight is not None,
        has_direct_ctf_rfloat=real.direct_ctf_rfloat_recon is not None,
        has_highres_xi2=real.highres_xi2_half is not None,
        has_relion_norm_high_shell=real.relion_norm_high_shell is not None,
    )
    mismatches = []
    for name in present:
        got, want = getattr(predicted, name), getattr(real, name)
        if tuple(int(d) for d in got.shape) != tuple(int(d) for d in want.shape):
            mismatches.append(f"{name}: shape {got.shape} vs {want.shape}")
        if jnp.dtype(got.dtype) != jnp.dtype(want.dtype):
            mismatches.append(f"{name}: dtype {got.dtype} vs {want.dtype}")
    assert not mismatches, "the optional operands disagree:\n  " + "\n  ".join(mismatches)
