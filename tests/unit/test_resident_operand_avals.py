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
    # float64 because that is what the source places: relion_ctf.py returns the
    # exact RELION CTF as binary64 and the window slice does not cast it, so the
    # admission check's rfloat_ctf_bytes=8 is exact rather than conservative.
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


def test_the_avals_agree_with_the_byte_estimate_in_the_production_combination():
    """float32 images with source-faithful normalization on: the norm high-shell
    term is float64 while every other real operand is float32. Both the byte
    estimate and the avals have to know that, or the admission check admits a
    half whose operands are four bytes per image larger than it thought."""

    operands = resident_half_operand_avals(
        **SHAPE_ARGS,
        **DTYPES,
        norm_high_shell_dtype=jnp.float64,
        has_recon_weight=True,
        has_direct_ctf_rfloat=True,
    )
    counted = 0
    for name in ARRAY_FIELDS:
        value = getattr(operands, name)
        if value is None:
            continue
        counted += int(np.prod(value.shape)) * jnp.dtype(value.dtype).itemsize
    estimate = resident_half_operand_bytes(**SHAPE_ARGS, norm_high_shell_bytes=8)
    assert counted == estimate, (
        f"the avals sum to {counted} bytes and the admission check estimates "
        f"{estimate} in the production dtype combination"
    )


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
    are really built.

    Their dtypes are settled from the source, not from here: the exact RELION
    CTF is float64 because `relion_ctf.py` places binary64 and the window slice
    does not cast it, and the norm high-shell term follows
    `relion_powerclass_noise_dtypes`. This case runs with source-faithful
    normalization off, so its norm term is float32; the prediction is built with
    that same rule rather than with a literal, so the test still holds if the
    case's flags change.
    """

    from test_resident_operands import N_FINE_TRANS, N_IMAGES, _case, _gpu_case

    from recovar.em.sparse_pass2.resident_operands import (
        ResidentOperandsUnsupported,
        prepare_resident_half_operands,
    )
    from recovar.em.sparse_pass2.sparse_pass2_scoring import (
        relion_powerclass_noise_dtypes,
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
        # The mock dataset is not STAR-backed and the exact RELION CTF is read
        # from the source STAR, so this fixture may not reach the configuration
        # that builds the optional operands. Recorded rather than worked around.
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
        norm_high_shell_dtype=relion_powerclass_noise_dtypes(
            real_dtype=jnp.float32, source_faithful_spectrum_norm=False
        )[1],
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


@pytest.mark.gpu
def test_eval_shape_through_the_real_gather_predicts_the_chunk_operands(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """The recipe a compile-ahead consumer uses, end to end.

    The pass-2 chunk programs take the CHUNK's operands, which
    `gather_resident_chunk_operands` produces from the half's. Rather than
    hand-write a second aval constructor for those -- a second place to drift,
    which is how the 476-byte disagreement above happened -- the consumer runs
    `jax.eval_shape` through the real gather, starting from
    `resident_half_operand_avals`. This test asserts that predicting the chunk
    operands that way gives exactly what gathering the real operands gives.

    It needs a GPU because the gather calls the RELION translate FFI, which
    refuses a non-GPU backend; that is also why the CPU tests above cannot
    cover it.
    """

    import dataclasses

    from test_resident_operands import (
        IMAGE_SHAPE,
        N_IMAGES,
        _case,
        _gpu_case,
        _resident_operands,
    )

    from recovar.em.sparse_pass2.resident_operands import gather_resident_chunk_operands

    _gpu_case(monkeypatch, custom_cuda_lib)
    with jax.default_device(gpu_device):
        case = _case(relion_angles=True)
        real = _resident_operands(case)

        image_capacity = 8
        slots = jnp.asarray(
            np.concatenate([np.arange(min(image_capacity, N_IMAGES)),
                            -np.ones(max(0, image_capacity - N_IMAGES), int)])[:image_capacity],
            dtype=jnp.int32,
        )
        angles = jnp.asarray(case["translation_angles"], dtype=jnp.float32)
        window = jnp.asarray(case["window_indices"], dtype=jnp.int32)

        def gather(operands):
            return gather_resident_chunk_operands(
                operands, slots, translation_angles=angles, rect_indices=window,
                exact_positions=window, image_shape=IMAGE_SHAPE,
            )

        observed = gather(real)

        # the consumer's path: avals only, through the same function
        predicted_half = resident_half_operand_avals(
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
        names = [f.name for f in dataclasses.fields(ResidentHalfOperands)
                 if not f.name.startswith("n_")]
        present = [n for n in names if getattr(predicted_half, n) is not None]
        scalars = {f.name: getattr(predicted_half, f.name)
                   for f in dataclasses.fields(ResidentHalfOperands)
                   if f.name.startswith("n_")}

        def rebuild(arrays):
            kwargs = dict(scalars)
            kwargs.update({n: None for n in names})
            kwargs.update(dict(zip(present, arrays)))
            return ResidentHalfOperands(**kwargs)

        predicted = jax.eval_shape(
            lambda arrays: gather(rebuild(arrays)),
            tuple(getattr(predicted_half, n) for n in present),
        )

    assert set(predicted) == set(observed), (
        f"predicted keys {sorted(set(predicted) ^ set(observed))} differ from the gather's"
    )
    mismatches = []
    for key in sorted(observed):
        got, want = predicted[key], observed[key]
        if (got is None) != (want is None):
            mismatches.append(f"{key}: predicted {got!r}, gathered {want!r}")
            continue
        if want is None:
            continue
        if tuple(int(d) for d in got.shape) != tuple(int(d) for d in want.shape):
            mismatches.append(f"{key}: shape {got.shape} vs {want.shape}")
        if jnp.dtype(got.dtype) != jnp.dtype(want.dtype):
            mismatches.append(f"{key}: dtype {got.dtype} vs {want.dtype}")
    assert not mismatches, (
        "eval_shape through the real gather does not predict the chunk operands:\n  "
        + "\n  ".join(mismatches)
    )


@pytest.mark.gpu
def test_the_optional_operands_against_a_star_backed_preparation(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """The exact-BPref operands, built for real, against the prediction.

    The mock dataset cannot reach this path on its own: the exact RELION CTF is
    read from a source STAR and the mock has none, so the test above skips.
    `RECOVAR_K1_RELION_EXACT_CTF_STAR` is the supported way in, and pointing it
    at a real RELION particles STAR makes the preparation build all four
    optional operands with real CTF evaluation.

    This is what settles their dtypes against a run rather than against a
    reading of the source. It covers both settings of source-faithful
    normalization, because that is the flag on which the two `powerClass` terms
    stop sharing a dtype, and production has it on.

    Two things have to line up for the operands to be built at all: the CTF
    source, which resolves from the dataset's own `particles_file` when that is
    a STAR and otherwise from `RECOVAR_K1_RELION_EXACT_CTF_STAR`, and the RELION
    CUDA Fourier backend, without which the preprocess kwargs do not exist and
    the preparation refuses. A mock dataset supplies neither, so point
    `RECOVAR_P4J_STAR_FIXTURE` at a real STAR and run this where a dataset built
    from it can be used. Unset, the test says so rather than passing quietly.
    """

    import os

    from test_resident_operands import N_FINE_TRANS, N_IMAGES, _case, _gpu_case

    from recovar.em.sparse_pass2.resident_operands import (
        ResidentOperandsUnsupported,
        describe_resident_operand_mismatch,
        prepare_resident_half_operands,
        resident_half_operand_presence,
    )
    from recovar.em.sparse_pass2.sparse_pass2_scoring import (
        relion_powerclass_noise_dtypes,
    )

    star = os.environ.get("RECOVAR_P4J_STAR_FIXTURE", "").strip()
    if not star:
        pytest.skip(
            "set RECOVAR_P4J_STAR_FIXTURE to a RELION particles STAR with an "
            "optics table to run the STAR-backed operand check"
        )
    monkeypatch.setenv("RECOVAR_K1_RELION_EXACT_CTF_STAR", star)
    _gpu_case(monkeypatch, custom_cuda_lib)

    case = _case(relion_angles=True)
    kwargs = dict(case["bucket_io_kwargs"])
    kwargs["relion_exact_bpref_operands"] = True

    for source_faithful in (False, True):
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
                use_exact_relion_gaussian=True,
                accumulate_noise=True,
                source_faithful_spectrum_norm=source_faithful,
                image_batch_size=N_IMAGES,
            )
        except ResidentOperandsUnsupported as exc:
            # Declared refusal. The driver catches this and keeps the per-chunk
            # path, which is the module's stated contract, so there is nothing
            # for this test to compare.
            pytest.xfail(f"the preparation declares this unsupported: {exc}")
        except ValueError as exc:
            # The exact-BPref operands need the RELION CUDA Fourier backend:
            # `prepare_batch_preprocess_operands` produces the preprocess
            # kwargs only when the dataset's preprocess backend is
            # `relion_cuda`, and without them `prepare_unshifted_bucket_operands`
            # refuses. That guard is correct and not specific to the resident
            # path: the per-chunk oracle calls the same helper and refuses the
            # same way. A mock dataset has no such backend, so this fixture
            # cannot reach the operands; a dataset built the way production
            # builds one can, which is what `RECOVAR_P4J_STAR_FIXTURE` is for.
            pytest.xfail(f"this dataset has no RELION CUDA preprocess backend: {exc}")

        presence = resident_half_operand_presence(
            relion_exact_bpref_operands=True,
            use_exact_relion_gaussian=True,
            accumulate_noise=True,
            current_size=case["current_size"],
        )
        built = {
            name: getattr(real, name) is not None
            for name in ("recon_weight", "direct_ctf_rfloat_recon",
                         "highres_xi2_half", "relion_norm_high_shell")
        }
        assert built == {
            "recon_weight": presence.has_recon_weight,
            "direct_ctf_rfloat_recon": presence.has_direct_ctf_rfloat,
            "highres_xi2_half": presence.has_highres_xi2,
            "relion_norm_high_shell": presence.has_relion_norm_high_shell,
        }, f"source_faithful={source_faithful}: the presence predicate is wrong"
        assert all(built.values()), (
            f"source_faithful={source_faithful}: this fixture built only {built}, "
            "so it does not cover the optional operands"
        )

        _, norm_dtype = relion_powerclass_noise_dtypes(
            real_dtype=jnp.float32, source_faithful_spectrum_norm=source_faithful
        )
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
            norm_high_shell_dtype=norm_dtype,
            has_recon_weight=presence.has_recon_weight,
            has_direct_ctf_rfloat=presence.has_direct_ctf_rfloat,
            has_highres_xi2=presence.has_highres_xi2,
            has_relion_norm_high_shell=presence.has_relion_norm_high_shell,
        )
        difference = describe_resident_operand_mismatch(predicted, real)
        assert difference == "", (
            f"source_faithful={source_faithful}: {difference}"
        )

        # the two facts the source reading claimed, now against a real run
        assert jnp.dtype(real.direct_ctf_rfloat_recon.dtype) == jnp.dtype(jnp.float64)
        assert jnp.dtype(real.relion_norm_high_shell.dtype) == jnp.dtype(
            jnp.float64 if source_faithful else jnp.float32
        )
