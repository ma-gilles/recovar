"""Bitwise tests for the jitted RELION coarse operand assemblies (P3-I).

``RECOVAR_COARSE_OPERAND_PROGRAM=1`` traces the coarse operand assembly once
per batch shape instead of dispatching one compiled program per primitive. The
eager path stays the oracle, so every test here compares the program's output
against the eager function it is ``jax.jit`` of, with
``np.testing.assert_array_equal`` rather than a tolerance.

The three assemblies are the generic coarse sincosf operands, the exact-source
operands, and the normalized-CC (``--firstiter_cc``) tree-rescore operands. The
padded last coarse batch and the inactive-support mask are covered because both
reach the production path that P3-B's ``RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH``
and T18b's fix ``1431919a7`` created.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.relion.relion_coarse_operands import (
    _COARSE_OPERAND_PROGRAM_ENV,
    _coarse_operand_program_enabled,
    _relion_cc_coarse_operand_program,
    _relion_cc_coarse_operands,
    _relion_cc_inverse_power_from_processed,
    _relion_coarse_sincosf_operand_program,
    _relion_coarse_sincosf_operands,
    _relion_exact_coarse_operand_program,
    _relion_exact_coarse_operands,
    _repeat_pad_batch_axis,
    assemble_relion_cc_coarse_operands,
)

IMAGE_SIZE = 16
N_HALF = IMAGE_SIZE * (IMAGE_SIZE // 2 + 1)
N_SCORE = 37
ACTUAL_BATCH = 6
PADDED_BATCH = 8


def _identical(actual, expected, what):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, f"{what}: {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"{what}: {actual.shape} != {expected.shape}"
    np.testing.assert_array_equal(actual, expected, err_msg=what)


def _score_indices(rng):
    return jnp.asarray(
        np.sort(rng.choice(N_HALF, size=N_SCORE, replace=False)).astype(np.int32)
    )


def _active_mask(rng):
    mask = rng.random(N_SCORE) > 0.25
    # The support is never empty and never complete on a real current size.
    mask[0] = True
    mask[-1] = False
    return jnp.asarray(mask)


def _sincosf_inputs(rng, *, batch, complex_dtype, real_dtype, zero_weights=True):
    weights = rng.uniform(0.2, 3.0, (batch, N_HALF))
    if zero_weights:
        # RELION's coarse support leaves exact zeros in ``score_weight_half``;
        # they select the ``safe_weight`` branch of the division.
        weights[:, ::7] = 0.0
    unshifted = rng.normal(size=(batch, N_HALF)) + 1j * rng.normal(size=(batch, N_HALF))
    return (
        jnp.asarray(unshifted, dtype=complex_dtype),
        jnp.asarray(weights, dtype=real_dtype),
        jnp.asarray(rng.uniform(0.5, 1.5, N_HALF), dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    ("complex_dtype", "real_dtype"),
    [(jnp.complex128, jnp.float64), (jnp.complex64, jnp.float32)],
)
def test_sincosf_program_is_bitwise_against_the_eager_assembly(complex_dtype, real_dtype):
    rng = np.random.default_rng(20260920)
    unshifted, weights, half_weights = _sincosf_inputs(
        rng, batch=ACTUAL_BATCH, complex_dtype=complex_dtype, real_dtype=real_dtype
    )
    indices = _score_indices(rng)
    mask = _active_mask(rng)

    eager = _relion_coarse_sincosf_operands(
        unshifted, weights, half_weights, indices, mask
    )
    program = _relion_coarse_sincosf_operand_program(
        unshifted, weights, half_weights, indices, mask
    )
    _identical(program[0], eager[0], "sincosf unshifted_corrected")
    _identical(program[1], eager[1], "sincosf pixel_weight")
    # The dtype pairing the caller reads back off the returned operand.
    assert eager[0].dtype == complex_dtype
    assert eager[1].dtype == real_dtype


def test_sincosf_program_zeroes_the_inactive_support_and_the_zero_weight_rows():
    rng = np.random.default_rng(7)
    unshifted, weights, half_weights = _sincosf_inputs(
        rng, batch=ACTUAL_BATCH, complex_dtype=jnp.complex128, real_dtype=jnp.float64
    )
    indices = _score_indices(rng)
    mask = _active_mask(rng)
    corrected, pixel_weight = _relion_coarse_sincosf_operand_program(
        unshifted, weights, half_weights, indices, mask
    )
    inactive = ~np.asarray(mask)
    assert np.all(np.asarray(corrected)[:, inactive] == 0)
    assert np.all(np.asarray(pixel_weight)[:, inactive] == 0)
    zero_weight = np.asarray(weights)[:, np.asarray(indices)] == 0.0
    assert zero_weight.any()
    assert np.all(np.asarray(corrected)[zero_weight] == 0)


@pytest.mark.parametrize("use_float64_scoring", [False, True])
@pytest.mark.parametrize("scale_corrections_enabled", [False, True])
def test_exact_program_is_bitwise_against_the_eager_assembly(
    use_float64_scoring, scale_corrections_enabled
):
    rng = np.random.default_rng(20260921)
    real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
    complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
    ctf = jnp.asarray(rng.uniform(-1.5, 1.5, (ACTUAL_BATCH, N_SCORE)), dtype=jnp.float64)
    # RELION's CTF crosses zero; the pixel correction has a magnitude guard there.
    ctf = ctf.at[:, ::9].set(0.0)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=real_dtype)
    processed = jnp.asarray(
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF)),
        dtype=complex_dtype,
    )
    indices = _score_indices(rng)
    mask = _active_mask(rng)
    noise = jnp.asarray(rng.uniform(1e3, 1e6, N_HALF), dtype=jnp.float64)
    half_weights = jnp.asarray(rng.uniform(0.5, 1.5, N_HALF), dtype=jnp.float64)

    kwargs = dict(
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        use_float64_scoring=use_float64_scoring,
        scale_corrections_enabled=scale_corrections_enabled,
    )
    eager = _relion_exact_coarse_operands(
        ctf, scale, processed, indices, mask, noise, half_weights, **kwargs
    )
    program = _relion_exact_coarse_operand_program(
        ctf, scale, processed, indices, mask, noise, half_weights, **kwargs
    )
    _identical(program[0], eager[0], "exact unshifted_corrected")
    _identical(program[1], eager[1], "exact pixel_weight")
    assert eager[0].dtype == complex_dtype
    assert eager[1].dtype == real_dtype


def test_exact_program_holds_on_a_repeat_padded_last_batch():
    """The padded coarse batch must give the live rows the unpadded answer.

    ``RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH`` repeats image row zero to fill the
    last batch of a half set. Padding may not move a science row, and the
    program must agree with the eager assembly on the padded extent as well.
    """

    rng = np.random.default_rng(20260922)
    ctf = np.asarray(rng.uniform(0.3, 1.5, (ACTUAL_BATCH, N_SCORE)), dtype=np.float64)
    scale = np.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=np.float32)
    processed = (
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF))
    ).astype(np.complex64)
    indices = _score_indices(rng)
    mask = _active_mask(rng)
    noise = jnp.asarray(rng.uniform(1e3, 1e6, N_HALF), dtype=jnp.float64)
    half_weights = jnp.asarray(rng.uniform(0.5, 1.5, N_HALF), dtype=jnp.float64)
    kwargs = dict(
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        use_float64_scoring=False,
        scale_corrections_enabled=True,
    )

    padded_ctf = jnp.asarray(_repeat_pad_batch_axis(ctf, PADDED_BATCH))
    padded_scale = jnp.asarray(_repeat_pad_batch_axis(scale, PADDED_BATCH))
    padded_processed = jnp.asarray(_repeat_pad_batch_axis(processed, PADDED_BATCH))
    assert padded_ctf.shape == (PADDED_BATCH, N_SCORE)

    unpadded = _relion_exact_coarse_operands(
        jnp.asarray(ctf), jnp.asarray(scale), jnp.asarray(processed),
        indices, mask, noise, half_weights, **kwargs
    )
    padded_eager = _relion_exact_coarse_operands(
        padded_ctf, padded_scale, padded_processed,
        indices, mask, noise, half_weights, **kwargs
    )
    padded_program = _relion_exact_coarse_operand_program(
        padded_ctf, padded_scale, padded_processed,
        indices, mask, noise, half_weights, **kwargs
    )
    for name, i in (("unshifted_corrected", 0), ("pixel_weight", 1)):
        _identical(padded_program[i], padded_eager[i], f"padded exact {name}")
        _identical(
            padded_program[i][:ACTUAL_BATCH], unpadded[i], f"padded exact live rows {name}"
        )
        _identical(
            padded_program[i][ACTUAL_BATCH:],
            np.repeat(np.asarray(unpadded[i][:1]), PADDED_BATCH - ACTUAL_BATCH, axis=0),
            f"padded exact repeated rows {name}",
        )


@pytest.mark.parametrize("with_window", [False, True])
@pytest.mark.parametrize("with_phase_factors", [False, True])
@pytest.mark.parametrize("scale_corrections_enabled", [False, True])
def test_cc_program_is_bitwise_against_the_eager_assembly(
    with_window, with_phase_factors, scale_corrections_enabled
):
    rng = np.random.default_rng(20260923)
    processed = jnp.asarray(
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF)),
        dtype=jnp.complex64,
    )
    ctf = jnp.asarray(rng.uniform(-1.5, 1.5, (ACTUAL_BATCH, N_HALF)), dtype=jnp.float64)
    ctf = ctf.at[:, ::11].set(0.0)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=jnp.float32)
    window = _score_indices(rng) if with_window else None
    inverse_power = _relion_cc_inverse_power_from_processed(processed, window)
    phase = (
        jnp.asarray(
            np.exp(1j * rng.uniform(-np.pi, np.pi, (ACTUAL_BATCH, N_HALF))),
            dtype=jnp.complex64,
        )
        if with_phase_factors
        else None
    )

    eager = _relion_cc_coarse_operands(
        processed, ctf, inverse_power, scale, phase, window,
        scale_corrections_enabled=scale_corrections_enabled,
    )
    program = _relion_cc_coarse_operand_program(
        processed, ctf, inverse_power, scale, phase, window,
        scale_corrections_enabled=scale_corrections_enabled,
    )
    for name in eager._fields:
        _identical(getattr(program, name), getattr(eager, name), f"cc {name}")
    if window is None:
        _identical(eager.windowed_unshifted, eager.unshifted_corrected, "cc unwindowed")
    else:
        assert eager.windowed_unshifted.shape == (ACTUAL_BATCH, N_SCORE)
        assert eager.windowed_corr_img.shape == (ACTUAL_BATCH, N_SCORE)


def test_cc_entry_point_follows_the_flag(monkeypatch):
    """The public entry point selects the program only when the flag is on."""

    rng = np.random.default_rng(20260924)
    processed = jnp.asarray(
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF)),
        dtype=jnp.complex64,
    )
    ctf = jnp.asarray(rng.uniform(0.3, 1.5, (ACTUAL_BATCH, N_HALF)), dtype=jnp.float64)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=jnp.float32)
    inverse_power = _relion_cc_inverse_power_from_processed(processed, None)

    monkeypatch.delenv(_COARSE_OPERAND_PROGRAM_ENV, raising=False)
    assert not _coarse_operand_program_enabled()
    off = assemble_relion_cc_coarse_operands(
        processed, ctf, inverse_power, scale, scale_corrections_enabled=True
    )
    monkeypatch.setenv(_COARSE_OPERAND_PROGRAM_ENV, "1")
    assert _coarse_operand_program_enabled()
    on = assemble_relion_cc_coarse_operands(
        processed, ctf, inverse_power, scale, scale_corrections_enabled=True
    )
    for name in off._fields:
        _identical(getattr(on, name), getattr(off, name), f"cc entry {name}")


def test_coarse_operand_program_flag_fails_closed(monkeypatch):
    monkeypatch.setenv(_COARSE_OPERAND_PROGRAM_ENV, "maybe")
    with pytest.raises(ValueError, match=_COARSE_OPERAND_PROGRAM_ENV):
        _coarse_operand_program_enabled()


def test_each_assembly_is_one_program_per_batch_shape():
    """The point of the flag: one traced program, not one program per primitive.

    The eager path dispatches one compiled program per primitive it runs, so
    the number of equations in the assembly's jaxpr is the number of eager
    dispatches it costs per image batch (before JAX's index renormalization
    multiplies each fancy index by five). Under ``jax.jit`` the same chain is a
    single ``pjit`` equation.
    """

    rng = np.random.default_rng(20260925)
    unshifted, weights, half_weights = _sincosf_inputs(
        rng, batch=ACTUAL_BATCH, complex_dtype=jnp.complex128, real_dtype=jnp.float64
    )
    indices = _score_indices(rng)
    mask = _active_mask(rng)
    args = (unshifted, weights, half_weights, indices, mask)

    eager_jaxpr = jax.make_jaxpr(_relion_coarse_sincosf_operands)(*args)
    program_jaxpr = jax.make_jaxpr(_relion_coarse_sincosf_operand_program)(*args)
    assert len(eager_jaxpr.eqns) >= 10, len(eager_jaxpr.eqns)
    assert [str(eqn.primitive) for eqn in program_jaxpr.eqns] in (["pjit"], ["jit"])

    ctf = jnp.asarray(rng.uniform(0.3, 1.5, (ACTUAL_BATCH, N_HALF)), dtype=jnp.float64)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=jnp.float32)
    processed = jnp.asarray(
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF)),
        dtype=jnp.complex64,
    )
    inverse_power = _relion_cc_inverse_power_from_processed(processed, None)
    cc_args = (processed, ctf, inverse_power, scale, None, None)
    cc_eager = jax.make_jaxpr(
        lambda *a: _relion_cc_coarse_operands(*a, scale_corrections_enabled=True)
    )(*cc_args)
    cc_program = jax.make_jaxpr(
        lambda *a: _relion_cc_coarse_operand_program(*a, scale_corrections_enabled=True)
    )(*cc_args)
    assert len(cc_eager.eqns) >= 8, len(cc_eager.eqns)
    assert [str(eqn.primitive) for eqn in cc_program.eqns] in (["pjit"], ["jit"])
