"""P4-J: the warm-up must describe the programs the chunk loop actually runs.

A compile-ahead warm-up cannot change a result: it hands a helper thread
shape/dtype stand-ins, and a program it describes wrongly is simply never used.
What it can do is waste the compile time it was added to save, silently. These
tests cover the three places that could go wrong without anyone noticing.

1. The chunk inputs are built by one constructor with two placements, on the
   device for the loop and as avals for the warm-up. The two placements must
   agree field for field.
2. The optional operand set is predicted from the configuration flags before
   any image exists, so the prediction must agree with the function that
   decides it while holding the image.
3. The driver compares its prediction against the real operands after the
   preparation. That comparison has to name a real difference and stay quiet
   when there is none.

All CPU: none of this touches a device.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from test_resident_candidates import (  # noqa: E402
    IMAGE_CAPACITY_LADDER,
    ROW_CAPACITY_LADDER,
    _synthetic_tables,
)

import recovar.em.sparse_pass2.resident_pass2 as rp  # noqa: E402
from recovar.em.sparse_pass2.compile_ahead import (  # noqa: E402
    CompileAheadConfig,
    CompileAheadPool,
)
from recovar.em.sparse_pass2.resident_candidates import (  # noqa: E402
    plan_capacity_chunks,
)
from recovar.em.sparse_pass2.resident_operands import (  # noqa: E402
    ResidentHalfOperands,
    describe_resident_operand_mismatch,
    resident_half_operand_avals,
    resident_half_operand_presence,
)
from recovar.em.sparse_pass2.sparse_pass2_scoring import (  # noqa: E402
    _relion_powerclass_noise_terms,
    relion_powerclass_noise_presence,
)

pytestmark = pytest.mark.unit

N_FINE_TRANS = 4


def _chunks():
    tables = _synthetic_tables([3, 5, 2, 7, 1, 4, 6, 2, 1, 1, 9])
    chunks = plan_capacity_chunks(
        tables,
        row_capacity_ladder=ROW_CAPACITY_LADDER,
        image_capacity_ladder=IMAGE_CAPACITY_LADDER,
    )
    return tables, chunks


# ------------------------------------------------- one constructor, two ways ---


def test_the_aval_placement_matches_the_device_placement():
    """The warm-up's row inputs must be the loop's, described.

    Both come out of `_make_chunk_row_arrays`; only the placement differs. If
    a field's aval and its device array ever disagreed, the warm-up would
    compile a program keyed on the wrong signature and the loop would compile
    its own anyway.
    """

    tables, chunks = _chunks()
    for chunk in chunks:
        real = rp._make_chunk_row_arrays(
            tables, chunk, N_FINE_TRANS, place=rp._PLACE_ON_DEVICE
        )
        predicted = rp._make_chunk_row_arrays(
            tables, chunk, N_FINE_TRANS, place=rp._PLACE_AS_AVAL
        )
        for name in type(real)._fields:
            got, want = getattr(predicted, name), getattr(real, name)
            assert tuple(int(d) for d in got.shape) == tuple(int(d) for d in want.shape), name
            assert jnp.dtype(got.dtype) == jnp.dtype(want.dtype), name


def test_the_aval_placement_does_no_device_work():
    """It must be usable before the operands exist, so it cannot allocate."""

    tables, chunks = _chunks()
    predicted = rp._make_chunk_row_arrays(
        tables, chunks[0], N_FINE_TRANS, place=rp._PLACE_AS_AVAL
    )
    for name in type(predicted)._fields:
        value = getattr(predicted, name)
        assert isinstance(value, jax.ShapeDtypeStruct), (name, type(value))


# ------------------------------------------------------ the presence predicate ---


@pytest.mark.parametrize("use_exact_relion_gaussian", [False, True])
@pytest.mark.parametrize("accumulate_noise", [False, True])
@pytest.mark.parametrize("current_size", [None, 6])
@pytest.mark.parametrize("source_faithful", [False, True])
def test_the_powerclass_predicate_agrees_with_the_function_that_decides(
    use_exact_relion_gaussian, accumulate_noise, current_size, source_faithful
):
    """Predicting from the flags must match deciding with the image in hand.

    `_relion_powerclass_noise_terms` returns None for a term the batch does not
    need. The warm-up has to know that before any batch exists, so the rule
    lives in `relion_powerclass_noise_presence` and this asserts the two
    readers of it never diverge.
    """

    images = jnp.asarray(
        np.random.default_rng(0).normal(size=(3, 8 * 5)).astype(np.complex64)
    )
    xi2, norm = _relion_powerclass_noise_terms(
        images,
        image_shape=(8, 8),
        current_size=current_size,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        accumulate_noise=accumulate_noise,
        source_faithful_spectrum_norm=source_faithful,
    )
    predicted = relion_powerclass_noise_presence(
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        accumulate_noise=accumulate_noise,
        current_size=current_size,
    )
    assert predicted == (xi2 is not None, norm is not None)


@pytest.mark.parametrize("exact_bpref", [False, True])
def test_the_bpref_operands_follow_the_exact_bpref_flag(exact_bpref):
    """Both exact-RELION reconstruction operands come from one flag."""

    presence = resident_half_operand_presence(
        relion_exact_bpref_operands=exact_bpref,
        use_exact_relion_gaussian=True,
        accumulate_noise=True,
        current_size=8,
    )
    assert presence.has_recon_weight is exact_bpref
    assert presence.has_direct_ctf_rfloat is exact_bpref
    assert presence.has_highres_xi2 is True
    assert presence.has_relion_norm_high_shell is True


# --------------------------------------------------- the after-the-fact check ---


def _half_avals(**overrides):
    kwargs = dict(
        n_images=5,
        n_score_pixels=11,
        n_recon_pixels=7,
        n_half_pixels=13,
        n_fine_trans=N_FINE_TRANS,
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        acc_real_dtype=jnp.float32,
        has_recon_weight=True,
        has_direct_ctf_rfloat=True,
        has_highres_xi2=True,
        has_relion_norm_high_shell=True,
    )
    kwargs.update(overrides)
    return resident_half_operand_avals(**kwargs)


def test_the_mismatch_report_is_quiet_when_they_agree():
    assert describe_resident_operand_mismatch(_half_avals(), _half_avals()) == ""


def test_the_mismatch_report_names_a_wrong_shape():
    message = describe_resident_operand_mismatch(
        _half_avals(), _half_avals(n_score_pixels=12)
    )
    assert "score_input" in message
    assert "n_score_pixels" in message


def test_the_mismatch_report_names_a_wrong_dtype():
    message = describe_resident_operand_mismatch(
        _half_avals(), _half_avals(score_real_dtype=jnp.float64)
    )
    assert "corr_img_score" in message
    assert "dtype" in message


def test_the_mismatch_report_names_an_operand_that_was_not_produced():
    message = describe_resident_operand_mismatch(
        _half_avals(), _half_avals(has_recon_weight=False)
    )
    assert "recon_weight" in message
    assert "really absent" in message


def test_the_rfloat_ctf_operand_is_float64_as_the_admission_check_assumes():
    """`_relion_exact_ctf_half_from_source_star` places binary64 and the window
    slice does not cast it, so the admission check's 8 bytes is exact rather
    than conservative. A change to either side shows up here."""

    assert _half_avals().direct_ctf_rfloat_recon.dtype == jnp.dtype(jnp.float64)


# ---------------------------------------------------------------- the pool ---


def test_a_thunk_runs_on_the_helper_and_its_result_is_used():
    calls = []

    class FakeProgram:
        def lower(self, *avals, **static):
            calls.append((avals, static))
            return self

        def compile(self):
            return self

    program = FakeProgram()
    aval = jax.ShapeDtypeStruct((3,), jnp.int32)
    with CompileAheadPool(CompileAheadConfig(enabled=True)) as pool:
        assert pool.submit_thunk("one", lambda: (program, (aval,), {"spec": 7}))
    assert pool.summary.compiled == 1
    assert calls == [((aval,), {"spec": 7})]


def test_a_thunk_that_raises_is_recorded_and_dropped():
    def explode():
        raise RuntimeError("no such signature")

    with CompileAheadPool(CompileAheadConfig(enabled=True)) as pool:
        pool.submit_thunk("bad", explode)
    assert pool.summary.failed == 1
    assert pool.summary.compiled == 0
    assert "no such signature" in pool.summary.errors[0]


def test_a_disabled_pool_never_runs_the_thunk():
    ran = []
    with CompileAheadPool(CompileAheadConfig(enabled=False)) as pool:
        assert pool.submit_thunk("one", lambda: ran.append(1)) is False
    assert ran == []
    assert pool.summary.submitted == 0


def test_one_label_is_queued_once():
    with CompileAheadPool(CompileAheadConfig(enabled=True)) as pool:
        first = pool.submit_thunk("same", lambda: (_Noop(), (), {}))
        second = pool.submit_thunk("same", lambda: (_Noop(), (), {}))
    assert first is True and second is False
    assert pool.summary.submitted == 1


class _Noop:
    def lower(self, *avals, **static):
        return self

    def compile(self):
        return self
