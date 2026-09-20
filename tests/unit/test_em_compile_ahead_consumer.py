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
    describe_resident_operand_mismatch,
    resident_half_operand_avals,
    resident_half_operand_presence,
)
from recovar.em.sparse_pass2.sparse_pass2_scoring import (  # noqa: E402
    _relion_powerclass_noise_terms,
    relion_powerclass_noise_dtypes,
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


@pytest.mark.parametrize("real_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("source_faithful", [False, True])
def test_the_powerclass_dtypes_agree_with_the_function_that_produces_them(
    real_dtype, source_faithful
):
    """The two terms do not share a dtype in production.

    `highres_Xi2` keeps the image's real dtype; the norm high-shell term is
    accumulated in float64 under source-faithful normalization. Production runs
    float32 images with that mode on, so predicting the norm term from the
    policy's real dtype is wrong there. It was, and the resident driver's own
    after-the-fact comparison is what reported it, in a run, on 2026-09-20.
    """

    complex_dtype = jnp.complex64 if jnp.dtype(real_dtype) == jnp.float32 else jnp.complex128
    images = jnp.asarray(
        np.random.default_rng(0).normal(size=(3, 8 * 5)), dtype=complex_dtype
    )
    xi2, norm = _relion_powerclass_noise_terms(
        images,
        image_shape=(8, 8),
        current_size=6,
        use_exact_relion_gaussian=True,
        accumulate_noise=True,
        source_faithful_spectrum_norm=source_faithful,
    )
    predicted = relion_powerclass_noise_dtypes(
        real_dtype=real_dtype, source_faithful_spectrum_norm=source_faithful
    )
    assert predicted == (jnp.dtype(xi2.dtype), jnp.dtype(norm.dtype))


def test_the_production_combination_has_two_different_dtypes():
    """Guard the case that actually broke: float32 images, source-faithful on."""

    xi2_dtype, norm_dtype = relion_powerclass_noise_dtypes(
        real_dtype=jnp.float32, source_faithful_spectrum_norm=True
    )
    assert xi2_dtype == jnp.dtype(jnp.float32)
    assert norm_dtype == jnp.dtype(jnp.float64)


def test_the_avals_take_the_norm_dtype_from_the_caller():
    operands = _half_avals(norm_high_shell_dtype=jnp.float64)
    assert operands.relion_norm_high_shell.dtype == jnp.dtype(jnp.float64)
    assert operands.highres_xi2_half.dtype == jnp.dtype(jnp.float32)


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


# ------------------------------------------------------- warming which path ---


def test_the_warm_up_reads_the_same_path_decision_as_the_chunk_loop(monkeypatch):
    """The fused chunk program is opt-in; production runs the per-stage path.

    The warm-up warmed the fused program regardless until 2026-09-20, so under
    the production flag set it compiled three programs per capacity class that
    the loop never called and bought nothing. `chunk_program_path` is the single
    statement both sides now read.
    """

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_JIT", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT", "1")
    assert rp.chunk_program_path() == "fused"

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_JIT", "0")
    assert rp.chunk_program_path() == "per-stage"

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT", "0")
    assert rp.chunk_program_path() == "eager"


def test_the_default_path_is_the_one_production_runs(monkeypatch):
    """No flags set: the fused program is off, so the per-stage path is it."""

    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_JIT", raising=False)
    assert rp._chunk_jit_enabled() is False


def test_nothing_is_queued_when_there_is_no_program_to_warm(monkeypatch):
    """The eager path submits no program, so the warm-up must submit no job."""

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_JIT", "0")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT", "0")

    class Refuse:
        def submit_thunk(self, *a, **k):
            raise AssertionError("the eager path has no program to warm")

    tables, chunks = _chunks()
    assert rp._submit_resident_chunk_warmup(
        Refuse(),
        chunks=chunks,
        tables=tables,
        n_fine_trans=N_FINE_TRANS,
        half_operand_avals=None,
        stage_tables=None,
        carry=None,
        translation_angles=None,
        rect_indices=None,
        exact_positions=None,
        image_shape=(8, 8),
        spec_kwargs={},
        translation_prior_centers_np=None,
        fine_translations=None,
        voxel_size=1.0,
        default_translation_sqdist=None,
    ) == ()


def test_every_program_a_runner_submits_is_one_the_warm_up_warms():
    """The list the warm-up reads must match what the runners actually call.

    The warm-up warmed the fused chunk program while the per-stage runner was
    the one submitting, for a whole commit. Nothing failed; the option simply
    bought nothing, and it took reading a compile census to notice. This reads
    the program names each runner references straight out of its bytecode and
    holds them to `chunk_programs_for_path`, so the next such split fails here.
    """

    def referenced(fn):
        return {n for n in fn.__code__.co_names if n.endswith("_program")}

    for path, runner in (
        ("per-stage", rp._run_resident_chunk_stages),
        ("fused", rp._run_resident_chunk),
    ):
        warmed = {f.__name__ for f in rp.chunk_programs_for_path(path)}
        assert warmed == referenced(runner), (path, warmed, referenced(runner))


def test_the_eager_path_warms_nothing():
    assert rp.chunk_programs_for_path("eager") == ()


def test_an_unknown_path_is_refused_rather_than_silently_warming_nothing():
    with pytest.raises(ValueError, match="unknown chunk program path"):
        rp.chunk_programs_for_path("something-else")


# ------------------------------------------------------- the hit rate's unit ---


class _Spec(str):
    """Stand-in for `_ChunkProgramSpec`: hashable, compares by value."""


def test_a_program_key_is_the_function_and_its_static_argument():
    """Capacity classes alone are not enough to say a warm-up was used.

    Two chunks of one class share a class but key their programs on the whole
    spec, so a spec boolean predicted wrongly gives a warmed program the loop
    never submits while the class looks covered. The key is what the hit rate
    is counted in.
    """

    a, b = _Spec("rows=131072,rfloat=1"), _Spec("rows=131072,rfloat=0")
    keys_a = rp.chunk_program_keys("per-stage", a)
    keys_b = rp.chunk_program_keys("per-stage", b)
    assert len(keys_a) == 3
    assert keys_a.isdisjoint(keys_b), "a different spec must give different keys"
    assert {name for name, _ in keys_a} == {
        f.__name__ for f in rp.chunk_programs_for_path("per-stage")
    }


def test_the_fused_path_has_one_key_and_the_eager_path_none():
    spec = _Spec("s")
    assert len(rp.chunk_program_keys("fused", spec)) == 1
    assert rp.chunk_program_keys("eager", spec) == frozenset()


# --------------------------------------------- the spec prediction's unit ---


class _Operands:
    def __init__(self, direct_ctf_rfloat_recon=None, recon_weight=None):
        self.direct_ctf_rfloat_recon = direct_ctf_rfloat_recon
        self.recon_weight = recon_weight


def test_the_spec_prediction_is_quiet_when_it_agrees():
    message = rp.describe_chunk_spec_prediction(
        predicted_rfloat_ctf_wavg=True,
        predicted_bpref_recon_operand=True,
        predicted_translate_sum_kernel=True,
        operands=_Operands(direct_ctf_rfloat_recon=object(), recon_weight=object()),
    )
    assert message == ""


def test_the_spec_prediction_names_a_wrong_rfloat_ctf_flag():
    """The case that would warm a program the loop never submits.

    Every operand aval can be right and the program still be keyed differently,
    because `use_rfloat_ctf_wavg` is part of the static argument. The operand
    comparison cannot see this; that is why it has its own check.
    """

    message = rp.describe_chunk_spec_prediction(
        predicted_rfloat_ctf_wavg=True,
        predicted_bpref_recon_operand=False,
        predicted_translate_sum_kernel=True,
        operands=_Operands(direct_ctf_rfloat_recon=None),
    )
    assert "use_rfloat_ctf_wavg" in message
    assert "predicted True, really False" in message


def test_the_spec_prediction_names_a_wrong_bpref_flag():
    message = rp.describe_chunk_spec_prediction(
        predicted_rfloat_ctf_wavg=False,
        predicted_bpref_recon_operand=True,
        predicted_translate_sum_kernel=True,
        operands=_Operands(recon_weight=None),
    )
    assert "bpref_recon_operand" in message


def test_the_spec_prediction_says_so_when_no_operands_were_prepared():
    message = rp.describe_chunk_spec_prediction(
        predicted_rfloat_ctf_wavg=False,
        predicted_bpref_recon_operand=False,
        predicted_translate_sum_kernel=True,
        operands=None,
    )
    assert "no resident operands" in message


def test_the_presence_predicate_and_the_spec_check_tell_one_story():
    """What the warm-up predicts must be what the loop reads, by construction.

    The warm-up feeds `resident_half_operand_presence` into the spec, and the
    loop reads the same two facts off the prepared operands. This runs both
    sides of that over the exact-BPref flag and asserts they never disagree.
    """

    for exact_bpref in (False, True):
        presence = resident_half_operand_presence(
            relion_exact_bpref_operands=exact_bpref,
            use_exact_relion_gaussian=True,
            accumulate_noise=True,
            current_size=8,
        )
        operands = _Operands(
            direct_ctf_rfloat_recon=object() if exact_bpref else None,
            recon_weight=object() if exact_bpref else None,
        )
        assert rp.describe_chunk_spec_prediction(
            predicted_rfloat_ctf_wavg=presence.has_direct_ctf_rfloat,
            predicted_bpref_recon_operand=presence.has_recon_weight,
            predicted_translate_sum_kernel=True,
            operands=operands,
        ) == ""


def _spec_kwargs():
    """The driver-level half of the spec, identical for both callers."""

    return dict(
        n_fine_trans=N_FINE_TRANS,
        n_score_pixels=1024,
        n_recon_pixels=512,
        n_rect=2048,
        mstep_block_rows=8192,
        adaptive_fraction=0.999,
        current_size=32,
        mstep_current_size=32,
        image_shape=(64, 64),
        recon_volume_shape=(64, 64, 64),
        max_adjoint_block_bytes=1 << 30,
        stats_config=("stats", 33),
    )


def test_the_warm_ups_spec_equals_the_loops_spec():
    """The whole point of one spec constructor: both callers build one key.

    The warm-up takes the three data-dependent booleans from
    `resident_half_operand_presence` before the preparation; the loop takes them
    from the prepared operands after it. Given a configuration where those agree,
    the two specs must be equal, because the spec is the program's static
    argument and an unequal spec means a warmed program the loop never submits.
    """

    for exact_bpref in (False, True):
        presence = resident_half_operand_presence(
            relion_exact_bpref_operands=exact_bpref,
            use_exact_relion_gaussian=True,
            accumulate_noise=True,
            current_size=32,
        )
        warm = rp._make_chunk_program_spec(
            row_capacity=131072,
            image_capacity=128,
            use_rfloat_ctf_wavg=presence.has_direct_ctf_rfloat,
            use_translate_sum_kernel=True,
            bpref_recon_operand=presence.has_recon_weight,
            **_spec_kwargs(),
        )
        # what the loop builds, from the prepared operands rather than the flags
        loop = rp._make_chunk_program_spec(
            row_capacity=131072,
            image_capacity=128,
            use_rfloat_ctf_wavg=exact_bpref,
            use_translate_sum_kernel=True,
            bpref_recon_operand=exact_bpref,
            **_spec_kwargs(),
        )
        assert warm == loop
        assert hash(warm) == hash(loop)
        assert rp.chunk_program_keys("per-stage", warm) == rp.chunk_program_keys(
            "per-stage", loop
        )


def test_one_wrong_boolean_makes_the_keys_disjoint():
    """So the hit rate, not an assertion, is what reports a mis-predicted spec."""

    right = rp._make_chunk_program_spec(
        row_capacity=131072, image_capacity=128, use_rfloat_ctf_wavg=True,
        use_translate_sum_kernel=True, bpref_recon_operand=True, **_spec_kwargs(),
    )
    wrong = rp._make_chunk_program_spec(
        row_capacity=131072, image_capacity=128, use_rfloat_ctf_wavg=False,
        use_translate_sum_kernel=True, bpref_recon_operand=True, **_spec_kwargs(),
    )
    assert right != wrong
    assert rp.chunk_program_keys("per-stage", right).isdisjoint(
        rp.chunk_program_keys("per-stage", wrong)
    )


def test_the_environment_fields_are_read_by_the_constructor(monkeypatch):
    """The three environment-read fields are why one constructor had to exist.

    Both callers must read them at the same moment. This pins that they come
    from the constructor rather than from either call site, so the only way to
    get two different values is to change the environment mid-iteration.
    """

    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_BLOCK_UNROLL", "1")
    one = rp._make_chunk_program_spec(
        row_capacity=8192, image_capacity=32, use_rfloat_ctf_wavg=False,
        use_translate_sum_kernel=True, bpref_recon_operand=False, **_spec_kwargs(),
    )
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_BLOCK_UNROLL", "2")
    two = rp._make_chunk_program_spec(
        row_capacity=8192, image_capacity=32, use_rfloat_ctf_wavg=False,
        use_translate_sum_kernel=True, bpref_recon_operand=False, **_spec_kwargs(),
    )
    assert one.block_unroll != two.block_unroll
