"""P3-G/H/J: the local-search glue folds, held bitwise against the path they replace.

Ticket: ``em_parity_tickets_20260918/P3G_local_glue_programs.md``.

Three opt-in changes, each measured by P3-E's census at the local state D of the
10k/256 matched harness and each with the current path kept as its oracle:

* **G** hands :func:`recovar.em.sparse_pass2.resident_pass2.run_resident_mstep_blocks`
  the chunk's whole row arrays instead of a Python callback that slices a block
  out of them per block (2466 eager dispatches per local iteration);
* **H** trims a bucket's per-image rows in one program instead of fourteen
  single-primitive ``dynamic_slice`` programs (1.11 s of trace/lower/compile
  **per local iteration**, because ``run_local_em_exact`` calls
  ``jax.clear_caches()`` at the end of every bucket);
* **J** builds a bucket's ten loop-invariant constant operands in one program
  instead of ten eager ``jnp.zeros``/``jnp.full`` calls, each of which
  dispatches a ``convert_element_type`` and a ``broadcast_in_dim`` (1620
  dispatches per local iteration).

Every assertion here is bitwise (``assert_array_equal`` plus dtype and shape),
not a tolerance: none of the three changes touches an arithmetic expression, so
a single differing bit is a defect and not rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.local import local_bucket_stages as lbs
from recovar.em.sparse_pass2 import resident_pass2 as rp

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# G: the M-step block program reads the chunk's rows instead of a callback
# --------------------------------------------------------------------------


def _chunk_rows(row_capacity=16, n_recon=5, seed=20260920):
    rng = np.random.default_rng(seed)
    proj = (
        rng.standard_normal((row_capacity, n_recon))
        + 1j * rng.standard_normal((row_capacity, n_recon))
    ).astype(np.complex64)
    proj_abs2 = rng.standard_normal((row_capacity, n_recon)).astype(np.float32)
    rotations = rng.standard_normal((row_capacity, 3, 3)).astype(np.float32)
    return jnp.asarray(proj), jnp.asarray(proj_abs2), jnp.asarray(rotations)


class _Spec:
    """The two fields ``_resident_mstep_block_at`` reads off the spec."""

    def __init__(self, mstep_block_rows):
        self.mstep_block_rows = int(mstep_block_rows)


def _tables_with_chunk_rows(proj, proj_abs2, rotations):
    fields = {name: None for name in rp._ChunkStageTables._fields}
    fields["projection_recon_cache"] = proj
    fields["projection_recon_abs2_cache"] = proj_abs2
    fields["mstep_grid"] = rotations
    return rp._ChunkStageTables(**fields)


def _record_block_projections(monkeypatch):
    seen = []

    def recorder(*, block_projections, carry, **kwargs):
        seen.append(tuple(np.asarray(value) for value in block_projections))
        seen[-1] = seen[-1] + (
            np.asarray(kwargs["block_row_image"]),
            np.asarray(kwargs["block_kernel_ids"]),
            np.asarray(kwargs["block_posterior"]),
        )
        return carry

    monkeypatch.setattr(rp, "_resident_mstep_block", recorder)
    return seen


def test_chunk_row_arrays_and_a_block_offset_read_the_rows_the_callback_sliced(
    monkeypatch,
):
    """The identity gather of a block's row ids is that block's Python slice.

    ``_resident_mstep_block_at`` already slices the chunk's row ids with
    ``dynamic_slice_in_dim`` and reads the global pass's caches at them. G puts
    the chunk's own projections where those caches sit, so the read is the
    identity gather of rows ``start .. start+block_rows`` -- the rows the
    callback returned. This compares the two, block by block, bitwise.
    """

    row_capacity, block_rows = 16, 4
    proj, proj_abs2, rotations = _chunk_rows(row_capacity=row_capacity)
    row_image = jnp.asarray(np.arange(row_capacity, dtype=np.int32) % 3)
    kernel_ids = jnp.asarray(np.arange(row_capacity, dtype=np.int32) % 2)
    posterior = jnp.asarray(
        np.linspace(0.0, 1.0, row_capacity * 2, dtype=np.float32).reshape(
            row_capacity, 2
        )
    )
    row_ids = jnp.asarray(np.arange(row_capacity, dtype=np.int32))
    spec = _Spec(block_rows)
    tables = _tables_with_chunk_rows(proj, proj_abs2, rotations)
    empty_tables = rp._ChunkStageTables(
        **{name: None for name in rp._ChunkStageTables._fields}
    )

    seen = _record_block_projections(monkeypatch)
    for start in range(0, row_capacity, block_rows):
        stop = start + block_rows
        # the current path: the caller slices, the program takes the block
        rp._resident_mstep_block_at(
            jnp.asarray(start, dtype=jnp.int32),
            rp._MstepBlockInputs(
                row_image_local=row_image,
                kernel_row_image_ids=kernel_ids,
                row_posterior=posterior,
                row_fine_rot=None,
                projections=(proj[start:stop], proj_abs2[start:stop], rotations[start:stop]),
            ),
            None,
            empty_tables,
            carry=None,
            spec=spec,
            cuda_backproject=None,
        )
        # G: the chunk's arrays ride in the tables, the program reads its rows
        rp._resident_mstep_block_at(
            jnp.asarray(start, dtype=jnp.int32),
            rp._MstepBlockInputs(
                row_image_local=row_image,
                kernel_row_image_ids=kernel_ids,
                row_posterior=posterior,
                row_fine_rot=row_ids,
                projections=None,
            ),
            None,
            tables,
            carry=None,
            spec=spec,
            cuda_backproject=None,
        )

    assert len(seen) == 2 * (row_capacity // block_rows)
    for callback_block, program_block in zip(seen[0::2], seen[1::2]):
        assert len(callback_block) == len(program_block) == 6
        for expected, actual in zip(callback_block, program_block):
            assert expected.dtype == actual.dtype
            assert expected.shape == actual.shape
            np.testing.assert_array_equal(expected, actual)


def _adapter_kwargs(row_capacity, block_rows, posterior):
    return dict(
        row_capacity=row_capacity,
        n_valid_rows=row_capacity,
        mstep_block_rows=block_rows,
        image_capacity=2,
        row_image_local=jnp.asarray(np.arange(row_capacity, dtype=np.int32) % 2),
        kernel_row_image_ids=jnp.asarray(np.arange(row_capacity, dtype=np.int32) % 2),
        row_posterior=posterior,
        recon={
            "shifted_recon": object(),
            "shifted_noise": object(),
            "ctf2_over_nv_recon": object(),
            "direct_ctf_rfloat_recon": None,
            "raw_translated_wavg_rectangle": object(),
            "raw_translated_wavg_for_atomic": object(),
            "scale": object(),
        },
        n_rect=1,
        n_shells=2,
        n_recon_windowed=5,
        noise_variance_for_noise=None,
        shell_indices_noise=None,
        exact_positions_device=None,
        Ft_y_total=None,
        Ft_ctf_total=None,
        image_shape=(4, 4),
        recon_volume_shape=(4, 4, 4),
        mstep_current_size=4,
        relion_x_half_recon_indices=None,
        max_adjoint_block_bytes=1 << 20,
        cuda_backproject=None,
    )


def test_the_adapter_walks_the_same_blocks_in_both_forms(monkeypatch):
    """``run_resident_mstep_blocks`` itself, loose path, both argument forms.

    The driver loop, its block bounds, its row slices and the arrays each block
    receives are compared between the callback form and the chunk-array form.
    ``_resident_mstep_block`` and the carry are stubbed, so this is the loop and
    its slicing, on CPU, without the CUDA M-step body.
    """

    row_capacity, block_rows, n_recon = 16, 4, 5
    proj, proj_abs2, rotations = _chunk_rows(row_capacity=row_capacity, n_recon=n_recon)
    posterior = jnp.asarray(
        np.linspace(-1.0, 1.0, row_capacity * 2, dtype=np.float32).reshape(
            row_capacity, 2
        )
    )
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT", "0")

    carry = rp._ChunkMstepCarry(
        **{name: None for name in rp._ChunkMstepCarry._fields}
    )
    monkeypatch.setattr(rp, "_initial_mstep_carry", lambda *a, **k: carry)
    seen = _record_block_projections(monkeypatch)

    kwargs = _adapter_kwargs(row_capacity, block_rows, posterior)
    rp.run_resident_mstep_blocks(
        lambda start, stop: (
            proj[start:stop],
            proj_abs2[start:stop],
            rotations[start:stop],
        ),
        **kwargs,
    )
    callback_blocks = list(seen)
    seen.clear()
    rp.run_resident_mstep_blocks(
        chunk_projections=(proj, proj_abs2, rotations), **kwargs
    )
    program_blocks = list(seen)

    assert len(callback_blocks) == row_capacity // block_rows
    assert len(program_blocks) == len(callback_blocks)
    for expected_block, actual_block in zip(callback_blocks, program_blocks):
        for expected, actual in zip(expected_block, actual_block):
            assert expected.dtype == actual.dtype
            assert expected.shape == actual.shape
            np.testing.assert_array_equal(expected, actual)


def test_the_adapter_refuses_both_and_neither_projection_form():
    """Exactly one projection source, and the chunk arrays carry the row axis."""

    row_capacity, block_rows = 8, 4
    proj, proj_abs2, rotations = _chunk_rows(row_capacity=row_capacity, n_recon=3)
    posterior = jnp.zeros((row_capacity, 2), dtype=jnp.float32)
    kwargs = _adapter_kwargs(row_capacity, block_rows, posterior)

    with pytest.raises(ValueError, match="exactly one of block_projections"):
        rp.run_resident_mstep_blocks(**kwargs)
    with pytest.raises(ValueError, match="exactly one of block_projections"):
        rp.run_resident_mstep_blocks(
            lambda start, stop: None,
            chunk_projections=(proj, proj_abs2, rotations),
            **kwargs,
        )
    with pytest.raises(ValueError, match="whole row axis"):
        rp.run_resident_mstep_blocks(
            chunk_projections=(proj[:4], proj_abs2, rotations), **kwargs
        )


def test_the_adapter_still_fails_closed_on_the_once_per_half_operands():
    """G does not weaken T18b's fail-closed contract, in either form."""

    row_capacity, block_rows = 8, 4
    proj, proj_abs2, rotations = _chunk_rows(row_capacity=row_capacity, n_recon=3)
    kwargs = _adapter_kwargs(
        row_capacity, block_rows, jnp.zeros((row_capacity, 2), dtype=jnp.float32)
    )
    kwargs["recon"] = dict(
        kwargs["recon"], shifted_recon=None, shifted_noise=None, recon_image=object()
    )
    with pytest.raises(ValueError, match="pre-shifted"):
        rp.run_resident_mstep_blocks(lambda start, stop: None, **kwargs)
    with pytest.raises(ValueError, match="pre-shifted"):
        rp.run_resident_mstep_blocks(
            chunk_projections=(proj, proj_abs2, rotations), **kwargs
        )


# --------------------------------------------------------------------------
# H: a bucket's per-image row trims in one program
# --------------------------------------------------------------------------


def _count_primitive(jaxpr, name):
    """Count one primitive through the ``pjit`` wrappers ``make_jaxpr`` leaves."""

    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == name:
            total += 1
        for value in eqn.params.values():
            inner = getattr(value, "jaxpr", None)
            if inner is not None:
                total += _count_primitive(inner, name)
    return total


def _row_values(batch=8, rotations=5, trans=3, seed=20260921):
    rng = np.random.default_rng(seed)
    return {
        "batch_norm": jnp.asarray(rng.standard_normal((batch, 1)).astype(np.float32)),
        "log_Z": jnp.asarray(rng.standard_normal(batch)),
        "best_argmax": jnp.asarray(rng.integers(0, 7, batch).astype(np.int64)),
        "best_log_score": jnp.asarray(rng.standard_normal(batch).astype(np.float32)),
        "max_posterior": jnp.asarray(rng.standard_normal(batch).astype(np.float32)),
        "n_significant_samples": jnp.asarray(rng.integers(0, 4, batch).astype(np.int32)),
        "stats_probs_sum_t": jnp.asarray(rng.standard_normal((batch, rotations))),
        "reconstruction_sample_mask": jnp.asarray(
            rng.integers(0, 2, (batch, rotations, trans)).astype(bool)
        ),
    }


def test_the_row_trim_program_writes_the_bytes_the_per_value_slices_wrote():
    """Every trimmed array, bitwise against ``value[:unpadded_batch_size]``."""

    batch, unpadded = 8, 5
    values = _row_values(batch=batch)
    trimmed = lbs.trim_local_postprocess_rows(values, unpadded_batch_size=unpadded)

    assert set(trimmed) == set(values)
    for name, value in values.items():
        expected = value[:unpadded]
        actual = trimmed[name]
        assert expected.dtype == actual.dtype, name
        assert expected.shape == actual.shape, name
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual), err_msg=name)


def test_the_row_trim_keeps_host_arrays_on_the_host_and_none_as_none():
    """A NumPy row array must not be moved to the device by the fold.

    ``stats_probs_sum_t`` is a host array on the branch where the engine has
    already pulled ``probs_sum_t``; handing it to the program would device_put
    it, which is a behaviour change and not a fold.
    """

    host = np.arange(12, dtype=np.float64).reshape(6, 2)
    values = {
        "host": host,
        "device": jnp.asarray(np.arange(6, dtype=np.float32)),
        "absent": None,
    }
    trimmed = lbs.trim_local_postprocess_rows(values, unpadded_batch_size=4)
    assert isinstance(trimmed["host"], np.ndarray)
    assert not isinstance(trimmed["host"], jax.Array)
    np.testing.assert_array_equal(trimmed["host"], host[:4])
    assert isinstance(trimmed["device"], jax.Array)
    np.testing.assert_array_equal(np.asarray(trimmed["device"]), np.arange(4, dtype=np.float32))
    assert trimmed["absent"] is None


def test_the_row_trim_is_one_program_for_the_whole_bucket():
    """One ``dynamic_slice`` program per bucket shape class, not one per array.

    This is the measured quantity: P3-E's census counted 28 single-primitive
    programs at ``postprocess_rows``, retraced every local iteration because
    the bucket loop clears JAX's caches. Counting jaxpr equations of the folded
    program shows the eight slices live in one program.
    """

    values = _row_values()
    names = tuple(sorted(values))
    closed = jax.make_jaxpr(
        lambda arrays: lbs._local_postprocess_row_trim_program(
            arrays, unpadded_batch_size=5
        )
    )(tuple(values[name] for name in names))
    assert _count_primitive(closed.jaxpr, "slice") == len(names)
    assert len(closed.out_avals) == len(names)


def test_every_name_the_engine_trims_is_built_by_the_engine():
    """The folded dict and the ``postprocess_rows`` call sites cannot drift.

    Every ``postprocess_rows(value, name="X")`` in ``run_local_em_exact`` must
    have ``"X"`` among the keys the same function puts into
    ``postprocess_row_inputs``; otherwise the flag-on path raises ``KeyError``
    on a branch no unit test drives.
    """

    import inspect
    import re

    from recovar.em.local import local_em_engine

    source = inspect.getsource(local_em_engine.run_local_em_exact)
    used = set(re.findall(r'postprocess_rows\([^()]*name="([a-z_A-Z]+)"', source))
    built = set(re.findall(r'^\s+"([a-z_A-Z]+)": [a-z_A-Z]+,$', source, re.MULTILINE))
    built |= set(re.findall(r"^\s+([a-z_A-Z]+)=\1,$", source, re.MULTILINE))
    built |= set(re.findall(r'postprocess_row_inputs\["([a-z_A-Z]+)"\]', source))
    assert used, "the call sites must name what they trim"
    assert used <= built, sorted(used - built)


# --------------------------------------------------------------------------
# J: a bucket's constant operands in one program
# --------------------------------------------------------------------------


_SCORE_REAL = np.float32
_NORMALIZATION_REAL = np.float64


def _engine_constant_expressions(*, batch_size, n_half, n_trans):
    """The statements ``run_local_em_exact`` falls back to, copied verbatim.

    Kept here rather than imported so the test is an independent statement of
    what each name has to equal, which is what makes the comparison meaningful.
    """

    return {
        "ctf_rfloat_half": lambda: jnp.zeros((batch_size, n_half), dtype=jnp.float64),
        "inverse_noise_rfloat_cast": lambda: jnp.zeros((n_half,), dtype=jnp.float32),
        "corr_img_rfloat_square": lambda: jnp.zeros(
            (batch_size, n_half), dtype=jnp.float32
        ),
        "integer_pre_shifts_zero": lambda: jnp.zeros((batch_size, 2), dtype=jnp.int32),
        "fourier_pre_shifts_zero": lambda: jnp.zeros((batch_size, 2), dtype=_SCORE_REAL),
        "image_corrections_one": lambda: jnp.ones(batch_size, dtype=_SCORE_REAL),
        "image_only_corrections_one": lambda: jnp.ones(batch_size, dtype=_SCORE_REAL),
        "scale_corrections_one": lambda: jnp.ones(batch_size, dtype=_SCORE_REAL),
        "translation_sqdist_zero": lambda: jnp.zeros(
            (batch_size, n_trans), dtype=_SCORE_REAL
        ),
        "normalization_log_z_zero": lambda: jnp.zeros(
            batch_size, dtype=_NORMALIZATION_REAL
        ),
        "normalization_log_evidence_zero": lambda: jnp.zeros(
            batch_size, dtype=_NORMALIZATION_REAL
        ),
        "normalization_max_posterior_zero": lambda: jnp.zeros(
            batch_size, dtype=jnp.float32
        ),
        "group_ids_zero": lambda: jnp.zeros(batch_size, dtype=jnp.int32),
        "scale_correction_pixel_mask_zero": lambda: jnp.zeros(n_half, dtype=bool),
        "reconstruction_probability_threshold_zero": lambda: jnp.zeros(
            (batch_size,), dtype=jnp.float64
        ),
        "fused_fine_job_plan_empty": lambda: jnp.full((1, 4), -1, dtype=jnp.int32),
    }


def _all_branch_combinations():
    """Every branch arm of the constant sites, in four covering configurations."""

    base = dict(
        batch_size=6,
        n_half=7,
        n_trans=3,
        score_real_dtype=_SCORE_REAL,
        normalization_real_dtype=_NORMALIZATION_REAL,
    )
    return [
        dict(
            base,
            relion_exact_bpref_operands=False,
            apply_integer_pre_shift=True,
            has_image_pre_shifts=True,
            has_image_corrections=True,
            has_scale_corrections=True,
            has_translation_sqdist=True,
            has_normalization_log_z=False,
            has_normalization_log_evidence=False,
            has_normalization_max_posterior=False,
            accumulate_noise=True,
            has_group_ids=True,
            has_reconstruction_probability_threshold=False,
            fused_pair_fine_score_enabled=False,
        ),
        dict(
            base,
            relion_exact_bpref_operands=True,
            apply_integer_pre_shift=False,
            has_image_pre_shifts=True,
            has_image_corrections=False,
            has_scale_corrections=False,
            has_translation_sqdist=False,
            has_normalization_log_z=True,
            has_normalization_log_evidence=True,
            has_normalization_max_posterior=True,
            accumulate_noise=True,
            has_group_ids=False,
            has_reconstruction_probability_threshold=True,
            fused_pair_fine_score_enabled=True,
        ),
        dict(
            base,
            relion_exact_bpref_operands=True,
            apply_integer_pre_shift=False,
            has_image_pre_shifts=False,
            has_image_corrections=False,
            has_scale_corrections=True,
            has_translation_sqdist=True,
            has_normalization_log_z=False,
            has_normalization_log_evidence=True,
            has_normalization_max_posterior=False,
            accumulate_noise=False,
            has_group_ids=False,
            has_reconstruction_probability_threshold=False,
            fused_pair_fine_score_enabled=False,
        ),
        dict(
            base,
            relion_exact_bpref_operands=False,
            apply_integer_pre_shift=True,
            has_image_pre_shifts=False,
            has_image_corrections=True,
            has_scale_corrections=False,
            has_translation_sqdist=False,
            has_normalization_log_z=True,
            has_normalization_log_evidence=False,
            has_normalization_max_posterior=True,
            accumulate_noise=False,
            has_group_ids=True,
            has_reconstruction_probability_threshold=True,
            fused_pair_fine_score_enabled=True,
        ),
    ]


@pytest.mark.parametrize("case_index", range(4))
def test_the_bucket_constant_program_writes_the_engine_statements(case_index):
    """Every constant, bitwise against the ``jnp.zeros``/``ones``/``full`` it replaces."""

    config = _all_branch_combinations()[case_index]
    specs = lbs.local_bucket_constant_specs(**config)
    built = lbs.local_bucket_constant_operands(specs)
    expressions = _engine_constant_expressions(
        batch_size=config["batch_size"],
        n_half=config["n_half"],
        n_trans=config["n_trans"],
    )

    assert set(built) == set(specs)
    for name, value in built.items():
        expected = expressions[name]()
        assert expected.dtype == value.dtype, name
        assert expected.shape == value.shape, name
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(value), err_msg=name)


def test_the_bucket_constant_specs_cover_every_name_the_engine_asks_for():
    """No ``bucket_constant("X", ...)`` call site without a spec that makes X.

    A missing name would be a ``KeyError`` on a branch combination no test
    drives, which is exactly the failure mode that cost this programme two
    end-to-end runs this week.
    """

    import inspect
    import re

    from recovar.em.local import local_em_engine

    source = inspect.getsource(local_em_engine.run_local_em_exact)
    used = set(re.findall(r'bucket_constant\(\s*"([a-z_A-Z]+)"', source))
    produced = set()
    for config in _all_branch_combinations():
        produced |= set(lbs.local_bucket_constant_specs(**config))
    assert used, "the constant sites must name what they build"
    assert used <= produced, sorted(used - produced)
    assert produced <= set(
        _engine_constant_expressions(batch_size=1, n_half=1, n_trans=1)
    )


def test_the_bucket_constants_are_one_program_and_fresh_buffers():
    """One program for the set, and a new buffer every call.

    The big JIT takes these operands as arguments; a memoized constant would be
    a buffer the previous call may have donated, so the fold builds them rather
    than caching them.
    """

    config = _all_branch_combinations()[0]
    specs = lbs.local_bucket_constant_specs(**config)
    first = lbs.local_bucket_constant_operands(specs)
    second = lbs.local_bucket_constant_operands(specs)
    for name, value in first.items():
        assert value is not second[name], name
        np.testing.assert_array_equal(np.asarray(value), np.asarray(second[name]))

    key = tuple(
        (
            tuple(int(dim) for dim in specs[name][0]),
            specs[name][1],
            np.dtype(specs[name][2]).name,
        )
        for name in sorted(specs)
    )
    closed = jax.make_jaxpr(
        lambda: lbs._local_bucket_constant_program(key=key)
    )()
    assert len(closed.out_avals) == len(specs)
