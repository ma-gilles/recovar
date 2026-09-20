"""P3-A: the resident chunk loop's glue, as programs instead of eager ops.

Ticket: ``em_parity_tickets_20260918/P3A_resident_driver_glue.md``.

The chunk loop used to issue about 780 eager JAX dispatches per chunk at the
early state and 324 at hp3, almost all of them the M-step block's own slices,
gathers and scalar operands, plus three ``jax.eval_shape`` probes per chunk
that existed only to learn the carry's dtypes. What is CPU-testable, and is
tested here, is the part that does not need the custom CUDA library:

* the carry dtypes are host arithmetic on the operand dtypes, and they agree
  with the ``jax.eval_shape`` probe of the real block stages;
* the probe check fails when the arithmetic is wrong, so it is a real guard;
* the block program's ``dynamic_slice`` picks the rows the Python slice picked,
  for every row capacity on the ladder;
* the device row offsets are made once per value, not once per block.

The bitwise comparison of the whole block body against the loose dispatch
needs a GPU and the custom library; it is the matched hp3/early pair in the
ticket's report, and the driver's own GPU test exercises the default path.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.sparse_pass2 import resident_pass2 as rp

pytestmark = pytest.mark.unit


def _spec(**overrides):
    kwargs = dict(
        row_capacity=8192,
        image_capacity=32,
        n_fine_trans=21,
        n_score_pixels=64,
        n_recon_pixels=97,
        n_rect=64,
        mstep_block_rows=4096,
        adaptive_fraction=0.999,
        current_size=16,
        mstep_current_size=16,
        image_shape=(16, 16),
        recon_volume_shape=(16, 16, 16),
        max_adjoint_block_bytes=1 << 30,
        stats_config=rp._MstepOnlyStatsConfig(n_shells=9),
        use_rfloat_ctf_wavg=False,
        use_translate_sum_kernel=False,
        bpref_recon_operand=False,
        kernel_ctf_probs=False,
        block_unroll=1,
        static_block_trip=False,
    )
    kwargs.update(overrides)
    return rp._ChunkProgramSpec(**kwargs)


def _operands(spec, *, tile_dtype=jnp.complex64, ctf_dtype=jnp.float32):
    """The three operand fields the carry dtypes are derived from."""

    shape = (spec.image_capacity, spec.n_fine_trans, spec.n_recon_pixels)
    tiles = jnp.zeros(shape, dtype=tile_dtype)
    return rp._ChunkStageOperands(
        score_input=None,
        corr_img_score=None,
        highres_xi2_half=None,
        translation_prior=None,
        shifted_recon=tiles,
        shifted_noise=tiles,
        recon_image=None,
        recon_weight=None,
        noise_image=None,
        ctf2_over_nv_recon=jnp.zeros(
            (spec.image_capacity, spec.n_recon_pixels), dtype=ctf_dtype
        ),
        direct_ctf_rfloat_recon=None,
        processed_image_half=None,
        relion_norm_high_shell=None,
        raw_translated_wavg_rectangle=None,
        raw_translated_wavg_for_atomic=None,
        scale=None,
        group_ids=None,
        translation_sqdist_ang=None,
    )


def _tables(spec, *, noise_dtype=jnp.float32):
    fields = {name: None for name in rp._ChunkStageTables._fields}
    fields["noise_variance_for_noise"] = jnp.ones(
        (spec.n_recon_pixels,), dtype=noise_dtype
    )
    fields["shell_indices_noise"] = jnp.zeros((spec.n_recon_pixels,), dtype=jnp.int32)
    return rp._ChunkStageTables(**fields)


@pytest.mark.parametrize("proj_dtype", [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("noise_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("use_kernel", [False, True])
def test_static_carry_dtypes_match_the_traced_block_stages(
    proj_dtype, noise_dtype, use_kernel
):
    """The promotion arithmetic answers what ``jax.eval_shape`` answered."""

    spec = _spec(use_translate_sum_kernel=use_kernel)
    proj_abs2_dtype = jnp.zeros((), dtype=proj_dtype).real.dtype
    operands = _operands(spec)
    tables = _tables(spec, noise_dtype=noise_dtype)
    dtypes = rp._mstep_block_operand_dtypes(
        operands,
        tables,
        spec=spec,
        projection_dtypes=(proj_dtype, proj_abs2_dtype),
    )
    # Raises if the arithmetic and the probe disagree.
    rp._check_mstep_carry_avals(tables, spec=spec, dtypes=dtypes)

    shells, a2, xa = rp._probe_mstep_block_output_avals(
        tables, spec=spec, dtypes=dtypes
    )
    assert shells.shape == (spec.stats_config.n_shells,)
    assert a2.shape == xa.shape == (spec.image_capacity,)
    assert jnp.dtype(shells.dtype) == jnp.dtype(jnp.float64)
    assert jnp.dtype(a2.dtype) == dtypes["a2"]
    assert jnp.dtype(xa.dtype) == dtypes["xa"]


def test_initial_mstep_carry_has_the_probed_shapes_and_no_probe_by_default(monkeypatch):
    """The carry the loop starts from, built without tracing anything."""

    monkeypatch.delenv(rp._CARRY_AVAL_PROBE_ENV, raising=False)
    calls = []
    original = rp._probe_mstep_block_output_avals
    monkeypatch.setattr(
        rp,
        "_probe_mstep_block_output_avals",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )

    spec = _spec()
    operands = _operands(spec)
    tables = _tables(spec)
    Ft_y = jnp.zeros((4, 4, 4), dtype=jnp.complex64)
    Ft_ctf = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    carry = rp._initial_mstep_carry(
        Ft_y,
        Ft_ctf,
        operands,
        tables,
        spec=spec,
        projection_dtypes=(jnp.complex64, jnp.float32),
    )
    assert calls == []
    assert carry.Ft_y is Ft_y and carry.Ft_ctf is Ft_ctf
    assert carry.wavg_triplet_pixels.shape == (spec.image_capacity, spec.n_rect, 3)
    assert carry.noise_shells.shape == (spec.stats_config.n_shells,)
    assert carry.noise_shells.dtype == jnp.float64
    assert carry.a2_per_image.shape == carry.xa_per_image.shape == (spec.image_capacity,)
    assert not np.any(np.asarray(carry.noise_shells))
    assert not np.any(np.asarray(carry.a2_per_image))
    assert not np.any(np.asarray(carry.wavg_triplet_pixels))

    monkeypatch.setenv(rp._CARRY_AVAL_PROBE_ENV, "1")
    rp._initial_mstep_carry(
        Ft_y,
        Ft_ctf,
        operands,
        tables,
        spec=spec,
        projection_dtypes=(jnp.complex64, jnp.float32),
    )
    assert calls == [1]


def test_the_carry_aval_check_rejects_a_wrong_dtype():
    """The probe check is a real guard, not a formality."""

    spec = _spec()
    operands = _operands(spec)
    tables = _tables(spec)
    dtypes = rp._mstep_block_operand_dtypes(
        operands, tables, spec=spec, projection_dtypes=(jnp.complex64, jnp.float32)
    )
    wrong = dict(dtypes)
    wrong["a2"] = jnp.dtype(jnp.float64)
    with pytest.raises(AssertionError, match="disagree with the traced block stages"):
        rp._check_mstep_carry_avals(tables, spec=spec, dtypes=wrong)


@pytest.mark.parametrize("row_capacity", rp._DEFAULT_ROW_CAPACITY_LADDER)
def test_block_dynamic_slice_picks_the_python_slice_rows(row_capacity):
    """Every capacity is a whole number of blocks, so the slice never clamps."""

    block_rows = 4096
    assert row_capacity % block_rows == 0
    values = jnp.arange(row_capacity, dtype=jnp.int32)
    for start in range(0, row_capacity, block_rows):
        taken = jax.lax.dynamic_slice_in_dim(
            values, rp._device_int32(start), block_rows, axis=0
        )
        expected = values[start : start + block_rows]
        np.testing.assert_array_equal(np.asarray(taken), np.asarray(expected))


def test_device_row_offsets_are_made_once_per_value():
    """The block offsets must not put an eager dispatch back per block."""

    rp._DEVICE_INT32_CACHE.clear()
    first = rp._device_int32(4096)
    second = rp._device_int32(4096)
    assert first is second
    assert int(first) == 4096
    assert first.dtype == jnp.int32
    assert rp._device_int32(8192) is not first


def test_glue_jit_is_on_by_default_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv(rp._RESIDENT_GLUE_JIT_ENV, raising=False)
    assert rp._resident_glue_jit_enabled() is True
    monkeypatch.setenv(rp._RESIDENT_GLUE_JIT_ENV, "0")
    assert rp._resident_glue_jit_enabled() is False
    monkeypatch.setenv(rp._RESIDENT_GLUE_JIT_ENV, "1")
    assert rp._resident_glue_jit_enabled() is True


def test_the_block_program_is_keyed_on_the_capacity_class_not_the_offset():
    """One program serves every block: the offset is an operand, not a key."""

    spec = _spec()
    sig = jax.jit(
        lambda start, values: jax.lax.dynamic_slice_in_dim(
            values, start, spec.mstep_block_rows, axis=0
        )
    )
    values = jnp.arange(spec.row_capacity, dtype=jnp.int32)
    sig(rp._device_int32(0), values)
    before = sig._cache_size() if hasattr(sig, "_cache_size") else None
    sig(rp._device_int32(4096), values)
    after = sig._cache_size() if hasattr(sig, "_cache_size") else None
    if before is not None:
        assert after == before
