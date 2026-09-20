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
