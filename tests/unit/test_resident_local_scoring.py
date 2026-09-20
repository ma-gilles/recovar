"""Projection-by-rows and the projected-row scoring program (T12 stage 2).

Ticket: ``em_parity_tickets_20260918/T12_resident_local_search.md``.

Local search cannot gather a per-iteration projection cache: its fine grid at
HEALPix order 5 holds 2.4M rotations. The chunk's own rows are projected
instead. Two properties make that a layout change rather than an arithmetic
change, and both are measured here:

1. scoring a chunk from an explicitly supplied reference is bitwise identical
   to scoring it from the same values gathered out of a cache, so the two
   routes share one arithmetic path;
2. projecting rows in blocks is bitwise identical to projecting them in one
   call, so the byte budget that sizes the blocks cannot move a number.

The first test reuses the T6 fixture builder, so the operands, the window and
the CUDA kernel are the production ones rather than a second-hand copy.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from test_resident_scoring import N_FINE_TRANS, _build_case
from test_sparse_pass2_bucketed_parity import IMAGE_SHAPE, VOLUME_SHAPE

from recovar.em.sparse_pass2.resident_candidates import expand_mask_rows, materialize_chunk
from recovar.em.sparse_pass2.resident_scoring import (
    project_resident_rows,
    resident_projection_block_rows,
    resident_row_projection_bytes,
    score_resident_chunk,
    score_resident_projected_chunk,
)

pytestmark = pytest.mark.unit


def _packed_row_mask(tables, chunk, n_fine_trans):
    """The chunk's candidate mask in the local layout's packing.

    The resident candidate table (T5) stores a per-parent uint32 bitset over
    coarse translations; the local layout stores per-row bytes over fine
    translations. Round-trip the T5 mask through its dense expansion so the two
    programs are handed the same candidate set in their own spellings.
    """

    dense = np.zeros((int(chunk.row_capacity), int(n_fine_trans)), dtype=bool)
    n_valid = int(chunk.n_valid_rows)
    if n_valid:
        rows = []
        for image in range(chunk.image_start, chunk.image_stop):
            rows.append(expand_mask_rows(tables, image, FINE_TRANS_PARENT))
        dense[:n_valid] = np.concatenate(rows, axis=0)
    return np.packbits(dense, axis=1, bitorder="little")


FINE_TRANS_PARENT = np.repeat(np.arange(4, dtype=np.int32), 2)


def test_projection_block_budget_is_at_least_one_row():
    per_row = resident_row_projection_bytes(n_score_pixels=3386, n_recon_pixels=4324)
    assert per_row == 3386 * 8 + 4324 * 12
    assert resident_projection_block_rows(
        n_score_pixels=3386, n_recon_pixels=4324, max_block_bytes=1
    ) == 1
    assert (
        resident_projection_block_rows(
            n_score_pixels=3386, n_recon_pixels=4324, max_block_bytes=10 * per_row
        )
        == 10
    )


@pytest.mark.gpu
def test_projected_scoring_equals_the_cached_gather_bitwise(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """One arithmetic path: the reference's source cannot change a score bit."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            6, row_capacity_ladder=(16, 64, 256), image_capacity_ladder=(2, 4, 8)
        )
        tables = case["tables"]
        operands = case["operands"]
        cache = case["projection_cache"]
        compared = 0
        for chunk in case["chunks"]:
            host = materialize_chunk(tables, chunk)
            cached = score_resident_chunk(
                jnp.asarray(host["row_image_local"]),
                jnp.asarray(host["row_fine_rot"]),
                jnp.asarray(host["row_log_prior"]),
                jnp.asarray(host["row_mask_bits"]),
                jnp.asarray(host["row_mask_mode"]),
                jnp.asarray(host["n_valid_rows"]),
                jnp.asarray(host["image_ids"]),
                cache,
                operands.score_input,
                operands.corr_img_score,
                operands.highres_xi2_half,
                operands.translation_prior,
                half_weights=operands.half_weights,
                translation_angles=operands.translation_angles,
                full_to_compact=operands.full_to_compact,
                fine_translation_parent=jnp.asarray(FINE_TRANS_PARENT),
                logical_current_size=jnp.asarray(operands.current_size, dtype=jnp.int32),
                row_capacity=int(chunk.row_capacity),
                image_capacity=int(chunk.image_capacity),
                n_fine_trans=N_FINE_TRANS,
                n_score_pixels=int(operands.n_score_pixels),
            )
            reference = cache[jnp.asarray(host["row_fine_rot"], dtype=jnp.int32)]
            projected = score_resident_projected_chunk(
                reference,
                jnp.asarray(host["row_image_local"]),
                jnp.asarray(host["row_log_prior"]),
                jnp.asarray(_packed_row_mask(tables, chunk, N_FINE_TRANS)),
                jnp.asarray(host["n_valid_rows"]),
                jnp.asarray(host["image_ids"]),
                operands.score_input,
                operands.corr_img_score,
                operands.highres_xi2_half,
                operands.translation_prior,
                half_weights=operands.half_weights,
                translation_angles=operands.translation_angles,
                full_to_compact=operands.full_to_compact,
                logical_current_size=jnp.asarray(operands.current_size, dtype=jnp.int32),
                row_capacity=int(chunk.row_capacity),
                image_capacity=int(chunk.image_capacity),
                n_fine_trans=N_FINE_TRANS,
                n_score_pixels=int(operands.n_score_pixels),
            )
            for field in ("raw_diff2", "scores", "min_diff2"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(cached, field)),
                    np.asarray(getattr(projected, field)),
                    err_msg=f"{field} on chunk {chunk}",
                )
            compared += 1
        assert compared, "the fixture must produce at least one chunk"


@pytest.mark.gpu
def test_full_support_mask_equals_an_all_ones_packed_mask(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """``row_mask_bits=None`` is the layout's compact spelling of full support."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            6, row_capacity_ladder=(16, 64, 256), image_capacity_ladder=(2, 4, 8)
        )
        operands = case["operands"]
        cache = case["projection_cache"]
        chunk = case["chunks"][0]
        host = materialize_chunk(case["tables"], chunk)
        reference = cache[jnp.asarray(host["row_fine_rot"], dtype=jnp.int32)]
        ones = np.packbits(
            np.ones((int(chunk.row_capacity), N_FINE_TRANS), dtype=bool),
            axis=1,
            bitorder="little",
        )

        def run(mask):
            return score_resident_projected_chunk(
                reference,
                jnp.asarray(host["row_image_local"]),
                jnp.asarray(host["row_log_prior"]),
                mask,
                jnp.asarray(host["n_valid_rows"]),
                jnp.asarray(host["image_ids"]),
                operands.score_input,
                operands.corr_img_score,
                operands.highres_xi2_half,
                operands.translation_prior,
                half_weights=operands.half_weights,
                translation_angles=operands.translation_angles,
                full_to_compact=operands.full_to_compact,
                logical_current_size=jnp.asarray(operands.current_size, dtype=jnp.int32),
                row_capacity=int(chunk.row_capacity),
                image_capacity=int(chunk.image_capacity),
                n_fine_trans=N_FINE_TRANS,
                n_score_pixels=int(operands.n_score_pixels),
            )

        none_mask = run(None)
        ones_mask = run(jnp.asarray(ones))
        for field in ("raw_diff2", "scores", "min_diff2"):
            np.testing.assert_array_equal(
                np.asarray(getattr(none_mask, field)),
                np.asarray(getattr(ones_mask, field)),
                err_msg=field,
            )


@pytest.mark.gpu
@pytest.mark.parametrize("block_rows", [1, 3, 7])
def test_row_projection_blocking_is_bitwise(monkeypatch, custom_cuda_lib, gpu_device, block_rows):
    """The byte budget that sizes a projector call cannot move a projection bit."""

    import recovar.cuda_backproject as cuda_backproject
    from helpers.em_arrays import _hermitian_volume

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    rng = np.random.default_rng(20260919)
    n_rows = 11
    angles = rng.uniform(0.0, 2 * np.pi, n_rows)
    rotations = np.stack(
        [
            np.array(
                [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            for a in angles
        ]
    )
    volume = _hermitian_volume(VOLUME_SHAPE, seed=5)
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    score_indices = np.arange(0, n_half, 2, dtype=np.int32)
    recon_indices = np.arange(n_half, dtype=np.int32)

    with jax.default_device(gpu_device):
        whole = project_resident_rows(
            jnp.asarray(volume),
            jnp.asarray(rotations),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            score_indices=score_indices,
            recon_indices=recon_indices,
            max_projected_rotations=n_rows,
            output_complex_dtype=jnp.complex64,
            output_abs2_dtype=jnp.float32,
            relion_texture_interp=False,
        )
        blocked = project_resident_rows(
            jnp.asarray(volume),
            jnp.asarray(rotations),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            score_indices=score_indices,
            recon_indices=recon_indices,
            max_projected_rotations=block_rows,
            output_complex_dtype=jnp.complex64,
            output_abs2_dtype=jnp.float32,
            relion_texture_interp=False,
        )
    for name, a, b in zip(("score", "recon", "recon_abs2"), whole, blocked, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    assert np.asarray(whole[0]).shape == (n_rows, score_indices.size)
    assert np.asarray(whole[1]).shape == (n_rows, recon_indices.size)
