"""P4-J: one warm-up per capacity class covers every chunk of that class.

A compile-ahead consumer builds the pass-2 chunk programs before the operand
preparation runs. For the chunk's row-aligned arrays it does not need a
predicted aval constructor at all: `materialize_chunk` is pure host numpy on
the plan, so a consumer can simply materialize one chunk per capacity class and
read the shapes off it.

That is only sound if the shapes and dtypes of a materialized chunk depend on
the capacity class alone and not on how full the chunk happens to be. This file
asserts exactly that, because it is the property the consumer rests on: if two
chunks of one class produced different signatures, warming from one of them
would compile a program the other never uses and leave the other cold.

Everything here is host numpy, so it runs on CPU.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

pytest.importorskip("jax")

from test_resident_candidates import (  # noqa: E402
    IMAGE_CAPACITY_LADDER,
    ROW_CAPACITY_LADDER,
    _synthetic_tables,
)

from recovar.em.sparse_pass2.resident_candidates import (  # noqa: E402
    materialize_chunk,
    plan_capacity_chunks,
)

pytestmark = pytest.mark.unit


def _signature(materialized):
    return {
        name: (None if value is None
               else (tuple(int(d) for d in np.shape(value)), np.asarray(value).dtype.str))
        for name, value in sorted(materialized.items())
    }


@pytest.mark.parametrize("row_counts", [
    [3, 5, 2, 7, 1, 4, 6, 2, 1, 1, 9],
    [1] * 40,
    [17, 1, 1, 33, 2, 9, 4, 4, 4, 12, 1, 1, 1, 25],
])
def test_one_chunk_per_class_gives_every_chunk_of_that_class(row_counts):
    tables = _synthetic_tables(row_counts)
    chunks = plan_capacity_chunks(
        tables,
        row_capacity_ladder=ROW_CAPACITY_LADDER,
        image_capacity_ladder=IMAGE_CAPACITY_LADDER,
    )
    by_class = collections.defaultdict(list)
    for chunk in chunks:
        by_class[(int(chunk.row_capacity), int(chunk.image_capacity))].append(chunk)

    classes_with_two = 0
    for klass, members in by_class.items():
        if len(members) < 2:
            continue
        classes_with_two += 1
        occupancies = {(int(c.n_valid_rows), int(c.n_valid_images)) for c in members}
        reference = _signature(materialize_chunk(tables, members[0]))
        for chunk in members[1:]:
            got = _signature(materialize_chunk(tables, chunk))
            assert got == reference, (
                f"class {klass} produced two different signatures at occupancies "
                f"{occupancies}: {got} vs {reference}"
            )
    if classes_with_two == 0:
        pytest.skip("this plan has no capacity class with two chunks to compare")


def test_the_shapes_are_the_capacities_and_nothing_else():
    tables = _synthetic_tables([3, 5, 2, 7, 1, 4, 6, 2, 1, 1, 9])
    chunks = plan_capacity_chunks(
        tables,
        row_capacity_ladder=ROW_CAPACITY_LADDER,
        image_capacity_ladder=IMAGE_CAPACITY_LADDER,
    )
    for chunk in chunks:
        m = materialize_chunk(tables, chunk)
        rows, images = int(chunk.row_capacity), int(chunk.image_capacity)
        for name in ("row_image_local", "row_fine_rot", "row_parent_local",
                     "row_log_prior", "row_mask_bits", "row_mask_mode"):
            assert np.shape(m[name]) == (rows,), (name, np.shape(m[name]), rows)
        assert np.shape(m["image_ids"]) == (images,)
        assert np.shape(m["n_valid_rows"]) == ()
        assert np.shape(m["n_valid_images"]) == ()


def test_row_mask_bits_is_one_packed_word_per_row():
    """The `_ChunkRowArrays` comment used to say `[C_R, W]`; it is `[C_R]`.

    `expand_chunk_mask_jnp` requires a 1-D array and expands it against the
    fine-translation parents, so a consumer that believed the comment would
    have built the wrong aval. The comment is corrected; this pins it.
    """

    tables = _synthetic_tables([3, 5, 2])
    chunks = plan_capacity_chunks(
        tables,
        row_capacity_ladder=ROW_CAPACITY_LADDER,
        image_capacity_ladder=IMAGE_CAPACITY_LADDER,
    )
    m = materialize_chunk(tables, chunks[0])
    assert np.asarray(m["row_mask_bits"]).ndim == 1
    assert np.asarray(m["row_mask_bits"]).dtype == np.uint32
