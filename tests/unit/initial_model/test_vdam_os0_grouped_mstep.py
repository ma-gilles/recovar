"""Focused guards for grouped zero-oversampling VDAM reconstruction."""

from __future__ import annotations

import pytest

from recovar.em.dense_single_volume.local_em_engine import (
    _return_local_big_jit_mstep_tensors,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    (
        "sparse_big_jit_backprojection",
        "source_faithful_bpref",
        "grouped_reconstruction",
        "reconstruct_significant_only",
        "disable_adjoint_y",
        "disable_adjoint_ctf",
        "expected",
    ),
    [
        (True, False, False, True, False, False, True),
        (False, True, True, False, False, False, True),
        (False, True, True, True, False, False, False),
        (False, True, False, False, False, False, False),
        (False, False, True, False, False, False, False),
        (True, True, True, False, True, True, False),
    ],
)
def test_grouped_os0_returns_physical_mstep_operands_for_group_aware_scatter(
    sparse_big_jit_backprojection,
    source_faithful_bpref,
    grouped_reconstruction,
    reconstruct_significant_only,
    disable_adjoint_y,
    disable_adjoint_ctf,
    expected,
):
    assert (
        _return_local_big_jit_mstep_tensors(
            sparse_big_jit_backprojection=sparse_big_jit_backprojection,
            source_faithful_bpref=source_faithful_bpref,
            grouped_reconstruction=grouped_reconstruction,
            reconstruct_significant_only=reconstruct_significant_only,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
        )
        is expected
    )
