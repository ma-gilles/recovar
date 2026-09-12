from __future__ import annotations

import hashlib
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.scoring import scoring
from recovar.em.scoring.coarse_gemm_hybrid import (
    plan_coarse_gemm_certificate_topology,
    validate_coarse_gemm_certificate_topology,
)

pytestmark = pytest.mark.unit


def _topology():
    return plan_coarse_gemm_certificate_topology(
        np.asarray([0, -1, 1, 2, -1, 3], dtype=np.int32),
        compact_pixel_count=4,
        translation_count=3,
    )


def _selected_block_operands(*, rotations=32, translations=3, pixels=4):
    return (
        jnp.zeros((rotations, pixels), dtype=jnp.complex64),
        jnp.zeros((2, translations, pixels), dtype=jnp.complex64),
        jnp.ones((2, pixels), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray([[1, -1], [-2, 99]], dtype=jnp.int32),
    )


def test_certificate_topology_owns_a_contiguous_immutable_lookup() -> None:
    source = np.asarray(
        [0, 99, -1, 99, 1, 99, 2, 99, -1, 99, 3, 99],
        dtype=np.int32,
    )
    source_view = source[::2]
    assert not source_view.flags.c_contiguous

    topology = plan_coarse_gemm_certificate_topology(
        source_view,
        compact_pixel_count=4,
        translation_count=3,
    )
    source_view[0] = -1

    assert topology.full_to_compact.tolist() == [0, -1, 1, 2, -1, 3]
    assert topology.full_to_compact.dtype == np.dtype(np.int32)
    assert topology.full_to_compact.flags.c_contiguous
    assert not topology.full_to_compact.flags.writeable
    assert not np.shares_memory(topology.full_to_compact, source)
    assert topology.full_to_compact_sha256 == hashlib.sha256(
        topology.full_to_compact.tobytes(order="C"),
    ).hexdigest()
    with pytest.raises(ValueError):
        topology.full_to_compact[0] = 3
    validate_coarse_gemm_certificate_topology(topology)


def test_certificate_topology_validator_rejects_a_writeable_lookup() -> None:
    topology = _topology()
    writeable = topology.full_to_compact.copy()

    with pytest.raises(ValueError, match="contiguous read-only snapshot"):
        validate_coarse_gemm_certificate_topology(
            topology._replace(full_to_compact=writeable),
        )


def test_certificate_topology_validator_rejects_lookup_digest_drift() -> None:
    topology = _topology()
    changed = topology.full_to_compact.copy()
    changed[[0, 2]] = changed[[2, 0]]
    changed.setflags(write=False)

    with pytest.raises(ValueError, match="digest, or gamma invariant"):
        validate_coarse_gemm_certificate_topology(
            topology._replace(full_to_compact=changed),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("full_position_count", 7),
        ("full_to_compact_sha256", "0" * 64),
        ("direct_f32_gamma", 0.0),
    ],
)
def test_certificate_topology_validator_recomputes_scalar_invariants(
    field: str,
    value,
) -> None:
    topology = _topology()

    with pytest.raises(ValueError, match="count, digest, or gamma invariant"):
        validate_coarse_gemm_certificate_topology(
            topology._replace(**{field: value}),
        )


def test_certificate_topology_validator_recomputes_fp64_gammas() -> None:
    topology = _topology()
    changed_gammas = topology.expanded_f64_gammas._replace(cross=0.0)

    with pytest.raises(ValueError, match="count, digest, or gamma invariant"):
        validate_coarse_gemm_certificate_topology(
            topology._replace(expanded_f64_gammas=changed_gammas),
        )


def test_selected_source16_wrapper_forwards_only_the_owned_lookup(monkeypatch) -> None:
    from recovar import cuda_backproject

    topology = _topology()
    operands = _selected_block_operands()
    marker = object()
    captured = {}

    def fake_selected_blocks(*args):
        captured["args"] = args
        return marker

    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rotation_blocks_f32",
        fake_selected_blocks,
    )
    result = scoring._relion_coarse_diff2_rotation_blocks_from_topology_f32(
        *operands,
        topology=topology,
    )

    assert result is marker
    assert all(
        actual is expected
        for actual, expected in zip(captured["args"][:5], operands)
    )
    # IDs stay on device; CUDA owns -1 padding and invalid-ID fail-close.
    assert captured["args"][4] is operands[4]
    assert np.array_equal(
        np.asarray(captured["args"][4]),
        np.asarray([[1, -1], [-2, 99]], dtype=np.int32),
    )
    assert np.array_equal(
        np.asarray(captured["args"][5]),
        topology.full_to_compact,
    )
    assert captured["args"][5].dtype == jnp.int32
    signature = inspect.signature(
        scoring._relion_coarse_diff2_rotation_blocks_from_topology_f32,
    )
    assert "full_to_compact" not in signature.parameters
    assert signature.parameters["topology"].kind is inspect.Parameter.KEYWORD_ONLY


def test_selected_source16_wrapper_rejects_tampering_before_dispatch(
    monkeypatch,
) -> None:
    from recovar import cuda_backproject

    topology = _topology()
    changed = topology.full_to_compact.copy()
    changed[[0, 2]] = changed[[2, 0]]
    changed.setflags(write=False)

    def unexpected_dispatch(*_args):
        raise AssertionError("invalid topology reached CUDA dispatch")

    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rotation_blocks_f32",
        unexpected_dispatch,
    )
    with pytest.raises(ValueError, match="digest, or gamma invariant"):
        scoring._relion_coarse_diff2_rotation_blocks_from_topology_f32(
            *_selected_block_operands(),
            topology=topology._replace(full_to_compact=changed),
        )


def test_selected_source16_wrapper_rejects_dtype_mismatch_before_dispatch(
    monkeypatch,
) -> None:
    from recovar import cuda_backproject

    def unexpected_dispatch(*_args):
        raise AssertionError("invalid operands reached CUDA dispatch")

    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rotation_blocks_f32",
        unexpected_dispatch,
    )
    operands = _selected_block_operands()
    invalid = operands[:-1] + (operands[-1].astype(jnp.float32),)
    with pytest.raises(TypeError, match="invalid dtypes"):
        scoring._relion_coarse_diff2_rotation_blocks_from_topology_f32(
            *invalid,
            topology=_topology(),
        )


@pytest.mark.parametrize(
    "operands",
    [
        _selected_block_operands(rotations=17),
        _selected_block_operands(translations=2),
        _selected_block_operands(pixels=5),
        _selected_block_operands()[:-1]
        + (jnp.zeros((1, 2), dtype=jnp.int32),),
    ],
)
def test_selected_source16_wrapper_rejects_shape_or_count_mismatch(
    monkeypatch,
    operands,
) -> None:
    from recovar import cuda_backproject

    def unexpected_dispatch(*_args):
        raise AssertionError("invalid operands reached CUDA dispatch")

    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rotation_blocks_f32",
        unexpected_dispatch,
    )
    with pytest.raises(ValueError):
        scoring._relion_coarse_diff2_rotation_blocks_from_topology_f32(
            *operands,
            topology=_topology(),
        )
