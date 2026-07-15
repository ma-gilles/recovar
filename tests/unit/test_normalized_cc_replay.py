"""Tests for the pure-NumPy normalized-CC reduction replay diagnostic."""

import numpy as np
import pytest

from recovar.em.normalized_cc_replay import (
    RELION_FINE_REDUCTION_LANES,
    REPLAY_SCHEMA,
    canonical_float32_reduce,
    canonical_float64_reduce,
    normalized_cc_pixel_contributions,
    recovar_logical_float32_reduce,
    relion_256lane_float32_reduce,
    replay_normalized_cc,
)

pytestmark = pytest.mark.unit


def _production_inputs():
    rng = np.random.default_rng(161552)
    batch, n_rot, n_trans, n_pix = 1, 3, 4, 17
    shifted = (rng.normal(size=(batch, n_trans, n_pix)) + 1j * rng.normal(size=(batch, n_trans, n_pix))).astype(
        np.complex64
    )
    projections = (rng.normal(size=(batch, n_rot, n_pix)) + 1j * rng.normal(size=(batch, n_rot, n_pix))).astype(
        np.complex64
    )
    score_weight = rng.uniform(0.1, 1.5, size=(batch, n_pix)).astype(np.float32)
    half_weight = rng.choice(np.asarray([1.0, 2.0], dtype=np.float32), size=n_pix)
    return shifted, score_weight, projections, half_weight


def test_recovar_logical_replay_matches_production_normalized_cc_score():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _score_pass2_pairs_normalized_cc,
    )

    shifted, score_weight, projections, half_weight = _production_inputs()
    production = np.asarray(
        _score_pass2_pairs_normalized_cc(
            jnp.asarray(shifted),
            jnp.asarray(score_weight),
            jnp.asarray(projections),
            jnp.asarray(half_weight),
            jnp.asarray([[0]], dtype=jnp.int32),
            jnp.asarray([[0]], dtype=jnp.int32),
            jnp.asarray([[True]]),
        )
    )[0, 0]
    contributions = normalized_cc_pixel_contributions(
        projections[0, 0],
        shifted[0, 0],
        score_weight[0],
        half_weight,
    )
    replay = replay_normalized_cc(contributions)

    replay_bits = np.asarray(replay.recovar_logical_float32.score, dtype=np.float32).view(np.uint32)
    assert replay_bits == production.view(np.uint32)
    assert replay.schema == REPLAY_SCHEMA
    assert replay.schema_version == 1


def test_relion_256lane_reduction_matches_hand_reference():
    rng = np.random.default_rng(256)
    values = rng.normal(size=777).astype(np.float32)

    # Independent transcription of diff2.cuh: each lane walks its passes,
    # followed by the explicit 128, 64, ..., 1 shared-memory tree.
    lanes = np.zeros(RELION_FINE_REDUCTION_LANES, dtype=np.float32)
    for lane in range(RELION_FINE_REDUCTION_LANES):
        for pixel in range(lane, values.size, RELION_FINE_REDUCTION_LANES):
            lanes[lane] = np.float32(lanes[lane] + values[pixel])
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        for lane in range(stride):
            lanes[lane] = np.float32(lanes[lane] + lanes[lane + stride])

    actual = relion_256lane_float32_reduce(values)
    assert actual.view(np.uint32) == lanes[0].view(np.uint32)


def test_reduction_order_can_flip_near_tie_while_float64_agrees():
    # A's exact sum is 1e-7, but a flat float32 fold loses it between the two
    # 1e8 terms.  RELION's lane tree combines the cancelling terms before the
    # small term.  Candidate B sits between the two results.
    candidate_a = np.asarray([1e8, 1e-7, -1e8], dtype=np.float32)
    candidate_b = np.asarray([5e-8, 0.0, 0.0], dtype=np.float32)

    recovar_scores = [
        recovar_logical_float32_reduce(candidate_a),
        recovar_logical_float32_reduce(candidate_b),
    ]
    relion_scores = [
        relion_256lane_float32_reduce(candidate_a),
        relion_256lane_float32_reduce(candidate_b),
    ]
    canonical_f64_scores = [
        canonical_float64_reduce(candidate_a),
        canonical_float64_reduce(candidate_b),
    ]

    assert np.argmax(recovar_scores) == 1
    assert np.argmax(relion_scores) == 0
    assert np.argmax(canonical_f64_scores) == 0
    assert canonical_f64_scores[0] > canonical_f64_scores[1]


def test_canonical_reductions_use_common_integer_identity_order():
    values = np.asarray([3.0, 1e8, -1e8], dtype=np.float32)
    identities = np.asarray([2, 0, 1], dtype=np.int64)
    canonical_values = values[np.argsort(identities)]

    assert canonical_float32_reduce(values, identities) == recovar_logical_float32_reduce(canonical_values)
    assert canonical_float64_reduce(values, identities) == np.float64(3.0)


def test_promoted_float64_is_marked_and_guarded_from_genuine_source_claim():
    shifted, score_weight, projections, half_weight = _production_inputs()
    captured = normalized_cc_pixel_contributions(projections[0, 0], shifted[0, 0], score_weight[0], half_weight)
    promoted_report = replay_normalized_cc(captured)

    promoted_provenance = promoted_report.canonical_float64.provenance
    assert promoted_provenance.accumulation_dtype == "float64"
    assert promoted_provenance.precision_origin == "promoted_captured"
    assert not promoted_report.has_genuine_source_float64
    with pytest.raises(ValueError, match="promotes captured low-precision operands"):
        promoted_report.require_genuine_source_float64()

    recomputed = normalized_cc_pixel_contributions(
        projections[0, 0].astype(np.complex128),
        shifted[0, 0].astype(np.complex128),
        score_weight[0].astype(np.float64),
        half_weight.astype(np.float64),
        arithmetic_dtype=np.float64,
        precision_origin="recomputed_high_precision",
    )
    recomputed_report = replay_normalized_cc(recomputed)
    assert recomputed_report.has_genuine_source_float64
    recomputed_report.require_genuine_source_float64()

    with pytest.raises(ValueError, match="must use float32 arithmetic"):
        normalized_cc_pixel_contributions(
            projections[0, 0],
            shifted[0, 0],
            score_weight[0],
            half_weight,
            arithmetic_dtype=np.float64,
            precision_origin="captured_production",
        )
    with pytest.raises(ValueError, match="complex128/float64 source operands"):
        normalized_cc_pixel_contributions(
            projections[0, 0],
            shifted[0, 0],
            score_weight[0],
            half_weight,
            arithmetic_dtype=np.float64,
            precision_origin="recomputed_high_precision",
        )
