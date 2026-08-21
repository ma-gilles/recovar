"""Tests for the pure-NumPy RELION coarse Gaussian reduction replay."""

import hashlib
import inspect
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from recovar.em.gaussian_reduction_replay import (
    CAPTURE_SCHEMA,
    RELION_COARSE_ATOMIC_LANES,
    RELION_COARSE_CHUNK_PIXELS,
    RELION_COARSE_NONZERO_LANES,
    RELION_COARSE_PACKED_PIXELS,
    RELION_V1_BLOCK_SIZE,
    RELION_V1_CURRENT_SIZE,
    RELION_V1_ENGINE,
    RELION_V1_KERNEL_ID,
    RELION_V1_PREFETCH,
    RELION_V1_SOURCE_COMMIT,
    RELION_V1_TRANSLATION_COUNT,
    REPLAY_SCHEMA,
    GaussianCaptureIdentity,
    GaussianContributionCapture,
    HighPrecisionGaussianOperands,
    enumerate_relion_coarse_atomic_float32,
    recompute_high_precision_gaussian_contributions,
    relion_coarse_lane_partials_float32,
    replay_relion_coarse_gaussian,
)

pytestmark = pytest.mark.unit


def _identity(**changes):
    values = {
        "engine": RELION_V1_ENGINE,
        "source_commit": RELION_V1_SOURCE_COMMIT,
        "source_diff_sha256": hashlib.sha256(b"source diff").hexdigest(),
        "executable_sha256": hashlib.sha256(b"executable").hexdigest(),
        "gpu_uuid": "GPU-a100-test-uuid",
        "dataset_id": "k1-exact-reference",
        "original_particle_id": 7881,
        "iteration": 2,
        "pass_index": 0,
        "current_size": RELION_V1_CURRENT_SIZE,
        "image_sha256": hashlib.sha256(b"image").hexdigest(),
        "map_sha256": hashlib.sha256(b"map").hexdigest(),
        "ctf_sha256": hashlib.sha256(b"ctf").hexdigest(),
        "noise_sha256": hashlib.sha256(b"noise").hexdigest(),
        "support_sha256": hashlib.sha256(b"support").hexdigest(),
        "class_index": 0,
        "candidate_role": "relion_only_boundary_parent_rank48",
        "rotation_id": "rotation-4215",
        "local_rotation_index": 4215,
        "translation_id": "translation-2",
        "local_translation_index": 2,
        "candidate_geometry_sha256": hashlib.sha256(b"candidate geometry").hexdigest(),
        "packed_count": RELION_COARSE_PACKED_PIXELS,
        "block_size": RELION_V1_BLOCK_SIZE,
        "prefetch": RELION_V1_PREFETCH,
        "translation_count": RELION_V1_TRANSLATION_COUNT,
        "kernel_id": RELION_V1_KERNEL_ID,
    }
    values.update(changes)
    return GaussianCaptureIdentity(**values)


def _synthetic_contributions():
    # Lane totals [1e8, 8, 4, 4] make atomic arrival order observable in
    # float32.  The second lane-1 term is beyond an intervening zero region, so
    # compact-and-reindex also changes lane ownership.
    values = np.zeros(RELION_COARSE_PACKED_PIXELS, dtype=np.float32)
    values[[0, 5, 10, 15, 33]] = np.asarray([1e8, 4, 4, 4, 4], dtype=np.float32)
    return values


def _independent_lanes(values, *, dtype):
    lanes = np.zeros(RELION_COARSE_ATOMIC_LANES, dtype=dtype)
    counts = np.zeros(RELION_COARSE_ATOMIC_LANES, dtype=np.int64)
    for chunk_start in range(0, RELION_COARSE_PACKED_PIXELS, RELION_COARSE_CHUNK_PIXELS):
        for lane in range(RELION_COARSE_NONZERO_LANES):
            pixel = chunk_start + lane
            while pixel < min(chunk_start + RELION_COARSE_CHUNK_PIXELS, RELION_COARSE_PACKED_PIXELS):
                lanes[lane] = dtype(lanes[lane] + dtype(values[pixel]))
                counts[lane] += 1
                pixel += RELION_COARSE_NONZERO_LANES
    return lanes, counts


def _independent_raw_diff2(values, initial, order=(0, 1, 2, 3)):
    lanes, _ = _independent_lanes(values, dtype=np.float32)
    total = np.float32(initial)
    for lane in order:
        total = np.float32(total + lanes[lane])
    return total


def _production_capture(
    *,
    values=None,
    identities=None,
    capture_identity=None,
    initial=np.float32(17.0),
    observed=None,
    captured_lanes=None,
    source_dtypes=("complex64", "float32"),
):
    if values is None:
        values = _synthetic_contributions()
    if identities is None:
        identities = np.arange(RELION_COARSE_PACKED_PIXELS, dtype=np.int64)
    if capture_identity is None:
        capture_identity = _identity()
    if np.asarray(values).shape == (RELION_COARSE_PACKED_PIXELS,) and np.asarray(identities).shape == (
        RELION_COARSE_PACKED_PIXELS,
    ):
        ordered = np.empty_like(values)
        ordered[np.asarray(identities)] = values
    else:
        ordered = np.zeros(RELION_COARSE_PACKED_PIXELS, dtype=np.float32)
    if captured_lanes is None:
        captured_lanes, _ = _independent_lanes(ordered, dtype=np.float32)
    if observed is None:
        observed = _independent_raw_diff2(ordered, initial)
    return GaussianContributionCapture(
        capture_identity=capture_identity,
        contributions=np.asarray(values),
        packed_identities=np.asarray(identities),
        source_dtypes=source_dtypes,
        captured_lane_partials_float32=captured_lanes,
        initial_highres_xi2_over_2=initial,
        observed_raw_diff2=observed,
    )


def _high_precision_arrays():
    pixel = np.arange(RELION_COARSE_PACKED_PIXELS, dtype=np.float64)
    reference = (1.1250000001 + pixel * 1.3e-10) + 1j * (0.3750000003 - pixel * 1.1e-10)
    shifted = (0.2500000002 - pixel * 0.7e-10) + 1j * (-0.1250000004 + pixel * 0.9e-10)
    weight = 0.5000000003 + pixel * 0.6e-10
    return reference.astype(np.complex128), shifted.astype(np.complex128), weight.astype(np.float64)


def _high_precision_operands(*, capture_identity=None, permutation=None):
    if capture_identity is None:
        capture_identity = _identity()
    reference, shifted, weight = _high_precision_arrays()
    identities = np.arange(RELION_COARSE_PACKED_PIXELS, dtype=np.int64)
    if permutation is not None:
        reference = reference[permutation]
        shifted = shifted[permutation]
        weight = weight[permutation]
        identities = identities[permutation]
    return HighPrecisionGaussianOperands(
        capture_identity=capture_identity,
        reference_complex=reference,
        shifted_image_complex=shifted,
        corr_over_2=weight,
        packed_identities=identities,
    )


def test_exact_four_lane_schedule_all_orders_and_observed_compatibility():
    values = _synthetic_contributions()
    capture = _production_capture(values=values)
    lanes = relion_coarse_lane_partials_float32(capture)
    expected, counts = _independent_lanes(values, dtype=np.float32)

    assert np.array_equal(np.asarray(lanes.values, dtype=np.float32), expected)
    assert lanes.pixel_counts == tuple(int(value) for value in counts)
    assert lanes.values[-1] == 0.0
    assert lanes.pixel_counts == (406, 406, 406, 406, 0)

    possible = enumerate_relion_coarse_atomic_float32(capture)
    assert len(possible) == 24
    assert {result.atomic_order for result in possible} == set(permutations(range(4)))
    assert len({np.float32(result.value).view(np.uint32).item() for result in possible}) > 1

    report = replay_relion_coarse_gaussian(capture)
    assert (0, 1, 2, 3) in report.compatible_atomic_orders_float32
    assert all(
        np.float32(result.value).view(np.uint32) == capture.observed_raw_diff2.view(np.uint32)
        for result in report.possible_atomic_float32
        if result.atomic_order in report.compatible_atomic_orders_float32
    )
    serialized = report.to_dict()
    assert serialized["observed_raw_diff2_float32"]["value"] == float(capture.observed_raw_diff2)
    assert serialized["observed_raw_diff2_float32"]["dtype"] == "float32"
    assert not serialized["observed_raw_diff2_float32"]["atomic_order_observed"]
    assert not serialized["production_atomic_order_observed"]


def test_incompatible_observed_raw_diff2_fails_closed():
    capture = _production_capture(observed=np.float32(1.0))
    with pytest.raises(ValueError, match="incompatible with all 24 RELION atomic orders"):
        replay_relion_coarse_gaussian(capture)


def test_device_captured_lane_mismatch_fails_closed():
    lanes, _ = _independent_lanes(_synthetic_contributions(), dtype=np.float32)
    lanes[2] = np.nextafter(lanes[2], np.float32(np.inf), dtype=np.float32)
    capture = _production_capture(captured_lanes=lanes)
    with pytest.raises(ValueError, match="lane partials do not bitwise match"):
        replay_relion_coarse_gaussian(capture)


def test_initial_and_observed_production_scalars_are_required_float32():
    signature = inspect.signature(GaussianContributionCapture)
    assert signature.parameters["initial_highres_xi2_over_2"].default is inspect.Parameter.empty
    assert signature.parameters["observed_raw_diff2"].default is inspect.Parameter.empty
    assert signature.parameters["captured_lane_partials_float32"].default is inspect.Parameter.empty
    for field, value in (
        ("initial", np.float64(17.0)),
        ("initial", 17.0),
        ("observed", np.float64(1e8)),
        ("observed", 1e8),
    ):
        kwargs = {field: value}
        with pytest.raises(ValueError, match="explicitly captured float32 scalar"):
            _production_capture(**kwargs)


def test_compaction_missing_and_nonunique_identities_fail_closed():
    values = _synthetic_contributions()
    active = values != 0
    with pytest.raises(ValueError, match="requires all 1624 contributions"):
        _production_capture(values=values[active], identities=np.flatnonzero(active))

    duplicate = np.arange(RELION_COARSE_PACKED_PIXELS, dtype=np.int64)
    duplicate[1] = 0
    with pytest.raises(ValueError, match="missing or duplicate identities"):
        _production_capture(identities=duplicate)


def test_capture_storage_permutation_is_identity_invariant_and_source_dtypes_are_copied():
    values = _synthetic_contributions()
    permutation = np.random.default_rng(1624).permutation(RELION_COARSE_PACKED_PIXELS)
    source_dtypes = ["complex64", "float32"]
    shuffled = _production_capture(
        values=values[permutation],
        identities=permutation,
        source_dtypes=source_dtypes,
    )
    source_dtypes[0] = "complex128"
    baseline = _production_capture(values=values)

    assert shuffled.source_dtypes == ("complex64", "float32")
    assert relion_coarse_lane_partials_float32(shuffled).values == relion_coarse_lane_partials_float32(baseline).values
    assert [value.value for value in enumerate_relion_coarse_atomic_float32(shuffled)] == [
        value.value for value in enumerate_relion_coarse_atomic_float32(baseline)
    ]


def test_promoted_float64_has_all_relion_orders_and_canonical_result():
    values = _synthetic_contributions()
    capture = _production_capture(values=values, initial=np.float32(3.25))
    report = replay_relion_coarse_gaussian(capture)

    assert report.schema == REPLAY_SCHEMA
    assert len(report.promoted_float64_relion_lane_orders) == 24
    assert {item.atomic_order for item in report.promoted_float64_relion_lane_orders} == set(permutations(range(4)))
    expected_lanes, _ = _independent_lanes(values, dtype=np.float64)
    for item in report.promoted_float64_relion_lane_orders:
        expected = np.float64(np.float32(3.25))
        for lane in item.atomic_order:
            expected = np.float64(expected + expected_lanes[lane])
        assert item.value == expected

    expected_canonical = np.float64(np.float32(3.25))
    for value in values:
        expected_canonical = np.float64(expected_canonical + np.float64(value))
    assert report.promoted_float64_canonical.value == expected_canonical
    assert report.promoted_float64_canonical.provenance.source_kind == "promoted_capture"
    assert report.promoted_float64_canonical.provenance.scope == "full_raw_diff2"


def test_genuine_float64_is_recomputed_from_operands_and_is_centered_only():
    capture = _production_capture()
    permutation = np.arange(RELION_COARSE_PACKED_PIXELS, dtype=np.int64)[::-1]
    operands = _high_precision_operands(permutation=permutation)
    report = replay_relion_coarse_gaussian(capture, high_precision_operands=operands)

    reference, shifted, weight = _high_precision_arrays()
    expected_contributions = ((reference.real - shifted.real) ** 2 + (reference.imag - shifted.imag) ** 2) * weight
    actual_contributions = recompute_high_precision_gaussian_contributions(operands)
    assert np.array_equal(actual_contributions, expected_contributions)

    assert report.has_genuine_centered_float64
    assert len(report.genuine_float64_centered_relion_lane_orders) == 24
    assert {item.atomic_order for item in report.genuine_float64_centered_relion_lane_orders} == set(
        permutations(range(4))
    )
    expected_lanes, _ = _independent_lanes(expected_contributions, dtype=np.float64)
    for item in report.genuine_float64_centered_relion_lane_orders:
        expected = np.float64(0.0)
        for lane in item.atomic_order:
            expected = np.float64(expected + expected_lanes[lane])
        assert item.value == expected
        assert item.provenance.scope == "centered_contribution_only_no_highres_xi2"
        assert item.provenance.initial_term is None

    expected_canonical = np.float64(0.0)
    for value in expected_contributions:
        expected_canonical = np.float64(expected_canonical + value)
    genuine = report.require_genuine_centered_float64()
    assert genuine.value == expected_canonical
    assert genuine.provenance.source_dtypes == ("complex128", "complex128", "float64")
    assert genuine.provenance.source_array_sha256 == operands.source_array_sha256
    serialized = report.to_dict()
    assert serialized["genuine_float64_scope"] == "centered_contribution_only_no_highres_xi2"
    assert serialized["genuine_float64_centered_canonical"]["dtype_provenance"]["initial_term"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"source_diff_sha256": hashlib.sha256(b"other diff").hexdigest()},
        {"executable_sha256": hashlib.sha256(b"other executable").hexdigest()},
        {"gpu_uuid": "GPU-other-uuid"},
        {"dataset_id": "different-dataset"},
        {"image_sha256": hashlib.sha256(b"other image").hexdigest()},
        {"map_sha256": hashlib.sha256(b"other map").hexdigest()},
        {"ctf_sha256": hashlib.sha256(b"other ctf").hexdigest()},
        {"noise_sha256": hashlib.sha256(b"other noise").hexdigest()},
        {"support_sha256": hashlib.sha256(b"other support").hexdigest()},
        {
            "candidate_role": "global_winner",
            "rotation_id": "rotation-8246",
            "local_rotation_index": 8246,
            "translation_id": "translation-8",
            "local_translation_index": 8,
        },
        {"candidate_geometry_sha256": hashlib.sha256(b"other geometry").hexdigest()},
    ],
)
def test_complete_capture_identity_mismatch_is_rejected(changes):
    production_identity = _identity()
    high_precision_identity = replace(production_identity, **changes)
    capture = _production_capture(capture_identity=production_identity)
    operands = _high_precision_operands(capture_identity=high_precision_identity)
    with pytest.raises(ValueError, match="capture identities differ"):
        replay_relion_coarse_gaussian(capture, high_precision_operands=operands)


def test_capture_identity_schema_and_version_are_enforced():
    with pytest.raises(ValueError, match="capture identity must use"):
        _identity(schema="wrong.schema")
    with pytest.raises(ValueError, match="capture identity must use"):
        _identity(version=2)
    assert _identity().schema == CAPTURE_SCHEMA


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"engine": "RECOVAR"}, "engine must be"),
        ({"source_commit": "0" * 40}, "source_commit must be"),
        ({"kernel_id": "other-kernel"}, "kernel_id must be"),
        ({"original_particle_id": 7882}, "original_particle_id must be 7881"),
        ({"iteration": 3}, "iteration must be 2"),
        ({"pass_index": 1}, "pass_index must be 0"),
        ({"class_index": 1}, "class_index must be 0"),
        ({"current_size": 128}, "current_size must be 56"),
        ({"packed_count": 8320}, "packed_count must be 1624"),
        ({"block_size": 256}, "block_size must be 128"),
        ({"prefetch": 2}, "prefetch must be 4"),
        ({"translation_count": 30}, "translation_count must be 29"),
        ({"local_rotation_index": 28}, "requires local_rotation_index 4215"),
        ({"rotation_id": "rotation-28"}, "rotation_id must be the numeric-derived"),
        ({"local_translation_index": 28}, "requires local_translation_index 2"),
        ({"translation_id": "translation-28"}, "translation_id must be the numeric-derived"),
    ],
)
def test_pinned_v1_kernel_layout_rejects_incompatible_identity(changes, message):
    with pytest.raises(ValueError, match=message):
        _identity(**changes)


def test_serialized_identity_contains_complete_schema_v1_provenance():
    serialized = _identity().to_dict()
    required = {
        "engine",
        "source_commit",
        "source_diff_sha256",
        "executable_sha256",
        "gpu_uuid",
        "iteration",
        "pass",
        "original_particle_id",
        "current_size",
        "image_sha256",
        "map_sha256",
        "ctf_sha256",
        "noise_sha256",
        "support_sha256",
        "candidate_role",
        "rotation_id",
        "local_rotation_index",
        "translation_id",
        "local_translation_index",
        "candidate_geometry_sha256",
        "packed_count",
        "block_size",
        "prefetch",
        "translation_count",
        "kernel_id",
    }
    assert required <= serialized.keys()


@pytest.mark.parametrize(
    ("role", "rotation_index", "translation_index"),
    [
        ("global_winner", 8246, 8),
        ("selected_cutoff_neighbor_rank47", 4504, 10),
        ("relion_only_boundary_parent_rank48", 4215, 2),
        ("first_excluded_control_rank49", 8246, 0),
    ],
)
def test_v1_target_roles_pin_numeric_rotation_and_translation_indices(role, rotation_index, translation_index):
    identity = _identity(
        candidate_role=role,
        rotation_id=f"rotation-{rotation_index}",
        local_rotation_index=rotation_index,
        translation_id=f"translation-{translation_index}",
        local_translation_index=translation_index,
    )
    assert identity.local_rotation_index == rotation_index
    assert identity.local_translation_index == translation_index


def test_cast_spoofed_contributions_and_high_precision_operands_are_rejected():
    with pytest.raises(ValueError, match="production contributions must be captured float32"):
        _production_capture(values=_synthetic_contributions().astype(np.float64))

    reference, shifted, weight = _high_precision_arrays()
    spoofed_reference = reference.astype(np.complex64).astype(np.complex128)
    spoofed_shifted = shifted.astype(np.complex64).astype(np.complex128)
    spoofed_weight = weight.astype(np.float32).astype(np.float64)
    with pytest.raises(ValueError, match="all-cast-spoofed capture"):
        HighPrecisionGaussianOperands(
            capture_identity=_identity(),
            reference_complex=spoofed_reference,
            shifted_image_complex=spoofed_shifted,
            corr_over_2=spoofed_weight,
            packed_identities=np.arange(RELION_COARSE_PACKED_PIXELS),
        )


def test_exactly_representable_float64_weights_are_allowed_and_flagged():
    reference, shifted, _ = _high_precision_arrays()
    exact_weight = np.full(RELION_COARSE_PACKED_PIXELS, 0.5, dtype=np.float64)
    operands = HighPrecisionGaussianOperands(
        capture_identity=_identity(),
        reference_complex=reference,
        shifted_image_complex=shifted,
        corr_over_2=exact_weight,
        packed_identities=np.arange(RELION_COARSE_PACKED_PIXELS),
    )
    assert operands.source_beyond_production_precision == (True, True, False)

    report = replay_relion_coarse_gaussian(
        _production_capture(),
        high_precision_operands=operands,
    )
    provenance = report.genuine_float64_centered_canonical.provenance
    assert provenance.source_beyond_production_precision == (True, True, False)
    assert provenance.source_array_sha256 == operands.source_array_sha256
    serialized = report.to_dict()["genuine_float64_centered_canonical"]["dtype_provenance"]
    assert serialized["source_beyond_production_precision"] == [True, True, False]
