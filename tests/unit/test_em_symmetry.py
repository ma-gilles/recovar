"""RELION rotational point-group contract tests."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.symmetry import (
    canonicalize_rotational_symmetry,
    parse_rotational_symmetry,
    relion_symmetry_operators,
    rotational_operators,
    symmetry_operator_sha256,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("family", ["C", "D"])
@pytest.mark.parametrize("order", range(1, 100))
def test_all_relion_rotational_orders_parse(family, order):
    parsed = parse_rotational_symmetry(f"{family}{order}")
    assert parsed.label == f"{family}{order}"
    assert parsed.operator_count == order * (1 if family == "C" else 2)
    operators = rotational_operators(parsed.label)
    assert operators.shape == (parsed.operator_count, 3, 3)
    np.testing.assert_allclose(
        np.linalg.det(operators), 1.0, rtol=0.0, atol=1e-9
    )


@pytest.mark.parametrize(
    ("requested", "canonical", "count"),
    [
        ("c1", "C1", 1),
        ("T", "T", 12),
        ("o", "O", 24),
        ("I", "I2", 60),
        ("i1", "I1", 60),
        ("I2", "I2", 60),
        ("i3", "I3", 60),
        ("I4", "I4", 60),
    ],
)
def test_relion_rotational_aliases(requested, canonical, count):
    parsed = parse_rotational_symmetry(requested)
    assert parsed.label == canonical
    assert parsed.operator_count == count
    assert canonicalize_rotational_symmetry(requested) == canonical


@pytest.mark.parametrize(
    "label",
    [
        "",
        "C0",
        "C100",
        "D0",
        "D100",
        "S2",
        "S7",
        "CI",
        "CS",
        "C2V",
        "C2H",
        "D7V",
        "D7H",
        "TD",
        "TH",
        "OH",
        "IH",
        "I1H",
        "I2H",
        "I3H",
        "I4H",
        "I5",
        "I5H",
        "P1",
    ],
)
def test_nonrotational_or_unimplemented_groups_fail_closed(label):
    with pytest.raises(ValueError):
        parse_rotational_symmetry(label)


@pytest.mark.parametrize(
    ("label", "count"),
    [("C1", 1), ("C7", 7), ("D1", 2), ("D5", 10), ("T", 12), ("O", 24)]
    + [(label, 60) for label in ("I1", "I2", "I3", "I4")],
)
def test_relion_operator_counts_and_properness(label, count):
    left, right = relion_symmetry_operators(label)
    np.testing.assert_array_equal(left[0], np.eye(3))
    np.testing.assert_array_equal(right[0], np.eye(3))
    assert left.shape == right.shape == (count, 3, 3)
    identity_stack = np.broadcast_to(np.eye(3), left.shape)
    np.testing.assert_allclose(left, identity_stack, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(right), 1.0, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        right @ np.swapaxes(right, -1, -2), identity_stack, rtol=0.0, atol=2e-7
    )
    np.testing.assert_allclose(rotational_operators(label), right, rtol=0.0, atol=0.0)


def test_c1_operator_lookup_does_not_require_relion_symmetry_files(tmp_path, monkeypatch):
    """The default C1 provenance path remains portable outside a RELION tree."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RELION_HOME", raising=False)
    left, right = relion_symmetry_operators("C1")
    np.testing.assert_array_equal(left, np.eye(3)[None, :, :])
    np.testing.assert_array_equal(right, np.eye(3)[None, :, :])


@pytest.mark.parametrize("label", ["C7", "D5", "T", "O", "I1", "I2", "I3", "I4"])
def test_relion_operator_sets_are_closed(label):
    operators = rotational_operators(label)
    products = np.einsum("aij,bjk->abik", operators, operators).reshape(-1, 3, 3)
    # Every product must match one of RELION's ordered operators.  Trace of
    # P Q^T gives the relative rotation angle without Euler singularities.
    traces = np.einsum("pij,qij->pq", products, operators)
    best_cosine = np.max(np.clip((traces - 1.0) / 2.0, -1.0, 1.0), axis=1)
    np.testing.assert_allclose(best_cosine, 1.0, rtol=0.0, atol=2e-7)


def test_i_alias_is_exact_and_digest_is_stable():
    left_i, right_i = relion_symmetry_operators("I")
    left_i2, right_i2 = relion_symmetry_operators("I2")
    np.testing.assert_array_equal(left_i, left_i2)
    np.testing.assert_array_equal(right_i, right_i2)
    assert symmetry_operator_sha256("I") == symmetry_operator_sha256("I2")


def test_icosahedral_conventions_are_distinct_ordered_sets():
    digests = [symmetry_operator_sha256(label) for label in ("I1", "I2", "I3", "I4")]
    assert len(set(digests)) == 4
    # Each convention represents the same abstract 60-element group, but in a
    # different coordinate frame.  Their non-identity arrays must not be
    # accidentally substituted for each other.
    for left, right in itertools.combinations(("I1", "I2", "I3", "I4"), 2):
        assert not np.array_equal(rotational_operators(left), rotational_operators(right))


@pytest.mark.parametrize(
    "label", ["C1", "C7", "D5", "T", "O", "I1", "I2", "I3", "I4"]
)
def test_symmetry_reduced_coarse_grid_matches_relion_binding(label):
    from recovar.relion_bind._relion_bind_core import (
        get_coarse_orientations,
        get_healpix_sampling_metadata,
    )

    from recovar import utils
    from recovar.em import sampling

    order = 3
    source = get_healpix_sampling_metadata(order, -1.0, label)
    source_eulers = np.asarray(
        get_coarse_orientations(order, -1.0, label), dtype=np.float64
    )
    n_directions = len(source["rot"])
    n_psi = len(source["psi"])

    native_eulers = sampling.get_relion_rotation_grid_eulers(
        order, rotation_index_order="relion", symmetry=label
    )
    native_rotations = sampling.get_relion_rotation_grid(
        order, rotation_index_order="relion", symmetry=label
    )
    recovar_rotations = sampling.get_relion_rotation_grid(
        order, rotation_index_order="recovar", symmetry=label
    )
    metadata = sampling.build_local_search_grid_metadata(order, symmetry=label)

    assert n_directions > 0
    assert native_eulers.shape == source_eulers.shape == (
        n_directions * n_psi,
        3,
    )
    assert sampling.rotation_grid_size(order, label) == n_directions * n_psi
    # The public Euler table is intentionally float32; pin that conversion
    # exactly while testing the matrix path against source-precision rows.
    np.testing.assert_array_equal(native_eulers, source_eulers.astype(np.float32))
    np.testing.assert_allclose(
        native_rotations,
        utils.R_from_relion(source_eulers, degrees=True),
        rtol=0.0,
        atol=4e-8,
    )
    np.testing.assert_allclose(
        recovar_rotations,
        native_rotations.reshape(n_directions, n_psi, 3, 3)
        .transpose(1, 0, 2, 3)
        .reshape(-1, 3, 3),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        metadata["directions_ipix"], source["directions_ipix"]
    )
    assert np.all(metadata["directions_ipix"] >= 0)
    assert np.all(metadata["directions_ipix"] < 12 * (2**order) ** 2)


@pytest.mark.parametrize(
    "label", ["C1", "C7", "D5", "T", "O", "I1", "I2", "I3", "I4"]
)
def test_symmetry_oversampling_matches_relion_binding(label):
    from recovar.relion_bind._relion_bind_core import (
        get_healpix_sampling_metadata,
        get_oversampled_orientations,
    )

    from recovar import utils
    from recovar.em import sampling

    order = 3
    source = get_healpix_sampling_metadata(order, -1.0, label)
    n_directions = len(source["rot"])
    n_psi = len(source["psi"])
    direction_index = min(1, n_directions - 1)
    psi_index = min(2, n_psi - 1)
    parent_index = np.asarray(
        [psi_index * n_directions + direction_index], dtype=np.int64
    )

    actual, parent_map = sampling.get_oversampled_rotation_grid_from_samples(
        parent_index,
        order,
        oversampling_order=1,
        rotation_index_order="recovar",
        symmetry=label,
    )
    expected_eulers = np.asarray(
        get_oversampled_orientations(
            order, 1, direction_index, psi_index, 0.0, label
        ),
        dtype=np.float64,
    )
    expected = utils.R_from_relion(expected_eulers, degrees=True)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(parent_map, np.zeros(8, dtype=np.int64))


@pytest.mark.parametrize(
    ("label", "child_ids", "expected_n_global", "expected_posterior_ids"),
    [
        ("C1", np.asarray([3, 4, 5, 6]), 20, None),
        ("I1", np.asarray([100, 101, 102, 103]), 8, np.asarray([2, 2, 7, 7])),
    ],
)
def test_parent_expanded_local_layout_maps_non_c1_children_to_reduced_parent_ids(
    monkeypatch,
    label,
    child_ids,
    expected_n_global,
    expected_posterior_ids,
):
    from recovar.em.local import local_layout

    parent_ids = np.asarray([2, 7], dtype=np.int32)
    parent_map = np.asarray([0, 0, 1, 1], dtype=np.int64)

    monkeypatch.setattr(
        local_layout,
        "build_local_search_grid_metadata",
        lambda _order, *, symmetry: {"symmetry": symmetry},
    )
    monkeypatch.setattr(
        local_layout,
        "_build_factorized_local_entries",
        lambda *_args, **_kwargs: (
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([2], dtype=np.int32),
            parent_ids,
            np.asarray([-0.25, -0.75], dtype=np.float32),
        ),
    )

    def fake_oversampled(_parent_ids, _parent_order, **kwargs):
        assert kwargs["symmetry"] == label
        return (
            np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy(),
            parent_map,
            child_ids,
            np.zeros((4, 3), dtype=np.float64),
        )

    monkeypatch.setattr(
        local_layout,
        "get_oversampled_rotation_grid_from_samples",
        fake_oversampled,
    )
    monkeypatch.setattr(
        local_layout,
        "rotation_grid_size",
        lambda _order, symmetry: 8 if symmetry == "I1" else 20,
    )

    layout = local_layout.build_local_hypothesis_layout(
        np.zeros((1, 3), dtype=np.float32),
        None,
        sigma_rot=0.0,
        sigma_psi=0.0,
        healpix_order=4,
        translations=np.zeros((1, 2), dtype=np.float32),
        prior_translations=np.zeros((1, 2), dtype=np.float32),
        sigma_offset_angstrom=1.0,
        offset_range_pixels=None,
        voxel_size=1.0,
        grid_metadata={
            "mode": "factorized",
            "n_pixels": np.asarray(5),
            "n_psi": np.asarray(4),
            "symmetry": label,
        },
        local_parent_oversampling_order=1,
    )

    np.testing.assert_array_equal(layout.rotation_ids_flat, child_ids)
    assert layout.n_global_rotations == expected_n_global
    if expected_posterior_ids is None:
        assert layout.rotation_posterior_ids_flat is None
    else:
        np.testing.assert_array_equal(
            layout.rotation_posterior_ids_flat,
            expected_posterior_ids,
        )
        assert int(np.max(layout.rotation_ids_flat)) >= layout.n_global_rotations
        assert int(np.max(layout.rotation_posterior_ids_flat)) < layout.n_global_rotations


@pytest.mark.parametrize("label", ["I2", "I3"])
def test_empty_too_coarse_asymmetric_unit_fails_closed(label):
    from recovar.em.sampling import rotation_grid_size

    with pytest.raises(ValueError, match="increase the initial HEALPix order"):
        rotation_grid_size(1, label)


@pytest.mark.parametrize(
    "label", ["C7", "D5", "T", "O", "I1", "I2", "I3", "I4"]
)
def test_convergence_distance_minimizes_over_relion_symmetry_mates(label):
    from recovar import utils
    from recovar.em.helpers.convergence import (
        _relion_angular_distance_per_particle as relion_angular_distance_per_particle,
    )

    base = utils.R_from_relion(
        np.asarray([[31.0, 67.0, -43.0]], dtype=np.float64), degrees=True
    )
    symmetry_mate = base @ rotational_operators(label)[1][None, :, :]

    c1_distance = relion_angular_distance_per_particle(symmetry_mate, base)
    symmetry_distance = relion_angular_distance_per_particle(
        symmetry_mate, base, symmetry_label=label
    )

    assert c1_distance[0] > 1.0
    np.testing.assert_allclose(symmetry_distance, 0.0, rtol=0.0, atol=2e-6)


def test_explicit_c1_convergence_distance_preserves_default_path_exactly():
    from recovar import utils
    from recovar.em.helpers.convergence import (
        _relion_angular_distance_per_particle as relion_angular_distance_per_particle,
    )

    first = utils.R_from_relion(
        np.asarray([[17.0, 83.0, -29.0], [-91.0, 42.0, 113.0]]), degrees=True
    )
    second = utils.R_from_relion(
        np.asarray([[18.0, 81.0, -33.0], [-95.0, 39.0, 109.0]]), degrees=True
    )
    default = relion_angular_distance_per_particle(first, second)
    explicit = relion_angular_distance_per_particle(
        first, second, symmetry_label="C1"
    )
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize(
    "label", ["C7", "D5", "T", "O", "I1", "I2", "I3", "I4"]
)
def test_symmetry_direction_prior_collapse_and_expansion(label):
    from recovar.em.helpers.orientation_priors import (
        collapse_rotation_posterior_to_direction_prior,
        make_relion_direction_log_prior,
    )
    from recovar.em.sampling import rotation_grid_n_in_planes, rotation_grid_size

    order = 3
    n_rotations = rotation_grid_size(order, label)
    n_directions = n_rotations // rotation_grid_n_in_planes(order)
    posterior = np.arange(1, n_rotations + 1, dtype=np.float64)

    direction_prior = collapse_rotation_posterior_to_direction_prior(
        posterior,
        order,
        label,
    )
    expanded = make_relion_direction_log_prior(
        direction_prior,
        order,
        symmetry=label,
    )

    assert direction_prior.shape == (n_directions,)
    assert expanded.shape == (n_rotations,)
    np.testing.assert_allclose(direction_prior.sum(), 1.0, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(
        np.exp(expanded),
        np.tile(direction_prior, rotation_grid_n_in_planes(order)),
        rtol=2e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "label", ["C7", "D5", "T", "O", "I1", "I2", "I3", "I4"]
)
def test_symmetry_direction_prior_geometry_expansion_matches_canonical_ids(label):
    from recovar.em.helpers.orientation_priors import (
        make_relion_direction_log_prior,
    )
    from recovar.em.sampling import (
        get_relion_rotation_grid,
        rotation_grid_n_in_planes,
        rotation_grid_size,
    )

    order = 3
    n_rotations = rotation_grid_size(order, label)
    n_directions = n_rotations // rotation_grid_n_in_planes(order)
    direction_prior = np.arange(1, n_directions + 1, dtype=np.float32)
    direction_prior /= direction_prior.sum()
    rotations = get_relion_rotation_grid(order, symmetry=label)

    by_id = make_relion_direction_log_prior(
        direction_prior,
        order,
        symmetry=label,
    )
    by_geometry = make_relion_direction_log_prior(
        direction_prior,
        order,
        rotations=rotations,
        symmetry=label,
    )
    np.testing.assert_array_equal(by_geometry, by_id)


@pytest.mark.parametrize(
    "label", ["C7", "D5", "T", "O", "I1", "I2", "I3", "I4"]
)
def test_symmetry_direction_prior_remap_is_normalized_and_uniform_preserving(label):
    from recovar.em.helpers.orientation_priors import (
        infer_direction_prior_healpix_order,
        remap_direction_prior_to_healpix_order,
    )
    from recovar.em.sampling import build_local_search_grid_metadata

    src_order = 3
    dst_order = 4
    src_count = int(
        build_local_search_grid_metadata(src_order, symmetry=label)["n_pixels"]
    )
    dst_count = int(
        build_local_search_grid_metadata(dst_order, symmetry=label)["n_pixels"]
    )
    source = np.full(src_count, 1.0 / src_count, dtype=np.float32)

    assert (
        infer_direction_prior_healpix_order(
            source,
            symmetry=label,
            expected_order=src_order,
        )
        == src_order
    )
    upsampled = remap_direction_prior_to_healpix_order(
        source,
        src_order,
        dst_order,
        symmetry=label,
    )
    downsampled = remap_direction_prior_to_healpix_order(
        upsampled,
        dst_order,
        src_order,
        symmetry=label,
    )

    assert upsampled.shape == (dst_count,)
    assert downsampled.shape == (src_count,)
    assert np.all(np.isfinite(upsampled)) and np.all(upsampled >= 0.0)
    assert np.all(np.isfinite(downsampled)) and np.all(downsampled >= 0.0)
    np.testing.assert_allclose(upsampled.sum(), 1.0, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(downsampled.sum(), 1.0, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(
        upsampled,
        np.full(dst_count, 1.0 / dst_count, dtype=np.float32),
        rtol=2e-6,
        atol=2e-8,
    )


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [("c7", "C7"), ("d5", "D5"), ("t", "T"), ("o", "O"), ("i", "I2"), ("i1", "I1")],
)
def test_refinement_symmetry_options_canonicalize_at_construction(requested, canonical):
    from recovar.em.refinement.refinement_options import SymmetryOptions

    assert SymmetryOptions(point_group=requested).point_group == canonical


@pytest.mark.parametrize("label", ["C7", "D5", "T", "O", "I1", "I2", "I3", "I4"])
def test_firstiter_cc_fine_grid_uses_reduced_asymmetric_unit_ids(label):
    from recovar.em.helpers.oversampling import (
        build_adaptive_pass2_grids,
    )
    from recovar.em.sampling import (
        get_oversampled_rotation_grid_from_samples,
        get_relion_rotation_grid,
    )

    order = 3
    coarse = get_relion_rotation_grid(order, symmetry=label).astype(np.float32)
    coarse_ids = np.arange(coarse.shape[0], dtype=np.int64)
    expected_rotations, expected_parent = get_oversampled_rotation_grid_from_samples(
        coarse_ids,
        order,
        oversampling_order=1,
        symmetry=label,
    )
    actual = build_adaptive_pass2_grids(
        coarse,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float64),
        coarse_healpix_order=order,
        adaptive_oversampling=1,
        translation_step_px=1.0,
        random_perturbation=0.0,
        symmetry=label,
    )

    np.testing.assert_array_equal(actual[2], expected_rotations)
    np.testing.assert_array_equal(actual[4], expected_parent)


@pytest.mark.parametrize("label", ["C7", "D5", "T", "O", "I1", "I2", "I3", "I4"])
def test_sparse_pass2_fallback_uses_symmetry_reduced_parent_ids(label):
    from recovar.em.scoring.sparse_bucket_arrays import (
        _prepare_per_image_pass2_inputs,
    )
    from recovar.em.sampling import (
        get_oversampled_rotation_grid_from_samples,
        rotation_grid_size,
    )

    order = 3
    n_coarse_rotations = rotation_grid_size(order, label)
    expected_rotations, expected_parent = get_oversampled_rotation_grid_from_samples(
        np.arange(n_coarse_rotations, dtype=np.int64),
        order,
        oversampling_order=1,
        symmetry=label,
    )
    prepared = _prepare_per_image_pass2_inputs(
        [None],
        n_coarse_rot=n_coarse_rotations,
        n_coarse_trans=1,
        nside_level=order,
        oversampling_order=1,
        n_fine_trans=4,
        fine_translation_parent=np.zeros(4, dtype=np.int64),
        rotation_log_prior=None,
        random_perturbation=0.0,
        symmetry_label=label,
    )

    np.testing.assert_array_equal(prepared["oversampled_rots"][0], expected_rotations)
    np.testing.assert_array_equal(prepared["parent_map"][0], expected_parent)


def test_sparse_pass2_rejects_c1_sized_parent_grid_for_i1():
    from recovar.em.scoring.sparse_bucket_arrays import (
        _prepare_per_image_pass2_inputs,
    )
    from recovar.em.sampling import rotation_grid_size

    with pytest.raises(ValueError, match="I1 sparse pass-2 coarse rotation count mismatch"):
        _prepare_per_image_pass2_inputs(
            [None],
            n_coarse_rot=rotation_grid_size(3, "C1"),
            n_coarse_trans=1,
            nside_level=3,
            oversampling_order=1,
            n_fine_trans=4,
            fine_translation_parent=np.zeros(4, dtype=np.int64),
            rotation_log_prior=None,
            random_perturbation=0.0,
            symmetry_label="I1",
        )


@pytest.mark.parametrize("label", ["C7", "D5", "T", "O", "I1", "I2", "I3", "I4"])
def test_symmetry_reduced_coarse_tie_break_uses_relion_direction_major_order(label):
    from recovar.em.relion.relion_coarse_operands import (
        _infer_relion_coarse_healpix_order,
        _relion_coarse_pose_tie_break_keys,
        _select_relion_coarse_rescore_winner_slots,
    )
    from recovar.em.sampling import rotation_grid_n_in_planes, rotation_grid_size

    order = 3
    n_rotations = rotation_grid_size(order, label)
    n_directions = n_rotations // rotation_grid_n_in_planes(order)
    candidate_pose_ids = np.asarray([[1, n_directions]], dtype=np.int64)

    assert _infer_relion_coarse_healpix_order(n_rotations, label) == order
    keys = _relion_coarse_pose_tie_break_keys(
        candidate_pose_ids,
        n_trans=1,
        healpix_order=order,
        symmetry_label=label,
    )
    np.testing.assert_array_equal(
        keys,
        np.asarray([[rotation_grid_n_in_planes(order), 1]], dtype=np.int64),
    )
    slots, tie_count = _select_relion_coarse_rescore_winner_slots(
        np.asarray([[0.5, 0.5]], dtype=np.float32),
        candidate_pose_ids,
        n_trans=1,
        healpix_order=order,
        symmetry_label=label,
    )
    np.testing.assert_array_equal(slots, np.asarray([1], dtype=np.int32))
    assert tie_count == 1


@pytest.mark.parametrize(
    ("sparse_pass2", "x_half", "message"),
    [
        (False, True, "requires sparse pass 2"),
        (True, False, "requires RELION x-half BPref"),
    ],
)
def test_non_c1_adaptive_engine_rejects_unsupported_reconstruction_route_early(
    sparse_pass2,
    x_half,
    message,
):
    from recovar.em.classification.k_class import (
        run_dense_k_class_em_adaptive,
    )

    # Route validation precedes dataset access and pass-1 scoring.  Deliberately
    # invalid placeholders prove that an unsupported symmetry configuration
    # fails before doing expensive work.
    with pytest.raises(RuntimeError, match=message):
        run_dense_k_class_em_adaptive(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            symmetry_label="I1",
            sparse_pass2=sparse_pass2,
            mstep_relion_x_half=x_half,
        )


def test_non_c1_adaptive_engine_requires_explicit_oversampling_order():
    from recovar.em.classification.k_class import (
        run_dense_k_class_em_adaptive,
    )

    with pytest.raises(ValueError, match="requires explicit oversampling_order"):
        run_dense_k_class_em_adaptive(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            symmetry_label="I1",
            sparse_pass2=True,
            mstep_relion_x_half=True,
        )


@pytest.mark.parametrize("n_classes", [1, 2])
def test_non_c1_nonadaptive_dense_reconstruction_fails_before_run_em(
    monkeypatch,
    n_classes,
):
    from recovar.em.refinement import half_scoring as iteration_loop

    monkeypatch.setattr(
        iteration_loop,
        "run_em",
        lambda *_args, **_kwargs: pytest.fail("unsupported non-C1 route called run_em"),
    )
    from recovar.em.dense.score_outputs import PerHalfOutputs

    common = dict(
        outputs=PerHalfOutputs(),
        k=0,
        experiment_dataset=object(),
        means_k=(
            np.zeros(8, dtype=np.complex64)
            if n_classes == 1
            else np.zeros((n_classes, 8), dtype=np.complex64)
        ),
        mean_variance=np.ones(8, dtype=np.float32),
        noise_variance_k=np.ones(8, dtype=np.float32),
        effective_rotations=np.eye(3, dtype=np.float32)[None, :, :],
        current_translations=np.zeros((1, 2), dtype=np.float32),
        base_translations=np.zeros((1, 2), dtype=np.float32),
        current_healpix_order=0,
        state=SimpleNamespace(adaptive_oversampling=0, translation_step=1.0),
        random_perturbation=0.0,
        disc_type="linear_interp",
        image_batch_size=1,
        rotation_log_prior_k=None,
        class_rotation_log_prior_k=None,
        translation_log_prior=None,
        translation_search_base=None,
        trans_prior_center_for_engine=None,
        image_corrections_k=None,
        scale_corrections_k=None,
        firstiter_score_mode_this_iter="gaussian",
        firstiter_winner_take_all_this_iter=False,
        cs_for_engine=4,
        class_log_priors=(
            None
            if n_classes == 1
            else np.full(n_classes, -np.log(n_classes), dtype=np.float64)
        ),
        k_class_enabled=n_classes > 1,
        relion_firstiter_cc_this_iter=False,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        safe_batch_sizes=lambda *_args, **_kwargs: (1, 1),
        max_significants=-1,
        symmetry="I1",
    )
    with pytest.raises(NotImplementedError, match="non-adaptive dense"):
        iteration_loop._score_half_dense(**common)


@pytest.mark.parametrize("label", ["C1", "C7", "D5", "T", "O", "I1", "I2", "I3", "I4"])
def test_kclass_healpix_order_inference_uses_symmetry_reduced_grid(label):
    from recovar.em.classification.k_class import (
        _infer_healpix_order_from_rotation_count,
    )
    from recovar.em.sampling import rotation_grid_size

    # Very coarse grids can retain no ASU direction for some icosahedral
    # conventions. Order three is the production global-search start and is
    # valid for every supported proper rotational group.
    order = 3
    assert (
        _infer_healpix_order_from_rotation_count(
            rotation_grid_size(order, label),
            label,
        )
        == order
    )


def test_exact_local_precompute_policy_counts_the_symmetry_asu():
    from recovar.em.refinement.local_search_iteration import (
        _precompute_exact_local_fine_grid_enabled,
    )

    assert not _precompute_exact_local_fine_grid_enabled(6, "C1")
    assert _precompute_exact_local_fine_grid_enabled(6, "I1")


def _write_iteration_debug_metadata(
    output_dir,
    monkeypatch,
    *,
    symmetry: str | None,
):
    from recovar.em.diagnostics.iteration import (
        _save_iteration_intermediates,
    )
    from recovar.output import output as output_module

    monkeypatch.setattr(output_module, "save_volume", lambda *_args, **_kwargs: None)
    kwargs = dict(
        iteration=0,
        Ft_y_0=None,
        Ft_y_1=None,
        Ft_ctf_0=None,
        Ft_ctf_1=None,
        means=[np.zeros(8, dtype=np.complex64), np.zeros(8, dtype=np.complex64)],
        unreg_means=[None, None],
        fsc=np.ones(2, dtype=np.float32),
        noise_variance=np.ones(2, dtype=np.float32),
        noise_variance_per_half=[
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        ],
        mean_variance=np.ones(2, dtype=np.float32),
        hard_assignments=[None, None],
        coarse_ha=[None, None],
        effective_rotations=np.empty((0, 3, 3), dtype=np.float32),
        current_translations=np.zeros((1, 2), dtype=np.float32),
        use_local=True,
        local_search_order=3,
        cs=6,
        state=SimpleNamespace(healpix_order=3, sigma_rot=1.25),
        n_classes=1,
        k_class_enabled=False,
        volume_shape=(2, 2, 2),
        voxel_size=1.0,
    )
    if symmetry is not None:
        kwargs["symmetry"] = symmetry
    _save_iteration_intermediates(str(output_dir), **kwargs)
    return np.load(output_dir / "it000_meta.npy", allow_pickle=True).item()


def test_iteration_debug_metadata_records_symmetry_reduced_local_grid(
    tmp_path,
    monkeypatch,
):
    from recovar.em.sampling import rotation_grid_size

    metadata = _write_iteration_debug_metadata(
        tmp_path / "i1",
        monkeypatch,
        symmetry="i1",
    )

    assert metadata["n_rotations"] == rotation_grid_size(3, "I1")
    assert metadata["symmetry_label"] == "I1"
    assert metadata["symmetry_operator_sha256"] == symmetry_operator_sha256("I1")


def test_iteration_debug_metadata_default_preserves_explicit_c1(
    tmp_path,
    monkeypatch,
):
    default = _write_iteration_debug_metadata(
        tmp_path / "default",
        monkeypatch,
        symmetry=None,
    )
    explicit = _write_iteration_debug_metadata(
        tmp_path / "explicit",
        monkeypatch,
        symmetry="C1",
    )

    assert default == explicit
