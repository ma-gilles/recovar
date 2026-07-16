"""CLI-level sampling contract tests for ``scripts/run_full_refinement.py``."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.sampling import (
    advance_relion_perturbation_from_seed,
    relion_sampling_perturbation_for_iteration,
)
from scripts.run_full_refinement import (
    _add_significant_count_artifacts,
    _effective_perturb_seed,
    _explicit_relion_optimiser_for_seed,
    _jsonable_profile_rows,
    _pose_history_by_image,
    _refine_sampling_kwargs,
    _resolve_effective_max_healpix_order,
    _resolve_optimizer_random_seed,
    _resolve_relion_sampling_orders,
)

RUN_FULL_REFINEMENT = Path(__file__).resolve().parents[2] / "scripts" / "run_full_refinement.py"


def test_relion_healpix_order_is_coarse_pass1_order():
    coarse, fine = _resolve_relion_sampling_orders(healpix_order=2, adaptive_oversampling=1)

    assert coarse == 2
    assert fine == 3


def test_relion_sampling_order_rejects_negative_values():
    with pytest.raises(ValueError):
        _resolve_relion_sampling_orders(healpix_order=-1, adaptive_oversampling=1)

    with pytest.raises(ValueError):
        _resolve_relion_sampling_orders(healpix_order=1, adaptive_oversampling=-1)


def test_kclass_default_max_healpix_order_matches_relion_fixed_class3d_sampling():
    cap, source = _resolve_effective_max_healpix_order(
        n_classes=4,
        healpix_order=1,
        max_healpix_order=None,
    )

    assert cap == 1
    assert "Class3D fixed" in source


def test_k1_default_max_healpix_order_keeps_autorefine_cap():
    cap, source = _resolve_effective_max_healpix_order(
        n_classes=1,
        healpix_order=3,
        max_healpix_order=None,
    )

    assert cap == 7
    assert "auto-refine" in source


def test_explicit_max_healpix_order_allows_kclass_refinement():
    cap, source = _resolve_effective_max_healpix_order(
        n_classes=4,
        healpix_order=1,
        max_healpix_order=3,
    )

    assert cap == 3
    assert source == "explicit CLI"


def test_explicit_max_healpix_order_cannot_be_coarser_than_start():
    with pytest.raises(ValueError, match="max_healpix_order"):
        _resolve_effective_max_healpix_order(
            n_classes=4,
            healpix_order=2,
            max_healpix_order=1,
        )


def test_cli_translation_grid_parameters_seed_refinement_state():
    args = SimpleNamespace(
        adaptive_oversampling=1,
        offset_range=3.0,
        offset_step=1.0,
        auto_local_healpix_order=4,
    )

    kwargs = _refine_sampling_kwargs(args, init_healpix_order=2)

    assert kwargs["init_healpix_order"] == 2
    assert kwargs["init_translation_range"] == 3.0
    assert kwargs["init_translation_step"] == 1.0
    assert kwargs["translation_pixel_offset"] == 1.0


def test_cli_perturb_seed_defaults_to_relion_random_seed():
    assert _effective_perturb_seed(SimpleNamespace(seed=17, perturb_seed=None)) == 17
    assert _effective_perturb_seed(SimpleNamespace(seed=17, perturb_seed=23)) == 23
    assert _effective_perturb_seed(SimpleNamespace(seed=17, perturb_seed=-1)) is None


def test_optimizer_seed_explicit_cli_wins_over_relion_star(tmp_path):
    optimiser = tmp_path / "run_it000_optimiser.star"
    optimiser.write_text("_rlnRandomSeed 1713\n")

    assert _resolve_optimizer_random_seed(99, optimiser) == (99, "explicit CLI")


def test_optimizer_seed_inherits_explicit_relion_star_when_omitted(tmp_path):
    optimiser = tmp_path / "run_it000_optimiser.star"
    optimiser.write_text("_rlnRandomSeed 1713\n")

    seed, source = _resolve_optimizer_random_seed(None, optimiser)

    assert seed == 1713
    assert str(optimiser.resolve()) in source


def test_optimizer_seed_omitted_without_relion_state_keeps_default():
    assert _resolve_optimizer_random_seed(None, None) == (42, "standalone default")


def test_optimizer_seed_zero_is_preserved():
    assert _resolve_optimizer_random_seed(0, None) == (0, "explicit CLI")


def test_seed_source_ignores_incidental_data_dir_relion_discovery(tmp_path):
    relion_dir = tmp_path / "relion_ref"
    relion_dir.mkdir()
    (relion_dir / "run_optimiser.star").write_text("_rlnRandomSeed 1713\n")
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        relion_optimiser=None,
        relion_init_dir=None,
        perturb_replay_relion_dir=None,
        relion_half_sets=None,
    )

    assert _explicit_relion_optimiser_for_seed(args) is None


def test_optimizer_seed_is_resolved_before_halfset_splitting():
    source = RUN_FULL_REFINEMENT.read_text()

    assert "default=None" in source[source.index('parser.add_argument(\n        "--seed"') :]
    assert source.index("args.seed, optimizer_seed_source = _resolve_optimizer_random_seed") < source.index(
        "_default_refinement_subsets(n_images, args.seed"
    )


def test_relion_seeded_sampling_perturbation_sequence_matches_reference_star():
    values = [
        relion_sampling_perturbation_for_iteration(
            perturbation_factor=0.5,
            random_seed=1776701668,
            relion_iteration=iteration,
        )
        for iteration in range(3)
    ]

    assert values == pytest.approx([0.460047, -0.25278, 0.125066], abs=5e-6)


def test_relion_seeded_sampling_perturbation_preserves_scaled_rnd_unif_rounding():
    # Source-level RELION evaluates rnd_unif(0.25f, 0.5f) directly.  Replacing
    # it with 0.25 + 0.25*rnd_unif() gives -0.049614354968070984 here and
    # changes axial outer-rim backprojection decisions by one ulp.
    initial = advance_relion_perturbation_from_seed(0.0, 0.5, seed=1)
    iteration_one = advance_relion_perturbation_from_seed(initial, 0.5, seed=20260713)

    assert initial == 0.4600469470024109
    assert iteration_one == -0.04961434006690979
    assert relion_sampling_perturbation_for_iteration(0.5, 1731, 1) == -0.11648395657539368


def test_pose_history_by_image_restores_original_particle_order():
    half_indices = [
        np.asarray([2, 0], dtype=np.int64),
        np.asarray([3, 1], dtype=np.int64),
    ]
    half_pose_arrays = [
        np.asarray([[20.0, 21.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[30.0, 31.0], [10.0, 11.0]], dtype=np.float32),
    ]

    by_image = _pose_history_by_image(
        half_pose_arrays,
        half_indices,
        n_images=4,
        trailing_shape=(2,),
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        by_image,
        np.asarray(
            [
                [0.0, 1.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ],
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError, match="does not match half-set index length"):
        _pose_history_by_image(
            [half_pose_arrays[0][:1], half_pose_arrays[1]],
            half_indices,
            n_images=4,
            trailing_shape=(2,),
            dtype=np.float32,
        )


def test_significant_count_artifacts_preserve_legacy_half_order_and_add_image_order():
    half_indices = [
        np.asarray([2, 0, 4], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
    ]
    counts_half_order = np.asarray([20, 0, 40, 10, 30], dtype=np.int32)
    artifacts = {}

    _add_significant_count_artifacts(
        artifacts,
        [counts_half_order, None],
        half_indices,
        n_images=5,
    )

    np.testing.assert_array_equal(artifacts["sig_counts_iter_000"], counts_half_order)
    np.testing.assert_array_equal(
        artifacts["sig_counts_half_order_iter_000"],
        counts_half_order,
    )
    np.testing.assert_array_equal(
        artifacts["sig_counts_by_image_iter_000"],
        np.asarray([0, 10, 20, 30, 40], dtype=np.int32),
    )
    assert artifacts["sig_counts_iter_000"].dtype == counts_half_order.dtype
    assert "sig_counts_iter_001" not in artifacts
    assert "sig_counts_half_order_iter_001" not in artifacts
    assert "sig_counts_by_image_iter_001" not in artifacts


def test_profile_rows_are_jsonable_for_nested_numpy_values():
    rows = [
        {
            "phase": "iteration",
            "sparse_kclass_fused_s": np.float64(3.5),
            "local_adaptive_pass2_parent_mode": "pruned_parent",
            "local_adaptive_pass2_full_parent": np.bool_(False),
            "counts": np.asarray([1, 2, 3], dtype=np.int32),
            "per_class_profile_summary": (
                {"class_index": np.int64(1), "score_s": np.float32(1.25)},
                {"class_index": np.int64(2), "score_s": np.float32(2.25)},
            ),
        }
    ]

    jsonable = _jsonable_profile_rows(rows)

    assert jsonable == [
        {
            "phase": "iteration",
            "sparse_kclass_fused_s": 3.5,
            "local_adaptive_pass2_parent_mode": "pruned_parent",
            "local_adaptive_pass2_full_parent": False,
            "counts": [1, 2, 3],
            "per_class_profile_summary": [
                {"class_index": 1, "score_s": pytest.approx(float(np.float32(1.25)))},
                {"class_index": 2, "score_s": pytest.approx(float(np.float32(2.25)))},
            ],
        }
    ]
