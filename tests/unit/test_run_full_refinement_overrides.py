"""Unit tests for ``scripts/run_full_refinement.py::_build_replay_iteration_overrides``.

Locks down the parity-critical contract that the per-iter replay override
dict always carries ``translation_sigma_angstrom`` sourced from RELION's
``rlnSigmaOffsetsAngst``. Without that, recovar's iter-1 leaves
``current_sigma_offset_angstrom`` at the 10 Å default and iter-2's
translation prior is ~6× too wide → iter-2 ave_Pmax is depressed by ~22 %
relative to RELION (cf. iteration_loop.py:4667-4703).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.run_full_refinement import (
    _build_replay_iteration_overrides,
    _default_refinement_subsets,
    _format_replay_mean_for_log,
    _load_initial_noise_cache,
    _load_init_noise_radial_npz,
    _load_init_previous_best_poses_npz,
    _load_relion_it000_model_stars,
    _load_native_group_ids_per_half,
    _parse_relion_cli_ini_high,
    _parse_relion_tau2_fudge,
    _replay_complete_initial_particle_state,
    _resolve_replay_normcorr,
    _resolve_tau2_fudge,
    _save_initial_noise_cache,
)

FIXTURE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0")
RUN_FULL_REFINEMENT = Path(__file__).resolve().parents[2] / "scripts" / "run_full_refinement.py"
ITERATION_LOOP = (
    Path(__file__).resolve().parents[2] / "recovar" / "em" / "dense_single_volume" / "iteration_loop.py"
)


def test_complete_initial_particle_state_is_autorefine_only():
    assert _replay_complete_initial_particle_state(1, 0)
    assert not _replay_complete_initial_particle_state(4, 0)
    assert not _replay_complete_initial_particle_state(1, 1)


def _read_relion_sigma(model_star: Path) -> float:
    import starfile

    m = starfile.read(str(model_star))
    mg = m["model_general"]
    val = mg["rlnSigmaOffsetsAngst"]
    if hasattr(val, "iloc"):
        val = val.iloc[0]
    elif hasattr(val, "__len__") and not isinstance(val, str):
        val = val[0]
    return float(val)


def test_parse_relion_cli_ini_high_reads_positive_cli_value():
    text = "# --i particles.star --firstiter_cc --ini_high 30 --ctf\n"
    assert _parse_relion_cli_ini_high(text) == pytest.approx(30.0)


def test_parse_relion_cli_ini_high_is_none_when_absent_or_disabled():
    assert _parse_relion_cli_ini_high("# --i particles.star --firstiter_cc --ctf\n") is None
    assert _parse_relion_cli_ini_high("# --i particles.star --firstiter_cc --ini_high -1 --ctf\n") is None


def test_firstiter_cc_passes_relion_cli_ini_high_to_refinement_loop():
    """RELION ``--firstiter_cc`` and ``--ini_high`` are distinct knobs.

    ``--firstiter_cc`` enables normalized-CC scoring in iter 1. RELION only
    reapplies the post-iter1 low-pass when the optimiser command has a
    positive ``--ini_high``. Do not substitute RECOVAR's ``--init_resolution``.
    """

    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "relion_firstiter_ini_high_angstrom" in keywords
    value = keywords["relion_firstiter_ini_high_angstrom"]
    assert isinstance(value, ast.IfExp)
    assert isinstance(value.test, ast.Attribute)
    assert value.test.attr == "firstiter_cc"
    assert isinstance(value.body, ast.Name)
    assert value.body.id == "relion_firstiter_ini_high_angstrom"


def test_refinement_results_persist_final_tau2_weight_combination():
    source = RUN_FULL_REFINEMENT.read_text()

    assert '"tau2_weight_combination_final_all_data"' in source
    assert 'save_dict["tau2_weight_combination_final_all_data"]' in source


def test_refinement_results_persist_class_assignment_history():
    source = RUN_FULL_REFINEMENT.read_text()

    assert '"class_assignment_history"' in source
    assert "class_assignments_iter_" in source
    assert "class_assignments_by_image_iter_" in source


def test_save_intermediates_skip_unregularized_passes_to_refinement_loop():
    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    value = keywords["save_intermediates_skip_unregularized"]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "bool"
    assert isinstance(value.args[0], ast.Attribute)
    assert value.args[0].attr == "save_intermediates_skip_unregularized"


def test_stop_after_local_search_passes_to_refinement_loop():
    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    assert "--stop_after_local_search" in RUN_FULL_REFINEMENT.read_text()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    value = keywords["stop_after_local_search"]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "bool"
    assert isinstance(value.args[0], ast.Attribute)
    assert value.args[0].attr == "stop_after_local_search"


def test_stop_after_local_search_score_only_passes_to_refinement_loop():
    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    assert "--stop_after_local_search_score_only" in RUN_FULL_REFINEMENT.read_text()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    value = keywords["stop_after_local_search_score_only"]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "bool"
    assert isinstance(value.args[0], ast.Attribute)
    assert value.args[0].attr == "stop_after_local_search_score_only"


def test_stop_after_local_search_score_only_is_diagnostic_score_only_path():
    source = ITERATION_LOOP.read_text()
    assert "if stop_after_local_search_score_only:\n        stop_after_local_search = True" in source
    assert "diagnostic_score_only=bool(stop_after_local_search_score_only)" in source
    assert "score_only=diagnostic_score_only" in source
    assert "accumulate_noise=local_accumulate_noise" in source
    assert '"stop_after_local_search_score_only": bool(stop_after_local_search_score_only)' in source


def test_diagnostic_single_half_is_guarded_to_local_search_stops():
    source = RUN_FULL_REFINEMENT.read_text()

    assert "--diagnostic_single_half" in source
    assert "local_stop_requested = (" in source
    assert "--diagnostic_single_half is only valid with --stop_after_local_search" in source
    assert "--diagnostic_single_half is K=1-only" in source
    assert "half2_idx = np.empty(0, dtype=np.int64)" in source
    assert '"diagnostic_single_half": bool(args.diagnostic_single_half)' in source
    assert "skipping Projector::data build for empty half-%d dataset" in ITERATION_LOOP.read_text()


def test_init_noise_from_npz_loader_uses_latest_numbered_spectrum(tmp_path):
    path = tmp_path / "refinement_results.npz"
    np.savez(
        path,
        noise_radial_iter_000=np.asarray([1.0, 2.0], dtype=np.float64),
        noise_radial_iter_003=np.asarray([3.0, 4.0], dtype=np.float64),
    )

    loaded = _load_init_noise_radial_npz(path, "last")

    assert loaded["iteration"] == "003"
    np.testing.assert_allclose(loaded["noise_radial"], np.asarray([3.0, 4.0], dtype=np.float64))


def test_init_noise_from_npz_is_diagnostic_cli_path():
    source = RUN_FULL_REFINEMENT.read_text()
    assert "--init_noise_from_npz" in source
    assert "--init_noise_iter" in source
    assert "_load_init_noise_radial_npz(args.init_noise_from_npz, args.init_noise_iter)" in source
    assert "estimate_initial_noise_spectrum_from_unaligned_images" in source


def test_initial_noise_cache_round_trips_and_marks_safe_to_delete(tmp_path):
    cache_path = _save_initial_noise_cache(
        tmp_path,
        "abc123",
        (8, 8),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
    )

    loaded, loaded_path = _load_initial_noise_cache(tmp_path, "abc123", (8, 8))

    assert loaded_path == cache_path
    assert (tmp_path / "SAFE_TO_DELETE").exists()
    np.testing.assert_allclose(loaded, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))


def test_initial_noise_cache_is_exact_bootstrap_cli_path():
    source = RUN_FULL_REFINEMENT.read_text()
    assert "--initial_noise_cache_dir" in source
    assert "_initial_noise_cache_key(" in source
    assert "_load_initial_noise_cache(" in source
    assert "_save_initial_noise_cache(" in source
    assert "estimate_initial_noise_spectrum_from_unaligned_images" in source
    assert "--init_noise_from_npz" in source


def test_relion_tau2_fudge_parser_accepts_class3d_arg_label():
    text = """
data_optimiser_general

_rlnDoSplitRandomHalves                                  0
_rlnTau2FudgeArg                                          4.000000
"""
    assert _parse_relion_tau2_fudge(text) == pytest.approx(4.0)


def test_load_native_group_ids_per_half_reads_particles_star(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    starfile.write(
        {
            "particles": pd.DataFrame(
                {
                    "rlnImageName": ["1@x.mrcs", "2@x.mrcs", "3@x.mrcs", "4@x.mrcs"],
                    "rlnGroupNumber": [1, 2, 3, 2],
                },
            ),
        },
        tmp_path / "particles.star",
    )

    got = _load_native_group_ids_per_half(
        tmp_path / "particles.star",
        half1_idx=np.asarray([0, 2], dtype=np.int64),
        half2_idx=np.asarray([1, 3], dtype=np.int64),
    )

    assert got is not None
    np.testing.assert_array_equal(got[0], np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(got[1], np.asarray([1, 1], dtype=np.int64))


def test_native_group_ids_are_available_to_k_class_refinement():
    source = RUN_FULL_REFINEMENT.read_text()
    group_block = source[
        source.index("native_group_ids_per_half = _load_native_group_ids_per_half") :
        source.index("optimiser_star = _find_relion_optimiser_star(args)")
    ]

    assert "args.n_classes == 1" not in group_block
    assert "Native group-scale updates remain disabled for K-class refinement" not in source
    assert "init_group_ids=native_group_ids_per_half" in source


def test_load_init_previous_best_poses_npz_selects_latest_numbered_iter(tmp_path):
    path = tmp_path / "refinement_results.npz"
    np.savez(
        path,
        best_rotation_eulers_iter_000_half0=np.ones((2, 3), dtype=np.float32),
        best_rotation_eulers_iter_000_half1=np.ones((1, 3), dtype=np.float32) * 2,
        best_translations_iter_000_half0=np.ones((2, 2), dtype=np.float32),
        best_translations_iter_000_half1=np.ones((1, 2), dtype=np.float32) * 2,
        best_rotation_eulers_iter_003_half0=np.ones((2, 3), dtype=np.float32) * 3,
        best_rotation_eulers_iter_003_half1=np.ones((1, 3), dtype=np.float32) * 4,
        best_translations_iter_003_half0=np.ones((2, 2), dtype=np.float32) * 5,
        best_translations_iter_003_half1=np.ones((1, 2), dtype=np.float32) * 6,
    )

    got = _load_init_previous_best_poses_npz(path, "last")

    assert got["iteration"] == "003"
    np.testing.assert_allclose(got["previous_best_rotation_eulers"][0], np.ones((2, 3)) * 3)
    np.testing.assert_allclose(got["previous_best_rotation_eulers"][1], np.ones((1, 3)) * 4)
    np.testing.assert_allclose(got["previous_best_translations"][0], np.ones((2, 2)) * 5)
    np.testing.assert_allclose(got["previous_best_translations"][1], np.ones((1, 2)) * 6)


def test_load_init_previous_best_poses_npz_accepts_final_all_data(tmp_path):
    path = tmp_path / "refinement_results.npz"
    np.savez(
        path,
        best_rotation_eulers_final_all_data_half0=np.ones((2, 3), dtype=np.float32),
        best_rotation_eulers_final_all_data_half1=np.ones((1, 3), dtype=np.float32) * 2,
        best_translations_final_all_data_half0=np.ones((2, 2), dtype=np.float32) * 3,
        best_translations_final_all_data_half1=np.ones((1, 2), dtype=np.float32) * 4,
    )

    got = _load_init_previous_best_poses_npz(path, "final_all_data")

    assert got["iteration"] == "final_all_data"
    np.testing.assert_allclose(got["previous_best_rotation_eulers"][0], np.ones((2, 3)))
    np.testing.assert_allclose(got["previous_best_translations"][1], np.ones((1, 2)) * 4)


def test_load_init_previous_best_poses_npz_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "refinement_results.npz"
    np.savez(
        path,
        best_rotation_eulers_iter_000_half0=np.ones((2, 3), dtype=np.float32),
        best_rotation_eulers_iter_000_half1=np.ones((1, 3), dtype=np.float32),
        best_translations_iter_000_half0=np.ones((3, 2), dtype=np.float32),
        best_translations_iter_000_half1=np.ones((1, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="row mismatch"):
        _load_init_previous_best_poses_npz(path, 0)


def _write_minimal_relion_model_star(path: Path, *, sigma2_noise_start: float) -> None:
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    starfile.write(
        {
            "model_general": pd.DataFrame({"rlnTau2FudgeFactor": [1.0]}),
            "model_optics_group_1": pd.DataFrame(
                {"rlnSigma2Noise": np.asarray([sigma2_noise_start, sigma2_noise_start + 1.0])},
            ),
            "model_class_1": pd.DataFrame(
                {
                    "rlnSpectralIndex": [0, 1],
                    "rlnReferenceTau2": [0.1, 0.2],
                },
            ),
        },
        path,
    )


def test_load_relion_it000_model_stars_prefers_shared_model(tmp_path):
    shared = tmp_path / "run_it000_model.star"
    _write_minimal_relion_model_star(shared, sigma2_noise_start=1.0)
    _write_minimal_relion_model_star(tmp_path / "run_it000_half1_model.star", sigma2_noise_start=2.0)
    _write_minimal_relion_model_star(tmp_path / "run_it000_half2_model.star", sigma2_noise_start=3.0)

    bundle = _load_relion_it000_model_stars(tmp_path, n_classes=1)

    assert bundle["source"] == "shared"
    assert bundle["model_paths"] == [shared]
    assert bundle["reference_model_path"] == shared
    assert len(bundle["models"]) == 1


def test_load_relion_it000_model_stars_accepts_autorefine_half_models(tmp_path):
    half1 = tmp_path / "run_it000_half1_model.star"
    half2 = tmp_path / "run_it000_half2_model.star"
    _write_minimal_relion_model_star(half1, sigma2_noise_start=2.0)
    _write_minimal_relion_model_star(half2, sigma2_noise_start=3.0)

    bundle = _load_relion_it000_model_stars(tmp_path, n_classes=1)

    assert bundle["source"] == "half-specific"
    assert bundle["model_paths"] == [half1, half2]
    assert bundle["reference_model_path"] == half1
    assert len(bundle["models"]) == 2
    np.testing.assert_allclose(bundle["models"][0]["model_optics_group_1"]["rlnSigma2Noise"], [2.0, 3.0])
    np.testing.assert_allclose(bundle["models"][1]["model_optics_group_1"]["rlnSigma2Noise"], [3.0, 4.0])


def test_load_relion_it000_model_stars_requires_shared_model_for_kclass(tmp_path):
    _write_minimal_relion_model_star(tmp_path / "run_it000_half1_model.star", sigma2_noise_start=2.0)
    _write_minimal_relion_model_star(tmp_path / "run_it000_half2_model.star", sigma2_noise_start=3.0)

    with pytest.raises(SystemExit, match="no compatible iter-0 model STAR"):
        _load_relion_it000_model_stars(tmp_path, n_classes=4)


def test_replay_mean_log_formatter_handles_empty_half_without_warning():
    assert _format_replay_mean_for_log(np.asarray([], dtype=np.float32)) == "empty"
    assert _format_replay_mean_for_log(np.asarray([1.0, 3.0], dtype=np.float32)) == "2.0000"


def test_relion_tau2_fudge_parser_accepts_factor_label():
    text = "_rlnTau2FudgeFactor 1.000000\n"
    assert _parse_relion_tau2_fudge(text) == pytest.approx(1.0)


def test_relion_tau2_fudge_parser_maps_arg_negative_one_to_none():
    """RELION's ``_rlnTau2FudgeArg=-1`` (in optimiser.star) is the sentinel
    for "user did not pass --tau2_fudge". RELION's ml_optimiser.cpp:881-882
    resolves it as ``tau2_fudge_factor = tau2_fudge_arg > 0 ? arg : 1``,
    i.e. -1 → 1.0 (the auto-refine binary default; Class3D's 4.0 comes from
    the GUI which always passes --tau2_fudge 4.0 — see
    pipeline_jobs.cpp::initialiseClass3DJob). The recovar parser must
    return None so _resolve_tau2_fudge falls back to the K-class default.
    Passing -1 downstream inverts the Wiener regularization
    (``inv_tau = 1 / (pf^3 * tau2_fudge * tau)`` becomes negative) which
    corrupts iter-1's reconstruction and collapses iter-2+ ave_Pmax —
    diagnosed on K=1 100k/256 replay job 8255968 (iter1 Pmax=0.94 at
    RELION parity, then iter2=0.32 vs RELION 0.98)."""
    text = """
data_optimiser_general

_rlnDoSplitRandomHalves                                  1
_rlnTau2FudgeArg                                          -1.000000
"""
    assert _parse_relion_tau2_fudge(text) is None


def test_relion_tau2_fudge_parser_prefers_factor_over_arg():
    """When both labels appear in the same text (combined parse), the
    Factor field from model.star is authoritative (actual value used)
    while Arg from optimiser.star is just the CLI input. RELION never
    writes both into the same file, but the parser must still prefer
    Factor to be robust."""
    text = """
_rlnTau2FudgeFactor 1.000000
_rlnTau2FudgeArg    -1.000000
"""
    assert _parse_relion_tau2_fudge(text) == pytest.approx(1.0)


def test_tau2_fudge_resolver_matches_relion_mode_defaults():
    assert _resolve_tau2_fudge(1, None, None) == (1.0, "RELION auto-refine default")
    assert _resolve_tau2_fudge(4, None, None) == (4.0, "RELION Class3D default")
    assert _resolve_tau2_fudge(4, 2.5, None) == (2.5, "explicit CLI")
    assert _resolve_tau2_fudge(4, 2.5, 4.0) == (4.0, "RELION it000 optimiser")


def test_replay_normcorr_defaults_to_strict_replay_only():
    assert _resolve_replay_normcorr(None, None) is False
    assert _resolve_replay_normcorr("/relion/run", None) is True
    assert _resolve_replay_normcorr("/relion/run", False) is False
    assert _resolve_replay_normcorr(None, True) is True


def test_default_refinement_subsets_keep_gold_standard_for_k1():
    half1, half2 = _default_refinement_subsets(9, seed=3, n_classes=1)

    assert half1.shape == (4,)
    assert half2.shape == (5,)
    np.testing.assert_array_equal(np.sort(np.concatenate([half1, half2])), np.arange(9))


def test_default_refinement_subsets_use_all_data_once_for_class3d():
    half1, half2 = _default_refinement_subsets(9, seed=3, n_classes=4)

    np.testing.assert_array_equal(half1, np.arange(9))
    assert half2.size == 0


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_replay_overrides_inject_per_iter_sigma_offset():
    half1_idx = np.arange(2515, dtype=np.int64)
    half2_idx = np.arange(2515, 5000, dtype=np.int64)

    overrides = _build_replay_iteration_overrides(
        FIXTURE,
        half1_idx,
        half2_idx,
        max_iter=8,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=False,
    )

    assert overrides[0] is None, "iter 0 (recovar iter 1) has no upstream RELION state"
    for recovar_iter in range(1, 8):
        assert overrides[recovar_iter] is not None, f"iter {recovar_iter} override missing"
        assert "translation_sigma_angstrom" in overrides[recovar_iter]
        # No normcorr/scale corrections when include_normcorr=False — only
        # sigma_offset, which is parity-critical regardless of normcorr replay.
        assert "image_corrections" not in overrides[recovar_iter]
        assert "scale_corrections" not in overrides[recovar_iter]

        m1 = FIXTURE / f"run_it{recovar_iter:03d}_half1_model.star"
        m2 = FIXTURE / f"run_it{recovar_iter:03d}_half2_model.star"
        relion_sigma = 0.5 * (_read_relion_sigma(m1) + _read_relion_sigma(m2))
        recovar_sigma = float(overrides[recovar_iter]["translation_sigma_angstrom"])
        assert recovar_sigma == pytest.approx(relion_sigma, abs=1e-6), (
            f"iter {recovar_iter}: recovar override sigma_offset {recovar_sigma:.6f} != "
            f"RELION rlnSigmaOffsetsAngst mean {relion_sigma:.6f}"
        )
        np.testing.assert_allclose(
            overrides[recovar_iter]["translation_sigma_angstrom_per_half"],
            np.asarray([_read_relion_sigma(m1), _read_relion_sigma(m2)], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_replay_overrides_can_load_complete_iter0_cold_start():
    half1_idx = np.arange(2515, dtype=np.int64)
    half2_idx = np.arange(2515, 5000, dtype=np.int64)

    overrides = _build_replay_iteration_overrides(
        FIXTURE,
        half1_idx,
        half2_idx,
        max_iter=0,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=True,
        include_initial_state=True,
    )

    cold_start = overrides[0]
    assert cold_start is not None
    assert cold_start["previous_best_translations"][0].shape == (2515, 2)
    assert cold_start["previous_best_translations"][1].shape == (2485, 2)
    assert cold_start["previous_best_rotation_eulers"][0].shape == (2515, 3)
    assert cold_start["previous_best_rotation_eulers"][1].shape == (2485, 3)
    assert cold_start["image_corrections"][0].shape == (2515,)
    assert cold_start["scale_corrections"][1].shape == (2485,)
    assert cold_start["direction_prior"][0].ndim == 1
    assert cold_start["translation_sigma_angstrom"] == pytest.approx(
        0.5
        * (
            _read_relion_sigma(FIXTURE / "run_it000_half1_model.star")
            + _read_relion_sigma(FIXTURE / "run_it000_half2_model.star")
        ),
        abs=1e-6,
    )


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_replay_overrides_iter2_sigma_matches_relion_iter1():
    """Specifically lock down the iter-2 cliff fix.

    recovar iter 2 (i.e. ``overrides[1]``) must use RELION iter-1's
    ``rlnSigmaOffsetsAngst``, since RELION iter-2 loads the prior from
    iter-1's model.star at E-step entry.
    """
    half1_idx = np.arange(2515, dtype=np.int64)
    half2_idx = np.arange(2515, 5000, dtype=np.int64)

    overrides = _build_replay_iteration_overrides(
        FIXTURE,
        half1_idx,
        half2_idx,
        max_iter=3,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=False,
    )

    iter1_h1 = FIXTURE / "run_it001_half1_model.star"
    iter1_h2 = FIXTURE / "run_it001_half2_model.star"
    relion_iter1_sigma = 0.5 * (_read_relion_sigma(iter1_h1) + _read_relion_sigma(iter1_h2))

    recovar_iter2_sigma = float(overrides[1]["translation_sigma_angstrom"])
    assert recovar_iter2_sigma == pytest.approx(relion_iter1_sigma, abs=1e-6)
    np.testing.assert_allclose(
        overrides[1]["translation_sigma_angstrom_per_half"],
        np.asarray([_read_relion_sigma(iter1_h1), _read_relion_sigma(iter1_h2)], dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    )
    # Sanity: this is the data-driven RELION sigma, not the 10 Å init default.
    assert recovar_iter2_sigma < 5.0, (
        "recovar iter-2 should use RELION's iter-1 data-driven sigma (~2 Å), "
        f"not the init default (10 Å); got {recovar_iter2_sigma:.4f} Å"
    )


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_replay_overrides_include_normcorr_adds_image_corrections():
    half1_idx = np.arange(2515, dtype=np.int64)
    half2_idx = np.arange(2515, 5000, dtype=np.int64)

    overrides = _build_replay_iteration_overrides(
        FIXTURE,
        half1_idx,
        half2_idx,
        max_iter=2,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=True,
    )

    assert "image_corrections" in overrides[1]
    assert "scale_corrections" in overrides[1]
    assert "translation_sigma_angstrom" in overrides[1]
    h1, h2 = overrides[1]["image_corrections"]
    assert h1.shape == (2515,)
    assert h2.shape == (2485,)


def test_replay_overrides_use_shared_class3d_model_star(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": [
                "1@particles.mrcs",
                "2@particles.mrcs",
                "3@particles.mrcs",
                "4@particles.mrcs",
            ],
            "rlnAngleRot": [10.0, 20.0, 30.0, 40.0],
            "rlnAngleTilt": [11.0, 21.0, 31.0, 41.0],
            "rlnAnglePsi": [12.0, 22.0, 32.0, 42.0],
            "rlnOriginXAngst": [2.0, -4.0, 6.0, 8.0],
            "rlnOriginYAngst": [1.0, -3.0, 5.0, 7.0],
            "rlnNormCorrection": [1.0, 2.0, 4.0, 5.0],
            "rlnGroupNumber": [1, 2, 1, 2],
        }
    )
    model_general = pd.DataFrame(
        {
            "rlnNormCorrectionAverage": [3.0],
            "rlnSigmaOffsetsAngst": [6.5],
        }
    )
    model_groups = pd.DataFrame({"rlnGroupScaleCorrection": [10.0, 20.0]})
    model_class_1 = pd.DataFrame({"rlnReferenceTau2": [0.1, 0.2]})
    model_class_2 = pd.DataFrame({"rlnReferenceTau2": [0.3, 0.4]})
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")
    starfile.write(
        {
            "model_general": model_general,
            "model_groups": model_groups,
            "model_class_1": model_class_1,
            "model_class_2": model_class_2,
        },
        tmp_path / "run_it001_model.star",
    )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0, 2], dtype=np.int64),
        half2_idx=np.asarray([1, 3], dtype=np.int64),
        max_iter=2,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=True,
    )

    assert overrides[0] is None
    assert overrides[1]["translation_sigma_angstrom"] == pytest.approx(6.5)
    assert overrides[1]["translation_sigma_angstrom_per_half"] == pytest.approx([6.5, 6.5])
    h1, h2 = overrides[1]["image_corrections"]
    np.testing.assert_allclose(h1, np.asarray([30.0, 7.5], dtype=np.float32))
    np.testing.assert_allclose(h2, np.asarray([30.0, 12.0], dtype=np.float32))
    s1, s2 = overrides[1]["scale_corrections"]
    np.testing.assert_allclose(s1, np.asarray([10.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(s2, np.asarray([20.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(
        overrides[1]["class_tau2"],
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64) * 8**4,
    )

    t1, t2 = overrides[1]["previous_best_translations"]
    np.testing.assert_allclose(t1, np.asarray([[1.0, 0.5], [3.0, 2.5]], dtype=np.float32))
    np.testing.assert_allclose(t2, np.asarray([[-2.0, -1.5], [4.0, 3.5]], dtype=np.float32))

    e1, e2 = overrides[1]["previous_best_rotation_eulers"]
    expected_e1 = np.asarray([[10.0, 11.0, 12.0], [30.0, 31.0, 32.0]], dtype=np.float32)
    expected_e2 = np.asarray([[20.0, 21.0, 22.0], [40.0, 41.0, 42.0]], dtype=np.float32)
    np.testing.assert_allclose(e1, expected_e1)
    np.testing.assert_allclose(e2, expected_e2)

    from recovar import utils

    r1, r2 = overrides[1]["previous_best_rotations"]
    np.testing.assert_allclose(r1, utils.R_from_relion(expected_e1, degrees=True).astype(np.float32))
    np.testing.assert_allclose(r2, utils.R_from_relion(expected_e2, degrees=True).astype(np.float32))


def test_replay_overrides_map_noncontiguous_subset_stack_indices(tmp_path):
    """Real-data subsets retain original stack IDs instead of renumbering rows."""
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    # RELION may reorder rows; values are deliberately keyed to stack ID.
    relion_stack_ids = [126, 16, 200, 51]
    particles = pd.DataFrame(
        {
            "rlnImageName": [f"{i}@particles.mrcs" for i in relion_stack_ids],
            "rlnAngleRot": np.asarray(relion_stack_ids, dtype=float),
            "rlnAngleTilt": np.asarray(relion_stack_ids, dtype=float) + 0.25,
            "rlnAnglePsi": np.asarray(relion_stack_ids, dtype=float) + 0.5,
            "rlnOriginXAngst": np.asarray(relion_stack_ids, dtype=float) * 2.0,
            "rlnOriginYAngst": -np.asarray(relion_stack_ids, dtype=float) * 2.0,
            "rlnNormCorrection": np.asarray(relion_stack_ids, dtype=float),
            "rlnGroupNumber": [1, 1, 1, 1],
        }
    )
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")
    starfile.write(
        {
            "model_general": pd.DataFrame(
                {"rlnNormCorrectionAverage": [10.0], "rlnSigmaOffsetsAngst": [3.0]}
            ),
            "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
        },
        tmp_path / "run_it001_model.star",
    )

    input_names = [
        "16@particles.mrcs",
        "51@particles.mrcs",
        "126@particles.mrcs",
        "200@particles.mrcs",
    ]
    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0, 2], dtype=np.int64),
        half2_idx=np.asarray([1, 3], dtype=np.int64),
        max_iter=1,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=True,
        particle_names=input_names,
    )

    h1_corr, h2_corr = overrides[1]["image_corrections"]
    np.testing.assert_allclose(h1_corr, np.asarray([10.0 / 16.0, 10.0 / 126.0], dtype=np.float32))
    np.testing.assert_allclose(h2_corr, np.asarray([10.0 / 51.0, 10.0 / 200.0], dtype=np.float32))

    h1_eulers, h2_eulers = overrides[1]["previous_best_rotation_eulers"]
    np.testing.assert_allclose(h1_eulers[:, 0], np.asarray([16.0, 126.0], dtype=np.float32))
    np.testing.assert_allclose(h2_eulers[:, 0], np.asarray([51.0, 200.0], dtype=np.float32))

    h1_trans, h2_trans = overrides[1]["previous_best_translations"]
    np.testing.assert_allclose(h1_trans[:, 0], np.asarray([16.0, 126.0], dtype=np.float32))
    np.testing.assert_allclose(h2_trans[:, 0], np.asarray([51.0, 200.0], dtype=np.float32))


def test_replay_overrides_include_max_iter_state_for_final_all_data(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnAngleRot": [10.0, 20.0],
            "rlnAngleTilt": [11.0, 21.0],
            "rlnAnglePsi": [12.0, 22.0],
            "rlnOriginXAngst": [2.0, -4.0],
            "rlnOriginYAngst": [1.0, -3.0],
            "rlnNormCorrection": [2.0, 4.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    direction_prior = np.linspace(1.0, 12.0, 12, dtype=np.float32)
    direction_prior /= direction_prior.sum()
    for relion_iter, sigma in ((1, 6.5), (2, 9.5)):
        starfile.write({"particles": particles}, tmp_path / f"run_it{relion_iter:03d}_data.star")
        sigma2_noise = np.linspace(1.0, 5.0, 5, dtype=np.float64) * float(relion_iter)
        starfile.write(
            {
                "model_general": pd.DataFrame(
                    {
                        "rlnNormCorrectionAverage": [float(relion_iter)],
                        "rlnSigmaOffsetsAngst": [sigma],
                    }
                ),
                "model_optics_group_1": pd.DataFrame({"rlnSigma2Noise": sigma2_noise}),
                "model_pdf_orient_class_1": pd.DataFrame({"rlnOrientationDistribution": direction_prior}),
                "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
            },
            tmp_path / f"run_it{relion_iter:03d}_model.star",
        )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=2,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=True,
    )

    assert len(overrides) == 3
    assert overrides[0] is None
    assert overrides[1]["translation_sigma_angstrom"] == pytest.approx(6.5)
    assert overrides[2]["translation_sigma_angstrom"] == pytest.approx(9.5)
    assert overrides[1]["translation_sigma_angstrom_per_half"] == pytest.approx([6.5, 6.5])
    assert overrides[2]["translation_sigma_angstrom_per_half"] == pytest.approx([9.5, 9.5])
    h1, h2 = overrides[2]["image_corrections"]
    np.testing.assert_allclose(h1, np.asarray([1.0], dtype=np.float32))
    np.testing.assert_allclose(h2, np.asarray([0.5], dtype=np.float32))

    noise_h1, noise_h2 = overrides[2]["noise_variance"]
    assert noise_h1.shape == (64,)
    assert noise_h2.shape == (64,)
    # RELION sigma2_noise is stored in RELION frame and replay expands it to
    # RECOVAR's image-shaped N^4-scaled variance convention.
    assert float(np.min(noise_h1)) == pytest.approx(2.0 * 8**4)
    assert float(np.max(noise_h1)) == pytest.approx(10.0 * 8**4)
    np.testing.assert_allclose(noise_h1, noise_h2)

    prior_h1, prior_h2 = overrides[2]["direction_prior"]
    np.testing.assert_allclose(prior_h1, direction_prior, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(prior_h2, direction_prior, rtol=1e-5, atol=1e-6)


def test_replay_overrides_respect_init_relion_iteration_offset(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnAngleRot": [10.0, 20.0],
            "rlnAngleTilt": [11.0, 21.0],
            "rlnAnglePsi": [12.0, 22.0],
            "rlnOriginXAngst": [2.0, -4.0],
            "rlnOriginYAngst": [1.0, -3.0],
            "rlnNormCorrection": [2.0, 4.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    for relion_iter, sigma in ((6, 6.5), (7, 7.5)):
        starfile.write({"particles": particles}, tmp_path / f"run_it{relion_iter:03d}_data.star")
        starfile.write(
            {
                "model_general": pd.DataFrame(
                    {
                        "rlnNormCorrectionAverage": [float(relion_iter)],
                        "rlnSigmaOffsetsAngst": [sigma],
                    }
                ),
                "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
            },
            tmp_path / f"run_it{relion_iter:03d}_model.star",
        )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=1,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=False,
        init_relion_iteration=6,
    )

    assert len(overrides) == 2
    assert overrides[0]["translation_sigma_angstrom"] == pytest.approx(6.5)
    assert overrides[1]["translation_sigma_angstrom"] == pytest.approx(7.5)
    np.testing.assert_allclose(
        overrides[0]["previous_best_rotation_eulers"][0],
        np.asarray([[10.0, 11.0, 12.0]], dtype=np.float32),
    )
