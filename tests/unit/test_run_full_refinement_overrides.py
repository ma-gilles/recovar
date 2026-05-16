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
    _parse_relion_tau2_fudge,
    _resolve_replay_normcorr,
    _resolve_tau2_fudge,
)

FIXTURE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0")
RUN_FULL_REFINEMENT = Path(__file__).resolve().parents[2] / "scripts" / "run_full_refinement.py"


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


def test_firstiter_cc_passes_ini_high_to_refinement_loop():
    """Lock down RELION's firstiter_cc low-pass handoff.

    RELION reapplies ``--ini_high`` to references after the first
    normalized-CC iteration. The full refinement entry point must forward
    that value to ``refine_single_volume``; otherwise iter 2 starts from
    unfiltered class maps even though the direct iter-1 oracle is correct.
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
    assert isinstance(value.body, ast.Attribute)
    assert value.body.attr == "init_resolution"


def test_relion_tau2_fudge_parser_accepts_class3d_arg_label():
    text = """
data_optimiser_general

_rlnDoSplitRandomHalves                                  0
_rlnTau2FudgeArg                                          4.000000
"""
    assert _parse_relion_tau2_fudge(text) == pytest.approx(4.0)


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
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")
    starfile.write(
        {"model_general": model_general, "model_groups": model_groups},
        tmp_path / "run_it001_model.star",
    )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0, 2], dtype=np.int64),
        half2_idx=np.asarray([1, 3], dtype=np.int64),
        max_iter=2,
        ds_voxel=1.0,
        ds_grid=8,
        include_normcorr=True,
    )

    assert overrides[0] is None
    assert overrides[1]["translation_sigma_angstrom"] == pytest.approx(6.5)
    h1, h2 = overrides[1]["image_corrections"]
    np.testing.assert_allclose(h1, np.asarray([30.0, 7.5], dtype=np.float32))
    np.testing.assert_allclose(h2, np.asarray([30.0, 12.0], dtype=np.float32))
    s1, s2 = overrides[1]["scale_corrections"]
    np.testing.assert_allclose(s1, np.asarray([10.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(s2, np.asarray([20.0, 20.0], dtype=np.float32))
