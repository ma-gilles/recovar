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
    _parse_relion_tau2_fudge,
    _resolve_tau2_fudge,
)

FIXTURE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0")
RUN_FULL_REFINEMENT = Path(__file__).resolve().parents[2] / "scripts" / "run_full_refinement.py"
ITERATION_LOOP = Path(__file__).resolve().parents[2] / "recovar" / "em" / "dense_single_volume" / "iteration_loop.py"


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


def test_class3d_direction_prior_replay_accepts_single_model_star():
    """Class3D writes one run_itNNN_model.star, not split-half model stars."""

    source = ITERATION_LOOP.read_text()
    assert "run_it{_prior_iter:03d}_half{_half_idx + 1}_model.star" in source
    assert "run_it{_prior_iter:03d}_model.star" in source
    assert "Replay override: class direction prior half-%d <- %s" in source


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


def test_tau2_fudge_resolver_matches_relion_mode_defaults():
    assert _resolve_tau2_fudge(1, None, None) == (1.0, "RELION auto-refine default")
    assert _resolve_tau2_fudge(4, None, None) == (4.0, "RELION Class3D default")
    assert _resolve_tau2_fudge(4, 2.5, None) == (2.5, "explicit CLI")
    assert _resolve_tau2_fudge(4, 2.5, 4.0) == (4.0, "RELION it000 optimiser")


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


def test_replay_overrides_accept_class3d_single_model_star(tmp_path):
    import pandas as pd
    import starfile

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs", "3@particles.mrcs", "4@particles.mrcs"],
            "rlnNormCorrection": [2.0, 4.0, 5.0, 10.0],
            "rlnGroupNumber": [1, 2, 1, 2],
        }
    )
    model_general = pd.DataFrame(
        {
            "rlnNormCorrectionAverage": [20.0],
            "rlnSigmaOffsetsAngst": [8.890755],
        }
    )
    model_groups = pd.DataFrame({"rlnGroupScaleCorrection": [1.0, 0.5]})
    model_optics_group_1 = pd.DataFrame({"rlnSigma2Noise": [3.0]})
    model_classes = pd.DataFrame({"rlnClassDistribution": [0.25, 0.75]})

    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star", overwrite=True)
    starfile.write(
        {
            "model_general": model_general,
            "model_groups": model_groups,
            "model_optics_group_1": model_optics_group_1,
            "model_classes": model_classes,
        },
        tmp_path / "run_it001_model.star",
        overwrite=True,
    )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0, 2], dtype=np.int64),
        half2_idx=np.asarray([1, 3], dtype=np.int64),
        max_iter=2,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=True,
        include_noise=True,
    )

    assert overrides[0] is None
    assert overrides[1]["translation_sigma_angstrom"] == pytest.approx(8.890755)
    h1, h2 = overrides[1]["image_corrections"]
    np.testing.assert_allclose(h1, np.asarray([10.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(h2, np.asarray([2.5, 1.0], dtype=np.float32))
    s1, s2 = overrides[1]["scale_corrections"]
    np.testing.assert_allclose(s1, np.asarray([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(s2, np.asarray([0.5, 0.5], dtype=np.float32))
    n1, n2 = overrides[1]["noise_variance"]
    assert n1.shape == (128 * 128,)
    assert n2.shape == (128 * 128,)
    np.testing.assert_allclose(n1, np.full(128 * 128, 3.0 * 128**4, dtype=np.float32))
    np.testing.assert_allclose(n2, np.full(128 * 128, 3.0 * 128**4, dtype=np.float32))
    np.testing.assert_allclose(overrides[1]["class_weights"], np.asarray([0.25, 0.75]))


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
