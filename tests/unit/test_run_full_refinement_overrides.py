"""Unit tests for ``scripts/run_full_refinement.py::_build_replay_iteration_overrides``.

Locks down the parity-critical contract that the per-iter replay override
dict always carries ``translation_sigma_angstrom`` sourced from RELION's
``rlnSigmaOffsetsAngst``. Without that, recovar's iter-1 leaves
``current_sigma_offset_angstrom`` at the 10 Å default and iter-2's
translation prior is ~6× too wide → iter-2 ave_Pmax is depressed by ~22 %
relative to RELION (cf. iteration_loop.py:4667-4703).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_full_refinement import _build_replay_iteration_overrides

FIXTURE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0")


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
