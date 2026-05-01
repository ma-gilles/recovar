"""Phase B integration test: production CLI on Ribosembly bundle.

Tests the ``recovar ppca_refine --pose-mode dense`` end-to-end path via
``main(argv, _bundle_override=...)``. The synthetic ``cryo``'s
``_FourierImageStack`` class is loaded via importlib spec and isn't
picklable, so we use the ``_bundle_override`` test hook to bypass
pickle.

Verifies output files: per-iter MRCs (mu_score, W_*_score), final basis
(U_*.mrc + S.npy), state.pkl, iter_log.pkl.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from recovar.em.ppca_refinement.cli import main  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _load_synthetic_helpers():
    src = Path(__file__).resolve().parents[2] / "unit" / "test_ppca_multimask_synthetic.py"
    spec = importlib.util.spec_from_file_location("_ppca_synthetic_helpers", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_ribosembly_bundle():
    """Return an in-memory bundle. NOT pickled — the synthetic ``cryo``'s
    _FourierImageStack class isn't picklable. Tests use the
    ``_bundle_override`` test hook on ``main`` to bypass pickle.
    """
    helpers = _load_synthetic_helpers()
    vols_real, vols_fourier, vol_shape = helpers._load_ribosembly_volumes(
        n_states=4,
        grid_size=64,
    )
    mask_left, mask_right, _ = helpers._make_split_masks(vol_shape, vols_real)
    cryo, _ = helpers._simulate_dataset(
        vols_fourier,
        vol_shape,
        n_images=80,
        noise_level=1.0,
        seed=42,
    )
    mu_init = np.real(np.fft.ifftn(vols_fourier.mean(axis=0).reshape(vol_shape))).astype(np.float32)
    mask = np.maximum(mask_left, mask_right).astype(np.float32)
    rotation_grid = np.asarray(cryo.rotation_matrices[:4], dtype=np.float32)
    translation_grid = np.zeros((1, 2), dtype=np.float32)
    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )
    return {
        "cryo": cryo,
        "mu_init": mu_init,
        "W_init": None,
        "mask": mask,
        "halfset_indices": halfset_indices,
        "rotation_grid": rotation_grid,
        "translation_grid": translation_grid,
    }


def test_cli_pose_mode_dense_e2e():
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        bundle = _build_ribosembly_bundle()

        rc = main(
            [
                "particles_unused.star",
                "--out",
                str(out_dir),
                "--init-mean",
                "consensus_unused.mrc",
                "--zdim",
                "1",
                "--pose-mode",
                "dense",
                "--em-iters",
                "2",
                "--pcg-maxiter",
                "5",
                "--input-bundle",
                "test_override",
                "--image-batch-size",
                "16",
                "--rotation-block-size",
                "4",
                "--halfset-combine",
                "mean",
            ],
            _bundle_override=bundle,
        )
        assert rc == 0
        for it in range(2):
            iter_dir = out_dir / f"iter_{it:03d}"
            assert (iter_dir / "mu_score.mrc").exists()
            assert (iter_dir / "W_00_score.mrc").exists()
        assert (out_dir / "U_00.mrc").exists()
        assert (out_dir / "S.npy").exists()
        assert (out_dir / "state.pkl").exists()
        assert (out_dir / "iter_log.pkl").exists()


def test_cli_state_pkl_round_trip_runs():
    """Run 1 iter and verify the state.pkl save artifact is loadable."""
    from recovar.em.ppca_refinement.postprocess import load_state

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        bundle = _build_ribosembly_bundle()

        rc = main(
            [
                "p.star",
                "--out",
                str(out_dir),
                "--init-mean",
                "m.mrc",
                "--zdim",
                "1",
                "--pose-mode",
                "dense",
                "--em-iters",
                "1",
                "--input-bundle",
                "test_override",
                "--image-batch-size",
                "16",
                "--rotation-block-size",
                "4",
            ],
            _bundle_override=bundle,
        )
        assert rc == 0
        state_path = out_dir / "state.pkl"
        assert state_path.exists()
        state = load_state(state_path)
        assert state.W_score.shape == (1, 64, 64, 64)
        assert state.mu_score.shape == (64, 64, 64)


def test_cli_pose_mode_fixed_still_works_without_input_bundle():
    """Pre-Phase-B fixed-pose CLI path still works without --input-bundle."""
    rc = main(
        [
            "p.star",
            "--out",
            "/tmp/unused_ppca_refine_test",
            "--init-mean",
            "m.mrc",
            "--zdim",
            "2",
            "--pose-mode",
            "fixed",
        ]
    )
    assert rc == 0
