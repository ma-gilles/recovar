"""
Unit tests for recovar.commands.reconstruct_from_external_embedding.

Covers argument registration and external embedding loading helpers.
"""

import argparse
import numpy as np
import pytest

pytest.importorskip("jax")  # module imports jax at top level

import recovar.commands.reconstruct_from_external_embedding as rfe_cmd

pytestmark = pytest.mark.unit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    rfe_cmd.add_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Required arguments
# ---------------------------------------------------------------------------


def test_registers_particles_positional():
    positionals = [a.dest for a in _parser()._actions if not a.option_strings]
    assert "particles" in positionals


def test_registers_outdir():
    actions = _parser()._option_string_actions
    assert "-o" in actions or "--outdir" in actions


def test_registers_poses():
    assert "--poses" in _parser()._option_string_actions


def test_registers_ctf():
    assert "--ctf" in _parser()._option_string_actions


def test_registers_embedding_arg():
    """--embedding is required for external reconstruction."""
    actions = _parser()._option_string_actions
    assert "--embedding" in actions


def test_registers_target_arg():
    """--target specifies which latent coordinates to reconstruct at."""
    assert "--target" in _parser()._option_string_actions


# ---------------------------------------------------------------------------
# Optional arguments with sensible defaults
# ---------------------------------------------------------------------------


def test_zdim_default_is_list():
    """Default zdim must be a list (same as pipeline)."""
    action = _parser()._option_string_actions["--zdim"]
    assert isinstance(action.default, list)


def test_n_images_registered():
    actions = _parser()._option_string_actions
    assert "--n-images" in actions or "--n_images" in actions


def test_bfactor_default_is_zero():
    action = _parser()._option_string_actions.get("--Bfactor")
    if action is None:
        pytest.skip("--Bfactor not in parser")
    assert action.default == 0


def test_tilt_series_is_boolean_flag():
    action = _parser()._option_string_actions.get("--tilt-series")
    if action is None:
        pytest.skip("--tilt-series not in parser")
    assert action.const is True  # store_true


def test_load_external_embeddings_accepts_npy(tmp_path):
    emb = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "embedding.npy"
    np.save(path, emb)

    out = rfe_cmd._load_external_embeddings(str(path))

    np.testing.assert_array_equal(out, emb)


def test_load_external_embeddings_accepts_npz_named_key(tmp_path):
    emb = np.array([[1.0, 2.0]], dtype=np.float32)
    path = tmp_path / "embedding.npz"
    np.savez(path, latent_coords=emb)

    out = rfe_cmd._load_external_embeddings(str(path))

    np.testing.assert_array_equal(out, emb)


def test_load_external_embeddings_rejects_bad_shape(tmp_path):
    emb = np.zeros((2, 2, 2), dtype=np.float32)
    path = tmp_path / "embedding.npy"
    np.save(path, emb)

    with pytest.raises(ValueError, match="must have shape"):
        rfe_cmd._load_external_embeddings(str(path))
