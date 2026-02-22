"""
Unit tests for recovar.commands.reconstruct_from_external_embedding.

Covers argument registration via add_args() only – no actual reconstruction.
"""
import argparse
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
    assert action.const is True   # store_true
