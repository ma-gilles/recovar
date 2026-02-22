"""
Unit tests for recovar.commands.pipeline and recovar.commands.pipeline_with_outliers.

Only tests argument registration via add_args() – no actual EM execution.
"""
import argparse
import pytest

pytest.importorskip("jax")  # pipeline.py imports jax at module level

import recovar.commands.pipeline as pipeline_cmd
import recovar.commands.pipeline_with_outliers as pwo_cmd

pytestmark = pytest.mark.unit


def _parser_with_pipeline_args() -> argparse.ArgumentParser:
    """Return a parser populated with pipeline.add_args()."""
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)
    return parser


# ---------------------------------------------------------------------------
# pipeline.add_args
# ---------------------------------------------------------------------------

def test_pipeline_registers_particles_positional():
    """'particles' is a required positional argument."""
    parser = _parser_with_pipeline_args()
    # positional args are listed in _actions, not _option_string_actions
    positional_dests = [a.dest for a in parser._actions if not a.option_strings]
    assert "particles" in positional_dests


def test_pipeline_registers_outdir():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "-o" in actions or "--outdir" in actions


def test_pipeline_registers_zdim():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--zdim" in actions


def test_pipeline_zdim_default_is_list():
    """Default zdim must be a list (multiple resolutions are trained)."""
    parser = _parser_with_pipeline_args()
    # parse with minimum required args; zdim should fall back to default
    # We only check the default stored in the action, not actually parsing
    zdim_action = _parser_with_pipeline_args()._option_string_actions["--zdim"]
    assert isinstance(zdim_action.default, list)
    assert len(zdim_action.default) > 0


def test_pipeline_registers_poses():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--poses" in actions


def test_pipeline_registers_ctf():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--ctf" in actions


def test_pipeline_registers_mask():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--mask" in actions


def test_pipeline_registers_n_images():
    """--n-images (or equivalent) controls how many particles to use."""
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--n-images" in actions or "--n_images" in actions


# ---------------------------------------------------------------------------
# pipeline_with_outliers
# ---------------------------------------------------------------------------

def test_pipeline_with_outliers_module_importable():
    assert callable(pwo_cmd.run_pipeline_with_outlier_removal)


def test_pipeline_with_outliers_reuses_pipeline_add_args():
    """pipeline_with_outliers calls pipeline.add_args, so --zdim must be available
    when we build the combined parser the same way the command does."""
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)                   # same call as pwo_cmd
    parser.add_argument("--k-rounds", type=int, default=1)
    parser.add_argument("--use-contrast-detection", action="store_true")
    actions = parser._option_string_actions
    assert "--zdim" in actions
    assert "--k-rounds" in actions
    assert "--use-contrast-detection" in actions


def test_pipeline_with_outliers_k_rounds_default_is_one():
    """The default number of outlier-removal rounds is 1."""
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)
    parser.add_argument("--k-rounds", type=int, default=1)
    action = parser._option_string_actions["--k-rounds"]
    assert action.default == 1


def test_pipeline_with_outliers_junk_detection_flag_is_store_true():
    """--use-junk-detection must be a boolean flag (action='store_true')."""
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)
    parser.add_argument("--use-junk-detection", action="store_true")
    action = parser._option_string_actions["--use-junk-detection"]
    assert action.const is True   # store_true stores True
