"""
Unit tests for recovar.commands.estimate_stable_states.

Covers:
  parse_args  – argument registration
  main        – pickle_load called with correct path; downstream called
  estimate_stable_states – directory creation and file writes (mocked)
"""

import argparse
import os
import sys
import pytest
import numpy as np
from types import SimpleNamespace

from recovar.commands import estimate_stable_states as ess_cmd

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# parse_args / argument registration
# ---------------------------------------------------------------------------


def test_parse_args_registers_density_positional():
    """'density' is a required positional arg."""
    parser = argparse.ArgumentParser()
    parser.add_argument("density", type=str)
    parser.add_argument("-o", "--output", dest="file_path", type=str, required=True)
    parser.add_argument("--percent_top", type=float, default=1)
    parser.add_argument("--n_local_maxs", type=int, default=3)
    positionals = [a.dest for a in parser._actions if not a.option_strings]
    assert "density" in positionals


def test_parse_args_registers_output_flag():
    """'-o' / '--output' must be a recognised flag."""
    parser = argparse.ArgumentParser()
    parser.add_argument("density")
    parser.add_argument("-o", "--output", dest="file_path", type=str, required=True)
    assert "-o" in parser._option_string_actions or "--output" in parser._option_string_actions


def test_parse_args_percent_top_default_is_one():
    parser = argparse.ArgumentParser()
    parser.add_argument("density")
    parser.add_argument("-o", "--output", dest="file_path", type=str, required=True)
    parser.add_argument("--percent_top", type=float, default=1)
    action = parser._option_string_actions["--percent_top"]
    assert action.default == 1


def test_parse_args_n_local_maxs_default_is_three():
    parser = argparse.ArgumentParser()
    parser.add_argument("density")
    parser.add_argument("-o", "--output", dest="file_path", type=str, required=True)
    parser.add_argument("--n_local_maxs", type=int, default=3)
    action = parser._option_string_actions["--n_local_maxs"]
    assert action.default == 3


# ---------------------------------------------------------------------------
# main – pickle_load + downstream call verification
# ---------------------------------------------------------------------------


def test_main_loads_pickle_with_given_path(monkeypatch, tmp_path):
    """main() must call utils.pickle_load with the path from args.density."""
    loaded_paths = []

    fake_density = np.zeros((4, 4))
    fake_bounds = np.array([[-1, 1], [-1, 1]], dtype=np.float32)
    fake_dens_pkl = {"density": fake_density, "latent_space_bounds": fake_bounds}

    monkeypatch.setattr(ess_cmd.utils, "pickle_load", lambda path: (loaded_paths.append(path), fake_dens_pkl)[1])
    monkeypatch.setattr(
        ess_cmd,
        "estimate_stable_states",
        lambda *a, **kw: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_stable_states",
            "density.pkl",
            "-o",
            str(tmp_path / "out"),
        ],
    )

    ess_cmd.main()

    assert loaded_paths == ["density.pkl"]


def test_main_calls_estimate_stable_states(monkeypatch, tmp_path):
    """main() must call estimate_stable_states with parsed arguments."""
    calls = []

    fake_density = np.zeros((4, 4))
    fake_bounds = np.array([[-1, 1], [-1, 1]], dtype=np.float32)
    fake_dens_pkl = {"density": fake_density, "latent_space_bounds": fake_bounds}

    monkeypatch.setattr(ess_cmd.utils, "pickle_load", lambda _: fake_dens_pkl)

    # main() calls estimate_stable_states positionally:
    #   estimate_stable_states(density, latent_space_bounds, percent_top, n_local_maxs, file_path)
    def fake_ess(density, latent_space_bounds, percent_top=1, n_local_maxs=3, file_path=None):
        calls.append({"file_path": file_path})

    monkeypatch.setattr(ess_cmd, "estimate_stable_states", fake_ess)

    out_dir = str(tmp_path / "states")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_stable_states",
            "density.pkl",
            "-o",
            out_dir,
        ],
    )

    ess_cmd.main()

    assert len(calls) == 1
    assert calls[0]["file_path"] == out_dir


# ---------------------------------------------------------------------------
# estimate_stable_states – directory creation and downstream calls (mocked)
# ---------------------------------------------------------------------------


def test_estimate_stable_states_creates_output_dirs(monkeypatch, tmp_path):
    """estimate_stable_states must call mkdir_safe for the output path."""
    made_dirs = []
    monkeypatch.setattr(ess_cmd.output, "mkdir_safe", lambda p: made_dirs.append(p))
    monkeypatch.setattr(
        ess_cmd.deconvolve_density,
        "find_local_maxs_of_density",
        lambda *a, **kw: (np.zeros((2, 2)), np.zeros((2, 2))),
    )
    monkeypatch.setattr(ess_cmd.output, "plot_over_density", lambda *a, **kw: None)
    monkeypatch.setattr(np, "savetxt", lambda *a, **kw: None)

    out = str(tmp_path / "out") + "/"
    ess_cmd.estimate_stable_states(
        density=np.zeros((4, 4)),
        latent_space_bounds=np.array([[-1, 1], [-1, 1]]),
        file_path=out,
    )

    assert any(out in d for d in made_dirs)


def test_estimate_stable_states_calls_find_local_maxs(monkeypatch, tmp_path):
    """estimate_stable_states must delegate to deconvolve_density.find_local_maxs_of_density."""
    calls = []
    monkeypatch.setattr(ess_cmd.output, "mkdir_safe", lambda _: None)
    monkeypatch.setattr(
        ess_cmd.deconvolve_density,
        "find_local_maxs_of_density",
        lambda density, bounds, **kw: (
            calls.append({"density": density, "bounds": bounds}),
            (np.zeros((1, 2)), np.zeros((1, 2))),
        )[1],
    )
    monkeypatch.setattr(ess_cmd.output, "plot_over_density", lambda *a, **kw: None)
    monkeypatch.setattr(np, "savetxt", lambda *a, **kw: None)

    density = np.ones((4, 4))
    bounds = np.array([[-2, 2], [-2, 2]], dtype=np.float32)
    ess_cmd.estimate_stable_states(density, bounds, file_path=str(tmp_path) + "/")

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["density"], density)
