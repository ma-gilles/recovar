import argparse

import pytest

from recovar.utils import parser_args

pytestmark = pytest.mark.unit


def test_standard_downstream_args_outdir_optional():
    """--outdir is always optional (auto-numbering fills in the default)."""
    parser = parser_args.standard_downstream_args(argparse.ArgumentParser(), analyze=False)
    parsed = parser.parse_args(["/tmp/results", "-o", "/tmp/out"])
    assert parsed.result_dir == "/tmp/results"
    assert parsed.outdir == "/tmp/out"

    # Without -o, outdir should be None (auto-numbering kicks in at runtime)
    parsed2 = parser.parse_args(["/tmp/results"])
    assert parsed2.outdir is None


def test_standard_downstream_args_outdir_optional_for_analyze():
    parser = parser_args.standard_downstream_args(argparse.ArgumentParser(), analyze=True)
    parsed = parser.parse_args(["/tmp/results"])
    assert parsed.result_dir == "/tmp/results"
    assert parsed.outdir is None


def test_standard_downstream_args_defaults_and_flags():
    parser = parser_args.standard_downstream_args(argparse.ArgumentParser(), analyze=True)
    parsed = parser.parse_args(
        [
            "/tmp/results",
            "--zdim1",
            "--no-z-regularization",
            "--lazy",
            "--apply-global-filtering",
        ]
    )
    assert parsed.n_bins == 50
    assert parsed.maskrad_fraction == 20
    assert parsed.zdim1 is True
    assert parsed.no_z_regularization is True
    assert parsed.lazy is True
    assert parsed.apply_global_filtering is True



def test_standard_downstream_args_project_mode_defaults():
    parser = parser_args.standard_downstream_args(argparse.ArgumentParser(), analyze=True)
    parsed = parser.parse_args(["--project", "/tmp/project", "--output-name", "embedding_k10"])
    assert parsed.result_dir is None
    assert parsed.project == "/tmp/project"
    assert parsed.output_name == "embedding_k10"
