from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import run_vdam_relion_parity_case as runner

pytestmark = pytest.mark.unit


DEFINITION = {
    "nr_classes": 1,
    "nr_iter": 8,
    "random_seed": 0,
    "tau2_fudge": 4.0,
    "healpix_order": 1,
    "oversampling": 1,
    "offset_range_px": 6.0,
    "offset_step_px": 2.0,
    "padding_factor": 1,
}


def _value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1
    return argv[argv.index(flag) + 1]


def test_relion_command_is_the_frozen_gui_initialmodel_contract():
    argv = runner.build_relion_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/relion/run"),
        definition=DEFINITION,
        relion_refine=Path("/opt/relion_refine"),
        threads=8,
    )

    assert argv[0] == "/opt/relion_refine"
    assert {"--grad", "--denovo_3dref", "--flatten_solvent", "--zero_mask", "--auto_sampling"} <= set(argv)
    assert _value(argv, "--iter") == "8"
    assert _value(argv, "--grad_write_iter") == "1"
    assert _value(argv, "--K") == "1"
    assert _value(argv, "--random_seed") == "0"
    assert _value(argv, "--tau2_fudge") == "4.0"
    assert _value(argv, "--healpix_order") == "1"
    assert _value(argv, "--oversampling") == "1"
    assert _value(argv, "--offset_range") == "6.0"
    assert _value(argv, "--offset_step") == "2.0"
    assert _value(argv, "--pad") == "1"
    assert _value(argv, "--gpu") == "0"


def test_recovar_command_maps_the_same_frozen_definition(monkeypatch):
    monkeypatch.setattr(runner.sys, "executable", "/env/python")

    argv = runner.build_recovar_command(
        input_star=Path("/data/particles.star"),
        output_prefix=Path("/out/recovar/run"),
        fixture_dir=Path("/data"),
        definition=DEFINITION,
    )

    assert argv[:2] == ["/env/python", "scripts/run_ab_initio.py"]
    assert _value(argv, "--nr_iter") == "8"
    assert _value(argv, "--K") == "1"
    assert _value(argv, "--random_seed") == "0"
    assert _value(argv, "--tau2_fudge") == "4.0"
    assert _value(argv, "--healpix_order") == "1"
    assert _value(argv, "--oversampling") == "1"
    assert _value(argv, "--offset_range") == "6.0"
    assert _value(argv, "--offset_step") == "2.0"
    assert _value(argv, "--padding_factor") == "1"
    assert _value(argv, "--gpu") == "0"


def test_gpu_capture_requires_exactly_one_visible_uuid(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "GPU-one\nGPU-two\n", ""),
    )

    with pytest.raises(runner.RunError, match="exactly one visible"):
        runner._physical_gpu_uuid()


def test_gpu_capture_rejects_slurm_uuid_mismatch(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_GPUS", "GPU-slurm")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "GPU-visible\n", ""),
    )

    with pytest.raises(runner.RunError, match="differs from visible UUID"):
        runner._physical_gpu_uuid()
