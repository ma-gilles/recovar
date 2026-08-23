from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest

import recovar
from recovar.commands import initial_model
from recovar.em.initial_model import driver
from recovar.em.initial_model.schedules import GuiInitialModelDefaults


@pytest.mark.unit
def test_public_defaults_match_native_option_defaults():
    defaults = initial_model.initial_model_defaults_dict()
    native = driver.NativeInitialModelOptions(fn_img="particles.star")

    assert defaults == {
        field.name: getattr(GuiInitialModelDefaults(), field.name) for field in fields(GuiInitialModelDefaults)
    }
    for name in (
        "nr_iter",
        "nr_classes",
        "tau2_fudge",
        "sym_name",
        "do_run_C1",
        "particle_diameter",
        "do_solvent",
        "do_zero_mask",
        "do_ctf_correction",
        "random_seed",
        "healpix_order",
        "oversampling",
        "offset_range_px",
        "offset_step_px",
        "perturbation_factor",
        "random_perturbation",
        "image_batch_size",
        "rotation_block_size",
        "pass2_engine",
        "bootstrap_min_particles",
        "sigma2_min_particles",
        "padding_factor",
        "deterministic_cuda",
        "lazy",
        "translation_sigma_angstrom",
        "write_iter_artifacts",
        "grad_write_iter",
    ):
        assert getattr(native, name) == defaults[name]


@pytest.mark.unit
def test_parser_resolves_gui_defaults_and_auto_gpu_backend():
    args = initial_model.make_parser().parse_args(["--i", "particles.star"])
    options = initial_model._native_options_dict(args)

    assert options["nr_iter"] == 200
    assert options["nr_classes"] == 1
    assert options["image_fourier_backend"] == "relion_cuda"
    assert args.require_custom_cuda is True
    assert args.gpu_ids == "0"


@pytest.mark.unit
def test_parser_accepts_important_overrides():
    args = initial_model.make_parser().parse_args(
        [
            "--i",
            "particles.star",
            "--nr-iter",
            "17",
            "--K",
            "3",
            "--tau2-fudge",
            "2.5",
            "--sym",
            "C3",
            "--no-run-in-c1",
            "--particle-diameter",
            "280",
            "--no-solvent",
            "--no-zero-mask",
            "--no-ctf",
            "--random-seed",
            "9",
            "--healpix-order",
            "2",
            "--oversampling",
            "0",
            "--offset-range",
            "4.5",
            "--offset-step",
            "1.5",
            "--padding-factor",
            "2",
            "--pass2-engine",
            "compact",
            "--image-fourier-backend",
            "host_numpy",
            "--gpu",
            "",
            "--no-require-custom-cuda",
            "--deterministic-cuda",
        ]
    )
    options = initial_model._native_options_dict(args)

    assert options["nr_iter"] == 17
    assert options["nr_classes"] == 3
    assert options["tau2_fudge"] == 2.5
    assert options["sym_name"] == "C3"
    assert options["do_run_C1"] is False
    assert options["particle_diameter"] == 280.0
    assert options["do_solvent"] is False
    assert options["do_zero_mask"] is False
    assert options["do_ctf_correction"] is False
    assert options["random_seed"] == 9
    assert options["healpix_order"] == 2
    assert options["oversampling"] == 0
    assert options["offset_range_px"] == 4.5
    assert options["offset_step_px"] == 1.5
    assert options["padding_factor"] == 2
    assert options["pass2_engine"] == "compact"
    assert options["image_fourier_backend"] == "host_numpy"
    assert options["deterministic_cuda"] is True
    assert args.require_custom_cuda is False


@pytest.mark.unit
def test_dry_run_prints_resolved_native_options(capsys):
    rc = initial_model.main(["--i", "particles.star", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fn_img"] == "particles.star"
    assert payload["nr_classes"] == 1
    assert payload["image_fourier_backend"] == "relion_cuda"
    assert "resolved_cuda_allocator" in payload


@pytest.mark.unit
def test_module_entrypoint_keeps_default_allocator_without_override():
    environ = dict(os.environ)
    environ["JAX_PLATFORMS"] = "cpu"
    environ.pop("TF_GPU_ALLOCATOR", None)
    environ.pop("RECOVAR_INITIAL_MODEL_CUDA_ALLOCATOR", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "recovar.commands.initial_model",
            "--i",
            "particles.star",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environ,
    )

    assert json.loads(completed.stdout)["resolved_cuda_allocator"] == "default"


@pytest.mark.unit
def test_non_dry_run_calls_native_driver(monkeypatch, capsys):
    calls = {}

    def fake_run(options):
        calls["options"] = options
        return SimpleNamespace(final_mrc="out/initial_model.mrc", final_model_star="out/run_it003_model.star")

    monkeypatch.setattr(driver, "run_native_initial_model", fake_run)
    rc = initial_model.main(
        [
            "--i",
            "particles.star",
            "--o",
            "out/run",
            "--nr-iter",
            "3",
            "--gpu",
            "",
            "--no-require-custom-cuda",
            "--no-write-iter-artifacts",
        ]
    )

    assert rc == 0
    assert calls["options"].fn_img == "particles.star"
    assert calls["options"].nr_iter == 3
    assert calls["options"].write_iter_artifacts is False
    assert calls["options"].image_fourier_backend == "host_numpy"
    assert "recovar InitialModel complete" in capsys.readouterr().out


@pytest.mark.unit
def test_rejects_mpi_before_cuda_runtime_gate(monkeypatch):
    monkeypatch.setattr(initial_model, "_require_custom_cuda_runtime", lambda: pytest.fail("CUDA gate called"))
    with pytest.raises(SystemExit, match="not supported together with MPI"):
        initial_model.main(["--i", "particles.star", "--nr-mpi", "2"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "orig_argv"),
    [
        (["recovar", "initial_model", "--i", "particles.star"], ["python", "recovar"]),
        (
            ["recovar.commands.initial_model", "--i", "particles.star"],
            ["python", "-m", "recovar.commands.initial_model", "--i", "particles.star"],
        ),
        (
            ["scripts/run_ab_initio.py", "--i", "particles.star"],
            ["python", "scripts/run_ab_initio.py", "--i", "particles.star"],
        ),
    ],
)
def test_initial_model_bootstrap_keeps_default_allocator(argv, orig_argv):
    environ = {}

    resolved = recovar._configure_initial_model_cuda_allocator(
        argv=argv,
        orig_argv=orig_argv,
        environ=environ,
    )

    assert resolved is None
    assert "TF_GPU_ALLOCATOR" not in environ


@pytest.mark.unit
def test_initial_model_bootstrap_applies_requested_allocator():
    environ = {"RECOVAR_INITIAL_MODEL_CUDA_ALLOCATOR": "cuda_malloc_async"}

    resolved = recovar._configure_initial_model_cuda_allocator(
        argv=["recovar", "initial_model"],
        orig_argv=(),
        environ=environ,
    )

    assert resolved == "cuda_malloc_async"
    assert environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"


@pytest.mark.unit
def test_initial_model_bootstrap_respects_allocator_override_and_optout():
    argv = ["recovar", "initial_model"]
    caller_selected = {
        "TF_GPU_ALLOCATOR": "platform",
        "RECOVAR_INITIAL_MODEL_CUDA_ALLOCATOR": "cuda_malloc_async",
    }
    assert (
        recovar._configure_initial_model_cuda_allocator(
            argv=argv,
            orig_argv=(),
            environ=caller_selected,
        )
        == "platform"
    )

    opted_out = {"RECOVAR_INITIAL_MODEL_CUDA_ALLOCATOR": "default"}
    assert (
        recovar._configure_initial_model_cuda_allocator(
            argv=argv,
            orig_argv=(),
            environ=opted_out,
        )
        is None
    )
    assert "TF_GPU_ALLOCATOR" not in opted_out


@pytest.mark.unit
def test_initial_model_allocator_bootstrap_is_scoped_to_initial_model():
    environ = {}

    assert (
        recovar._configure_initial_model_cuda_allocator(
            argv=["recovar", "pipeline"],
            orig_argv=["python", "-m", "pytest"],
            environ=environ,
        )
        is None
    )
    assert environ == {}
