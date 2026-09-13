"""G.1 command-snapshot test (plan v3 § G.1).

Asserts that `tests/helpers/relion_initial_model_command.py::build_command` produces a
command string matching the GUI-generated InitialModel command at
`pipeline_jobs.cpp::getCommandsInimodelJob` (lines 3428-3613).

This is the single most important parity test for avoiding silent drift
toward auto-refine defaults.
"""

from __future__ import annotations

import pytest
from helpers import relion_initial_model_command

from recovar.commands import initial_model


@pytest.fixture(scope="module")
def relion_command():
    return relion_initial_model_command


def _basic_opts(relion_command, **overrides):
    opts = relion_command.InitialModelJobOptions(
        fn_img="particles.star",
        outputname="ab_initio/run",
        nr_iter=200,
        nr_classes=1,
        tau2_fudge=4.0,
        sym_name="C1",
        do_run_C1=True,
        particle_diameter=200.0,
        nr_threads=1,
    )
    for key, val in overrides.items():
        setattr(opts, key, val)
    return opts


# ---------------------------------------------------------------------------
# Required tokens (tokens that MUST be present)
# ---------------------------------------------------------------------------


class TestCommandContainsRequiredFlags:
    def test_grad_and_denovo_3dref(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command))
        assert "--grad" in cmd
        assert "--denovo_3dref" in cmd
        assert cmd[cmd.index("--grad_write_iter") + 1] == "10"

    def test_grad_write_interval_is_configurable(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command, grad_write_iter=1))
        assert cmd[cmd.index("--grad_write_iter") + 1] == "1"

    def test_pad_1(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command))
        # --pad is immediately followed by "1" (not "2")
        idx = cmd.index("--pad")
        assert cmd[idx + 1] == "1"

    def test_auto_sampling(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command))
        assert "--auto_sampling" in cmd

    def test_oversampling_and_healpix_defaults(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command))
        # Exact token sequence from pipeline_jobs.cpp:3548
        assert cmd[cmd.index("--oversampling") + 1] == "1"
        assert cmd[cmd.index("--healpix_order") + 1] == "1"
        assert cmd[cmd.index("--offset_range") + 1] == "6"
        assert cmd[cmd.index("--offset_step") + 1] == "2"

    def test_tau2_fudge_from_opts(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command, tau2_fudge=4.0))
        assert cmd[cmd.index("--tau2_fudge") + 1] == "4.0"

    def test_zero_mask_and_flatten_solvent(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command))
        assert "--zero_mask" in cmd
        assert "--flatten_solvent" in cmd

    def test_ctf_enabled_by_default(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command))
        assert "--ctf" in cmd

    def test_K_sym_particle_diameter(self, relion_command):
        cmd = relion_command.build_command(_basic_opts(relion_command, nr_classes=2, sym_name="C4", do_run_C1=False))
        assert cmd[cmd.index("--K") + 1] == "2"
        assert cmd[cmd.index("--sym") + 1] == "C4"
        assert "--particle_diameter" in cmd


# ---------------------------------------------------------------------------
# Rejected tokens (tokens that MUST NOT be present)
# ---------------------------------------------------------------------------


class TestCommandDoesNotContainForbiddenFlags:
    """Any token from this list appearing means we've silently absorbed
    auto-refine defaults and broken InitialModel parity."""

    FORBIDDEN = [
        "--split_random_halves",
        "--auto_refine",
        "--low_resol_join_halves",
        "--norm",
        "--scale",
        "--firstiter_cc",
        "--ini_high",
        "--grad_ini_resol",
        "--grad_fin_resol",
    ]

    @pytest.mark.parametrize("token", FORBIDDEN)
    def test_absent(self, relion_command, token):
        cmd = relion_command.build_command(_basic_opts(relion_command))
        assert token not in cmd, f"{token} must not appear in InitialModel command — check pipeline_jobs.cpp:3428-3613"


# ---------------------------------------------------------------------------
# MPI rejection
# ---------------------------------------------------------------------------


class TestMpiRejected:
    def test_nr_mpi_gt_1_raises(self, relion_command):
        opts = _basic_opts(relion_command, nr_mpi=2)
        with pytest.raises(SystemExit):
            relion_command.build_command(opts)

    def test_nr_mpi_1_ok(self, relion_command):
        opts = _basic_opts(relion_command, nr_mpi=1)
        cmd = relion_command.build_command(opts)
        assert cmd[0] == "relion_refine"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_fn_img_raises(self, relion_command):
        opts = relion_command.InitialModelJobOptions(fn_img="")
        with pytest.raises(SystemExit):
            relion_command.build_command(opts)

    def test_ctf_intact_first_peak_added_when_true(self, relion_command):
        opts = _basic_opts(relion_command, ctf_intact_first_peak=True)
        cmd = relion_command.build_command(opts)
        assert "--ctf_intact_first_peak" in cmd

    def test_ctf_intact_first_peak_absent_when_false(self, relion_command):
        opts = _basic_opts(relion_command, ctf_intact_first_peak=False)
        cmd = relion_command.build_command(opts)
        assert "--ctf_intact_first_peak" not in cmd


class TestRecovarRuntimeOptions:
    def test_stable_fourier_window_shapes_defaults_off(self):
        args = initial_model.make_parser().parse_args(["--no-require-custom-cuda", "--no-jax-compilation-cache", "--gpu", "", "--i", "particles.star"])
        assert args.stable_fourier_window_shapes is False

    def test_stable_fourier_window_shapes_reaches_native_driver(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        import recovar.em.vdam.driver as driver

        captured = {}

        def fake_run_native_initial_model(options):
            captured["options"] = options
            return SimpleNamespace(final_mrc="initial_model.mrc", final_model_star="model.star")

        monkeypatch.setattr(driver, "run_native_initial_model", fake_run_native_initial_model)
        assert (
            initial_model.main(
                [
                    "--no-require-custom-cuda", "--no-jax-compilation-cache", "--gpu", "", "--i",
                    "particles.star",
                    "--stable-fourier-window-shapes",
                    "--no-write-iter-artifacts",
                ]
            )
            == 0
        )
        assert captured["options"].stable_fourier_window_shapes is True


# ---------------------------------------------------------------------------
# align_symmetry command
# ---------------------------------------------------------------------------


class TestAlignSymmetryCommand:
    def test_c1_run_emits_sym_c1(self, relion_command):
        cmd = relion_command.build_align_symmetry_command(
            outputname="ab_initio/run",
            nr_iter=200,
            sym_name="C1",
            do_run_C1=True,
        )
        assert cmd[0] == "relion_align_symmetry"
        # --i <last_model.star>
        assert cmd[1] == "--i"
        assert cmd[2] == "ab_initio/run_it200_model.star"
        # When do_run_C1 and sym==C1, RELION emits --sym C1
        assert cmd[cmd.index("--sym") + 1] == "C1"
        assert "--apply_sym" in cmd
        assert "--select_largest_class" in cmd

    def test_user_sym_c4_with_run_c1_true_emits_c4(self, relion_command):
        cmd = relion_command.build_align_symmetry_command(
            outputname="ab_initio/run",
            nr_iter=200,
            sym_name="C4",
            do_run_C1=True,
        )
        assert cmd[cmd.index("--sym") + 1] == "C4"

    def test_user_sym_c4_with_run_c1_false_emits_c1(self, relion_command):
        # When not running in C1 (optimisation done in C4 directly), the
        # align-symmetry step aligns to C1 axes (no-op for a C4 volume
        # already in the right frame — matches RELION's else branch).
        cmd = relion_command.build_align_symmetry_command(
            outputname="ab_initio/run",
            nr_iter=200,
            sym_name="C4",
            do_run_C1=False,
        )
        assert cmd[cmd.index("--sym") + 1] == "C1"
