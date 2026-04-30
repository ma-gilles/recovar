"""G.1 command-snapshot test (plan v3 § G.1).

Asserts that `scripts/run_ab_initio.py::build_command` produces a
command string matching the GUI-generated InitialModel command at
`pipeline_jobs.cpp::getCommandsInimodelJob` (lines 3428-3613).

This is the single most important parity test for avoiding silent drift
toward auto-refine defaults.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_ab_initio.py"


@pytest.fixture(scope="module")
def run_ab_initio():
    """Load `scripts/run_ab_initio.py` as a module without importing
    `scripts.*` (it isn't a package).

    We have to register the module in `sys.modules` before `exec_module`
    so `@dataclass`'s forward-ref resolution can find it via
    `sys.modules[cls.__module__].__dict__`.
    """
    import sys

    spec = importlib.util.spec_from_file_location("run_ab_initio", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ab_initio"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("run_ab_initio", None)
        raise
    return module


def _basic_opts(run_ab_initio, **overrides):
    opts = run_ab_initio.InitialModelJobOptions(
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
    def test_grad_and_denovo_3dref(self, run_ab_initio):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio))
        assert "--grad" in cmd
        assert "--denovo_3dref" in cmd

    def test_pad_1(self, run_ab_initio):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio))
        # --pad is immediately followed by "1" (not "2")
        idx = cmd.index("--pad")
        assert cmd[idx + 1] == "1"

    def test_auto_sampling(self, run_ab_initio):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio))
        assert "--auto_sampling" in cmd

    def test_oversampling_and_healpix_defaults(self, run_ab_initio):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio))
        # Exact token sequence from pipeline_jobs.cpp:3548
        assert cmd[cmd.index("--oversampling") + 1] == "1"
        assert cmd[cmd.index("--healpix_order") + 1] == "1"
        assert cmd[cmd.index("--offset_range") + 1] == "6"
        assert cmd[cmd.index("--offset_step") + 1] == "2"

    def test_tau2_fudge_from_opts(self, run_ab_initio):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio, tau2_fudge=4.0))
        assert cmd[cmd.index("--tau2_fudge") + 1] == "4.0"

    def test_zero_mask_and_flatten_solvent(self, run_ab_initio):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio))
        assert "--zero_mask" in cmd
        assert "--flatten_solvent" in cmd

    def test_ctf_enabled_by_default(self, run_ab_initio):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio))
        assert "--ctf" in cmd

    def test_K_sym_particle_diameter(self, run_ab_initio):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio, nr_classes=2, sym_name="C4", do_run_C1=False))
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
    def test_absent(self, run_ab_initio, token):
        cmd = run_ab_initio.build_command(_basic_opts(run_ab_initio))
        assert token not in cmd, f"{token} must not appear in InitialModel command — check pipeline_jobs.cpp:3428-3613"


# ---------------------------------------------------------------------------
# MPI rejection
# ---------------------------------------------------------------------------


class TestMpiRejected:
    def test_nr_mpi_gt_1_raises(self, run_ab_initio):
        opts = _basic_opts(run_ab_initio, nr_mpi=2)
        with pytest.raises(SystemExit):
            run_ab_initio.build_command(opts)

    def test_nr_mpi_1_ok(self, run_ab_initio):
        opts = _basic_opts(run_ab_initio, nr_mpi=1)
        cmd = run_ab_initio.build_command(opts)
        assert cmd[0] == "relion_refine"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_fn_img_raises(self, run_ab_initio):
        opts = run_ab_initio.InitialModelJobOptions(fn_img="")
        with pytest.raises(SystemExit):
            run_ab_initio.build_command(opts)

    def test_ctf_intact_first_peak_added_when_true(self, run_ab_initio):
        opts = _basic_opts(run_ab_initio, ctf_intact_first_peak=True)
        cmd = run_ab_initio.build_command(opts)
        assert "--ctf_intact_first_peak" in cmd

    def test_ctf_intact_first_peak_absent_when_false(self, run_ab_initio):
        opts = _basic_opts(run_ab_initio, ctf_intact_first_peak=False)
        cmd = run_ab_initio.build_command(opts)
        assert "--ctf_intact_first_peak" not in cmd


# ---------------------------------------------------------------------------
# align_symmetry command
# ---------------------------------------------------------------------------


class TestAlignSymmetryCommand:
    def test_c1_run_emits_sym_c1(self, run_ab_initio):
        cmd = run_ab_initio.build_align_symmetry_command(
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

    def test_user_sym_c4_with_run_c1_true_emits_c4(self, run_ab_initio):
        cmd = run_ab_initio.build_align_symmetry_command(
            outputname="ab_initio/run",
            nr_iter=200,
            sym_name="C4",
            do_run_C1=True,
        )
        assert cmd[cmd.index("--sym") + 1] == "C4"

    def test_user_sym_c4_with_run_c1_false_emits_c1(self, run_ab_initio):
        # When not running in C1 (optimisation done in C4 directly), the
        # align-symmetry step aligns to C1 axes (no-op for a C4 volume
        # already in the right frame — matches RELION's else branch).
        cmd = run_ab_initio.build_align_symmetry_command(
            outputname="ab_initio/run",
            nr_iter=200,
            sym_name="C4",
            do_run_C1=False,
        )
        assert cmd[cmd.index("--sym") + 1] == "C1"


# ---------------------------------------------------------------------------
# align_symmetry module API
# ---------------------------------------------------------------------------


class TestAlignSymmetryModule:
    def test_build_tokens_matches_driver(self, run_ab_initio):
        """The standalone module in recovar.em.initial_model.align_symmetry
        must produce the same tokens as the driver script."""
        from recovar.em.initial_model.align_symmetry import (
            AlignSymmetrySpec,
            build_align_symmetry_tokens,
        )

        spec = AlignSymmetrySpec(
            last_model_star="ab_initio/run_it200_model.star",
            out_mrc="ab_initio/initial_model.mrc",
            sym_name="C4",
            do_run_C1=True,
        )
        module_tokens = build_align_symmetry_tokens(spec)
        # The driver composes out_mrc differently (strips 'run' from
        # outputname); we just check the semantic invariants.
        assert module_tokens[0] == "relion_align_symmetry"
        assert "--apply_sym" in module_tokens
        assert "--select_largest_class" in module_tokens
        assert module_tokens[module_tokens.index("--sym") + 1] == "C4"

    def test_c1_noop(self, run_ab_initio):
        from recovar.em.initial_model.align_symmetry import (
            AlignSymmetrySpec,
            build_align_symmetry_tokens,
        )

        spec = AlignSymmetrySpec(
            last_model_star="a.star",
            out_mrc="b.mrc",
            sym_name="C1",
            do_run_C1=True,
        )
        tokens = build_align_symmetry_tokens(spec)
        assert tokens[tokens.index("--sym") + 1] == "C1"
