"""Regression guards for the EM-parity fixes shipped on
``codex/vdam-initial-volume-parity`` ahead of the EM / VDAM / PPCA merge.

Locks down two parity-critical changes that are easy to silently revert
during a merge:

1. ``RELION_WIDTH_FMASK_EDGE = 2`` (Fourier mask edge for
   ``initialLowPassFilterReferences``) is distinct from the real-space
   ``RELION_WIDTH_MASK_EDGE = 5`` (RELION's ``--maskedge``). Conflating
   them gives a softer LP filter than RELION applies — see
   ``ml_optimiser.h:91`` and ``ml_optimiser.cpp:3556``.

2. ``--apply-initial-lowpass`` CLI flag exists on
   ``scripts/run_full_refinement.py`` and triggers the LP filter via
   ``recovar.heterogeneity.locres.low_pass_filter_map`` with the
   Fourier mask edge width. RELION's ``initialLowPassFilterReferences``
   runs whenever ``--ini_high > 0`` regardless of ``--firstiter_cc``;
   recovar previously only mirrored that under ``--firstiter_cc``,
   leaving an iter-1 reconstruction gap on auto-refine fixtures.

3. ``_parse_relion_tau2_fudge`` maps ``_rlnTau2FudgeArg <= 0`` to
   ``None`` (RELION's sentinel for "user did not pass --tau2_fudge")
   and prefers ``_rlnTau2FudgeFactor`` from model.star. The
   ``ml_optimiser.cpp:881-882`` resolution is
   ``tau2_fudge_factor = arg > 0 ? arg : 1``; recovar previously
   passed the raw -1 sentinel into the Wiener regularization, which
   inverts the prior term and corrupts iter-1's reconstruction.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ITERATION_LOOP_PY = REPO_ROOT / "recovar" / "em" / "dense_single_volume" / "iteration_loop.py"
MEAN_HELPERS_PY = REPO_ROOT / "recovar" / "em" / "dense_single_volume" / "mean_helpers.py"
RUN_FULL_REFINEMENT_PY = REPO_ROOT / "scripts" / "run_full_refinement.py"


# ---------------------------------------------------------------------------
# Bug #1: RELION_WIDTH_FMASK_EDGE = 2 (Fourier mask edge, distinct from
# real-space mask edge).
# ---------------------------------------------------------------------------


def test_relion_width_fmask_edge_constant_is_two():
    """Lock the Fourier mask edge constant at 2 shells.

    Matches RELION's ``WIDTH_FMASK_EDGE`` (``ml_optimiser.h:91``). A merge
    that reverts this to 5 or removes the constant entirely would silently
    soften recovar's iter-1 LP filter (taper 5 shells wide instead of 2).
    """

    source = ITERATION_LOOP_PY.read_text()
    match = re.search(r"RELION_WIDTH_FMASK_EDGE\s*=\s*([0-9]+)", source)
    assert match is not None, (
        "RELION_WIDTH_FMASK_EDGE constant missing from iteration_loop.py. "
        "Required for the iter-1 ini_high low-pass parity fix."
    )
    assert int(match.group(1)) == 2, (
        f"RELION_WIDTH_FMASK_EDGE = {match.group(1)} but RELION's WIDTH_FMASK_EDGE "
        f"(ml_optimiser.h:91) is 2. Do not conflate with the real-space "
        f"RELION_WIDTH_MASK_EDGE = 5."
    )


def test_relion_width_mask_edge_constant_is_five():
    """Lock the real-space mask edge constant at 5 px (RELION's ``--maskedge``)."""

    source = ITERATION_LOOP_PY.read_text()
    match = re.search(r"RELION_WIDTH_MASK_EDGE\s*=\s*([0-9]+)", source)
    assert match is not None, "RELION_WIDTH_MASK_EDGE constant missing from iteration_loop.py."
    assert int(match.group(1)) == 5, (
        f"RELION_WIDTH_MASK_EDGE = {match.group(1)} but RELION's --maskedge default "
        f"(ml_optimiser.cpp:1235) is 5."
    )


def test_mean_helpers_lp_filter_uses_fmask_edge_not_mask_edge():
    """Lock the post-iter-1 LP filter to use the Fourier mask edge param.

    Mixing the real-space ``relion_width_mask_edge`` with the Fourier
    ``low_pass_filter_map`` is the original bug. Search the post-recon LP
    filter call sites for ``filter_edgewidth=relion_fmask_edge``.
    """

    source = MEAN_HELPERS_PY.read_text()
    bad_pattern = re.compile(r"filter_edgewidth\s*=\s*relion_width_mask_edge")
    good_pattern = re.compile(r"filter_edgewidth\s*=\s*relion_fmask_edge")
    assert bad_pattern.search(source) is None, (
        "_apply_relion_initial_lowpass_filter is still being called with "
        "filter_edgewidth=relion_width_mask_edge in mean_helpers.py. "
        "That conflates the real-space mask edge (5 px) with the Fourier "
        "mask edge (2 shells). Use filter_edgewidth=relion_fmask_edge instead."
    )
    assert len(good_pattern.findall(source)) >= 2, (
        "Expected at least two filter_edgewidth=relion_fmask_edge call "
        "sites (K-class and K=1 branches) in mean_helpers.py."
    )


def test_reconstruct_postprocess_means_threads_fmask_edge():
    """Lock the new ``relion_fmask_edge`` kwarg on the recon helper signature."""

    source = MEAN_HELPERS_PY.read_text()
    tree = ast.parse(source)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_reconstruct_and_postprocess_means":
            fn = node
            break
    assert fn is not None, "_reconstruct_and_postprocess_means missing from mean_helpers.py"
    kwarg_names = {arg.arg for arg in fn.args.kwonlyargs}
    assert "relion_fmask_edge" in kwarg_names, (
        "Expected kwarg 'relion_fmask_edge' on _reconstruct_and_postprocess_means; "
        "a merge that drops it will silently revert the Fourier-vs-real-space "
        "mask-edge distinction."
    )
    assert "relion_width_mask_edge" in kwarg_names, (
        "Expected kwarg 'relion_width_mask_edge' to remain (real-space mask edge)."
    )


# ---------------------------------------------------------------------------
# Bug #2: --apply-initial-lowpass CLI flag.
# ---------------------------------------------------------------------------


def _build_run_full_refinement_parser() -> argparse.ArgumentParser:
    """Import the actual argparse parser from scripts/run_full_refinement.py.

    The script defines the parser inside ``main()`` so we can't directly
    import it. AST-walk the source instead to extract the
    ``add_argument`` calls relevant to this regression and reconstruct a
    minimal parser.
    """

    source = RUN_FULL_REFINEMENT_PY.read_text()
    # Use direct regex parsing for the flag specifically. This avoids
    # executing run_full_refinement.py (which has heavy imports).
    return source


def test_apply_initial_lowpass_flag_exists():
    """Lock the --apply-initial-lowpass CLI flag in run_full_refinement.py."""

    source = _build_run_full_refinement_parser()
    assert re.search(r'"--apply-initial-lowpass"', source) is not None, (
        "Missing CLI flag '--apply-initial-lowpass' on scripts/run_full_refinement.py. "
        "Required to opt into RELION's initialLowPassFilterReferences on iter-1 "
        "input reference."
    )


def test_apply_initial_lowpass_flag_default_off():
    """Lock the default value of --apply-initial-lowpass to False.

    Backward-compatibility guarantee: existing recovar callers (and the
    K=4 sbatch which doesn't want LP filtering since RELION K=4 Class3D
    has no --ini_high) should keep their current behavior.
    """

    source = _build_run_full_refinement_parser()
    # Match the add_argument block (multi-line). Look for the literal
    # default=False and dest=apply_initial_lowpass.
    flag_match = re.search(
        r'add_argument\(\s*"--apply-initial-lowpass"[^\)]*?dest="apply_initial_lowpass"[^\)]*?default=False',
        source, re.DOTALL,
    )
    assert flag_match is not None, (
        "Expected --apply-initial-lowpass to be defined with dest='apply_initial_lowpass' "
        "and default=False."
    )


def test_apply_initial_lowpass_fmask_edge_is_two_in_script():
    """Lock the Fourier mask edge used by the in-script LP helper at 2."""

    source = _build_run_full_refinement_parser()
    match = re.search(r"_RELION_FMASK_EDGE\s*=\s*([0-9]+)", source)
    assert match is not None, (
        "Missing _RELION_FMASK_EDGE constant in scripts/run_full_refinement.py. "
        "The --apply-initial-lowpass helper needs RELION's WIDTH_FMASK_EDGE = 2."
    )
    assert int(match.group(1)) == 2, (
        f"_RELION_FMASK_EDGE = {match.group(1)} in run_full_refinement.py; "
        f"RELION's WIDTH_FMASK_EDGE (ml_optimiser.h:91) is 2."
    )


def test_apply_initial_lowpass_helper_calls_low_pass_filter_map():
    """Lock the call site to ``low_pass_filter_map`` (not a custom variant)."""

    source = _build_run_full_refinement_parser()
    assert re.search(r"from recovar\.heterogeneity\.locres import low_pass_filter_map", source) is not None, (
        "Expected --apply-initial-lowpass helper to import "
        "recovar.heterogeneity.locres.low_pass_filter_map. A custom in-script "
        "filter would risk drifting from the parity-tested implementation."
    )


def test_apply_initial_lowpass_helper_uses_init_resolution_as_ini_high():
    """Lock the LP cutoff to ``--init_resolution`` (not a separate flag)."""

    source = _build_run_full_refinement_parser()
    # The helper should bind ini_high_for_lowpass from args.init_resolution.
    assert re.search(
        r"_ini_high_for_lowpass\s*=\s*\(\s*float\(args\.init_resolution\)",
        source,
    ) is not None, (
        "Expected --apply-initial-lowpass to use --init_resolution as the LP cutoff. "
        "Splitting them into two CLI flags would be confusing and break parity."
    )


# ---------------------------------------------------------------------------
# Bug #3: tau2_fudge=-1 sentinel handling.
# ---------------------------------------------------------------------------


def test_parse_tau2_fudge_neg_one_arg_returns_none():
    """Lock the parser behavior: _rlnTau2FudgeArg=-1 (RELION's "unset"
    sentinel) returns None so _resolve_tau2_fudge falls back to the
    K-class binary default (1.0 auto-refine, 4.0 Class3D).

    RELION's ml_optimiser.cpp:881-882 resolves arg<=0 to default 1.0;
    passing -1 verbatim into recovar's Wiener inverts the regularization
    and collapses iter-2+ ave_Pmax.
    """

    from scripts.run_full_refinement import _parse_relion_tau2_fudge

    text = "_rlnTau2FudgeArg                                          -1.000000\n"
    assert _parse_relion_tau2_fudge(text) is None


def test_parse_tau2_fudge_prefers_factor_over_arg():
    """Lock model.star Factor (actual value used) over optimiser.star Arg."""

    from scripts.run_full_refinement import _parse_relion_tau2_fudge

    text = """
_rlnTau2FudgeFactor 1.000000
_rlnTau2FudgeArg    -1.000000
"""
    assert _parse_relion_tau2_fudge(text) == pytest.approx(1.0)


def test_parse_tau2_fudge_passes_positive_arg_through():
    """Lock K-class CLI value: explicit Arg=4 passes through to recovar."""

    from scripts.run_full_refinement import _parse_relion_tau2_fudge

    text = "_rlnTau2FudgeArg                                          4.000000\n"
    assert _parse_relion_tau2_fudge(text) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Completion-benchmark baselines: assert the locked JSONs exist + are read-able
# so future merges that delete them break this test rather than silently
# losing the parity evidence.
# ---------------------------------------------------------------------------


BASELINES = REPO_ROOT / "tests" / "baselines"


@pytest.mark.parametrize(
    "name",
    [
        "em_parity_completion_k1_100k256_os0_strict",
        "em_parity_completion_k4_100k256",
        "em_parity_completion_k1_5k128_apply_lowpass",
    ],
)
def test_completion_baselines_exist_and_are_readable(name):
    import json

    path = BASELINES / f"{name}.json"
    assert path.exists(), f"Missing locked baseline {path}"
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict), f"Baseline {path} is not a JSON object"
    assert "fixture" in payload, f"Baseline {path} missing 'fixture' key"
    assert "notes" in payload, f"Baseline {path} missing 'notes' key"


def test_k1_completion_baseline_locks_bit_for_bit_parity_thresholds():
    """The K=1 100k256 os=0 strict result is the strongest parity signal we
    have. Any merge that breaks Wiener / tau2 / LP-filter parity will trip
    this on the next K=1 run."""

    import json

    path = BASELINES / "em_parity_completion_k1_100k256_os0_strict.json"
    payload = json.loads(path.read_text())
    # FSC=1.0 first 30 shells, merged corr 0.999802 is RELION parity.
    assert payload["k1_strict_merged_corr_vs_relion"] >= 0.9995, (
        f"K=1 100k256 strict baseline merged_corr_vs_relion = "
        f"{payload['k1_strict_merged_corr_vs_relion']} but bit-for-bit parity "
        f"required ≥ 0.9995. Did a merge silently revert tau2_fudge=-1 fix?"
    )
    assert payload["k1_strict_fsc_vs_relion_min_first_30_shells"] >= 0.99, (
        f"K=1 100k256 strict FSC vs RELION in first 30 shells fell to "
        f"{payload['k1_strict_fsc_vs_relion_min_first_30_shells']}; "
        f"recovar↔RELION should stay above 0.99 in low shells."
    )


def test_k4_completion_baseline_locks_per_class_quality():
    """K=4 Class3D quality on the realistic-noise fixture. Locks
    Hungarian-matched mean per-class corr at ≥ 0.99."""

    import json

    path = BASELINES / "em_parity_completion_k4_100k256.json"
    payload = json.loads(path.read_text())
    assert payload["kclass_completion_mean_corr_vs_relion"] >= 0.99, (
        f"K=4 completion mean per-class corr = "
        f"{payload['kclass_completion_mean_corr_vs_relion']} but locked threshold "
        f"is 0.99. The merge likely broke K-class M-step / fused pass-2."
    )
    assert payload["kclass_completion_hungarian_swap_count"] == 0, (
        f"K=4 completion Hungarian swap count = "
        f"{payload['kclass_completion_hungarian_swap_count']}; expected 0 "
        f"(diagonal class match, no permutation)."
    )


def test_k1_5k128_lowpass_baseline_locks_lp_applied():
    """The --apply-initial-lowpass + tau2_fudge=-1 + Fourier-mask-edge=2
    fixes are all exercised end-to-end by the K=1 5k128 verification job.
    Locks the verified state."""

    import json

    path = BASELINES / "em_parity_completion_k1_5k128_apply_lowpass.json"
    payload = json.loads(path.read_text())
    assert payload["k1_5k128_lp_lp_applied"] is True
    assert payload["k1_5k128_lp_fmask_edge_shells"] == 2
    assert payload["apply_initial_lowpass"] is True
    assert payload["tau2_fudge_resolved"] == 1.0, (
        "--apply-initial-lowpass run on K=1 5k128 (which has no --tau2_fudge "
        "in the sbatch) should resolve tau2_fudge to 1.0 from model.star, "
        "NOT -1.0 from optimiser.star."
    )
    assert payload["k1_5k128_lp_final_half1_corr_vs_relion_it003"] >= 0.999
    assert payload["k1_5k128_lp_final_half2_corr_vs_relion_it003"] >= 0.999
