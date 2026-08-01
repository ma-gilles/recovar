#!/usr/bin/env python3
"""Report all fixed RECOVAR/RELION EM parity panels in one table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_em_k1_continuation_initializer_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_CONTINUATION_INITIALIZER_SCORECARD,
)
from scripts.summarize_em_k1_continuation_initializer_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_continuation_initializer,
)
from scripts.summarize_em_k1_live_noise_counterfactual_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_LIVE_NOISE_COUNTERFACTUAL_SCORECARD,
)
from scripts.summarize_em_k1_live_noise_counterfactual_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_live_noise_counterfactual,
)
from scripts.summarize_em_k1_mask_deterministic_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_MASK_DETERMINISTIC_SCORECARD,
)
from scripts.summarize_em_k1_mask_deterministic_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_mask_deterministic,
)
from scripts.summarize_em_k1_norm_roundtrip_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_NORM_ROUNDTRIP_SCORECARD,
)
from scripts.summarize_em_k1_norm_roundtrip_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_norm_roundtrip,
)
from scripts.summarize_em_k1_reference_roundtrip_rejection_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_REFERENCE_ROUNDTRIP_REJECTION_SCORECARD,
)
from scripts.summarize_em_k1_reference_roundtrip_rejection_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_reference_roundtrip_rejection,
)
from scripts.summarize_em_k1_sampling_perturbation_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_SAMPLING_PERTURBATION_SCORECARD,
)
from scripts.summarize_em_k1_sampling_perturbation_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_sampling_perturbation,
)
from scripts.summarize_em_k1_sampling_roundtrip_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_SAMPLING_ROUNDTRIP_SCORECARD,
)
from scripts.summarize_em_k1_sampling_roundtrip_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_sampling_roundtrip,
)
from scripts.summarize_em_k1_serialized_restart_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_RESTART_SCORECARD,
)
from scripts.summarize_em_k1_serialized_restart_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_restart,
)
from scripts.summarize_em_k1_shared_checkpoint_fp64_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K1_SHARED_CHECKPOINT_FP64_SCORECARD,
)
from scripts.summarize_em_k1_shared_checkpoint_fp64_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k1_shared_checkpoint_fp64,
)
from scripts.summarize_em_k4_causal_boundary_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K4_CAUSAL_SCORECARD,
)
from scripts.summarize_em_k4_causal_boundary_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k4_causal,
)
from scripts.summarize_em_k4_class_fsc_auc_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K4_CLASS_SCORECARD,
)
from scripts.summarize_em_k4_class_fsc_auc_scorecard import (  # noqa: E402
    failed_checks as k4_failed_checks,
)
from scripts.summarize_em_k4_class_fsc_auc_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k4_classes,
)
from scripts.summarize_em_k4_contribution_repeatability_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K4_CONTRIBUTION_REPEATABILITY_SCORECARD,
)
from scripts.summarize_em_k4_contribution_repeatability_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k4_contribution_repeatability,
)
from scripts.summarize_em_k4_deterministic_contribution_repeatability_candidate_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K4_DETERMINISTIC_CONTRIBUTION_REPEATABILITY_CANDIDATE_SCORECARD,
)
from scripts.summarize_em_k4_deterministic_contribution_repeatability_candidate_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k4_deterministic_contribution_repeatability_candidate,
)
from scripts.summarize_em_k4_preprocess_replay_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K4_PREPROCESS_SCORECARD,
)
from scripts.summarize_em_k4_preprocess_replay_scorecard import (  # noqa: E402
    load_and_validate as load_and_validate_k4_preprocess,
)
from scripts.summarize_em_relion_parity_scorecard import (  # noqa: E402
    DEFAULT_FIXTURE_MANIFEST,
    DEFAULT_K4_SNAPSHOT,
    DEFAULT_SCORECARD,
    load_and_validate,
    load_and_validate_fixture_manifest,
    load_and_validate_k4_snapshot,
    sha256_file,
)

SCHEMA = "recovar.em_parity_progress.v14"


def _panel(
    panel_id: str,
    label: str,
    passed: int,
    evaluated: int,
    denominator: int,
    *,
    scoring: bool,
) -> dict[str, object]:
    if not (0 <= passed <= evaluated <= denominator):
        raise ValueError(f"{panel_id}: invalid counts passed={passed} evaluated={evaluated} denominator={denominator}")
    return {
        "id": panel_id,
        "label": label,
        "passed": passed,
        "evaluated": evaluated,
        "denominator": denominator,
        "rate_percent": round(100.0 * passed / denominator, 1),
        "scoring": scoring,
    }


def _input_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT.resolve())),
        "sha256": sha256_file(path),
    }


def _case_identity(case: dict[str, object]) -> dict[str, str]:
    return {
        "id": str(case["id"]),
        "name": str(case["name"]),
    }


def build_progress(
    *,
    scorecard_path: Path = DEFAULT_SCORECARD,
    fixture_manifest_path: Path = DEFAULT_FIXTURE_MANIFEST,
    k4_snapshot_path: Path = DEFAULT_K4_SNAPSHOT,
    k4_class_path: Path = DEFAULT_K4_CLASS_SCORECARD,
    k4_causal_path: Path = DEFAULT_K4_CAUSAL_SCORECARD,
    k4_contribution_repeatability_path: Path = (
        DEFAULT_K4_CONTRIBUTION_REPEATABILITY_SCORECARD
    ),
    k4_deterministic_contribution_repeatability_candidate_path: Path = (
        DEFAULT_K4_DETERMINISTIC_CONTRIBUTION_REPEATABILITY_CANDIDATE_SCORECARD
    ),
    k4_preprocess_path: Path = DEFAULT_K4_PREPROCESS_SCORECARD,
    k1_restart_path: Path = DEFAULT_K1_RESTART_SCORECARD,
    k1_continuation_initializer_path: Path = (DEFAULT_K1_CONTINUATION_INITIALIZER_SCORECARD),
    k1_sampling_perturbation_path: Path = DEFAULT_K1_SAMPLING_PERTURBATION_SCORECARD,
    k1_sampling_roundtrip_path: Path = DEFAULT_K1_SAMPLING_ROUNDTRIP_SCORECARD,
    k1_norm_roundtrip_path: Path = DEFAULT_K1_NORM_ROUNDTRIP_SCORECARD,
    k1_mask_deterministic_path: Path = DEFAULT_K1_MASK_DETERMINISTIC_SCORECARD,
    k1_live_noise_counterfactual_path: Path = (
        DEFAULT_K1_LIVE_NOISE_COUNTERFACTUAL_SCORECARD
    ),
    k1_reference_roundtrip_rejection_path: Path = (
        DEFAULT_K1_REFERENCE_ROUNDTRIP_REJECTION_SCORECARD
    ),
    k1_shared_checkpoint_fp64_path: Path = DEFAULT_K1_SHARED_CHECKPOINT_FP64_SCORECARD,
) -> dict[str, object]:
    """Validate every fixed source and return the consolidated progress report."""

    scorecard = load_and_validate(scorecard_path)
    load_and_validate_fixture_manifest(fixture_manifest_path, scorecard)
    k4_snapshot = load_and_validate_k4_snapshot(k4_snapshot_path)
    k4_class_scorecard = load_and_validate_k4_classes(k4_class_path)
    k4_causal = load_and_validate_k4_causal(k4_causal_path)
    k4_contribution_repeatability = (
        load_and_validate_k4_contribution_repeatability(
            k4_contribution_repeatability_path
        )
    )
    k4_deterministic_contribution_repeatability_candidate = (
        load_and_validate_k4_deterministic_contribution_repeatability_candidate(
            k4_deterministic_contribution_repeatability_candidate_path
        )
    )
    k4_preprocess = load_and_validate_k4_preprocess(k4_preprocess_path)
    k1_restart = load_and_validate_k1_restart(k1_restart_path)
    k1_continuation_initializer = load_and_validate_k1_continuation_initializer(k1_continuation_initializer_path)
    k1_sampling_perturbation = load_and_validate_k1_sampling_perturbation(k1_sampling_perturbation_path)
    k1_sampling_roundtrip = load_and_validate_k1_sampling_roundtrip(k1_sampling_roundtrip_path)
    k1_norm_roundtrip = load_and_validate_k1_norm_roundtrip(k1_norm_roundtrip_path)
    k1_mask_deterministic = load_and_validate_k1_mask_deterministic(k1_mask_deterministic_path)
    k1_live_noise_counterfactual = load_and_validate_k1_live_noise_counterfactual(
        k1_live_noise_counterfactual_path
    )
    k1_reference_roundtrip_rejection = (
        load_and_validate_k1_reference_roundtrip_rejection(
            k1_reference_roundtrip_rejection_path
        )
    )
    k1_shared_checkpoint_fp64 = load_and_validate_k1_shared_checkpoint_fp64(
        k1_shared_checkpoint_fp64_path
    )

    k1_counts = scorecard["current_snapshot"]["counts"]
    k1_denominator = scorecard["frozen_denominator"]
    k1_evaluated = k1_counts["pass"] + k1_counts["fail"]
    k1_topology_passed = sum(case["intermediate_result"] == "pass" for case in scorecard["cases"])
    k4_direct_denominator = k4_snapshot["direct_fsc_auc_checks_total"]
    k4_iteration_denominator = k4_snapshot["numbered_iterations"]
    k4_classes = k4_snapshot["classes"]
    k4_causal_summary = k4_causal["summary"]
    k4_contribution_repeatability_summary = k4_contribution_repeatability[
        "summary"
    ]
    k4_deterministic_contribution_repeatability_candidate_summary = (
        k4_deterministic_contribution_repeatability_candidate["summary"]
    )
    k4_preprocess_summary = k4_preprocess["summary"]
    k1_restart_summary = k1_restart["summary"]
    k1_continuation_initializer_baseline = k1_continuation_initializer["baseline_summary"]
    k1_continuation_initializer_treatment = k1_continuation_initializer["treatment_summary"]
    k1_sampling_geometry = k1_sampling_perturbation["geometry_summary"]
    k1_sampling_score_map = k1_sampling_perturbation["score_map_summary"]
    k1_roundtrip_geometry = k1_sampling_roundtrip["geometry_summary"]
    k1_roundtrip_score_map = k1_sampling_roundtrip["score_map_summary"]
    k1_norm_preprocess = k1_norm_roundtrip["preprocess_summary"]
    k1_norm_geometry = k1_norm_roundtrip["geometry_summary"]
    k1_norm_score_map = k1_norm_roundtrip["score_map_summary"]
    k1_mask_preprocess = k1_mask_deterministic["preprocess_summary"]
    k1_mask_geometry = k1_mask_deterministic["geometry_summary"]
    k1_mask_score_map = k1_mask_deterministic["score_map_summary"]
    k1_live_noise_counterfactual_summary = k1_live_noise_counterfactual["summary"]
    k1_reference_roundtrip_rejection_summary = (
        k1_reference_roundtrip_rejection["summary"]
    )
    k1_shared_checkpoint_fp64_summary = k1_shared_checkpoint_fp64["summary"]
    if (
        k4_class_scorecard["summary"]["pass"] != k4_snapshot["direct_fsc_auc_checks_passed"]
        or k4_class_scorecard["summary"]["evaluated"] != k4_snapshot["direct_fsc_auc_checks_total"]
        or k4_class_scorecard["summary"]["iterations_all_classes_passed"]
        != k4_snapshot["iterations_all_classes_passed"]
    ):
        raise ValueError("K=4 class scorecard does not replay the fixed trajectory snapshot")
    k1_strict_failures = [
        _case_identity(case) | {"intermediate_result": case["intermediate_result"]}
        for case in scorecard["cases"]
        if case["result"] != "pass"
    ]
    k1_topology_failures = [
        _case_identity(case) for case in scorecard["cases"] if case["intermediate_result"] != "pass"
    ]
    k4_direct_failures = [
        {
            "iteration": iteration,
            "passed": passed,
            "failed": k4_classes - passed,
        }
        for iteration, passed in enumerate(
            k4_snapshot["direct_fsc_auc_passes_by_iteration"],
            start=1,
        )
        if passed != k4_classes
    ]
    k4_causal_failures = [_case_identity(case) for case in k4_causal["cases"] if case["result"] != "pass"]
    k1_restart_failures = [_case_identity(case) for case in k1_restart["cases"] if case["result"] != "pass"]
    k1_continuation_initializer_failures = [
        _case_identity(case) for case in k1_continuation_initializer["cases"] if case["treatment_result"] != "pass"
    ]

    panels = [
        _panel(
            "k1_strict",
            "K=1 strict FSC/FSC-AUC",
            k1_counts["pass"],
            k1_evaluated,
            k1_denominator,
            scoring=True,
        ),
        _panel(
            "k1_topology",
            "K=1 topology",
            k1_topology_passed,
            k1_evaluated,
            k1_denominator,
            scoring=False,
        ),
        _panel(
            "k1_evaluated",
            "K=1 evaluated",
            k1_evaluated,
            k1_evaluated,
            k1_denominator,
            scoring=False,
        ),
        _panel(
            "k1_restart_causal",
            "K=1 serialized-restart causal gates",
            k1_restart_summary["pass"],
            k1_restart_summary["evaluated"],
            k1_restart["frozen_denominator"],
            scoring=False,
        ),
        _panel(
            "k1_continuation_initializer",
            "K=1 continuation initializer patched arm",
            k1_continuation_initializer_treatment["pass"],
            k1_continuation_initializer_treatment["evaluated"],
            k1_continuation_initializer["frozen_denominator"],
            scoring=False,
        ),
        _panel(
            "k1_sampling_perturbation_geometry",
            "K=1 sampling-perturbation geometry identity",
            k1_sampling_geometry["treatment_pass"],
            k1_sampling_geometry["evaluated"],
            k1_sampling_geometry["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_sampling_perturbation_score_map",
            "K=1 sampling-perturbation score/map gates",
            k1_sampling_score_map["treatment_pass"],
            k1_sampling_score_map["evaluated"],
            k1_sampling_score_map["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_sampling_roundtrip_geometry",
            "K=1 sampling-roundtrip geometry identity",
            k1_roundtrip_geometry["treatment_pass"],
            k1_roundtrip_geometry["evaluated"],
            k1_roundtrip_geometry["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_sampling_roundtrip_score_map",
            "K=1 sampling-roundtrip score/map gates",
            k1_roundtrip_score_map["treatment_pass"],
            k1_roundtrip_score_map["evaluated"],
            k1_roundtrip_score_map["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_norm_roundtrip_preprocess",
            "K=1 normalization-roundtrip preprocessing exactness",
            k1_norm_preprocess["treatment_pass"],
            k1_norm_preprocess["evaluated"],
            k1_norm_preprocess["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_norm_roundtrip_geometry",
            "K=1 normalization-roundtrip geometry identity",
            k1_norm_geometry["treatment_pass"],
            k1_norm_geometry["evaluated"],
            k1_norm_geometry["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_norm_roundtrip_score_map",
            "K=1 normalization-roundtrip score/map gates",
            k1_norm_score_map["treatment_pass"],
            k1_norm_score_map["evaluated"],
            k1_norm_score_map["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_mask_deterministic_preprocess",
            "K=1 deterministic-mask preprocessing exactness",
            k1_mask_preprocess["treatment_pass"],
            k1_mask_preprocess["evaluated"],
            k1_mask_preprocess["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_mask_deterministic_geometry",
            "K=1 deterministic-mask geometry identity",
            k1_mask_geometry["treatment_pass"],
            k1_mask_geometry["evaluated"],
            k1_mask_geometry["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_mask_deterministic_score_map",
            "K=1 deterministic-mask score/map gates",
            k1_mask_score_map["treatment_pass"],
            k1_mask_score_map["evaluated"],
            k1_mask_score_map["denominator"],
            scoring=False,
        ),
        _panel(
            "k1_live_noise_counterfactual",
            "K=1 live-noise sealed counterfactual gates",
            k1_live_noise_counterfactual_summary["pass"],
            k1_live_noise_counterfactual_summary["evaluated"],
            k1_live_noise_counterfactual["frozen_denominator"],
            scoring=False,
        ),
        _panel(
            "k1_reference_roundtrip_rejection",
            "K=1 reference-roundtrip rejected gates",
            k1_reference_roundtrip_rejection_summary["pass"],
            k1_reference_roundtrip_rejection_summary["evaluated"],
            k1_reference_roundtrip_rejection["frozen_denominator"],
            scoring=False,
        ),
        _panel(
            "k1_shared_checkpoint_fp64",
            "K=1 shared-checkpoint FP64 reference gates",
            k1_shared_checkpoint_fp64_summary["pass"],
            k1_shared_checkpoint_fp64_summary["evaluated"],
            k1_shared_checkpoint_fp64["frozen_denominator"],
            scoring=False,
        ),
        _panel(
            "k4_direct",
            "K=4 direct per-class FSC-AUC",
            k4_snapshot["direct_fsc_auc_checks_passed"],
            k4_direct_denominator,
            k4_direct_denominator,
            scoring=True,
        ),
        _panel(
            "k4_all_class",
            "K=4 all-class iterations",
            k4_snapshot["iterations_all_classes_passed"],
            k4_iteration_denominator,
            k4_iteration_denominator,
            scoring=True,
        ),
        _panel(
            "k4_causal",
            "K=4 exact-device causal boundary",
            k4_causal_summary["pass"],
            k4_causal_summary["evaluated"],
            k4_causal["frozen_denominator"],
            scoring=False,
        ),
        _panel(
            "k4_contribution_repeatability",
            "K=4 contribution archive repeatability",
            k4_contribution_repeatability_summary["pass"],
            k4_contribution_repeatability_summary["evaluated"],
            k4_contribution_repeatability["frozen_denominator"],
            scoring=False,
        ),
        _panel(
            "k4_deterministic_contribution_repeatability_candidate",
            "K=4 deterministic contribution candidate repeatability",
            k4_deterministic_contribution_repeatability_candidate_summary["pass"],
            k4_deterministic_contribution_repeatability_candidate_summary[
                "evaluated"
            ],
            k4_deterministic_contribution_repeatability_candidate[
                "frozen_denominator"
            ],
            scoring=False,
        ),
        _panel(
            "k4_preprocess_bitwise",
            "K=4 preprocess bitwise replay",
            k4_preprocess_summary["bitwise_equal"],
            k4_preprocess_summary["evaluated"],
            k4_preprocess["frozen_denominator"],
            scoring=False,
        ),
        _panel(
            "k4_preprocess_material",
            "K=4 preprocess within fixed material floor",
            k4_preprocess_summary["within_material_floor"],
            k4_preprocess_summary["evaluated"],
            k4_preprocess["frozen_denominator"],
            scoring=False,
        ),
    ]
    return {
        "schema": SCHEMA,
        "metric_policy": (
            "K=1 and K=4 quality panels use shellwise FSC/FSC-AUC; "
            "correlation is not used. The K=1 serialized-restart, K=1 "
            "continuation-initializer, K=1 sampling-perturbation, K=1 "
            "sampling-roundtrip, K=1 normalization-roundtrip, K=4 causal, "
            "K=1 deterministic-mask, K=1 live-noise counterfactual, K=1 "
            "reference-roundtrip rejection, K=1 "
            "shared-checkpoint FP64 reference, K=4 contribution-repeatability, "
            "K=4 deterministic contribution candidate, "
            "and K=4 preprocessing panels are non-scoring."
        ),
        "scorecard_change_admissible": False,
        "panels": panels,
        "k1_strict_history": [snapshot["counts"]["pass"] for snapshot in scorecard["history"]],
        "k1_continuation_initializer_progress": {
            "stock_pass": k1_continuation_initializer_baseline["pass"],
            "patched_pass": k1_continuation_initializer_treatment["pass"],
            "denominator": k1_continuation_initializer["frozen_denominator"],
            "paired_gain": k1_continuation_initializer["paired_gain"],
        },
        "k1_sampling_perturbation_progress": {
            "geometry_stock_pass": k1_sampling_geometry["baseline_pass"],
            "geometry_treatment_pass": k1_sampling_geometry["treatment_pass"],
            "geometry_denominator": k1_sampling_geometry["denominator"],
            "geometry_gain": k1_sampling_geometry["paired_gain"],
            "score_map_stock_pass": k1_sampling_score_map["baseline_pass"],
            "score_map_treatment_pass": k1_sampling_score_map["treatment_pass"],
            "score_map_denominator": k1_sampling_score_map["denominator"],
            "score_map_gain": k1_sampling_score_map["paired_gain"],
        },
        "k1_sampling_roundtrip_progress": {
            "geometry_stock_pass": k1_roundtrip_geometry["baseline_pass"],
            "geometry_treatment_pass": k1_roundtrip_geometry["treatment_pass"],
            "geometry_denominator": k1_roundtrip_geometry["denominator"],
            "geometry_gain": k1_roundtrip_geometry["paired_gain"],
            "score_map_stock_pass": k1_roundtrip_score_map["baseline_pass"],
            "score_map_treatment_pass": k1_roundtrip_score_map["treatment_pass"],
            "score_map_denominator": k1_roundtrip_score_map["denominator"],
            "score_map_gain": k1_roundtrip_score_map["paired_gain"],
        },
        "k1_norm_roundtrip_progress": {
            "preprocess_baseline_pass": k1_norm_preprocess["baseline_pass"],
            "preprocess_treatment_pass": k1_norm_preprocess["treatment_pass"],
            "preprocess_denominator": k1_norm_preprocess["denominator"],
            "preprocess_gain": k1_norm_preprocess["paired_gain"],
            "geometry_baseline_pass": k1_norm_geometry["baseline_pass"],
            "geometry_treatment_pass": k1_norm_geometry["treatment_pass"],
            "geometry_denominator": k1_norm_geometry["denominator"],
            "geometry_gain": k1_norm_geometry["paired_gain"],
            "score_map_baseline_pass": k1_norm_score_map["baseline_pass"],
            "score_map_treatment_pass": k1_norm_score_map["treatment_pass"],
            "score_map_denominator": k1_norm_score_map["denominator"],
            "score_map_gain": k1_norm_score_map["paired_gain"],
        },
        "k1_mask_deterministic_progress": {
            "preprocess_baseline_pass": k1_mask_preprocess["baseline_pass"],
            "preprocess_treatment_pass": k1_mask_preprocess["treatment_pass"],
            "preprocess_denominator": k1_mask_preprocess["denominator"],
            "preprocess_gain": k1_mask_preprocess["paired_gain"],
            "geometry_baseline_pass": k1_mask_geometry["baseline_pass"],
            "geometry_treatment_pass": k1_mask_geometry["treatment_pass"],
            "geometry_denominator": k1_mask_geometry["denominator"],
            "geometry_gain": k1_mask_geometry["paired_gain"],
            "score_map_baseline_pass": k1_mask_score_map["baseline_pass"],
            "score_map_treatment_pass": k1_mask_score_map["treatment_pass"],
            "score_map_denominator": k1_mask_score_map["denominator"],
            "score_map_gain": k1_mask_score_map["paired_gain"],
        },
        "remaining": {
            "k1_strict_failures": k1_strict_failures,
            "k1_topology_failures": k1_topology_failures,
            "k1_restart_causal_failures": k1_restart_failures,
            "k1_continuation_initializer_failures": (k1_continuation_initializer_failures),
            "k4_direct_failures_by_iteration": k4_direct_failures,
            "k4_direct_failures": k4_failed_checks(k4_class_scorecard),
            "k4_all_class_failure_iterations": [record["iteration"] for record in k4_direct_failures],
            "k4_causal_failures": k4_causal_failures,
        },
        "inputs": {
            "k1_scorecard": _input_record(scorecard_path),
            "k1_fixture_manifest": _input_record(fixture_manifest_path),
            "k1_restart_scorecard": _input_record(k1_restart_path),
            "k1_continuation_initializer_scorecard": _input_record(k1_continuation_initializer_path),
            "k1_sampling_perturbation_scorecard": _input_record(k1_sampling_perturbation_path),
            "k1_sampling_roundtrip_scorecard": _input_record(k1_sampling_roundtrip_path),
            "k1_norm_roundtrip_scorecard": _input_record(k1_norm_roundtrip_path),
            "k1_mask_deterministic_scorecard": _input_record(k1_mask_deterministic_path),
            "k1_live_noise_counterfactual_scorecard": _input_record(
                k1_live_noise_counterfactual_path
            ),
            "k1_reference_roundtrip_rejection_scorecard": _input_record(
                k1_reference_roundtrip_rejection_path
            ),
            "k1_shared_checkpoint_fp64_scorecard": _input_record(
                k1_shared_checkpoint_fp64_path
            ),
            "k4_trajectory_snapshot": _input_record(k4_snapshot_path),
            "k4_class_scorecard": _input_record(k4_class_path),
            "k4_causal_scorecard": _input_record(k4_causal_path),
            "k4_contribution_repeatability_scorecard": _input_record(
                k4_contribution_repeatability_path
            ),
            "k4_deterministic_contribution_repeatability_candidate_scorecard": _input_record(
                k4_deterministic_contribution_repeatability_candidate_path
            ),
            "k4_preprocess_scorecard": _input_record(k4_preprocess_path),
        },
    }


def render_markdown(progress: dict[str, object]) -> str:
    """Render the consolidated fixed panels as a compact PR-ready table."""

    lines = [
        "| Fixed panel | Passed | Evaluated | Denominator | Rate | Scoring |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for panel in progress["panels"]:
        scoring = "yes" if panel["scoring"] else "no"
        lines.append(
            f"| {panel['label']} | **{panel['passed']}** | "
            f"{panel['evaluated']} | {panel['denominator']} | "
            f"{panel['rate_percent']:.1f}% | {scoring} |"
        )
    history = " → ".join(str(value) for value in progress["k1_strict_history"])
    initializer = progress["k1_continuation_initializer_progress"]
    sampling = progress["k1_sampling_perturbation_progress"]
    roundtrip = progress["k1_sampling_roundtrip_progress"]
    norm_roundtrip = progress["k1_norm_roundtrip_progress"]
    mask_deterministic = progress["k1_mask_deterministic_progress"]
    remaining = progress["remaining"]
    k1_strict = ", ".join(case["id"] for case in remaining["k1_strict_failures"]) or "none"
    k1_topology = ", ".join(case["id"] for case in remaining["k1_topology_failures"]) or "none"
    k1_restart = ", ".join(case["id"] for case in remaining["k1_restart_causal_failures"]) or "none"
    k1_continuation_initializer = (
        ", ".join(case["id"] for case in remaining["k1_continuation_initializer_failures"]) or "none"
    )
    k4_direct = (
        ", ".join(
            f"{record['iteration']} ({record['failed']}/{record['passed'] + record['failed']} failed)"
            for record in remaining["k4_direct_failures_by_iteration"]
        )
        or "none"
    )
    k4_direct_classes = ", ".join(record["id"] for record in remaining["k4_direct_failures"]) or "none"
    k4_causal = ", ".join(case["id"] for case in remaining["k4_causal_failures"]) or "none"
    lines.extend(
        [
            "",
            f"K=1 strict progress on the unchanged denominator: **{history}**.",
            "",
            (
                "K=1 continuation-initializer paired progress on the unchanged "
                f"denominator: **{initializer['stock_pass']}/"
                f"{initializer['denominator']} stock → "
                f"{initializer['patched_pass']}/"
                f"{initializer['denominator']} patched "
                f"(+{initializer['paired_gain']})**."
            ),
            "",
            (
                "K=1 sampling-perturbation geometry on the unchanged denominator: "
                f"**{sampling['geometry_stock_pass']}/"
                f"{sampling['geometry_denominator']} stock → "
                f"{sampling['geometry_treatment_pass']}/"
                f"{sampling['geometry_denominator']} treatment "
                f"({sampling['geometry_gain']:+d})**."
            ),
            (
                "K=1 sampling-perturbation score/map gates on the unchanged denominator: "
                f"**{sampling['score_map_stock_pass']}/"
                f"{sampling['score_map_denominator']} stock → "
                f"{sampling['score_map_treatment_pass']}/"
                f"{sampling['score_map_denominator']} treatment "
                f"({sampling['score_map_gain']:+d})**."
            ),
            "",
            (
                "K=1 sampling-roundtrip geometry on the unchanged denominator: "
                f"**{roundtrip['geometry_stock_pass']}/"
                f"{roundtrip['geometry_denominator']} stock → "
                f"{roundtrip['geometry_treatment_pass']}/"
                f"{roundtrip['geometry_denominator']} treatment "
                f"({roundtrip['geometry_gain']:+d})**."
            ),
            (
                "K=1 sampling-roundtrip score/map gates on the unchanged denominator: "
                f"**{roundtrip['score_map_stock_pass']}/"
                f"{roundtrip['score_map_denominator']} stock → "
                f"{roundtrip['score_map_treatment_pass']}/"
                f"{roundtrip['score_map_denominator']} treatment "
                f"({roundtrip['score_map_gain']:+d})**."
            ),
            "",
            (
                "K=1 normalization-roundtrip preprocessing on the unchanged denominator: "
                f"**{norm_roundtrip['preprocess_baseline_pass']}/"
                f"{norm_roundtrip['preprocess_denominator']} sampling-only → "
                f"{norm_roundtrip['preprocess_treatment_pass']}/"
                f"{norm_roundtrip['preprocess_denominator']} treatment "
                f"({norm_roundtrip['preprocess_gain']:+d})**."
            ),
            (
                "K=1 normalization-roundtrip geometry on the unchanged denominator: "
                f"**{norm_roundtrip['geometry_baseline_pass']}/"
                f"{norm_roundtrip['geometry_denominator']} sampling-only → "
                f"{norm_roundtrip['geometry_treatment_pass']}/"
                f"{norm_roundtrip['geometry_denominator']} treatment "
                f"({norm_roundtrip['geometry_gain']:+d})**."
            ),
            (
                "K=1 normalization-roundtrip score/map gates on the unchanged denominator: "
                f"**{norm_roundtrip['score_map_baseline_pass']}/"
                f"{norm_roundtrip['score_map_denominator']} sampling-only → "
                f"{norm_roundtrip['score_map_treatment_pass']}/"
                f"{norm_roundtrip['score_map_denominator']} treatment "
                f"({norm_roundtrip['score_map_gain']:+d})**."
            ),
            "",
            (
                "K=1 deterministic-mask preprocessing on the unchanged denominator: "
                f"**{mask_deterministic['preprocess_baseline_pass']}/"
                f"{mask_deterministic['preprocess_denominator']} normalization-only → "
                f"{mask_deterministic['preprocess_treatment_pass']}/"
                f"{mask_deterministic['preprocess_denominator']} treatment "
                f"({mask_deterministic['preprocess_gain']:+d})**."
            ),
            (
                "K=1 deterministic-mask geometry on the unchanged denominator: "
                f"**{mask_deterministic['geometry_baseline_pass']}/"
                f"{mask_deterministic['geometry_denominator']} normalization-only → "
                f"{mask_deterministic['geometry_treatment_pass']}/"
                f"{mask_deterministic['geometry_denominator']} treatment "
                f"({mask_deterministic['geometry_gain']:+d})**."
            ),
            (
                "K=1 deterministic-mask score/map gates on the unchanged denominator: "
                f"**{mask_deterministic['score_map_baseline_pass']}/"
                f"{mask_deterministic['score_map_denominator']} normalization-only → "
                f"{mask_deterministic['score_map_treatment_pass']}/"
                f"{mask_deterministic['score_map_denominator']} treatment "
                f"({mask_deterministic['score_map_gain']:+d})**."
            ),
            "",
            f"Remaining K=1 strict cases: {k1_strict}.",
            f"Remaining K=1 topology cases: {k1_topology}.",
            f"Remaining K=1 serialized-restart causal cases: {k1_restart}.",
            (f"Remaining K=1 continuation-initializer patched cases: {k1_continuation_initializer}."),
            f"Remaining K=4 direct iterations: {k4_direct}.",
            f"Remaining K=4 direct classes: {k4_direct_classes}.",
            f"Remaining K=4 causal cases: {k4_causal}.",
            "",
            str(progress["metric_policy"]),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    progress = build_progress()
    if args.format == "json":
        rendered = json.dumps(progress, indent=2, sort_keys=True) + "\n"
    else:
        rendered = render_markdown(progress)
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
