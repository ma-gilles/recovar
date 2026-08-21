#!/usr/bin/env python3
"""Classify the Case-22 live-versus-serialized noise-state boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovar.em.dense_single_volume.helpers.half_spectrum import (
    make_shell_indices_half,
)
from scripts.analyze_em_k1_corr_img_conditioning import (
    SHELL_PARTITION_CLASSIFICATION,
    _load_ctf_half,
    _load_recovar,
    _sha256,
    _sigma2_noise_tokens,
    load_operand,
    load_preprocess,
    relion_values_on_recovar_window,
)

SCHEMA = "em-k1-noise-serialization-boundary-v1"
PARENT_SCHEMA = "em-k1-corr-img-conditioning-v2"
EXPECTED_PARENT_SHA256 = (
    "9d6b8cf39c9abe21c71d5c3d0dc0ef73b381566b439328748d92c32efa473073"
)
EXPECTED_PARTICLES = 14
FULL_IMAGE_SIZE = 128
EFFECTIVE_CTF_THRESHOLD = 1.0e-2
FIXED_DECIMAL_SHELLS = (1, 2, 3, 4)
SCIENTIFIC_CONTROL_SHELL = 5
SERIALIZED_ABS_ERROR_MAX = 5.0e-9
LIVE_FIXED_SHELL_ABS_ERROR_MIN = 2.0e-8
SERIALIZED_CLOSENESS_RATIO_MIN = 100.0
SCIENTIFIC_CONTROL_ABS_ERROR_MAX = 5.0e-9
WITHIN_SHELL_PTP_MAX = 5.0e-9
CLASSIFICATION = (
    "recovar_score_weight_matches_serialized_star_noise_while_"
    "live_relion_retains_pre_serialization_shells_1_to_4"
)
NEXT_BOUNDARY = (
    "restart_relion_from_the_same_serialized_model_star_and_compare_"
    "physical_iteration_2"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def classify_serialization_boundary(
    *,
    parent_qualified: bool,
    shell_metrics: dict[int, dict[str, float | int]],
) -> str:
    """Classify fixed shell gates without fitting a tolerance or scale."""

    if not parent_qualified:
        return "noise_serialization_parent_not_qualified"
    expected_shells = set((*FIXED_DECIMAL_SHELLS, SCIENTIFIC_CONTROL_SHELL))
    if set(shell_metrics) != expected_shells:
        return "noise_serialization_shell_denominator_changed"
    fixed_pass = all(
        int(shell_metrics[shell]["evaluated_particles"])
        == EXPECTED_PARTICLES
        and float(shell_metrics[shell]["recovar_vs_token_max_abs"])
        <= SERIALIZED_ABS_ERROR_MAX
        and float(shell_metrics[shell]["live_relion_vs_token_min_abs"])
        >= LIVE_FIXED_SHELL_ABS_ERROR_MIN
        and float(shell_metrics[shell]["serialized_closeness_ratio_min"])
        >= SERIALIZED_CLOSENESS_RATIO_MIN
        and float(shell_metrics[shell]["recovar_within_shell_ptp_max"])
        <= WITHIN_SHELL_PTP_MAX
        and float(shell_metrics[shell]["live_relion_within_shell_ptp_max"])
        <= WITHIN_SHELL_PTP_MAX
        for shell in FIXED_DECIMAL_SHELLS
    )
    control = shell_metrics[SCIENTIFIC_CONTROL_SHELL]
    control_pass = (
        int(control["evaluated_particles"]) == EXPECTED_PARTICLES
        and float(control["recovar_vs_token_max_abs"])
        <= SCIENTIFIC_CONTROL_ABS_ERROR_MAX
        and float(control["live_relion_vs_token_max_abs"])
        <= SCIENTIFIC_CONTROL_ABS_ERROR_MAX
        and float(control["recovar_within_shell_ptp_max"])
        <= WITHIN_SHELL_PTP_MAX
        and float(control["live_relion_within_shell_ptp_max"])
        <= WITHIN_SHELL_PTP_MAX
    )
    if fixed_pass and control_pass:
        return CLASSIFICATION
    return "live_and_serialized_noise_state_boundary_did_not_close"


def _summarize_shell(
    *,
    shell: int,
    token: float,
    particle_rows: list[dict[str, float | int]],
) -> dict[str, Any]:
    _require(
        len(particle_rows) == EXPECTED_PARTICLES,
        f"shell {shell} particle denominator changed",
    )
    recovar = np.asarray(
        [row["recovar_sigma2_median"] for row in particle_rows],
        dtype=np.float64,
    )
    live = np.asarray(
        [row["live_relion_sigma2_median"] for row in particle_rows],
        dtype=np.float64,
    )
    recovar_error = np.abs(recovar - token)
    live_error = np.abs(live - token)
    denominator = np.maximum(recovar_error, np.finfo(np.float64).tiny)
    ratios = live_error / denominator
    return {
        "shell": shell,
        "serialized_token": token,
        "evaluated_particles": len(particle_rows),
        "recovar_sigma2": {
            "min": float(np.min(recovar)),
            "median": float(np.median(recovar)),
            "max": float(np.max(recovar)),
        },
        "live_relion_sigma2": {
            "min": float(np.min(live)),
            "median": float(np.median(live)),
            "max": float(np.max(live)),
        },
        "recovar_vs_token_max_abs": float(np.max(recovar_error)),
        "live_relion_vs_token_min_abs": float(np.min(live_error)),
        "live_relion_vs_token_max_abs": float(np.max(live_error)),
        "serialized_closeness_ratio_min": float(np.min(ratios)),
        "recovar_within_shell_ptp_max": float(
            max(row["recovar_within_shell_ptp"] for row in particle_rows)
        ),
        "live_relion_within_shell_ptp_max": float(
            max(
                row["live_relion_within_shell_ptp"]
                for row in particle_rows
            )
        ),
    }


def _validate_parent(parent_path: Path) -> dict[str, Any]:
    _require(parent_path.is_file(), "noise shell-partition parent is missing")
    _require(
        _sha256(parent_path) == EXPECTED_PARENT_SHA256,
        "noise shell-partition parent hash changed",
    )
    parent = json.loads(parent_path.read_text())
    _require(parent.get("schema") == PARENT_SCHEMA, "parent schema changed")
    _require(
        parent.get("status") == "complete"
        and parent.get("classification_ready") is True,
        "parent report is incomplete",
    )
    shell_parent = parent.get("shell_partition_metric", {})
    _require(
        shell_parent.get("classification")
        == SHELL_PARTITION_CLASSIFICATION,
        "parent inverse-noise shell classification changed",
    )
    _require(
        shell_parent.get("fixed_decimal_shells")
        == list(FIXED_DECIMAL_SHELLS)
        and shell_parent.get("evaluated_particles") == EXPECTED_PARTICLES
        and shell_parent.get("expected_particles") == EXPECTED_PARTICLES,
        "parent shell denominator changed",
    )
    _require(
        int(parent.get("full_image_size", -1)) == FULL_IMAGE_SIZE,
        "parent full image size changed",
    )
    return parent


def build_report(*, parent_path: Path) -> dict[str, Any]:
    parent_path = parent_path.resolve()
    parent = _validate_parent(parent_path)
    for binding_name in ("relion_model_star", "ctf_pickle"):
        binding = parent.get(binding_name, {})
        path = Path(binding.get("path", "")).resolve()
        _require(path.is_file(), f"{binding_name} is missing")
        _require(
            _sha256(path) == binding.get("sha256"),
            f"{binding_name} hash changed",
        )
    model_star = Path(parent["relion_model_star"]["path"]).resolve()
    raw_tokens = _sigma2_noise_tokens(model_star)
    expected_shells = (*FIXED_DECIMAL_SHELLS, SCIENTIFIC_CONTROL_SHELL)
    _require(
        all(shell in raw_tokens for shell in expected_shells),
        "serialized noise tokens are incomplete",
    )
    ctf_half, voxel_size = _load_ctf_half(
        Path(parent["ctf_pickle"]["path"]),
        full_image_size=FULL_IMAGE_SIZE,
    )

    particle_metrics: list[dict[str, Any]] = []
    rows_by_shell: dict[int, list[dict[str, float | int]]] = {
        shell: [] for shell in expected_shells
    }
    for parent_row in parent["particles"]:
        paths = {
            key: Path(value).resolve()
            for key, value in parent_row["artifact_paths"].items()
        }
        for key in ("preprocess", "operand", "recovar"):
            _require(paths[key].is_file(), f"particle {key} input is missing")
            _require(
                _sha256(paths[key]) == parent_row["artifact_sha256"][key],
                f"particle {key} input hash changed",
            )
        preprocess = load_preprocess(paths["preprocess"])
        operand = load_operand(paths["operand"])
        recovar = _load_recovar(paths["recovar"])
        original_index = int(parent_row["original_index_zero_based"])
        stack_index = int(parent_row["stack_index_one_based"])
        _require(
            preprocess.stack_index == operand.stack_index == stack_index,
            "RELION stack identity changed",
        )
        _require(
            int(recovar["original_index"]) == original_index
            and stack_index == original_index + 1,
            "RECOVAR/RELION particle identity changed",
        )

        window_indices = recovar["window_indices"]
        current_size = int(recovar["current_size"])
        postoptics = relion_values_on_recovar_window(
            preprocess.masked_fourier_post_optics[0].reshape(1, -1),
            window_indices,
            full_image_size=FULL_IMAGE_SIZE,
            current_size=current_size,
        )[0]
        relion_pixel = relion_values_on_recovar_window(
            (
                operand.image_real.astype(np.float64)
                + np.complex128(1j)
                * operand.image_imag.astype(np.float64)
            ).reshape(1, -1),
            window_indices,
            full_image_size=FULL_IMAGE_SIZE,
            current_size=current_size,
        )[0]
        relion_corr = relion_values_on_recovar_window(
            operand.correction.reshape(1, -1),
            window_indices,
            full_image_size=FULL_IMAGE_SIZE,
            current_size=current_size,
        )[0].real
        relion_effective_ctf_complex = postoptics / relion_pixel
        _require(
            float(np.max(np.abs(relion_effective_ctf_complex.imag)))
            <= 2.0e-5,
            "RELION effective CTF has a material imaginary component",
        )
        relion_effective_ctf = relion_effective_ctf_complex.real
        recovar_ctf = ctf_half[original_index].reshape(-1)[window_indices]
        recovar_effective_ctf = (
            -float(parent_row["scale_correction"]) * recovar_ctf
        )
        recovar_corr = (
            float(FULL_IMAGE_SIZE**4)
            * recovar["half_weights"]
            * recovar["ctf2_data"]
        )
        shells = np.asarray(
            make_shell_indices_half((FULL_IMAGE_SIZE, FULL_IMAGE_SIZE)),
            dtype=np.int64,
        )[window_indices]
        valid = (
            np.abs(relion_effective_ctf) > EFFECTIVE_CTF_THRESHOLD
        ) & (np.abs(recovar_effective_ctf) > EFFECTIVE_CTF_THRESHOLD)
        _require(np.any(valid), "effective CTF gate excludes every pixel")
        _require(
            np.all(np.isfinite(relion_corr))
            and np.all(np.isfinite(recovar_corr)),
            "captured score operands are non-finite",
        )

        shell_rows = {}
        for shell in expected_shells:
            mask = valid & (shells == shell)
            _require(np.any(mask), f"particle shell {shell} is empty")
            live_sigma2 = (
                relion_effective_ctf[mask] ** 2 / relion_corr[mask]
            )
            recovar_sigma2 = (
                recovar_effective_ctf[mask] ** 2 / recovar_corr[mask]
            )
            _require(
                np.all(np.isfinite(live_sigma2))
                and np.all(np.isfinite(recovar_sigma2))
                and np.all(live_sigma2 > 0.0)
                and np.all(recovar_sigma2 > 0.0),
                "recovered shell noise is non-finite or nonpositive",
            )
            shell_row = {
                "shell": shell,
                "valid_pixel_count": int(np.count_nonzero(mask)),
                "serialized_token": float(raw_tokens[shell]),
                "live_relion_sigma2_median": float(
                    np.median(live_sigma2)
                ),
                "recovar_sigma2_median": float(
                    np.median(recovar_sigma2)
                ),
                "live_relion_within_shell_ptp": float(
                    np.ptp(live_sigma2)
                ),
                "recovar_within_shell_ptp": float(
                    np.ptp(recovar_sigma2)
                ),
            }
            shell_rows[str(shell)] = shell_row
            rows_by_shell[shell].append(shell_row)
        particle_metrics.append(
            {
                "original_index_zero_based": original_index,
                "stack_index_one_based": stack_index,
                "shells": shell_rows,
                "artifact_paths": {
                    key: str(paths[key])
                    for key in ("preprocess", "operand", "recovar")
                },
                "artifact_sha256": {
                    key: parent_row["artifact_sha256"][key]
                    for key in ("preprocess", "operand", "recovar")
                },
            }
        )

    _require(
        len(particle_metrics) == EXPECTED_PARTICLES,
        "particle denominator changed",
    )
    shell_metrics = {
        shell: _summarize_shell(
            shell=shell,
            token=float(raw_tokens[shell]),
            particle_rows=rows_by_shell[shell],
        )
        for shell in expected_shells
    }
    classification = classify_serialization_boundary(
        parent_qualified=True,
        shell_metrics=shell_metrics,
    )
    _require(
        classification == CLASSIFICATION,
        "noise serialization boundary gates did not pass",
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "next_causal_boundary": NEXT_BOUNDARY,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed 14-particle, shells-1-through-5 effective-noise "
            "recovery above absolute effective-CTF 0.01; exact bound STAR "
            "tokens; fixed absolute and relative-closeness gates; shell 5 "
            "scientific-serialization control; no fitted scale/sign, FSC "
            "claim, or correlation"
        ),
        "fixed_gates": {
            "expected_particles": EXPECTED_PARTICLES,
            "fixed_decimal_shells": list(FIXED_DECIMAL_SHELLS),
            "scientific_control_shell": SCIENTIFIC_CONTROL_SHELL,
            "effective_ctf_threshold": EFFECTIVE_CTF_THRESHOLD,
            "serialized_abs_error_max": SERIALIZED_ABS_ERROR_MAX,
            "live_fixed_shell_abs_error_min": (
                LIVE_FIXED_SHELL_ABS_ERROR_MIN
            ),
            "serialized_closeness_ratio_min": (
                SERIALIZED_CLOSENESS_RATIO_MIN
            ),
            "scientific_control_abs_error_max": (
                SCIENTIFIC_CONTROL_ABS_ERROR_MAX
            ),
            "within_shell_ptp_max": WITHIN_SHELL_PTP_MAX,
        },
        "fixed_metric": {
            "evaluated_particles": len(particle_metrics),
            "expected_particles": EXPECTED_PARTICLES,
            "shells": {str(key): value for key, value in shell_metrics.items()},
        },
        "parent": {
            "path": str(parent_path),
            "sha256": EXPECTED_PARENT_SHA256,
            "shell_partition_classification": (
                parent["shell_partition_metric"]["classification"]
            ),
        },
        "relion_model_star": parent["relion_model_star"],
        "ctf_pickle": parent["ctf_pickle"],
        "voxel_size": voxel_size,
        "particles": particle_metrics,
        "notes": [
            (
                "The parent report already establishes that shells 1-4 "
                "carry the score-relevant inverse-noise residual."
            ),
            (
                "This report classifies the live-versus-serialized state "
                "boundary only; the queued exact-device RELION restart is "
                "the causal end-to-end test."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(parent_path=args.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "next_causal_boundary": report["next_causal_boundary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
