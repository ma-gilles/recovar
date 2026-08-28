#!/usr/bin/env python3
"""Localize one final K=1 particle from score through BPref reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .analyze_k1_bpref_factor_boundary import (
        RELATIVE_L2_BOUND,
        _compare_particle,
        _first_cross_engine_boundary,
    )
    from .validate_relion_bpref_factor_capture import load_factor_capture
else:
    from analyze_k1_bpref_factor_boundary import (  # type: ignore[no-redef]
        RELATIVE_L2_BOUND,
        _compare_particle,
        _first_cross_engine_boundary,
    )
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )


REPORT_SCHEMA = "recovar.em.k1_final_bpref_particle.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(values: dict[str, np.ndarray], name: str) -> int:
    array = np.asarray(values[name])
    _require(array.size == 1, f"{name} is not scalar")
    return int(array.reshape(-1)[0])


def load_recovar_pair(local_score_path: Path, contribution_path: Path) -> dict[str, Any]:
    """Join independently written score and M-step operands by immutable identity."""

    with np.load(local_score_path, allow_pickle=False) as archive:
        score = {name: archive[name] for name in archive.files}
    with np.load(contribution_path, allow_pickle=False) as archive:
        contribution = {name: archive[name] for name in archive.files}

    original_index = _scalar(score, "selected_global_image_indices")
    _require(
        np.asarray(contribution["original_indices"]).tolist() == [original_index],
        "score/contribution original identity changed",
    )
    current_size = _scalar(score, "current_size")
    _require(_scalar(contribution, "current_size") == current_size, "current size changed")
    iteration = _scalar(score, "debug_iteration")
    _require(_scalar(contribution, "iteration") == iteration, "physical iteration changed")
    rotations = np.asarray(score["local_rotation_matrices"], dtype=np.float32)
    rotation_ids = np.asarray(score["local_rotation_indices"], dtype=np.int64)
    translations = np.asarray(score["translations"], dtype=np.float32)
    posterior = np.asarray(score["posterior"])
    mask = np.asarray(score["reconstruction_sample_mask"], dtype=bool)
    shifted = np.asarray(contribution["mstep_shifted_recon"])
    ctf2 = np.asarray(contribution["mstep_ctf2_over_nv"])
    window = np.asarray(contribution["window_indices"], dtype=np.int64)
    _require(rotations.shape == (rotation_ids.size, 3, 3), "rotation table shape changed")
    _require(translations.ndim == 2 and translations.shape[1] == 2, "translation table shape changed")
    _require(
        posterior.shape == mask.shape == (1, rotation_ids.size, translations.shape[0]),
        "posterior/support shape changed",
    )
    _require(
        shifted.shape == (1, translations.shape[0], window.size),
        "shifted reconstruction operand shape changed",
    )
    _require(ctf2.shape == (1, window.size), "CTF-squared operand shape changed")
    _require(
        np.array_equal(
            rotation_ids,
            np.asarray(contribution["active_global_rotation_indices"], dtype=np.int64),
        ),
        "score/contribution active rotation sequence changed",
    )
    padded_rotation_count = np.asarray(contribution["posterior_probs"]).shape[1]
    _require(padded_rotation_count >= rotation_ids.size, "contribution rotation axis was truncated")
    _require(
        np.array_equal(posterior[0], contribution["posterior_probs"][0, : rotation_ids.size]),
        "score/contribution posterior changed",
    )
    _require(
        np.array_equal(mask[0], contribution["reconstruction_mask"][0, : rotation_ids.size]),
        "score/contribution support changed",
    )
    _require(
        np.array_equal(score["debug_shifted_recon"], shifted[0]),
        "score/contribution shifted reconstruction operand changed",
    )
    _require(
        np.array_equal(score["debug_ctf2_over_nv_recon"], ctf2[0]),
        "score/contribution CTF-squared operand changed",
    )
    return {
        "original_index": np.asarray(original_index, dtype=np.int64),
        "current_size": np.asarray(current_size, dtype=np.int64),
        "physical_iteration": iteration,
        "rotations": rotations,
        "fine_translations": translations,
        "reconstruction_probs": posterior[0],
        "reconstruction_mask": mask[0],
        "shifted_recon": np.asarray(shifted[0], dtype=np.complex64),
        "ctf2_over_nv_recon": np.asarray(ctf2[0], dtype=np.float32),
        "recon_window_indices": window,
        "window_indices": window,
        "_path": str(contribution_path),
    }


def analyze(
    *,
    factor_path: Path,
    local_score_path: Path,
    contribution_path: Path,
    physical_image_size: int,
    require_h100: bool,
) -> dict[str, Any]:
    factor = load_factor_capture(factor_path)
    recovar = load_recovar_pair(local_score_path, contribution_path)
    stack_index = int(factor.stack_index)
    original_index = int(np.asarray(recovar["original_index"]).item())
    _require(stack_index == original_index + 1, "native/RECOVAR stack identity changed")
    target = {
        "original_index_zero_based": original_index,
        "stack_index_one_based": stack_index,
        "physical_iteration": int(recovar["physical_iteration"]),
        "role": "focused_final_score_to_bpref_boundary",
    }
    particle = _compare_particle(
        target=target,
        factor=factor,
        recovar=recovar,
        physical_image_size=physical_image_size,
        current_size=int(np.asarray(recovar["current_size"]).item()),
        require_h100=require_h100,
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "metric_policy": (
            "exact and relative-L2 intermediate metrics; no correlation; "
            "signed shellwise FSC/FSC-AUC remains the map acceptance metric"
        ),
        "relative_l2_bound": RELATIVE_L2_BOUND,
        "first_cross_engine_boundary": _first_cross_engine_boundary([particle]),
        "production_authorized": False,
        "fixed_scorecard_changed": False,
        "sources": {
            "native_factor": {"path": str(factor_path.resolve()), "sha256": _sha256(factor_path)},
            "recovar_local_score": {
                "path": str(local_score_path.resolve()),
                "sha256": _sha256(local_score_path),
            },
            "recovar_contribution": {
                "path": str(contribution_path.resolve()),
                "sha256": _sha256(contribution_path),
            },
        },
        "particle": particle,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-path", type=Path, required=True)
    parser.add_argument("--local-score-path", type=Path, required=True)
    parser.add_argument("--contribution-path", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-non-h100", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        factor_path=args.factor_path,
        local_score_path=args.local_score_path,
        contribution_path=args.contribution_path,
        physical_image_size=args.physical_image_size,
        require_h100=not args.allow_non_h100,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "first_cross_engine_boundary": report["first_cross_engine_boundary"],
                "stack_index_one_based": report["particle"]["stack_index_one_based"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
