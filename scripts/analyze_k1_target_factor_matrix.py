#!/usr/bin/env python3
"""Summarize fixed K=1 coarse-to-fine factor arms for one target particle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _scalar(values: dict[str, np.ndarray], key: str):
    return np.asarray(values[key]).item()


def summarize_arm(
    *,
    name: str,
    root: Path,
    expected_native_coarse_count: int,
    extra_coarse_index: int,
    extra_fine_rotation_first: int,
    extra_fine_rotation_last: int,
) -> dict[str, object]:
    pass1_path = root / "pass1" / "significance_orig082009_it002_cs100.npz"
    pass2_path = root / "pass2" / "pass2_orig082009_cs100.npz"
    if not pass1_path.is_file() or not pass2_path.is_file():
        raise FileNotFoundError(f"incomplete target arm {name}: {root}")

    coarse = _load_npz(pass1_path)
    fine = _load_npz(pass2_path)
    if int(_scalar(coarse, "original_index")) != 82009:
        raise ValueError(f"{name}: unexpected coarse source row")
    if int(_scalar(fine, "original_index")) != 82009:
        raise ValueError(f"{name}: unexpected fine source row")

    weights = np.asarray(coarse["weights_full"], dtype=np.float64).reshape(-1)
    significant = np.asarray(coarse["significant_mask"], dtype=bool).reshape(-1)
    if weights.shape != significant.shape or not 0 <= extra_coarse_index < weights.size:
        raise ValueError(f"{name}: invalid coarse topology")
    order = np.argsort(-weights, kind="stable")
    total = float(np.sum(weights, dtype=np.float64))
    adaptive_fraction = float(_scalar(coarse, "adaptive_fraction"))
    retained = order[:expected_native_coarse_count]
    retained_mass = float(np.sum(weights[retained], dtype=np.float64) / total)
    extra_rank = int(np.flatnonzero(order == extra_coarse_index)[0]) + 1

    candidate_mask = np.asarray(fine["candidate_mask"], dtype=bool)
    probabilities = np.asarray(fine["probs"], dtype=np.float64)
    if candidate_mask.shape != probabilities.shape:
        raise ValueError(f"{name}: fine candidate/probability topology differs")
    global_rotations = np.asarray(fine["oversampled_rot_indices"], dtype=np.int64)
    if global_rotations.shape != (candidate_mask.shape[0],):
        raise ValueError(f"{name}: fine rotation identity topology differs")
    extra_rows = (global_rotations >= extra_fine_rotation_first) & (
        global_rotations <= extra_fine_rotation_last
    )
    extra_mask = candidate_mask & extra_rows[:, None]
    winner_flat = int(np.argmax(np.where(candidate_mask, probabilities, -np.inf)))
    winner_row, winner_translation = np.unravel_index(winner_flat, probabilities.shape)

    return {
        "name": name,
        "root": str(root.resolve()),
        "coarse_capture": str(pass1_path.resolve()),
        "coarse_capture_sha256": _sha256(pass1_path),
        "fine_capture": str(pass2_path.resolve()),
        "fine_capture_sha256": _sha256(pass2_path),
        "coarse": {
            "selected_count": int(np.count_nonzero(significant)),
            "selected_indices": np.flatnonzero(significant).astype(int).tolist(),
            "adaptive_fraction": adaptive_fraction,
            "cumulative_mass_after_expected_native_count": retained_mass,
            "margin_after_expected_native_count": retained_mass - adaptive_fraction,
            "extra_index": extra_coarse_index,
            "extra_rank_one_based": extra_rank,
            "extra_probability": float(weights[extra_coarse_index]),
            "extra_selected": bool(significant[extra_coarse_index]),
            "sum_weight_float32": float(_scalar(coarse, "relion_f32_sum_weight")),
            "significant_weight_float32": float(
                _scalar(coarse, "relion_f32_significant_weight")
            ),
        },
        "fine": {
            "active_candidate_count": int(np.count_nonzero(candidate_mask)),
            "significant_count": int(_scalar(fine, "reconstruction_n_significant")),
            "pmax": float(np.max(probabilities[candidate_mask])),
            "winner_rotation_row": int(winner_row),
            "winner_global_rotation": int(global_rotations[winner_row]),
            "winner_translation_row": int(winner_translation),
            "extra_rotation_row_count": int(np.count_nonzero(extra_rows)),
            "extra_active_candidate_count": int(np.count_nonzero(extra_mask)),
            "extra_probability_mass": float(
                np.sum(probabilities[extra_mask], dtype=np.float64)
            ),
        },
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# K=1 source-82009 factor matrix",
        "",
        "| Arm | Coarse support | Top-8 margin | Extra parent | Fine candidates | Extra mass | Pmax | Fine support | Winner |",
        "| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in report["arms"]:
        coarse = arm["coarse"]
        fine = arm["fine"]
        lines.append(
            "| {name} | {coarse_count} | {margin:.9g} | {extra} | {fine_count} | "
            "{extra_mass:.9g} | {pmax:.9g} | {fine_support} | ({winner_rot}, {winner_trans}) |".format(
                name=arm["name"],
                coarse_count=coarse["selected_count"],
                margin=coarse["margin_after_expected_native_count"],
                extra="yes" if coarse["extra_selected"] else "no",
                fine_count=fine["active_candidate_count"],
                extra_mass=fine["extra_probability_mass"],
                pmax=fine["pmax"],
                fine_support=fine["significant_count"],
                winner_rot=fine["winner_global_rotation"],
                winner_trans=fine["winner_translation_row"],
            )
        )
    lines.extend(
        [
            "",
            "Positive top-8 margin means the expected native eight coarse tuples already cover the adaptive fraction.",
            "Acceptance remains the fixed cross-engine FSC-AUC/topology scorecard, not this particle diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=ROOT",
        help="Named factor-arm root; may be repeated",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--expected-native-coarse-count", type=int, default=8)
    parser.add_argument("--extra-coarse-index", type=int, default=355444)
    parser.add_argument("--extra-fine-rotation-first", type=int, default=98048)
    parser.add_argument("--extra-fine-rotation-last", type=int, default=98055)
    args = parser.parse_args()

    arms = []
    names = set()
    for item in args.arm:
        if "=" not in item:
            parser.error(f"invalid --arm {item!r}; expected NAME=ROOT")
        name, root = item.split("=", 1)
        if not name or name in names:
            parser.error(f"empty or duplicate arm name {name!r}")
        names.add(name)
        arms.append(
            summarize_arm(
                name=name,
                root=Path(root),
                expected_native_coarse_count=args.expected_native_coarse_count,
                extra_coarse_index=args.extra_coarse_index,
                extra_fine_rotation_first=args.extra_fine_rotation_first,
                extra_fine_rotation_last=args.extra_fine_rotation_last,
            )
        )

    report = {
        "schema": "recovar.em.k1_target_factor_matrix.v1",
        "status": "complete",
        "source_row_zero_based": 82009,
        "stack_index_one_based": 82010,
        "expected_native_coarse_count": args.expected_native_coarse_count,
        "metric_policy": "fixed source-aligned coarse cutoff and fine posterior metrics; FSC-AUC remains acceptance",
        "arms": arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
