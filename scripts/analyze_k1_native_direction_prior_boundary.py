#!/usr/bin/env python3
"""Compare native RELION and RECOVAR coarse rotation log priors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from scripts.parse_relion_dump_dir import parse_dump_dir
else:
    from parse_relion_dump_dir import parse_dump_dir


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    _require(absolute.size > 0, "cannot summarize an empty residual")
    _require(bool(np.all(np.isfinite(absolute))), "residual contains nonfinite values")
    return {
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
        "rms": float(np.sqrt(np.mean(np.square(absolute)))),
    }


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    """Compare log-prior arrays while treating matched infinities as equal."""

    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    _require(left.shape == right.shape, "log-prior arrays differ in shape")
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    common_finite = left_finite & right_finite
    _require(bool(np.any(common_finite)), "log-prior arrays have no common finite entries")
    return {
        "float32_exact": bool(np.array_equal(left, right)),
        "finite_support_mismatch_count": int(np.count_nonzero(left_finite != right_finite)),
        "common_finite_count": int(np.count_nonzero(common_finite)),
        "residual": _stats(
            left[common_finite].astype(np.float64) - right[common_finite].astype(np.float64)
        ),
    }


def _finite_difference(left: np.float32, right: np.float32) -> float | None:
    if not (np.isfinite(left) and np.isfinite(right)):
        return None
    return float(np.float64(left) - np.float64(right))


def native_direction_major_to_recovar_psi_major(
    native_log_prior: np.ndarray,
    *,
    n_directions: int,
    n_psi: int,
) -> np.ndarray:
    """Return RELION's direction-major prior in RECOVAR's psi-major order."""

    native = np.asarray(native_log_prior, dtype=np.float32).reshape(-1)
    n_rotations = int(n_directions) * int(n_psi)
    _require(n_directions > 0 and n_psi > 0, "rotation dimensions must be positive")
    _require(
        native.shape == (n_rotations,),
        f"native prior must have shape ({n_rotations},), got {native.shape}",
    )
    native_ids = np.arange(n_rotations, dtype=np.int64)
    directions = native_ids // int(n_psi)
    psi = native_ids % int(n_psi)
    recovar_ids = psi * int(n_directions) + directions
    output = np.empty_like(native)
    output[recovar_ids] = native
    return output


def _load_recovar_rotation_log_prior(path: Path, n_rotations: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        _require(
            "rotation_log_prior" in archive.files,
            f"RECOVAR capture lacks rotation_log_prior: {path}",
        )
        prior = np.asarray(archive["rotation_log_prior"], dtype=np.float32)
    if prior.ndim == 2:
        _require(prior.shape[0] == 1, f"expected K=1 rotation prior, got {prior.shape}")
        prior = prior[0]
    prior = prior.reshape(-1)
    _require(
        prior.shape == (n_rotations,),
        f"RECOVAR prior must have shape ({n_rotations},), got {prior.shape}",
    )
    return prior


def compare_rotation_log_priors(
    *,
    native_direction_major: np.ndarray,
    candidate_psi_major: np.ndarray,
    n_directions: int,
    n_psi: int,
    replay_psi_major: np.ndarray | None = None,
    target_rotations: tuple[int, ...] = (),
) -> dict[str, object]:
    native = native_direction_major_to_recovar_psi_major(
        native_direction_major,
        n_directions=n_directions,
        n_psi=n_psi,
    )
    # RELION's accelerated ``initOrientations`` stores a zero-probability
    # orientation as numeric 0 in ``pdf_orientation`` and records its semantic
    # value in a separate boolean zero mask.  The passive capture contains the
    # numeric buffer but not that mask.  With a multi-direction K=1 prior,
    # log-probability 0 cannot be a live orientation, so restore the semantic
    # -inf representation used by RECOVAR before comparing support.
    native = native.copy()
    native[native == np.float32(0.0)] = np.float32(-np.inf)
    candidate = np.asarray(candidate_psi_major, dtype=np.float32).reshape(-1)
    _require(candidate.shape == native.shape, "candidate and native priors differ in shape")
    n_rotations = native.size
    targets = tuple(int(value) for value in target_rotations)
    _require(
        all(0 <= value < n_rotations for value in targets),
        "target rotation is outside the coarse grid",
    )

    candidate_comparison = _comparison(candidate, native)
    report: dict[str, object] = {
        "n_directions": int(n_directions),
        "n_psi": int(n_psi),
        "n_rotations": int(n_rotations),
        "candidate_vs_native": candidate_comparison,
        "targets": [
            {
                "rotation": rotation,
                "native_log_prior": float(native[rotation]),
                "candidate_log_prior": float(candidate[rotation]),
                "candidate_minus_native": _finite_difference(
                    candidate[rotation], native[rotation]
                ),
            }
            for rotation in targets
        ],
    }

    if replay_psi_major is not None:
        replay = np.asarray(replay_psi_major, dtype=np.float32).reshape(-1)
        _require(replay.shape == native.shape, "replay and native priors differ in shape")
        replay_comparison = _comparison(replay, native)
        candidate_replay_comparison = _comparison(candidate, replay)
        report["replay_vs_native"] = replay_comparison
        report["candidate_vs_replay"] = candidate_replay_comparison
        candidate_key = (
            int(candidate_comparison["finite_support_mismatch_count"]),
            float(candidate_comparison["residual"]["rms"]),
        )
        replay_key = (
            int(replay_comparison["finite_support_mismatch_count"]),
            float(replay_comparison["residual"]["rms"]),
        )
        report["classification"] = {
            "candidate_closer_to_native_by_support_then_rms": bool(candidate_key < replay_key),
            "candidate_native_rms_over_replay_native_rms": (
                float(
                    candidate_comparison["residual"]["rms"]
                    / replay_comparison["residual"]["rms"]
                )
                if replay_comparison["residual"]["rms"] > 0.0
                else None
            ),
        }
        for row in report["targets"]:
            rotation = int(row["rotation"])
            row["replay_log_prior"] = float(replay[rotation])
            row["replay_minus_native"] = _finite_difference(
                replay[rotation], native[rotation]
            )
            row["candidate_minus_replay"] = _finite_difference(
                candidate[rotation], replay[rotation]
            )
    return report


def analyze(
    *,
    native_dump_dir: Path,
    candidate_path: Path,
    n_directions: int,
    n_psi: int,
    replay_path: Path | None = None,
    target_rotations: tuple[int, ...] = (),
) -> dict[str, object]:
    native_payload = parse_dump_dir(
        native_dump_dir,
        include_names={"pass0_pdf_orientation"},
    )
    _require(
        "pass0_pdf_orientation" in native_payload,
        "native dump lacks pass0_pdf_orientation",
    )
    n_rotations = int(n_directions) * int(n_psi)
    candidate = _load_recovar_rotation_log_prior(candidate_path, n_rotations)
    replay = (
        None
        if replay_path is None
        else _load_recovar_rotation_log_prior(replay_path, n_rotations)
    )
    report = compare_rotation_log_priors(
        native_direction_major=np.asarray(native_payload["pass0_pdf_orientation"]),
        candidate_psi_major=candidate,
        replay_psi_major=replay,
        n_directions=n_directions,
        n_psi=n_psi,
        target_rotations=target_rotations,
    )
    report.update(
        {
            "schema": "recovar.em.k1_native_direction_prior_boundary.v1",
            "status": "complete",
            "artifacts": {
                "native": {
                    "path": str((native_dump_dir / "pass0_pdf_orientation.bin").resolve()),
                    "sha256": _sha256(native_dump_dir / "pass0_pdf_orientation.bin"),
                },
                "candidate": {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                },
                "replay": (
                    None
                    if replay_path is None
                    else {
                        "path": str(replay_path.resolve()),
                        "sha256": _sha256(replay_path),
                    }
                ),
            },
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dump-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--n-directions", type=int, required=True)
    parser.add_argument("--n-psi", type=int, required=True)
    parser.add_argument("--target-rotation", type=int, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        native_dump_dir=args.native_dump_dir,
        candidate_path=args.candidate,
        replay_path=args.replay,
        n_directions=args.n_directions,
        n_psi=args.n_psi,
        target_rotations=tuple(args.target_rotation),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
