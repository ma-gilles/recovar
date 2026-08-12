#!/usr/bin/env python3
"""Compare live RELION scoring noise with RECOVAR and serialized STAR state."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import starfile


MAGIC = b"RLNSIGMAV1"
HEADER_WORDS = 16


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_capture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    prefix_size = 16 + HEADER_WORDS * 8
    _require(len(payload) >= prefix_size, "scoring-noise capture is truncated")
    _require(payload[:16].rstrip(b"\0") == MAGIC, "scoring-noise magic changed")
    header = np.frombuffer(payload, dtype="<u8", count=HEADER_WORDS, offset=16).copy()
    _require(int(header[0]) == 1, "scoring-noise schema changed")
    count = int(header[4])
    expected = prefix_size + count * 8 + count * 4
    _require(len(payload) == expected, "scoring-noise byte count changed")
    sigma2 = np.frombuffer(payload, dtype="<f8", count=count, offset=prefix_size).copy()
    minvsigma2 = np.frombuffer(
        payload,
        dtype="<f4",
        count=count,
        offset=prefix_size + count * 8,
    ).copy()
    _require(np.all(np.isfinite(sigma2)) and np.all(sigma2 > 0), "invalid captured sigma2")
    _require(np.all(np.isfinite(minvsigma2)) and np.all(minvsigma2 > 0), "invalid captured inverse noise")
    return header, sigma2, minvsigma2


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    _require(left.shape == right.shape and left.size > 0, "metric shape changed")
    delta = right - left
    return {
        "exact_equal": bool(np.array_equal(left, right)),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(left)),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _ordered_f32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return np.where(
        (bits & np.uint32(0x80000000)) != 0,
        ~bits,
        bits | np.uint32(0x80000000),
    ).astype(np.uint32)


def _ulp_distance(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    left = _ordered_f32(reference).astype(np.int64)
    right = _ordered_f32(candidate).astype(np.int64)
    return np.abs(right - left)


def _ulp_summary(distance: np.ndarray) -> dict[str, int]:
    values = np.asarray(distance, dtype=np.int64)
    _require(values.ndim == 1 and values.size > 0, "ULP summary input changed")
    return {
        "exact_shell_count": int(np.count_nonzero(values == 0)),
        "one_ulp_shell_count": int(np.count_nonzero(values == 1)),
        "greater_than_one_ulp_shell_count": int(np.count_nonzero(values > 1)),
        "max_ulp": int(np.max(values)),
    }


def analyze(
    capture_path: Path,
    recovar_results: Path,
    relion_model: Path,
    *,
    recovar_iteration: int,
    half: int,
) -> dict[str, object]:
    header, live_sigma2, live_minvsigma2 = _load_capture(capture_path)
    _require(int(header[1]) == recovar_iteration + 2, "capture iteration is not the requested scoring boundary")
    _require(int(header[3]) == 0, "only one optics group is supported")
    with np.load(recovar_results, allow_pickle=True) as archive:
        volume_shape = np.asarray(archive["volume_shape"], dtype=np.int64).reshape(-1)
        _require(volume_shape.size == 3 and len(set(volume_shape.tolist())) == 1, "volume shape changed")
        recovar_sigma2_native = np.asarray(
            archive[f"noise_radial_per_half_iter_{recovar_iteration:03d}"],
            dtype=np.float64,
        )[half - 1]
    model = starfile.read(relion_model)
    serialized_sigma2 = np.asarray(
        model["model_optics_group_1"]["rlnSigma2Noise"],
        dtype=np.float64,
    )
    count = live_sigma2.size
    _require(recovar_sigma2_native.size == count and serialized_sigma2.size == count, "shell count changed")
    n4 = float(int(volume_shape[0]) ** 4)
    recovar_sigma2 = recovar_sigma2_native / n4
    # The exact scoring path reciprocates the binary64 native variance and
    # only then stores Minvsigma2 as float32.  Casting sigma2 itself to
    # float32 here would manufacture a boundary that production does not use.
    recovar_score_sigma2 = recovar_sigma2
    reciprocal_replay = np.asarray(1.0 / live_sigma2, dtype=np.float32)
    _require(np.array_equal(reciprocal_replay, live_minvsigma2), "captured inverse noise does not replay")
    recovar_minvsigma2 = np.asarray(1.0 / recovar_score_sigma2, dtype=np.float32)
    serialized_minvsigma2 = np.asarray(1.0 / serialized_sigma2, dtype=np.float32)
    low = slice(1, 5)
    live_vs_rec_ulp = _ulp_distance(live_minvsigma2, recovar_minvsigma2)
    live_vs_star_ulp = _ulp_distance(live_minvsigma2, serialized_minvsigma2)
    shell_rows = []
    for shell in range(count):
        shell_rows.append(
            {
                "shell": shell,
                "live_sigma2": float(live_sigma2[shell]),
                "recovar_score_sigma2": float(recovar_score_sigma2[shell]),
                "serialized_sigma2": float(serialized_sigma2[shell]),
                "live_minvsigma2_f32": float(live_minvsigma2[shell]),
                "recovar_minvsigma2_f32": float(recovar_minvsigma2[shell]),
                "serialized_minvsigma2_f32": float(serialized_minvsigma2[shell]),
                "recovar_minvsigma2_ulp": int(live_vs_rec_ulp[shell]),
                "serialized_minvsigma2_ulp": int(live_vs_star_ulp[shell]),
            }
        )
    return {
        "schema": "recovar.em.k1_scoring_noise_boundary.v1",
        "status": "complete",
        "identity": {
            "relion_iteration": int(header[1]),
            "mpi_rank": int(header[2]),
            "optics_group_zero_based": int(header[3]),
            "recovar_post_update_iteration_zero_based": recovar_iteration,
            "half": half,
            "rfloat_bytes": int(header[5]),
            "shell_count": count,
        },
        "comparisons": {
            "live_vs_recovar_all_shells": _metrics(live_sigma2, recovar_score_sigma2),
            "live_vs_serialized_all_shells": _metrics(live_sigma2, serialized_sigma2),
            "live_vs_recovar_shells_1_through_4": _metrics(live_sigma2[low], recovar_score_sigma2[low]),
            "live_vs_serialized_shells_1_through_4": _metrics(live_sigma2[low], serialized_sigma2[low]),
            "inverse_noise_ulp_shells_1_through_4": {
                "recovar": live_vs_rec_ulp[low].astype(int).tolist(),
                "serialized": live_vs_star_ulp[low].astype(int).tolist(),
            },
            "inverse_noise_ulp_all_shells": {
                "recovar": _ulp_summary(live_vs_rec_ulp),
                "serialized": _ulp_summary(live_vs_star_ulp),
            },
        },
        "shells": shell_rows,
        "artifacts": {
            "capture": str(capture_path.resolve()),
            "capture_sha256": _sha256(capture_path),
            "recovar_results": str(recovar_results.resolve()),
            "recovar_results_sha256": _sha256(recovar_results),
            "relion_model": str(relion_model.resolve()),
            "relion_model_sha256": _sha256(relion_model),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--recovar-results", type=Path, required=True)
    parser.add_argument("--relion-model", type=Path, required=True)
    parser.add_argument("--recovar-iteration", type=int, default=0)
    parser.add_argument("--half", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.capture,
        args.recovar_results,
        args.relion_model,
        recovar_iteration=args.recovar_iteration,
        half=args.half,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
