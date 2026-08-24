#!/usr/bin/env python3
"""Compare RELION and RECOVAR K=1 noise-update sufficient statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

_INTEGER_KEYS = {"iter", "part_id", "halfset", "random_subset", "optics_group", "shell"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_native_rows(
    path: Path,
    *,
    iteration: int,
    half: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            phase = line.partition("\t")[0]
            if phase not in {"acc_particle", "final"}:
                continue
            if f"\titer={iteration}\t" not in line or f"\thalfset={half}\t" not in line:
                continue
            fields = line.rstrip("\n").split("\t")
            _require(fields and fields[0], f"empty native phase at line {line_number}")
            row: dict[str, float | int | str] = {"phase": fields[0]}
            for field in fields[1:]:
                _require("=" in field, f"malformed native field at line {line_number}: {field}")
                key, value = field.split("=", 1)
                _require(key not in row, f"duplicate native field at line {line_number}: {key}")
                row[key] = int(value) if key in _INTEGER_KEYS else float(value)
            rows.append(row)
    _require(rows, f"empty native noise dump: {path}")
    return rows


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    _require(candidate.shape == reference.shape and candidate.size > 0, "metric topology mismatch")
    residual = candidate - reference
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(reference), np.finfo(float).tiny)),
        "median_abs": float(np.median(np.abs(residual))),
        "p95_abs": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _positive_ratio(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int | None]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    _require(candidate.shape == reference.shape, "ratio topology mismatch")
    valid = np.isfinite(candidate) & np.isfinite(reference) & (candidate > 0.0) & (reference > 0.0)
    if not np.any(valid):
        return {
            "count": 0,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    ratio = candidate[valid] / reference[valid]
    return {
        "count": int(ratio.size),
        "median": float(np.median(ratio)),
        "p05": float(np.percentile(ratio, 5)),
        "p95": float(np.percentile(ratio, 95)),
        "min": float(np.min(ratio)),
        "max": float(np.max(ratio)),
    }


def _native_final_arrays(
    rows: list[dict[str, float | int | str]],
    *,
    iteration: int,
    half: int,
) -> dict[str, np.ndarray | float]:
    selected = [
        row
        for row in rows
        if row["phase"] == "final"
        and row.get("iter") == iteration
        and row.get("halfset") == half
        and row.get("optics_group") == 0
    ]
    _require(selected, f"no native final rows for iteration {iteration} half {half}")
    selected.sort(key=lambda row: int(row["shell"]))
    shells = np.asarray([row["shell"] for row in selected], dtype=np.int64)
    _require(np.array_equal(shells, np.arange(shells.size)), "native final shell sequence changed")
    sumw = np.asarray([row["sumw_group"] for row in selected], dtype=np.float64)
    _require(np.all(sumw == sumw[0]), "native sumw varies by shell")
    mu = np.asarray([row["my_mu"] for row in selected], dtype=np.float64)
    _require(np.all(mu == mu[0]), "native update mixing factor varies by shell")
    return {
        "shell": shells,
        "raw": np.asarray([row["raw_sigma2_accum"] for row in selected], dtype=np.float64),
        "sumw": float(sumw[0]),
        "npix": np.asarray([row["Npix_per_shell"] for row in selected], dtype=np.float64),
        "old": np.asarray([row["old_sigma2"] for row in selected], dtype=np.float64),
        "new": np.asarray([row["new_sigma2"] for row in selected], dtype=np.float64),
        "mu": float(mu[0]),
    }


def _native_particle_component_arrays(
    rows: list[dict[str, float | int | str]],
    *,
    iteration: int,
    half: int,
    shell_count: int,
) -> dict[str, np.ndarray | int]:
    selected = [
        row
        for row in rows
        if row["phase"] == "acc_particle"
        and row.get("iter") == iteration
        and row.get("halfset") == half
        and row.get("optics_group") == 0
    ]
    _require(selected, f"no native accelerated particle rows for iteration {iteration} half {half}")
    shells = np.asarray([row["shell"] for row in selected], dtype=np.int64)
    _require(np.all((shells >= 0) & (shells < shell_count)), "native particle shell outside final topology")
    counts = np.bincount(shells, minlength=shell_count)
    _require(np.all(counts == counts[0]), "native particle row count varies by shell")
    _require(int(counts[0]) > 0, "native particle component rows are empty")

    direct_residual = np.bincount(
        shells,
        weights=np.asarray([row["direct_residual"] for row in selected], dtype=np.float64),
        minlength=shell_count,
    )
    image_power = np.bincount(
        shells,
        weights=np.asarray([row["image_power"] for row in selected], dtype=np.float64),
        minlength=shell_count,
    )
    return {
        "particle_count": int(counts[0]),
        "direct_residual": direct_residual,
        "image_power": image_power,
        "raw": direct_residual + image_power,
    }


def _native_detailed_component_arrays(
    path: Path,
    *,
    iteration: int,
    half: int,
    shell_count: int,
) -> dict[str, np.ndarray | int]:
    values = {
        name: np.zeros(shell_count, dtype=np.float64)
        for name in ("direct_residual", "aa", "xa", "inferred_image_power")
    }
    counts = np.zeros(shell_count, dtype=np.int64)
    prefix = f"acc_components\titer={iteration}\t"
    half_token = f"\thalfset={half}\t"
    with path.open() as stream:
        for line in stream:
            if not line.startswith(prefix) or half_token not in line:
                continue
            fields = {key: value for key, value in (item.split("=", 1) for item in line.rstrip().split("\t")[1:])}
            shell = int(fields["shell"])
            _require(0 <= shell < shell_count, "native detailed component shell is out of range")
            counts[shell] += 1
            for name in values:
                values[name][shell] += float(fields[name])
    _require(np.all(counts > 0), "native detailed component selection is incomplete")
    _require(np.all(counts == counts[0]), "native detailed particle count varies by shell")
    return {**values, "particle_count": int(counts[0])}


def analyze(
    native_tsv: Path,
    recovar_npz: Path,
    *,
    iteration: int,
    half: int,
    image_size: int,
    native_components_tsv: Path | None = None,
    recovar_prefix: str | None = None,
) -> dict[str, object]:
    rows = _parse_native_rows(native_tsv, iteration=iteration, half=half)
    native = _native_final_arrays(rows, iteration=iteration, half=half)
    native_components = _native_particle_component_arrays(
        rows,
        iteration=iteration,
        half=half,
        shell_count=int(np.asarray(native["shell"]).size),
    )
    with np.load(recovar_npz, allow_pickle=False) as payload:
        prefix = recovar_prefix or f"half{half}"
        rec_raw = np.asarray(payload[f"{prefix}_wsum_total"], dtype=np.float64)
        rec_residual = np.asarray(payload[f"{prefix}_wsum_sigma2_noise"], dtype=np.float64)
        rec_image_power = np.asarray(payload[f"{prefix}_wsum_img_power"], dtype=np.float64)
        rec_sumw = float(np.asarray(payload[f"{prefix}_sumw"]).reshape(-1)[0])
        rec_npix = np.asarray(payload["relion_half_plane_shell_counts"], dtype=np.float64)
        rec_old = np.asarray(payload[f"{prefix}_previous_sigma2_noise"], dtype=np.float64)
        rec_new = np.asarray(payload[f"{prefix}_sigma2_noise"], dtype=np.float64)
        current_size = int(np.asarray(payload["current_size"]).reshape(-1)[0])
    count = int(np.asarray(native["shell"]).size)
    for name, values in (
        ("RECOVAR raw", rec_raw),
        ("RECOVAR residual", rec_residual),
        ("RECOVAR image power", rec_image_power),
        ("RECOVAR shell counts", rec_npix),
        ("RECOVAR old noise", rec_old),
        ("RECOVAR new noise", rec_new),
    ):
        _require(values.shape == (count,), f"{name} shell topology changed: {values.shape}")
    _require(np.allclose(rec_raw, rec_residual + rec_image_power, rtol=0.0, atol=0.0), "RECOVAR raw closure failed")

    n4 = float(image_size**4)
    rec_raw_relion = rec_raw / n4
    rec_old_relion = rec_old / n4
    rec_new_relion = rec_new / n4
    native_raw = np.asarray(native["raw"], dtype=np.float64)
    native_residual = np.asarray(native_components["direct_residual"], dtype=np.float64)
    native_image_power = np.asarray(native_components["image_power"], dtype=np.float64)
    native_particle_raw = np.asarray(native_components["raw"], dtype=np.float64)
    native_npix = np.asarray(native["npix"], dtype=np.float64)
    native_old = np.asarray(native["old"], dtype=np.float64)
    native_new = np.asarray(native["new"], dtype=np.float64)
    native_sumw = float(native["sumw"])
    mu = float(native["mu"])
    native_formula = mu * native_old + (1.0 - mu) * native_raw / (2.0 * native_sumw * native_npix)
    rec_formula = rec_raw_relion / (2.0 * rec_sumw * rec_npix)
    native_raw_rec_denominator = mu * native_old + (1.0 - mu) * native_raw / (
        2.0 * rec_sumw * rec_npix
    )
    rec_raw_native_denominator = mu * native_old + (1.0 - mu) * rec_raw_relion / (
        2.0 * native_sumw * native_npix
    )
    _require(np.all(native_npix > 0.0) and np.all(rec_npix > 0.0), "zero shell denominator")
    _require(np.all(np.isfinite(native_formula)), "non-finite native formula replay")

    active_stop = min(count, current_size // 2 + 1)
    low = slice(1, active_stop)
    high = slice(active_stop, count)
    comparisons: dict[str, object] = {
        "shell_counts_recovar_vs_native": _metric(rec_npix, native_npix),
        "old_noise_recovar_vs_native": _metric(rec_old_relion, native_old),
        "raw_total_recovar_vs_native": _metric(rec_raw_relion, native_raw),
        "native_particle_components_vs_final_raw": _metric(native_particle_raw, native_raw),
        "residual_recovar_vs_native_particles": _metric(rec_residual / n4, native_residual),
        "image_power_recovar_vs_native_particles": _metric(rec_image_power / n4, native_image_power),
        "new_noise_recovar_vs_native": _metric(rec_new_relion, native_new),
        "native_formula_replay": _metric(native_formula, native_new),
        "recovar_formula_replay": _metric(rec_formula, rec_new_relion),
        "native_raw_recovar_denominator_vs_native": _metric(native_raw_rec_denominator, native_new),
        "recovar_raw_native_denominator_vs_native": _metric(rec_raw_native_denominator, native_new),
        "low_shell_raw_total_recovar_vs_native": _metric(rec_raw_relion[low], native_raw[low]),
        "low_shell_native_over_recovar_raw_ratio": _positive_ratio(native_raw[low], rec_raw_relion[low]),
    }
    if active_stop < count:
        comparisons["high_shell_raw_total_recovar_vs_native"] = _metric(
            rec_raw_relion[high], native_raw[high]
        )
        comparisons["high_shell_residual_recovar_vs_native_particles"] = _metric(
            rec_residual[high] / n4,
            native_residual[high],
        )
        comparisons["high_shell_image_power_recovar_vs_native_particles"] = _metric(
            rec_image_power[high] / n4,
            native_image_power[high],
        )
        comparisons["high_shell_native_over_recovar_raw_ratio"] = _positive_ratio(
            native_raw[high],
            rec_raw_relion[high],
        )

    detailed_terms = None
    if native_components_tsv is not None:
        detailed = _native_detailed_component_arrays(
            native_components_tsv,
            iteration=iteration,
            half=half,
            shell_count=count,
        )
        native_detailed_raw = np.asarray(detailed["direct_residual"], dtype=np.float64)
        native_detailed_image = np.asarray(detailed["inferred_image_power"], dtype=np.float64)
        native_detailed_residual = np.asarray(detailed["aa"], dtype=np.float64) - 2.0 * np.asarray(
            detailed["xa"], dtype=np.float64
        )
        raw_delta_sum = float(np.sum(rec_raw_relion[low] - native_detailed_raw[low]))
        image_delta_sum = float(np.sum(rec_image_power[low] / n4 - native_detailed_image[low]))
        residual_delta_sum = float(np.sum(rec_residual[low] / n4 - native_detailed_residual[low]))
        comparisons.update(
            {
                "low_shell_image_power_recovar_vs_native_components": _metric(
                    rec_image_power[low] / n4, native_detailed_image[low]
                ),
                "low_shell_image_power_recovar_over_native_components": _positive_ratio(
                    rec_image_power[low] / n4, native_detailed_image[low]
                ),
                "low_shell_a2_minus_2xa_recovar_vs_native_components": _metric(
                    rec_residual[low] / n4, native_detailed_residual[low]
                ),
                "low_shell_a2_minus_2xa_recovar_over_native_components": _positive_ratio(
                    rec_residual[low] / n4, native_detailed_residual[low]
                ),
                "low_shell_native_component_closure": _metric(
                    native_detailed_image[low] + native_detailed_residual[low],
                    native_detailed_raw[low],
                ),
            }
        )
        detailed_terms = {
            "native_particle_count": int(detailed["particle_count"]),
            "active_shell_raw_delta_sum_recovar_minus_native": raw_delta_sum,
            "active_shell_image_power_delta_sum_recovar_minus_native": image_delta_sum,
            "active_shell_a2_minus_2xa_delta_sum_recovar_minus_native": residual_delta_sum,
            "image_power_fraction_of_signed_raw_delta": image_delta_sum / raw_delta_sum,
            "a2_minus_2xa_fraction_of_signed_raw_delta": residual_delta_sum / raw_delta_sum,
        }

    report = {
        "schema": "recovar.em.k1_noise_update_terms.v1",
        "identity": {
            "native_iteration": iteration,
            "half": half,
            "image_size": image_size,
            "current_size": current_size,
            "active_shell_stop_exclusive": active_stop,
            "shell_count": count,
            "native_particle_count": int(native_components["particle_count"]),
            "native_halfset": int(half),
            "recovar_prefix": prefix,
        },
        "denominator": {
            "native_sumw": native_sumw,
            "recovar_sumw": rec_sumw,
            "recovar_over_native_sumw": rec_sumw / native_sumw,
            "native_mu": mu,
            "native_particle_count_over_native_sumw": int(native_components["particle_count"]) / native_sumw,
            "native_particle_count_over_recovar_sumw": int(native_components["particle_count"]) / rec_sumw,
        },
        "comparisons": comparisons,
        "shells": [
            {
                "shell": shell,
                "native_npix": float(native_npix[shell]),
                "recovar_npix": float(rec_npix[shell]),
                "native_raw": float(native_raw[shell]),
                "native_residual": float(native_residual[shell]),
                "native_image_power": float(native_image_power[shell]),
                "recovar_raw_relion_units": float(rec_raw_relion[shell]),
                "recovar_residual_relion_units": float(rec_residual[shell] / n4),
                "recovar_image_power_relion_units": float(rec_image_power[shell] / n4),
                "native_new": float(native_new[shell]),
                "recovar_new_relion_units": float(rec_new_relion[shell]),
            }
            for shell in range(count)
        ],
        "artifacts": {
            "native_tsv": str(native_tsv.resolve()),
            "native_tsv_sha256": _sha256(native_tsv),
            "recovar_npz": str(recovar_npz.resolve()),
            "recovar_npz_sha256": _sha256(recovar_npz),
        },
    }
    if detailed_terms is not None:
        report["active_shell_numerator_decomposition"] = detailed_terms
        report["artifacts"]["native_components_tsv"] = str(native_components_tsv.resolve())
        report["artifacts"]["native_components_tsv_sha256"] = _sha256(native_components_tsv)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-tsv", type=Path, required=True)
    parser.add_argument("--native-components-tsv", type=Path)
    parser.add_argument("--recovar-npz", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--half", type=int, choices=(-1, 0, 1, 2), required=True)
    parser.add_argument("--recovar-prefix")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_tsv,
        args.recovar_npz,
        iteration=args.iteration,
        half=args.half,
        image_size=args.image_size,
        native_components_tsv=args.native_components_tsv,
        recovar_prefix=args.recovar_prefix,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
