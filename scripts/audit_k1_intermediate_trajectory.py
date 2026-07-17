#!/usr/bin/env python3
"""Audit K=1 RECOVAR/RELION intermediate trajectories without correlation.

Numbered RECOVAR iteration ``i`` is paired with RELION ``run_it{i+1}``.
Selected iteration topology is checked exactly. Numeric arrays are compared
directly with absolute and relative error metrics; no map-quality claim is made here.
Use ``audit_k1_fsc_trajectory`` for FSC/FSC-AUC map acceptance.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import starfile

SCHEMA = "em_k1_intermediate_trajectory_audit_v1"


class AuditError(RuntimeError):
    """Raised when a required trajectory artifact is absent or malformed."""


def array_metrics(left, right) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    left = left.reshape(-1)
    right = right.reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.all(finite):
        return {
            "shape_equal": True,
            "finite_pair_count": int(np.count_nonzero(finite)),
            "count": int(left.size),
            "all_finite": False,
        }
    delta = right - left
    abs_delta = np.abs(delta)
    left_l2 = float(np.linalg.norm(left))
    left_l1 = float(np.sum(np.abs(left), dtype=np.float64))
    return {
        "shape_equal": True,
        "count": int(left.size),
        "all_finite": True,
        "exact_equal": bool(np.array_equal(left, right)),
        "max_abs": float(np.max(abs_delta, initial=0.0)),
        "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "relative_l2": float(np.linalg.norm(delta) / left_l2) if left_l2 else None,
        "relative_l1": float(np.sum(abs_delta, dtype=np.float64) / left_l1) if left_l1 else None,
        "mean_delta": float(np.mean(delta)) if delta.size else 0.0,
        "p95_abs": float(np.quantile(abs_delta, 0.95)) if delta.size else 0.0,
        "p99_abs": float(np.quantile(abs_delta, 0.99)) if delta.size else 0.0,
    }


def _read_star(path: Path):
    if not path.is_file():
        raise AuditError(f"missing {path}")
    return starfile.read(path)


def _model(path: Path) -> dict:
    payload = _read_star(path)
    if not isinstance(payload, dict):
        raise AuditError(f"expected multi-block model STAR: {path}")
    return payload


def _particles(path: Path):
    payload = _read_star(path)
    if isinstance(payload, dict):
        if "particles" not in payload:
            raise AuditError(f"missing particles block in {path}")
        return payload["particles"]
    return payload


def _values_in_original_image_order(particles, column: str, reference_names=None):
    """Invert RELION's STAR row ordering using exact particle identities."""
    if column not in particles or "rlnImageName" not in particles:
        raise AuditError(f"RELION particles are missing {column} or rlnImageName")
    names = np.asarray(particles["rlnImageName"], dtype=str).reshape(-1)
    values = np.asarray(particles[column], dtype=np.float64).reshape(-1)
    if names.size != values.size:
        raise AuditError(f"RELION particle identity/value length mismatch for {column}")
    if reference_names is not None:
        reference_names = np.asarray(reference_names, dtype=str).reshape(-1)
        if reference_names.size != names.size:
            raise AuditError("RELION and reference particle identity lengths differ")
        if len(set(names.tolist())) != names.size or len(set(reference_names.tolist())) != names.size:
            raise AuditError("RELION or reference particle identities are not unique")
        row_by_name = {name: row for row, name in enumerate(names)}
        if set(row_by_name) != set(reference_names.tolist()):
            raise AuditError("RELION and reference particle identity sets differ")
        take = np.asarray([row_by_name[name] for name in reference_names], dtype=np.int64)
        return values[take], {
            "mapping": "exact rlnImageName identity to input particles.star order",
            "identity_count": int(names.size),
            "unique_identities": True,
            "exact_identity_set": True,
        }

    # Fail-closed fallback for older synthetic fixtures that do not retain the
    # input STAR. Numeric stack positions are only unambiguous for one stack.
    image_indices = np.empty(names.size, dtype=np.int64)
    stack_names = set()
    for row, name in enumerate(names):
        match = re.fullmatch(r"([1-9][0-9]*)@(.+)", name)
        if match is None:
            raise AuditError(f"unsupported RELION rlnImageName identity: {name!r}")
        image_indices[row] = int(match.group(1)) - 1
        stack_names.add(match.group(2))
    expected = np.arange(names.size, dtype=np.int64)
    if len(stack_names) != 1 or not np.array_equal(np.sort(image_indices), expected):
        raise AuditError(
            "RELION rlnImageName values are not an exact single-stack permutation "
            f"of 1..{names.size}"
        )
    original_order = np.empty_like(values)
    original_order[image_indices] = values
    return original_order, {
        "mapping": "numeric 1-based rlnImageName prefix to RECOVAR original image index",
        "single_stack": next(iter(stack_names)),
        "identity_count": int(names.size),
        "exact_permutation": True,
    }


def _sampling_general(path: Path) -> dict:
    payload = _read_star(path)
    if not isinstance(payload, dict):
        return dict(payload)
    for key in ("sampling_general", "data_sampling_general"):
        if key in payload:
            block = payload[key]
            return dict(block.iloc[0]) if hasattr(block, "iloc") else dict(block)
    return payload


def _class_table(model: dict):
    if "model_class_1" not in model:
        raise AuditError("RELION model is missing model_class_1")
    return model["model_class_1"]


def _noise_table(model: dict):
    if "model_optics_group_1" not in model:
        raise AuditError("RELION model is missing model_optics_group_1")
    return model["model_optics_group_1"]


def _general(model: dict) -> dict:
    if "model_general" not in model:
        raise AuditError("RELION model is missing model_general")
    block = model["model_general"]
    return dict(block.iloc[0]) if hasattr(block, "iloc") else dict(block)


def _rec_array(payload, key: str, *, required: bool = True):
    if key not in payload.files:
        if required:
            raise AuditError(f"RECOVAR results missing {key}")
        return None
    value = np.asarray(payload[key])
    if value.dtype == object:
        if value.shape == () and value.item() is None:
            return None
        try:
            value = np.asarray(value.item())
        except ValueError:
            pass
    return value


def _direction_prior(payload, iteration: int, half: int):
    if "direction_prior_trajectory_per_half" not in payload.files:
        return None
    trajectory = payload["direction_prior_trajectory_per_half"]
    if iteration >= len(trajectory) or trajectory[iteration] is None:
        return None
    value = trajectory[iteration][half - 1]
    return None if value is None else np.asarray(value, dtype=np.float64).reshape(-1)


def _trajectory_pair(payload, key: str, iteration: int):
    if key not in payload.files:
        return None
    trajectory = payload[key]
    if iteration >= len(trajectory) or trajectory[iteration] is None:
        return None
    try:
        pair = np.asarray(trajectory[iteration], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if pair.size != 2 or not np.all(np.isfinite(pair)):
        return None
    return pair


def _rel_direction_prior(model: dict):
    table = model.get("model_pdf_orient_class_1")
    if table is None or "rlnOrientationDistribution" not in table:
        return None
    return np.asarray(table["rlnOrientationDistribution"], dtype=np.float64)


def _compare_optional(left, right):
    if left is None or right is None:
        return {"available": False, "left_available": left is not None, "right_available": right is not None}
    return {"available": True, **array_metrics(left, right)}


def _shell_array(table, column: str):
    if column not in table:
        return None
    return np.asarray(table[column], dtype=np.float64)


def _scalar_pair(relion_value, recovar_value) -> dict[str, object]:
    relion_value = float(relion_value)
    recovar_value = float(recovar_value)
    if not np.isfinite(relion_value) or not np.isfinite(recovar_value):
        return {
            "all_finite": False,
            "relion": relion_value if np.isfinite(relion_value) else None,
            "recovar": recovar_value if np.isfinite(recovar_value) else None,
        }
    delta = recovar_value - relion_value
    return {
        "all_finite": True,
        "relion": relion_value,
        "recovar": recovar_value,
        "delta": delta,
        "abs_delta": abs(delta),
        "relative_abs": abs(delta) / max(abs(relion_value), np.finfo(np.float64).tiny),
        "exact_equal": relion_value == recovar_value,
    }


def audit(case_root: Path, recovar_dir: Path | None = None, relion_dir: Path | None = None) -> dict:
    case_root = case_root.resolve()
    recovar_dir = (recovar_dir or case_root / "recovar").resolve()
    relion_dir = (relion_dir or case_root / "relion_ref").resolve()
    results_path = recovar_dir / "refinement_results.npz"
    if not results_path.is_file():
        raise AuditError(f"missing {results_path}")
    with np.load(results_path, allow_pickle=True) as recovar:
        input_particles_path = case_root / "data" / "particles.star"
        reference_names = None
        if input_particles_path.is_file():
            input_particles = _particles(input_particles_path)
            if "rlnImageName" not in input_particles:
                raise AuditError(f"missing rlnImageName in {input_particles_path}")
            reference_names = np.asarray(input_particles["rlnImageName"], dtype=str)
        current_sizes = np.asarray(_rec_array(recovar, "current_sizes"), dtype=np.int64)
        healpix = np.asarray(_rec_array(recovar, "healpix_order_trajectory"), dtype=np.int64)
        n_iterations = int(current_sizes.size)
        if healpix.size != n_iterations:
            raise AuditError(f"healpix trajectory length {healpix.size} != {n_iterations}")
        volume_shape = np.asarray(_rec_array(recovar, "volume_shape"), dtype=np.int64)
        if volume_shape.size != 3 or len(set(volume_shape.tolist())) != 1:
            raise AuditError(f"unexpected volume shape {volume_shape}")
        n4 = float(int(volume_shape[0]) ** 4)
        rows = []
        topology_failures = []
        for index in range(n_iterations):
            relion_iteration = index + 1
            prefix = relion_dir / f"run_it{relion_iteration:03d}"
            model_h1 = _model(Path(f"{prefix}_half1_model.star"))
            model_h2 = _model(Path(f"{prefix}_half2_model.star"))
            general_h1 = _general(model_h1)
            sampling = _sampling_general(Path(f"{prefix}_sampling.star"))
            rel_current_size = int(general_h1["rlnCurrentImageSize"])
            rel_healpix = int(sampling["rlnHealpixOrder"])
            current_size_equal = rel_current_size == int(current_sizes[index])
            healpix_equal = rel_healpix == int(healpix[index])
            meta_path = recovar_dir / "intermediates" / f"it{index:03d}_meta.npy"
            if not meta_path.is_file():
                raise AuditError(f"missing {meta_path}")
            meta = np.load(meta_path, allow_pickle=True).item()
            meta_exact = {
                "iteration": int(meta["iteration"]) == index,
                "current_size": int(meta["current_size"]) == int(current_sizes[index]),
                "healpix_order": int(meta["healpix_order"]) == int(healpix[index]),
            }
            for field, exact in meta_exact.items():
                if not exact:
                    topology_failures.append(f"it{relion_iteration:03d} intermediate meta {field} mismatch")
            if not current_size_equal:
                topology_failures.append(
                    f"it{relion_iteration:03d} current_size RELION={rel_current_size} RECOVAR={current_sizes[index]}"
                )
            if not healpix_equal:
                topology_failures.append(
                    f"it{relion_iteration:03d} healpix RELION={rel_healpix} RECOVAR={healpix[index]}"
                )

            particles = _particles(Path(f"{prefix}_data.star"))
            rel_pmax, particle_identity = _values_in_original_image_order(
                particles, "rlnMaxValueProbDistribution", reference_names
            )
            rec_pmax = _rec_array(recovar, f"pmax_per_image_by_image_iter_{index:03d}")
            rec_sigma = _trajectory_pair(recovar, "sigma_offset_per_half_trajectory", index)
            rel_sigma = np.asarray(
                [
                    float(general_h1["rlnSigmaOffsetsAngst"]),
                    float(_general(model_h2)["rlnSigmaOffsetsAngst"]),
                ],
                dtype=np.float64,
            )

            class_h1 = _class_table(model_h1)
            noise_h1 = _noise_table(model_h1)
            noise_h2 = _noise_table(model_h2)
            rec_noise = _rec_array(recovar, f"noise_radial_per_half_iter_{index:03d}")
            if rec_noise is not None:
                rec_noise = np.asarray(rec_noise, dtype=np.float64)
                if rec_noise.shape[0] != 2:
                    raise AuditError(f"unexpected RECOVAR noise shape at iteration {index}: {rec_noise.shape}")

            shell_metrics = {
                "gold_standard_fsc": _compare_optional(
                    _shell_array(class_h1, "rlnGoldStandardFsc"),
                    _rec_array(recovar, f"fsc_iter_{index:03d}", required=False),
                ),
                "reference_tau2_scaled_n4": _compare_optional(
                    None
                    if _shell_array(class_h1, "rlnReferenceTau2") is None
                    else _shell_array(class_h1, "rlnReferenceTau2") * n4,
                    _rec_array(recovar, f"tau2_radial_iter_{index:03d}", required=False),
                ),
                "reference_sigma2_scaled_n4": _compare_optional(
                    None
                    if _shell_array(class_h1, "rlnReferenceSigma2") is None
                    else _shell_array(class_h1, "rlnReferenceSigma2") * n4,
                    _rec_array(recovar, f"tau2_sigma2_iter_{index:03d}", required=False),
                ),
                "ssnr_map": _compare_optional(
                    _shell_array(class_h1, "rlnSsnrMap"),
                    _rec_array(recovar, f"tau2_ssnr_iter_{index:03d}", required=False),
                ),
                "sigma2_noise_half1_scaled_n4": _compare_optional(
                    _shell_array(noise_h1, "rlnSigma2Noise") * n4,
                    None if rec_noise is None else rec_noise[0],
                ),
                "sigma2_noise_half2_scaled_n4": _compare_optional(
                    _shell_array(noise_h2, "rlnSigma2Noise") * n4,
                    None if rec_noise is None else rec_noise[1],
                ),
            }
            direction_metrics = {
                "half1": _compare_optional(_rel_direction_prior(model_h1), _direction_prior(recovar, index, 1)),
                "half2": _compare_optional(_rel_direction_prior(model_h2), _direction_prior(recovar, index, 2)),
            }
            rec_pmax_mean = float(np.mean(rec_pmax)) if rec_pmax is not None else float("nan")
            row = {
                "recovar_index": index,
                "relion_iteration": relion_iteration,
                "topology": {
                    "current_size": {
                        "relion": rel_current_size,
                        "recovar": int(current_sizes[index]),
                        "exact_equal": current_size_equal,
                    },
                    "healpix_order": {
                        "relion": rel_healpix,
                        "recovar": int(healpix[index]),
                        "exact_equal": healpix_equal,
                    },
                    "intermediate_meta_exact": meta_exact,
                    "n_rotations": int(meta["n_rotations"]),
                    "n_translations": int(meta["n_translations"]),
                    "local_search": bool(meta["local_search"]),
                },
                "pmax_per_particle": _compare_optional(rel_pmax, rec_pmax),
                "particle_identity": particle_identity,
                "pmax_mean": _scalar_pair(float(np.mean(rel_pmax)), rec_pmax_mean),
                "sigma_offset_per_half_angstrom": _compare_optional(rel_sigma, rec_sigma),
                "direction_prior": direction_metrics,
                "shell_arrays": shell_metrics,
            }
            rows.append(row)

        numeric = []
        numeric_artifact_failures = []
        for row in rows:
            for family_name, family in (
                ("direction_prior", row["direction_prior"]),
                ("shell_arrays", row["shell_arrays"]),
            ):
                for name, metrics in family.items():
                    if metrics.get("available") and not metrics.get("shape_equal", False):
                        numeric_artifact_failures.append(
                            f"it{row['relion_iteration']:03d} {family_name}.{name} shape mismatch"
                        )
                    elif metrics.get("available") and not metrics.get("all_finite", False):
                        numeric_artifact_failures.append(
                            f"it{row['relion_iteration']:03d} {family_name}.{name} has non-finite values"
                        )
                    if (
                        metrics.get("available")
                        and metrics.get("shape_equal")
                        and metrics.get("relative_l2") is not None
                    ):
                        numeric.append(
                            {
                                "relion_iteration": row["relion_iteration"],
                                "field": f"{family_name}.{name}",
                                "relative_l2": float(metrics["relative_l2"]),
                                "max_abs": float(metrics["max_abs"]),
                            }
                        )
            pmax = row["pmax_per_particle"]
            if pmax.get("available") and not pmax.get("shape_equal", False):
                numeric_artifact_failures.append(
                    f"it{row['relion_iteration']:03d} pmax_per_particle shape mismatch"
                )
            elif pmax.get("available") and not pmax.get("all_finite", False):
                numeric_artifact_failures.append(
                    f"it{row['relion_iteration']:03d} pmax_per_particle has non-finite values"
                )
            if pmax.get("available") and pmax.get("shape_equal") and pmax.get("relative_l2") is not None:
                numeric.append(
                    {
                        "relion_iteration": row["relion_iteration"],
                        "field": "pmax_per_particle",
                        "relative_l2": float(pmax["relative_l2"]),
                        "max_abs": float(pmax["max_abs"]),
                    }
                )
            sigma = row["sigma_offset_per_half_angstrom"]
            if sigma.get("available") and not sigma.get("shape_equal", False):
                numeric_artifact_failures.append(
                    f"it{row['relion_iteration']:03d} sigma_offset_per_half_angstrom shape mismatch"
                )
            elif sigma.get("available") and not sigma.get("all_finite", False):
                numeric_artifact_failures.append(
                    f"it{row['relion_iteration']:03d} sigma_offset_per_half_angstrom has non-finite values"
                )
            if sigma.get("available") and sigma.get("shape_equal") and sigma.get("relative_l2") is not None:
                numeric.append(
                    {
                        "relion_iteration": row["relion_iteration"],
                        "field": "sigma_offset_per_half_angstrom",
                        "relative_l2": float(sigma["relative_l2"]),
                        "max_abs": float(sigma["max_abs"]),
                    }
                )
        numeric.sort(key=lambda item: item["relative_l2"], reverse=True)
        if topology_failures:
            status = "topology_mismatch"
        elif numeric_artifact_failures:
            status = "numeric_artifact_error"
        else:
            status = "pass"
        return {
            "schema": SCHEMA,
            "status": status,
            "status_scope": (
                "required artifacts, selected iteration topology, and finite shape-compatible numeric arrays; "
                "finite numeric magnitudes are diagnostic and are not threshold-gated"
            ),
            "metric_policy": (
                "exact selected topology and direct array-error metrics for intermediates; no correlation; "
                "map quality is evaluated separately with FSC/FSC-AUC"
            ),
            "paths": {
                "case_root": str(case_root),
                "recovar_dir": str(recovar_dir),
                "relion_dir": str(relion_dir),
                "refinement_results": str(results_path),
            },
            "numbered_iteration_count": n_iterations,
            "topology_failures": topology_failures,
            "earliest_topology_failure": topology_failures[0] if topology_failures else None,
            "numeric_artifact_failures": numeric_artifact_failures,
            "earliest_numeric_artifact_failure": (
                numeric_artifact_failures[0] if numeric_artifact_failures else None
            ),
            "largest_numeric_relative_l2": numeric[:20],
            "numbered_iterations": rows,
        }


def render_markdown(report: dict) -> str:
    lines = [
        "# K=1 intermediate trajectory audit",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "Selected topology and direct array-error metrics are used. Correlation is not computed; map quality is handled by FSC/FSC-AUC.",
        "Finite numeric magnitudes are diagnostic and are not threshold-gated.",
        "",
        "| Iteration | Current size exact | HEALPix exact | Pmax relative L2 | Pmax max abs |",
        "|---:|:---:|:---:|---:|---:|",
    ]
    for row in report.get("numbered_iterations", []):
        pmax = row["pmax_per_particle"]
        relative_l2 = pmax.get("relative_l2")
        max_abs = pmax.get("max_abs")
        lines.append(
            f"| {row['relion_iteration']} | {row['topology']['current_size']['exact_equal']} | "
            f"{row['topology']['healpix_order']['exact_equal']} | "
            f"{'—' if relative_l2 is None else f'{float(relative_l2):.6e}'} | "
            f"{'—' if max_abs is None else f'{float(max_abs):.6e}'} |"
        )
    if report.get("topology_failures"):
        lines.extend(["", "## Topology failures", ""])
        lines.extend(f"- {failure}" for failure in report["topology_failures"])
    if report.get("numeric_artifact_failures"):
        lines.extend(["", "## Numeric artifact failures", ""])
        lines.extend(f"- {failure}" for failure in report["numeric_artifact_failures"])
    lines.extend(["", "## Largest numeric relative-L2 differences", ""])
    for item in report.get("largest_numeric_relative_l2", [])[:10]:
        lines.append(
            f"- Iteration {item['relion_iteration']} `{item['field']}`: "
            f"relative L2 `{item['relative_l2']:.6e}`, max absolute `{item['max_abs']:.6e}`."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--recovar-dir", type=Path)
    parser.add_argument("--relion-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    output_dir = args.case_root.resolve() / "trajectory_analysis"
    output_json = (args.output_json or output_dir / "k1_intermediate_trajectory.json").resolve()
    output_markdown = (args.output_markdown or output_dir / "k1_intermediate_trajectory.md").resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = audit(args.case_root, args.recovar_dir, args.relion_dir)
    except AuditError as exc:
        report = {
            "schema": SCHEMA,
            "status": "error",
            "metric_policy": "exact/array intermediate metrics; no correlation; FSC/FSC-AUC for maps",
            "topology_failures": [str(exc)],
            "earliest_topology_failure": str(exc),
            "numbered_iterations": [],
        }
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output_markdown.write_text(render_markdown(report))
    print(json.dumps({"status": report["status"], "output": str(output_json)}))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
