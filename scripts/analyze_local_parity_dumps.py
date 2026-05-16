#!/usr/bin/env python
"""Analyze one RELION operand dump against one RECOVAR local score dump."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_parity_analysis_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "recovar" / "utils" / "local_parity_analysis.py"
    spec = importlib.util.spec_from_file_location("local_parity_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parity = _load_parity_analysis_module()


def _load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _resolve_image_position(recovar, image_position, global_index):
    if global_index is not None:
        selected = np.asarray(recovar["selected_global_image_indices"], dtype=np.int64)
        matches = np.flatnonzero(selected == int(global_index))
        if matches.size == 0:
            raise ValueError(f"Global image index {global_index} not present in {selected.tolist()}")
        return int(matches[0])
    return int(image_position)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion", required=True, help="RELION operand NPZ")
    parser.add_argument("--recovar", required=True, help="RECOVAR local score-dump NPZ")
    parser.add_argument(
        "--mapping-recovar",
        default=None,
        help="Optional RECOVAR score dump whose support metadata is used for RELION candidate mapping",
    )
    parser.add_argument("--image-position", type=int, default=0, help="Selected-image position within the RECOVAR dump")
    parser.add_argument("--global-index", type=int, default=None, help="Selected global image index within the RECOVAR dump")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--candidate-csv-out", type=str, default=None, help="Optional per-candidate CSV output path")
    args = parser.parse_args()

    relion_raw = _load_npz(args.relion)
    recovar_raw = _load_npz(args.recovar)
    mapping_raw = _load_npz(args.mapping_recovar) if args.mapping_recovar is not None else recovar_raw
    image_position = _resolve_image_position(recovar_raw, args.image_position, args.global_index)
    mapping_image_position = _resolve_image_position(mapping_raw, args.image_position, args.global_index)

    relion_summary = parity.summarize_relion_operands(relion_raw)
    recovar_summary = parity.summarize_recovar_score_dump(recovar_raw, image_position=image_position)
    comparison = parity.compare_relion_recovar_pmax(relion_summary, recovar_summary)
    comparison["scale_to_match_relion_full_pmax"] = float(
        parity.solve_scale_for_target_pmax(recovar_raw["pass2_scores_total"][image_position], relion_summary["pmax"])
    )
    comparison["scale_to_match_relion_raw_pmax"] = float(
        parity.solve_scale_for_target_pmax(recovar_raw["pass2_scores_raw"][image_position], relion_summary["pmax"])
    )

    mapping = None
    mask_ladder = None
    score_deltas = None
    candidate_table = None
    component_summary = None
    if "local_rotation_pixel_indices" in mapping_raw and "local_rotation_psi_indices" in mapping_raw:
        mapping = parity.build_relion_recovar_candidate_mapping(
            relion_raw,
            mapping_raw,
            image_position=mapping_image_position,
        )
        mask_ladder = parity.summarize_recovar_mask_ladder(recovar_raw, mapping, image_position=image_position)
        score_deltas = parity.compare_shared_subset_score_deltas(
            relion_raw,
            recovar_raw,
            mapping,
            image_position=image_position,
        )
        candidate_table = parity.build_shared_subset_candidate_table(
            relion_raw,
            recovar_raw,
            mapping,
            image_position=image_position,
        )
        component_summary = parity.summarize_candidate_table_components(candidate_table)

    print("=== RELION semantics ===")
    print(f"  stored Pmax           : {relion_summary['stored_pmax']:.12f}")
    print(f"  max(raw)/sum(raw)     : {relion_summary['pmax_from_raw']:.12f}")
    print(f"  semantics gap         : {relion_summary['pmax_semantics_gap']:+.3e}")
    print(
        "  membership counts     : "
        f"D={relion_summary['denominator_count']} "
        f"S={relion_summary['fine_threshold_count']} "
        f"R={relion_summary['reconstruction_count']}"
    )
    print(
        "  threshold metadata    : "
        f"adaptive_fraction={relion_summary['adaptive_fraction']:.6f} "
        f"maximum_significants={relion_summary['maximum_significants']} "
        f"threshold_idx={relion_summary['candidate_threshold_idx']}"
    )

    print("\n=== RECOVAR semantics ===")
    print(f"  selected image        : pos={image_position} global={recovar_summary['selected_global_image_index']}")
    print(f"  saved Pmax            : {recovar_summary['saved_pmax']:.12f}")
    print(f"  recomputed Pmax       : {recovar_summary['recomputed_pmax']:.12f}")
    print(f"  saved log_Z           : {recovar_summary['saved_log_Z']:.6f}")
    print(f"  recomputed log_Z      : {recovar_summary['recomputed_log_Z']:.6f}")
    print(
        "  support / entropy     : "
        f"rot={recovar_summary['support_rotations']} "
        f"pairs<= {recovar_summary['support_pairs_upper_bound']} "
        f"effective={recovar_summary['full']['effective_support']:.3f} "
        f"entropy={recovar_summary['full']['entropy']:.6f}"
    )
    if recovar_summary["pass_raw_max_abs_diff"] is not None:
        print(f"  pass1-pass2 raw diff  : {recovar_summary['pass_raw_max_abs_diff']:.3e}")
    if recovar_summary["cross_norm_max_abs_diff"] is not None:
        print(f"  cross+norm consistency: {recovar_summary['cross_norm_max_abs_diff']:.3e}")

    print("\n=== RECOVAR decomposition ===")
    print(
        "  raw only              : "
        f"pmax={recovar_summary['raw_only']['pmax']:.6f} "
        f"top2={recovar_summary['raw_only']['top2_mass']:.6f} "
        f"gap12={recovar_summary['raw_only']['best_minus_runner_up']:.6f}"
    )
    print(
        "  raw + rot prior       : "
        f"pmax={recovar_summary['raw_plus_rotation_prior']['pmax']:.6f} "
        f"top2={recovar_summary['raw_plus_rotation_prior']['top2_mass']:.6f} "
        f"gap12={recovar_summary['raw_plus_rotation_prior']['best_minus_runner_up']:.6f}"
    )
    print(
        "  full posterior        : "
        f"pmax={recovar_summary['full']['pmax']:.6f} "
        f"top2={recovar_summary['full']['top2_mass']:.6f} "
        f"gap12={recovar_summary['full']['best_minus_runner_up']:.6f}"
    )

    if mapping is not None and mask_ladder is not None:
        print("\n=== Support Mapping ===")
        print(
            "  factorized support    : "
            f"pixel_equal={mapping.pixel_support_equal} "
            f"psi_equal={mapping.psi_support_equal} "
            f"relion_candidates={mapping.relion_rot_id.size}"
        )

        print("\n=== Mask Ladder ===")
        for label in [
            "full",
            "denominator_under_full",
            "threshold_under_full",
            "reconstruction_under_full",
            "denominator",
            "threshold_denorm",
            "threshold_renorm",
            "raw_denominator",
        ]:
            item = mask_ladder[label]
            print(
                f"  {label:20s}: "
                f"support={item['support_size']} "
                f"norm={item['normalization_support_size']} "
                f"mass={item['support_mass']:.6f} "
                f"pmax_norm={item['normalized_pmax']:.6f} "
                f"pmax_renorm={item['renormalized']['pmax']:.6f}"
            )

        print("\n=== Shared-Subset Score Deltas ===")
        print(
            "  best candidate        : "
            f"rot_id={score_deltas['best_relion_rot_id']} "
            f"trans_idx={score_deltas['best_relion_trans_idx']}"
        )
        print(
            "  fit / error           : "
            f"corr={score_deltas['corr']:.6f} "
            f"slope={score_deltas['slope']:.6f} "
            f"r2={score_deltas['r2']:.6f} "
            f"mean_abs={score_deltas['mean_abs_delta_error']:.6f} "
            f"max_abs={score_deltas['max_abs_delta_error']:.6f}"
        )
        if candidate_table is not None:
            print(
                "  candidate table       : "
                f"n={len(candidate_table['candidate_index'])} "
                f"relion_best={int(candidate_table['best_relion_candidate_index'])} "
                f"recovar_best={int(candidate_table['best_recovar_candidate_index'])}"
            )
        if component_summary is not None:
            print("\n=== Shared-Subset Component Errors ===")
            print(
                "  best candidate agree  : "
                f"{component_summary['same_best_candidate']} "
                f"(relion={component_summary['best_relion_candidate_index']} "
                f"recovar={component_summary['best_recovar_candidate_index']})"
            )
            for label in ["total_delta_error", "data_delta_error", "prior_delta_error", "prior_level_error"]:
                item = component_summary[label]
                print(
                    f"  {label:20s}: "
                    f"mean_abs={item['mean_abs']:.6f} "
                    f"rms={item['rms']:.6f} "
                    f"max_abs={item['max_abs']:.6f}"
                )

    print("\n=== Comparison ===")
    print(f"  RELION Pmax           : {comparison['relion_pmax']:.12f}")
    print(f"  RECOVAR Pmax          : {comparison['recovar_pmax']:.12f}")
    print(f"  gap (recovar-relion)  : {comparison['gap']:+.6f}")
    print(f"  RECOVAR raw-only Pmax : {comparison['recovar_raw_only_pmax']:.12f}")
    print(f"  prior delta on Pmax   : {comparison['prior_delta_pmax']:+.6f}")
    print(f"  scale to match full   : {comparison['scale_to_match_relion_full_pmax']:.6f}")
    print(f"  scale to match raw    : {comparison['scale_to_match_relion_raw_pmax']:.6f}")

    payload = {
        "relion": relion_summary,
        "recovar": recovar_summary,
        "comparison": comparison,
    }
    if mapping is not None and mask_ladder is not None:
        payload["mapping"] = {
            "pixel_support_equal": mapping.pixel_support_equal,
            "psi_support_equal": mapping.psi_support_equal,
            "relion_rot_id": mapping.relion_rot_id.tolist(),
            "relion_trans_idx": mapping.relion_trans_idx.tolist(),
            "relion_coarse_trans_idx": mapping.relion_coarse_trans_idx.tolist(),
            "recovar_rot_slot": mapping.recovar_rot_slot.tolist(),
            "recovar_trans_idx": mapping.recovar_trans_idx.tolist(),
        }
        payload["mask_ladder"] = mask_ladder
        payload["score_deltas"] = score_deltas
        payload["candidate_table"] = candidate_table
        payload["component_summary"] = component_summary
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        print(f"\nJSON summary written to {out_path}")
    if args.candidate_csv_out is not None and candidate_table is not None:
        out_path = Path(args.candidate_csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(candidate_table.keys())
        row_count = len(candidate_table["candidate_index"])
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(row_count):
                row = {}
                for key in fieldnames:
                    value = candidate_table[key]
                    if np.ndim(value) == 0:
                        row[key] = value.item() if hasattr(value, "item") else value
                    else:
                        item = value[idx]
                        row[key] = item.item() if hasattr(item, "item") else item
                writer.writerow(row)
        print(f"Candidate CSV written to {out_path}")


if __name__ == "__main__":
    main()
