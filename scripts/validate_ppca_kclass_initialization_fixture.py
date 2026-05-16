#!/usr/bin/env python
"""Validate K-class-to-PPCA initialization on a real parity fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovar.em.ppca_refinement.fixture_validation import validate_kclass_to_ppca_initialization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, help="K-class parity summary.json from run_k_class_parity.py")
    parser.add_argument("--class-manifest-json", help="Optional CryoBench/PDB class_manifest.json for GT comparison")
    parser.add_argument("--output-json", required=True, help="Path for validation summary JSON")
    parser.add_argument("--save-initialization-npz", help="Optional path for mu/W/weights arrays")
    parser.add_argument("--q", type=int, default=None, help="Number of PPCA components; default K-1")
    parser.add_argument("--map-key", default="output_maps", help="Summary key containing K-class volume paths")
    parser.add_argument(
        "--kclass-frame",
        choices=("relion", "recovar"),
        default="relion",
        help="Frame of K-class MRC paths. run_k_class_parity.py writes RELION-frame MRCs.",
    )
    parser.add_argument(
        "--gt-frame",
        choices=("recovar", "relion"),
        default="recovar",
        help="Frame of optional GT manifest MRC paths. CryoBench PDB manifests use RECOVAR-frame MRCs.",
    )
    parser.add_argument(
        "--weight-source",
        choices=("recovar", "relion", "uniform"),
        default="recovar",
        help="Class occupancy weights for PPCA initialization.",
    )
    parser.add_argument("--covariance-trace-rtol", type=float, default=1e-5)
    parser.add_argument("--double-load-rtol", type=float, default=1e-6)
    parser.add_argument("--double-load-subspace-atol", type=float, default=1e-4)
    parser.add_argument("--min-explained-covariance", type=float, default=0.999999)
    parser.add_argument("--min-gt-mean-correlation", type=float, default=None)
    parser.add_argument("--min-gt-subspace-agreement", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validation, init, gt_init = validate_kclass_to_ppca_initialization(
        args.summary_json,
        q=args.q,
        map_key=args.map_key,
        kclass_frame=args.kclass_frame,
        weight_source=args.weight_source,
        class_manifest_json=args.class_manifest_json,
        gt_frame=args.gt_frame,
        covariance_trace_rtol=args.covariance_trace_rtol,
        double_load_rtol=args.double_load_rtol,
        double_load_subspace_atol=args.double_load_subspace_atol,
        min_explained_covariance=args.min_explained_covariance,
        min_gt_mean_correlation=args.min_gt_mean_correlation,
        min_gt_subspace_agreement=args.min_gt_subspace_agreement,
    )

    output = {"passed": validation.passed, "failures": list(validation.failures), **validation.summary}
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))

    if args.save_initialization_npz:
        npz_path = Path(args.save_initialization_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "mu": np.asarray(init.mu),
            "W": np.asarray(init.W),
            "weights": np.asarray(init.weights),
        }
        if gt_init is not None:
            save_kwargs.update(
                {
                    "gt_mu": np.asarray(gt_init.mu),
                    "gt_W": np.asarray(gt_init.W),
                    "gt_weights": np.asarray(gt_init.weights),
                }
            )
        np.savez_compressed(npz_path, **save_kwargs)
        print(f"saved_initialization_npz={npz_path}")

    if not validation.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
