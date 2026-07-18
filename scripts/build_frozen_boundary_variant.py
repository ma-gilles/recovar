#!/usr/bin/env python
"""Build or validate a sealed single-component frozen-boundary variant."""

from __future__ import annotations

import argparse
import json

from recovar.em.dense_single_volume.frozen_boundary_variant import (
    build_frozen_boundary_variant,
    validate_frozen_boundary_variant,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--base-boundary-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--component", choices=("tau2",))
    parser.add_argument("--component-source")
    parser.add_argument("--source-results")
    args = parser.parse_args()

    if args.validate:
        forbidden = {
            "--base-boundary-dir": args.base_boundary_dir,
            "--component": args.component,
            "--component-source": args.component_source,
            "--source-results": args.source_results,
        }
        supplied = sorted(name for name, value in forbidden.items() if value is not None)
        if supplied:
            parser.error(f"--validate cannot be combined with {', '.join(supplied)}")
        attestation = validate_frozen_boundary_variant(args.output_dir)
    else:
        required = {
            "--base-boundary-dir": args.base_boundary_dir,
            "--component": args.component,
            "--component-source": args.component_source,
            "--source-results": args.source_results,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            parser.error(f"build mode requires {', '.join(missing)}")
        attestation = build_frozen_boundary_variant(
            base_boundary_dir=args.base_boundary_dir,
            output_dir=args.output_dir,
            component=args.component,
            component_source=args.component_source,
            source_results=args.source_results,
        )
    print(json.dumps(attestation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
