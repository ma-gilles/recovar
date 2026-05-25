#!/usr/bin/env python3
"""Run DynaMight optimize_deformations without importing its PyQt visualizer.

The RELION 5.0.1 DynaMight console script imports ``dynamight.__init__`` before
dispatching the training command. On this cluster that import pulls in a PyQt
visualization module with a broken Qt symbol, even for non-visual training. This
wrapper installs a namespace package for DynaMight and imports the training
module directly.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


DEFAULT_SITE = Path(
    "/projects/MOLBIO/local/pythonenv/relion-5.0.1-rhel9/lib/python3.10/site-packages"
)


def load_optimize_deformations(site_packages: Path):
    package_dir = site_packages / "dynamight"
    if not package_dir.exists():
        raise FileNotFoundError(package_dir)
    package = types.ModuleType("dynamight")
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules["dynamight"] = package
    from dynamight.deformations.optimize_deformations import optimize_deformations

    return optimize_deformations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--refinement-star-file", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--mask-file", type=Path, required=True)
    parser.add_argument("--n-latent-dimensions", type=int, required=True)
    parser.add_argument("--n-gaussians", type=int, default=8000)
    parser.add_argument("--initial-resolution", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=150)
    args = parser.parse_args()

    optimize_deformations = load_optimize_deformations(args.site_packages)
    optimize_deformations(
        refinement_star_file=args.refinement_star_file,
        output_directory=args.output_directory,
        initial_model=args.initial_model,
        mask_file=args.mask_file,
        n_latent_dimensions=args.n_latent_dimensions,
        n_gaussians=args.n_gaussians,
        initial_resolution=args.initial_resolution,
        batch_size=args.batch_size,
        n_threads=args.n_threads,
        n_workers=args.n_workers,
        gpu_id=args.gpu_id,
        n_epochs=args.n_epochs,
        preload_images=False,
    )


if __name__ == "__main__":
    main()

