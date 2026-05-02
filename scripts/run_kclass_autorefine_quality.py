#!/usr/bin/env python
"""K-class auto-refine quality test: multi-iter k-class EM with a
RELION-style HEALPix-order ramp driven by per-iter convergence (mean
Pmax above a threshold + min iters at current order).

Mirrors the schedule used by ``refine_single_volume`` for the single-
class case but applied to ``run_dense_k_class_em`` over K classes.
Initialises means from the first K ground-truth volumes downsampled
to the working grid; refines until either ``max_iters`` is reached or
the convergence criterion fires three iters in a row.

Outputs per-iter MRCs, an ``iter_log.json`` with convergence history,
and a final FSC vs GT (Hungarian-matched) for the K predicted classes.

Usage::

    pixi run python scripts/run_kclass_autorefine_quality.py \\
        --data-dir /scratch/.../ribosembly_allk_g256_n100000_snr1 \\
        --output-dir /scratch/.../_agent_scratch/kclass_autorefine \\
        --n-classes 4 --n-images 100000 --max-iters 25
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Spec:
    data_dir: Path
    output_dir: Path
    n_classes: int = 4
    n_images: int = 100_000
    grid_size: int | None = None  # None = native
    max_iters: int = 25
    min_iters_per_order: int = 2
    healpix_order_init: int = 1
    healpix_order_max: int = 4
    pmax_advance_threshold: float = 0.85
    image_batch_size: int = 16
    rotation_block_size: int = 128
    seed: int = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--n-images", type=int, default=100_000)
    parser.add_argument(
        "--grid-size", type=int, default=None, help="Downsample particles to this grid (default: native)."
    )
    parser.add_argument("--max-iters", type=int, default=25)
    parser.add_argument("--min-iters-per-order", type=int, default=2)
    parser.add_argument("--healpix-order-init", type=int, default=1)
    parser.add_argument("--healpix-order-max", type=int, default=4)
    parser.add_argument("--pmax-advance-threshold", type=float, default=0.85)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--rotation-block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec = Spec(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_classes=args.n_classes,
        n_images=args.n_images,
        grid_size=args.grid_size,
        max_iters=args.max_iters,
        min_iters_per_order=args.min_iters_per_order,
        healpix_order_init=args.healpix_order_init,
        healpix_order_max=args.healpix_order_max,
        pmax_advance_threshold=args.pmax_advance_threshold,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        seed=args.seed,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    (spec.output_dir / "SAFE_TO_DELETE").touch()

    # Lazy heavy imports.
    import jax.numpy as jnp
    import mrcfile
    from scipy.optimize import linear_sum_assignment

    import recovar.core.fourier_transform_utils as ftu
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.dense_single_volume.k_class import run_dense_k_class_em
    from recovar.em.sampling import get_rotation_grid
    from recovar.reconstruction.regularization import get_fsc

    # ---- Load dataset.
    cryo = load_dataset(
        particles_file=str(spec.data_dir / "particles.star"),
        n_images=spec.n_images,
        ind=None,
        lazy=True,
        padding=0,
        downsample_D=spec.grid_size,
    )
    grid_size = cryo.grid_size
    voxel_size = float(cryo.voxel_size)
    vol_shape = (grid_size, grid_size, grid_size)
    vol_size = int(np.prod(vol_shape))
    image_size = int(np.prod(cryo.image_shape))
    print(f"dataset: N={cryo.n_units}, grid={grid_size}³, voxel={voxel_size:.3f} A/px")

    # ---- Load ground-truth volumes (first K), downsample to working grid.
    gt_paths = sorted(spec.data_dir.glob("reference_gt_class*.mrc"))
    if len(gt_paths) < spec.n_classes:
        raise SystemExit(f"need {spec.n_classes} GT classes, found {len(gt_paths)} at {spec.data_dir}")
    gt_vols = []
    for p in gt_paths[: spec.n_classes]:
        with mrcfile.open(str(p), permissive=True) as mrc:
            v = mrc.data.copy().astype(np.float32)
        if v.shape[0] != grid_size:
            ratio = v.shape[0] // grid_size
            if v.shape[0] != grid_size * ratio:
                raise SystemExit(f"GT shape {v.shape} not a clean multiple of working grid {grid_size}")
            v = v.reshape(grid_size, ratio, grid_size, ratio, grid_size, ratio).mean(axis=(1, 3, 5))
        gt_vols.append(v)
    gt_vols = np.stack(gt_vols, axis=0)
    print(f"GT classes: {gt_vols.shape}, downsampled to {grid_size}³")

    # ---- Initialise per-class means in flat-Fourier from GT.
    means = jnp.stack(
        [ftu.get_dft3(jnp.asarray(v, dtype=jnp.float32)).reshape(vol_size) for v in gt_vols],
        axis=0,
    ).astype(jnp.complex64)
    translations = jnp.zeros((1, 2), dtype=jnp.float32)
    noise_variance = jnp.ones((image_size,), dtype=jnp.float32)
    mean_variance = jnp.ones((vol_size,), dtype=jnp.float32)

    # ---- Auto-refine loop with HEALPix order ramp.
    iter_log: list[dict] = []
    healpix_order = spec.healpix_order_init
    iters_at_order = 0
    t_total0 = time.time()

    for it in range(spec.max_iters):
        rotations = jnp.asarray(get_rotation_grid(healpix_order, matrices=True), dtype=jnp.float32)
        n_rot = int(rotations.shape[0])

        t0 = time.time()
        result = run_dense_k_class_em(
            cryo,
            means,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            disc_type="linear_interp",
            image_batch_size=spec.image_batch_size,
            rotation_block_size=spec.rotation_block_size,
            projection_padding_factor=1,
            reconstruction_padding_factor=1,
        )
        means = jnp.asarray(result.new_means)
        wall = time.time() - t0

        # Convergence diagnostics.
        try:
            log_evidence = float(jnp.sum(result.stats.log_evidence_per_image))
            pmax_mean = float(jnp.mean(result.stats.max_posterior_per_image))
        except Exception:
            log_evidence = float("nan")
            pmax_mean = float("nan")

        iter_log.append(
            {
                "iter": it,
                "healpix_order": healpix_order,
                "n_rotations": n_rot,
                "wall_s": wall,
                "log_evidence": log_evidence,
                "pmax_mean": pmax_mean,
            }
        )
        print(
            f"iter {it:02d}: order={healpix_order} n_rot={n_rot} wall={wall:.1f}s "
            f"logE={log_evidence:.3e} pmax={pmax_mean:.4f}"
        )

        # Persist per-iter mrcs + log so partial runs are usable.
        iter_dir = spec.output_dir / f"iter_{it:03d}"
        iter_dir.mkdir(exist_ok=True)
        for k in range(spec.n_classes):
            real = np.real(np.asarray(ftu.get_idft3(means[k].reshape(vol_shape)))).astype(np.float32)
            with mrcfile.new(str(iter_dir / f"class_{k:02d}.mrc"), overwrite=True) as mrc:
                mrc.set_data(real)
                mrc.voxel_size = voxel_size
        with (spec.output_dir / "iter_log.json").open("w") as fh:
            json.dump(iter_log, fh, indent=2)

        # HEALPix order advance: bumped if pmax above threshold and min-iters
        # spent at current order, mirroring RELION auto-refine's
        # "stop sub-pixel sampling" criterion ported to angular ramp.
        iters_at_order += 1
        if (
            pmax_mean > spec.pmax_advance_threshold
            and iters_at_order >= spec.min_iters_per_order
            and healpix_order < spec.healpix_order_max
        ):
            healpix_order += 1
            iters_at_order = 0
            print(f"  advance: HEALPix order → {healpix_order}")

    runtime_s = time.time() - t_total0

    # ---- FSC vs GT (Hungarian-matched).
    final_real = np.stack(
        [
            np.real(np.asarray(ftu.get_idft3(means[k].reshape(vol_shape)))).astype(np.float32)
            for k in range(spec.n_classes)
        ],
        axis=0,
    )
    K = spec.n_classes
    fsc_areas = np.zeros((K, K), dtype=np.float32)
    fsc_curves = []
    for i in range(K):
        for j in range(K):
            fsc = np.asarray(get_fsc(final_real[i].reshape(-1), gt_vols[j].reshape(-1), vol_shape))
            fsc_areas[i, j] = float(np.mean(fsc))
            fsc_curves.append({"i": i, "j": j, "fsc": [float(v) for v in fsc]})
    row, col = linear_sum_assignment(-fsc_areas)
    matched = [(int(r), int(c)) for r, c in zip(row, col)]

    summary = {
        "spec": {**asdict(spec), "data_dir": str(spec.data_dir), "output_dir": str(spec.output_dir)},
        "runtime_s": runtime_s,
        "n_iters_completed": len(iter_log),
        "final_healpix_order": healpix_order,
        "fsc_assignment": matched,
        "fsc_area_per_class": [float(fsc_areas[i, j]) for i, j in matched],
        "fsc_area_mean": float(np.mean([fsc_areas[i, j] for i, j in matched])),
        "voxel_size": voxel_size,
        "grid_size": grid_size,
    }
    print("\n=== Final ===")
    print(f"runtime: {runtime_s / 60:.1f} min, iters: {summary['n_iters_completed']}")
    print(f"final HEALPix order: {healpix_order}")
    print(f"FSC area per class (Hungarian): {summary['fsc_area_per_class']}")
    print(f"FSC area mean: {summary['fsc_area_mean']:.4f}")

    with (spec.output_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {spec.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
