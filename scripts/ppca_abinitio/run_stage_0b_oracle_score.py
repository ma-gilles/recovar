"""Stage 0B — oracle-score falsification.

Per spec Section 11.2, this is the first scientific gate of the
PPCA-ab-initio v0 project. It runs the score-only diagnostic with
the **true** `(mu, U, s)` and asks: does the marginalized PPCA
score rank the true pose better than the homogeneous score?

Exit criterion (all of):

1. On family **B** (matched-grid heterogeneous), PPCA improves
   validation `true_state_mass` over homogeneous by an amount whose
   95% bootstrap CI excludes zero, for **all 3 seeds**.
2. On family **A** (null), the absolute change in validation
   `true_state_mass` is `≤ 0.01` for all 3 seeds.
3. On family **C** (misspecified pose), PPCA gain is reduced but
   sign does not flip and magnitude ≥ 25% of the family-B gain.
   **Family C is not implemented in v0**; this exit clause is
   deferred until `synthetic.py` adds it.

If 1 or 2 fails, **stop the project**.

Output is JSON written to the path passed as `--out` (or printed
to stdout if no `--out` is given). The JSON has one entry per
(family, seed) and a top-level `passed` field summarizing the
exit-criterion check.

Run with toy sizes (default) for a quick sanity check, or with
larger sizes for a real Stage 0B gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.init import init_oracle
from recovar.em.ppca_abinitio.loop import run_score_diagnostic
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity-CTF / identity-process forward model config (no real dataset
# needed for the v0 score gate).
# ---------------------------------------------------------------------------


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(batch, apply_image_mask=False):
    return batch


class _SyntheticConfig(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    _ctf: object = eqx.field(static=True)
    _process: object = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, ctf_params, *, half_image=False):
        full = self._ctf(ctf_params, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, batch, apply_image_mask=False):
        return self._process(batch, apply_image_mask=apply_image_mask)


def _make_config(image_shape, volume_shape):
    return _SyntheticConfig(
        image_shape=tuple(image_shape),
        volume_shape=tuple(volume_shape),
        _ctf=_identity_ctf,
        _process=_identity_process,
        voxel_size=1.0,
    )


# ---------------------------------------------------------------------------
# Run one (family, seed) and return a JSON-friendly record
# ---------------------------------------------------------------------------


_S_FLOOR = 1e-6  # see _run_one docstring


def _run_one(
    family,
    seed,
    *,
    volume_shape,
    image_shape,
    grid,
    q,
    n_train,
    n_val,
    sigma_real,
    n_bootstrap,
):
    """Run one (family, seed). For family A (null), the synthetic
    `s_true = 0`, which makes `diag(1/s)` infinite and breaks the
    Cholesky in the PPCA branch. We clamp `s` from below by
    `_S_FLOOR` (~1e-6) when feeding the kernel: this is the
    "degenerate-PPCA" model whose score should be indistinguishable
    from the homogeneous score on null data, which is the actual
    Stage 0B null check (PPCA must NOT beat homogeneous on null
    data). The synthetic ground truth `s_true` is unchanged on the
    `SyntheticDataset` object — only the `init.s` fed to the kernel
    is clamped.
    """
    ds = make_synthetic_fixed_grid_dataset(
        family,
        volume_shape=volume_shape,
        image_shape=image_shape,
        grid=grid,
        q=q,
        n_images_train=n_train,
        n_images_val=n_val,
        sigma_real=sigma_real,
        seed=seed,
    )
    config = _make_config(image_shape, volume_shape)
    s_for_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    init = init_oracle(
        mu_half_true=ds.mu_half_true,
        U_half_true=ds.U_half_true,
        s_true=s_for_kernel,
        volume_shape=volume_shape,
    )
    res = run_score_diagnostic(config, ds, init, seed=seed, n_bootstrap=n_bootstrap)
    diag = res.diagnostic
    return {
        "family": diag.family,
        "seed": diag.seed,
        "n_val": diag.n_val,
        "homog": asdict(diag.homog_true_state_mass),
        "ppca": asdict(diag.ppca_true_state_mass),
        "delta": asdict(diag.delta_true_state_mass),
    }


# ---------------------------------------------------------------------------
# Aggregate Stage 0B exit criterion check
# ---------------------------------------------------------------------------


def evaluate_stage_0b_exit_criterion(records, *, family_b_min_gain_ci=0.0, family_a_max_abs_delta=0.01):
    """Per spec Section 11.2:

    1. Family B: delta CI excludes zero (low > family_b_min_gain_ci) on all seeds.
    2. Family A: |delta.mean| <= family_a_max_abs_delta on all seeds.

    Returns a dict with `passed` (bool) and the per-clause checks.
    """
    by_family = {"A": [], "B": []}
    for r in records:
        if r["family"] in by_family:
            by_family[r["family"]].append(r)

    fam_b_ok = True
    fam_b_details = []
    for r in by_family["B"]:
        ci_low = r["delta"]["ci_low"]
        ci_excludes_zero = bool(ci_low > family_b_min_gain_ci)
        fam_b_ok = fam_b_ok and ci_excludes_zero
        fam_b_details.append({"seed": r["seed"], "ci_low": ci_low, "ok": ci_excludes_zero})

    fam_a_ok = True
    fam_a_details = []
    for r in by_family["A"]:
        delta_mean = r["delta"]["mean"]
        within = bool(abs(delta_mean) <= family_a_max_abs_delta)
        fam_a_ok = fam_a_ok and within
        fam_a_details.append({"seed": r["seed"], "delta_mean": delta_mean, "ok": within})

    passed = fam_b_ok and fam_a_ok and (len(by_family["A"]) > 0) and (len(by_family["B"]) > 0)
    return {
        "passed": passed,
        "family_B": {"all_seeds_pass": fam_b_ok, "per_seed": fam_b_details},
        "family_A": {"all_seeds_pass": fam_a_ok, "per_seed": fam_a_details},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="Path to write JSON output. If omitted, print to stdout.")
    parser.add_argument("--volume-size", type=int, default=8, help="Cubic volume side length.")
    parser.add_argument("--healpix-order", type=int, default=2)
    parser.add_argument("--max-shift", type=int, default=1)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--sigma-real", type=float, default=0.5)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    volume_shape = (args.volume_size, args.volume_size, args.volume_size)
    image_shape = (args.volume_size, args.volume_size)
    grid = build_fixed_grid(healpix_order=args.healpix_order, max_shift=args.max_shift)

    records = []
    for family in (SyntheticFamily.NULL, SyntheticFamily.MATCHED_GRID_HET):
        for seed in args.seeds:
            logger.info("running family=%s seed=%d", family.value, seed)
            rec = _run_one(
                family,
                seed,
                volume_shape=volume_shape,
                image_shape=image_shape,
                grid=grid,
                q=args.q,
                n_train=args.n_train,
                n_val=args.n_val,
                sigma_real=args.sigma_real,
                n_bootstrap=args.n_bootstrap,
            )
            records.append(rec)
            logger.info(
                "  family=%s seed=%d  delta_mean=%.4f  delta_ci=[%.4f, %.4f]",
                rec["family"],
                rec["seed"],
                rec["delta"]["mean"],
                rec["delta"]["ci_low"],
                rec["delta"]["ci_high"],
            )

    exit_check = evaluate_stage_0b_exit_criterion(records)
    output = {
        "stage": "0B",
        "config": {
            "volume_shape": list(volume_shape),
            "image_shape": list(image_shape),
            "healpix_order": args.healpix_order,
            "max_shift": args.max_shift,
            "q": args.q,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "sigma_real": args.sigma_real,
            "n_bootstrap": args.n_bootstrap,
            "seeds": args.seeds,
        },
        "records": records,
        "exit_criterion": exit_check,
    }

    text = json.dumps(output, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        logger.info("wrote %s", args.out)
    else:
        print(text)

    return 0 if exit_check["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
