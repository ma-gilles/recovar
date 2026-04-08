"""Stage 1A — non-oracle score stress test (per spec Section 11.3).

After Stage 0B verifies that the oracle factors give a real PPCA gain
on family B, Stage 1A asks: does the gain survive when the factors
are not oracle?

Two initializations:

- **truth-perturbed** (positive control): real-space perturbation of
  the ground-truth `(mu, U)` with `eps_mu`, `eps_U` ~ 0.1. Per spec
  Section 11.3, this MUST preserve a positive PPCA-over-homogeneous
  score gain on family B with bootstrap CI excluding zero. This is
  the gating clause.

- **random-lowpass** (stress control): random low-pass real volumes
  for `U`, zero `mu`. Reported but **not** gating. Per spec, failure
  here is informative; success here does not mean the bootstrap
  problem is solved.

Output JSON has both records per (family, seed, init), with a
top-level `passed` field that reflects only the truth-perturbed
gating clause.
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
from recovar.em.ppca_abinitio.init import (
    init_random_lowpass,
    init_truth_perturbed,
)
from recovar.em.ppca_abinitio.loop import run_score_diagnostic
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)

logger = logging.getLogger(__name__)


_S_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Identity-CTF / identity-process forward model config
# (mirrors run_stage_0b_oracle_score.py for now)
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
# Single (family, seed, init_kind) run
# ---------------------------------------------------------------------------


def _make_init(init_kind, ds, *, eps_mu, eps_U, k_max, init_seed):
    s_for_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    if init_kind == "truth_perturbed":
        return init_truth_perturbed(
            mu_half_true=ds.mu_half_true,
            U_half_true=ds.U_half_true,
            s_true=s_for_kernel,
            volume_shape=ds.volume_shape,
            eps_mu=eps_mu,
            eps_U=eps_U,
            seed=init_seed,
        )
    if init_kind == "random_lowpass":
        return init_random_lowpass(
            volume_shape=ds.volume_shape,
            q=ds.q,
            k_max=k_max,
            s_init=s_for_kernel,
            seed=init_seed,
            orthonormalize=True,
        )
    raise ValueError(f"unknown init_kind: {init_kind}")


def _run_one(
    family,
    seed,
    init_kind,
    *,
    volume_shape,
    image_shape,
    grid,
    q,
    n_train,
    n_val,
    sigma_real,
    eps_mu,
    eps_U,
    k_max,
    n_bootstrap,
):
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
    # Use a different seed for init to avoid coupling synth seed with init noise
    init = _make_init(init_kind, ds, eps_mu=eps_mu, eps_U=eps_U, k_max=k_max, init_seed=seed + 1000)
    res = run_score_diagnostic(config, ds, init, seed=seed, n_bootstrap=n_bootstrap)
    diag = res.diagnostic
    return {
        "family": diag.family,
        "seed": diag.seed,
        "init_kind": init_kind,
        "n_val": diag.n_val,
        "homog": asdict(diag.homog_true_state_mass),
        "ppca": asdict(diag.ppca_true_state_mass),
        "delta": asdict(diag.delta_true_state_mass),
    }


# ---------------------------------------------------------------------------
# Stage 1A exit-criterion check
# ---------------------------------------------------------------------------


def evaluate_stage_1a_exit_criterion(records):
    """Per spec Section 11.3: only the **truth_perturbed** init on
    family B gates the stage. random_lowpass is reported but not
    gating.

    Returns a dict with `passed` (bool), the truth-perturbed B
    per-seed checks, and a flag for whether each random-lowpass
    case happened to also pass (informational only).
    """
    truth_b = [r for r in records if r["family"] == "B" and r["init_kind"] == "truth_perturbed"]
    random_b = [r for r in records if r["family"] == "B" and r["init_kind"] == "random_lowpass"]

    truth_b_ok = True
    truth_b_details = []
    for r in truth_b:
        ci_low = r["delta"]["ci_low"]
        ok = bool(ci_low > 0)
        truth_b_ok = truth_b_ok and ok
        truth_b_details.append({"seed": r["seed"], "ci_low": ci_low, "ok": ok})

    random_b_details = []
    for r in random_b:
        random_b_details.append(
            {
                "seed": r["seed"],
                "ci_low": r["delta"]["ci_low"],
                "delta_mean": r["delta"]["mean"],
                # Informational only — does NOT gate
                "incidentally_excludes_zero": bool(r["delta"]["ci_low"] > 0),
            }
        )

    passed = truth_b_ok and (len(truth_b) > 0)
    return {
        "passed": passed,
        "truth_perturbed_family_B": {
            "all_seeds_pass": truth_b_ok,
            "per_seed": truth_b_details,
        },
        "random_lowpass_family_B": {
            "informational_only": True,
            "per_seed": random_b_details,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--volume-size", type=int, default=8)
    parser.add_argument("--healpix-order", type=int, default=1)
    parser.add_argument("--max-shift", type=int, default=1)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--sigma-real", type=float, default=0.2)
    parser.add_argument("--eps-mu", type=float, default=0.1)
    parser.add_argument("--eps-U", type=float, default=0.1)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    volume_shape = (args.volume_size, args.volume_size, args.volume_size)
    image_shape = (args.volume_size, args.volume_size)
    grid = build_fixed_grid(healpix_order=args.healpix_order, max_shift=args.max_shift)

    records = []
    families = (SyntheticFamily.NULL, SyntheticFamily.MATCHED_GRID_HET)
    init_kinds = ("truth_perturbed", "random_lowpass")
    for family in families:
        for init_kind in init_kinds:
            for seed in args.seeds:
                logger.info("running family=%s init=%s seed=%d", family.value, init_kind, seed)
                rec = _run_one(
                    family,
                    seed,
                    init_kind,
                    volume_shape=volume_shape,
                    image_shape=image_shape,
                    grid=grid,
                    q=args.q,
                    n_train=args.n_train,
                    n_val=args.n_val,
                    sigma_real=args.sigma_real,
                    eps_mu=args.eps_mu,
                    eps_U=args.eps_U,
                    k_max=args.k_max,
                    n_bootstrap=args.n_bootstrap,
                )
                records.append(rec)
                logger.info(
                    "  family=%s init=%s seed=%d  delta_mean=%+.4f  delta_ci=[%+.4f, %+.4f]",
                    rec["family"],
                    rec["init_kind"],
                    rec["seed"],
                    rec["delta"]["mean"],
                    rec["delta"]["ci_low"],
                    rec["delta"]["ci_high"],
                )

    exit_check = evaluate_stage_1a_exit_criterion(records)
    output = {
        "stage": "1A",
        "config": {
            "volume_shape": list(volume_shape),
            "image_shape": list(image_shape),
            "healpix_order": args.healpix_order,
            "max_shift": args.max_shift,
            "q": args.q,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "sigma_real": args.sigma_real,
            "eps_mu": args.eps_mu,
            "eps_U": args.eps_U,
            "k_max": args.k_max,
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
