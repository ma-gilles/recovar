"""Negative-test verification for ``check_parity.py``.

Takes a known-good recovar dump and a frozen baseline, then synthesizes a
"corrupted" dump that should fail every metric in the baseline. Runs the
checker against both:

- the original (expected: exit 0)
- the corrupted (expected: exit nonzero, REGRESSED on multiple metrics)

If either expectation is violated, prints a diagnostic and exits 1. Used as
a smoke check that the regression detector actually fires when something
breaks.

Usage:
    pixi run python scripts/parity/negative_test.py \\
        --baseline tests/baselines/parity/quality_baseline_5k_128.json \\
        --scenario iter7_to_8_grouped_union \\
        --recovar-dump _agent_scratch/parity/recovar_iter7_grouped_union/iter_008.npz \\
        --relion-dump _agent_scratch/parity/relion/iter_008.npz
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CHECK_PARITY = REPO / "scripts/parity/check_parity.py"


def _run_check(baseline: Path, scenario: str, recovar_dump: Path, relion_dump: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(CHECK_PARITY),
            "--baseline",
            str(baseline),
            "--scenario",
            scenario,
            "--recovar-dump",
            str(recovar_dump),
            "--relion-dump",
            str(relion_dump),
            "--exit-code-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def _corrupt_dump(src: Path, dst: Path) -> None:
    """Write a corrupted copy that should fail every metric the baseline tracks.

    We perturb each field aggressively enough to push past any sane regression
    threshold:
    - ``ave_pmax`` set far outside any plausible tolerance band
    - per-particle eulers scrambled (90 deg rotation against ground truth)
    - half-volumes given a constant non-zero offset (correlation -> well below floor
      while still being numerically defined)
    - sigma2_noise inflated 1000x (ratio leaves any reasonable band)
    - sigma_offset bumped by 100 A
    """
    data = dict(np.load(src, allow_pickle=False))
    data["ave_pmax"] = np.float64(0.99)  # huge gap from any real RELION ave_pmax
    if "half1_best_eulers_total" in data:
        # Add 90 deg to every angle so the rel rotation against RELION is huge
        data["half1_best_eulers_total"] = data["half1_best_eulers_total"] + 90.0
    if "half2_best_eulers_total" in data:
        data["half2_best_eulers_total"] = data["half2_best_eulers_total"] + 90.0
    if "half1_max_posterior" in data:
        data["half1_max_posterior"] = np.zeros_like(data["half1_max_posterior"])
    if "half2_max_posterior" in data:
        data["half2_max_posterior"] = np.zeros_like(data["half2_max_posterior"])
    if "half1_mean_real_ds" in data:
        # Use noise (well-defined correlation, will be near zero)
        rng = np.random.default_rng(0)
        data["half1_mean_real_ds"] = rng.standard_normal(data["half1_mean_real_ds"].shape).astype(
            data["half1_mean_real_ds"].dtype
        )
    if "half2_mean_real_ds" in data:
        rng = np.random.default_rng(1)
        data["half2_mean_real_ds"] = rng.standard_normal(data["half2_mean_real_ds"].shape).astype(
            data["half2_mean_real_ds"].dtype
        )
    if "half1_wsum_sigma2_noise" in data:
        data["half1_wsum_sigma2_noise"] = data["half1_wsum_sigma2_noise"] * 1000.0
    if "half2_wsum_sigma2_noise" in data:
        data["half2_wsum_sigma2_noise"] = data["half2_wsum_sigma2_noise"] * 1000.0
    if "sigma_offset" in data:
        data["sigma_offset"] = np.float64(99.0)
    np.savez_compressed(dst, **data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--recovar-dump", required=True, type=Path)
    ap.add_argument("--relion-dump", required=True, type=Path)
    args = ap.parse_args()

    if not args.recovar_dump.exists():
        print(f"ERROR: recovar dump not found at {args.recovar_dump}", file=sys.stderr)
        return 1
    if not args.relion_dump.exists():
        print(f"ERROR: relion dump not found at {args.relion_dump}", file=sys.stderr)
        return 1

    print("== Positive test (baseline-conforming dump) ==")
    rc_pos, out_pos = _run_check(args.baseline, args.scenario, args.recovar_dump, args.relion_dump)
    print(out_pos)
    if rc_pos != 0:
        print(f"FAIL: positive test should have exited 0, got {rc_pos}")
        return 1
    print("OK: positive test exited 0")

    with tempfile.TemporaryDirectory() as td:
        corrupt = Path(td) / "corrupt.npz"
        _corrupt_dump(args.recovar_dump, corrupt)
        print()
        print("== Negative test (corrupted dump, expect REGRESSED) ==")
        rc_neg, out_neg = _run_check(args.baseline, args.scenario, corrupt, args.relion_dump)
        print(out_neg)
        if rc_neg == 0:
            print("FAIL: corrupted dump should have exited nonzero")
            return 1
        if "REGRESSED" not in out_neg:
            print("FAIL: corrupted dump should have at least one REGRESSED metric")
            return 1
        print(f"OK: negative test exited {rc_neg} with REGRESSED metrics present")

    print()
    print("== Negative test PASSED — regression detector is wired correctly ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
