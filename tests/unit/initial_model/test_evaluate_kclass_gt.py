"""Smoke + correctness tests for scripts/evaluate_kclass_gt.py.

The script wraps :func:`recovar.em.initial_model.gt_metrics.align_volume_to_reference`
to do best-permutation per-class FSC for K-class ab-initio outputs. These
tests pin:

* CLI accepts repeated ``--volume`` and ``--gt_volume`` and matches their
  count;
* on identical (recovar, GT) inputs with K=2, the picked permutation is
  identity and per-class fsc(1-8) is ~1.0;
* on swapped inputs (recovar class 0 = GT class 1 and vice versa) the
  picked permutation is (1, 0) — so the script does NOT silently fall
  back to identity ordering.

All tests run on tiny synthetic volumes (32^3) so they finish in a few
seconds even on CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "evaluate_kclass_gt.py"


def _make_synthetic_volume(seed: int, ori: int = 32) -> np.ndarray:
    """Build a non-symmetric 32^3 volume so rotation alignment has a unique optimum."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((ori, ori, ori))
    vol = np.zeros((ori, ori, ori), dtype=np.float32)
    centers = [(8, 12, 16), (20, 22, 10), (14, 6, 24)]
    for cz, cy, cx in centers:
        r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        vol += np.exp(-r2 / 6.0).astype(np.float32)
    vol += (0.02 * rng.standard_normal(vol.shape)).astype(np.float32)
    return vol


def _save_mrc(path: Path, vol: np.ndarray, voxel_size: float = 4.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(vol.astype(np.float32))
        m.voxel_size = voxel_size


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        pytest.fail(f"evaluate_kclass_gt.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def test_kclass_eval_identity_permutation(tmp_path):
    """recovar = GT for both classes → best perm is identity, fsc(1-8) ~ 1."""
    rec_a = _make_synthetic_volume(seed=1)
    rec_b = _make_synthetic_volume(seed=2)

    rec_a_path = tmp_path / "rec_class001.mrc"
    rec_b_path = tmp_path / "rec_class002.mrc"
    gt_a_path = tmp_path / "gt_class001.mrc"
    gt_b_path = tmp_path / "gt_class002.mrc"
    _save_mrc(rec_a_path, rec_a)
    _save_mrc(rec_b_path, rec_b)
    _save_mrc(gt_a_path, rec_a.copy())
    _save_mrc(gt_b_path, rec_b.copy())

    out_json = tmp_path / "summary.json"
    _run(
        [
            "--volume",
            str(rec_a_path),
            "--volume",
            str(rec_b_path),
            "--gt_volume",
            str(gt_a_path),
            "--gt_volume",
            str(gt_b_path),
            # Same frame for both — no contrast flip needed.
            "--volume_frame",
            "recovar",
            "--gt_frame",
            "recovar",
            # Refinement is unnecessary at zero-rotation truth, but exercising
            # it confirms the (3, 4) default doesn't break correctness.
            "--gt_align_refine_orders",
            "3",
            "4",
            "--output_json",
            str(out_json),
        ]
    )

    summary = json.loads(out_json.read_text())
    assert summary["K"] == 2
    primary = summary["primary"]
    assert tuple(primary["best_perm"]) == (0, 1)
    assert summary["comparisons"] == []  # no --compare_with → empty list
    # Even for a perfect (recovar = GT) input the alignment goes through
    # HEALPix-2 coarse + HEALPix-(3, 4) refinement, neither of which contains
    # exact identity, so the best on-grid rotation introduces some trilinear
    # blur. The threshold 0.7 is loose enough to survive that interpolation
    # noise but tight enough to fail loudly when alignment is broken
    # (the misaligned cross pairs above score ~0).
    for entry in primary["per_class"]:
        assert entry["mean_fsc_1_8"] >= 0.7, (
            f"identity-truth fsc(1-8) should be high after alignment, got class {entry['class']} = {entry['mean_fsc_1_8']:.4f}"
        )


def test_kclass_eval_swapped_classes_pick_swapped_permutation(tmp_path):
    """recovar c0 == GT c1 and recovar c1 == GT c0 → best perm is (1, 0)."""
    vol_a = _make_synthetic_volume(seed=3)
    vol_b = _make_synthetic_volume(seed=4)

    rec_a_path = tmp_path / "rec_class001.mrc"
    rec_b_path = tmp_path / "rec_class002.mrc"
    gt_a_path = tmp_path / "gt_class001.mrc"
    gt_b_path = tmp_path / "gt_class002.mrc"
    # recovar c0 = vol_a, recovar c1 = vol_b
    _save_mrc(rec_a_path, vol_a)
    _save_mrc(rec_b_path, vol_b)
    # GT c0 = vol_b, GT c1 = vol_a → identity is wrong, swap is right.
    _save_mrc(gt_a_path, vol_b.copy())
    _save_mrc(gt_b_path, vol_a.copy())

    out_json = tmp_path / "summary.json"
    _run(
        [
            "--volume",
            str(rec_a_path),
            "--volume",
            str(rec_b_path),
            "--gt_volume",
            str(gt_a_path),
            "--gt_volume",
            str(gt_b_path),
            "--volume_frame",
            "recovar",
            "--gt_frame",
            "recovar",
            "--gt_align_refine_orders",
            "3",
            "--output_json",
            str(out_json),
        ]
    )

    summary = json.loads(out_json.read_text())
    primary = summary["primary"]
    assert tuple(primary["best_perm"]) == (1, 0), f"expected swapped permutation (1, 0), got {primary['best_perm']}"
    # Same threshold rationale as test_kclass_eval_identity_permutation:
    # HEALPix grid coarseness + trilinear blur cap "perfect alignment" at
    # ~0.92 fsc(1-8) for these tiny (32^3) synthetic volumes.
    for entry in primary["per_class"]:
        assert entry["mean_fsc_1_8"] >= 0.7


def test_kclass_eval_count_mismatch_errors(tmp_path):
    """Mismatched --volume / --gt_volume count is rejected."""
    vol = _make_synthetic_volume(seed=5)
    rec_path = tmp_path / "rec_class001.mrc"
    gt1_path = tmp_path / "gt_class001.mrc"
    gt2_path = tmp_path / "gt_class002.mrc"
    _save_mrc(rec_path, vol)
    _save_mrc(gt1_path, vol)
    _save_mrc(gt2_path, vol)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--volume",
        str(rec_path),
        "--gt_volume",
        str(gt1_path),
        "--gt_volume",
        str(gt2_path),
        "--volume_frame",
        "recovar",
        "--gt_frame",
        "recovar",
        "--gt_align_refine_orders",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "must equal" in proc.stderr or "must equal" in proc.stdout


def test_kclass_eval_compare_with_produces_delta_table(tmp_path):
    """``--compare_with LABEL=p1,p2`` produces a comparison entry whose
    per-class metrics are independently best-permutation-aligned and whose
    deltas equal primary-minus-comparison.

    Setup:
      * primary = (vol_a, vol_b), GT = (vol_a, vol_b)              → identity
      * comparison "RELION" = (vol_a, vol_b_noisier), same GT      → identity
        but the c1 path uses a noisier copy so its fsc(1-8) is strictly
        lower than primary's. Locks the delta-table sign.
    """
    vol_a = _make_synthetic_volume(seed=11)
    vol_b = _make_synthetic_volume(seed=12)
    rng = np.random.default_rng(99)
    vol_b_noisy = vol_b + rng.standard_normal(vol_b.shape).astype(np.float32) * 0.5

    rec_a_path = tmp_path / "rec_class001.mrc"
    rec_b_path = tmp_path / "rec_class002.mrc"
    cmp_a_path = tmp_path / "cmp_class001.mrc"
    cmp_b_path = tmp_path / "cmp_class002.mrc"
    gt_a_path = tmp_path / "gt_class001.mrc"
    gt_b_path = tmp_path / "gt_class002.mrc"
    _save_mrc(rec_a_path, vol_a)
    _save_mrc(rec_b_path, vol_b)
    _save_mrc(cmp_a_path, vol_a)
    _save_mrc(cmp_b_path, vol_b_noisy)
    _save_mrc(gt_a_path, vol_a.copy())
    _save_mrc(gt_b_path, vol_b.copy())

    out_json = tmp_path / "summary.json"
    proc = _run(
        [
            "--label",
            "primary",
            "--volume",
            str(rec_a_path),
            "--volume",
            str(rec_b_path),
            "--gt_volume",
            str(gt_a_path),
            "--gt_volume",
            str(gt_b_path),
            "--compare_with",
            f"RELION={cmp_a_path},{cmp_b_path}",
            "--volume_frame",
            "recovar",
            "--gt_frame",
            "recovar",
            "--gt_align_refine_orders",
            "3",
            "--output_json",
            str(out_json),
        ]
    )

    summary = json.loads(out_json.read_text())
    assert summary["primary"]["label"] == "primary"
    assert len(summary["comparisons"]) == 1
    cmp = summary["comparisons"][0]
    assert cmp["label"] == "RELION"
    assert tuple(cmp["best_perm"]) == (0, 1)
    # primary class 0 should match the comparison closely (same vol_a);
    # primary class 1 should beat comparison c1 because comparison c1 is
    # the noisy copy.
    primary_c0 = summary["primary"]["per_class"][0]
    primary_c1 = summary["primary"]["per_class"][1]
    cmp_c0 = cmp["per_class"][0]
    cmp_c1 = cmp["per_class"][1]
    assert primary_c0["mean_fsc_1_8"] == pytest.approx(cmp_c0["mean_fsc_1_8"], abs=0.05)
    assert primary_c1["mean_fsc_1_8"] > cmp_c1["mean_fsc_1_8"], (
        f"primary c1 ({primary_c1['mean_fsc_1_8']:.4f}) should beat noisier RELION c1 ({cmp_c1['mean_fsc_1_8']:.4f})"
    )
    # Side-by-side delta table is printed to stdout.
    assert "[delta] primary vs RELION" in proc.stdout
