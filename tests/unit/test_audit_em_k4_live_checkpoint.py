from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import audit_em_k4_live_checkpoint as auditor


def _write_iteration(
    root: Path,
    *,
    iteration: int,
    coarse: np.ndarray,
    final: np.ndarray,
    noise: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    wall_time_s: float,
) -> None:
    intermediates = root / "intermediates"
    timing = root / "timing"
    intermediates.mkdir(parents=True, exist_ok=True)
    timing.mkdir(parents=True, exist_ok=True)
    tag = f"it{iteration - 1:03d}"
    metadata = {
        "iteration": iteration - 1,
        "current_size": 40 + 2 * iteration,
        "n_rotations": rotations.shape[0],
        "n_translations": translations.shape[0],
        "healpix_order": 1,
        "local_search": False,
        "sigma_rot": 0.0,
    }
    np.save(intermediates / f"{tag}_meta.npy", metadata, allow_pickle=True)
    np.save(intermediates / f"{tag}_coarse_ha_half1.npy", coarse)
    np.save(intermediates / f"{tag}_ha_half1.npy", final)
    np.save(intermediates / f"{tag}_noise.npy", noise)
    np.save(intermediates / f"{tag}_rotations.npy", rotations)
    np.save(intermediates / f"{tag}_translations.npy", translations)
    np.savez_compressed(
        timing / f"iter_{iteration:03d}.npz",
        iteration=np.int32(iteration - 1),
        relion_iteration=np.int32(iteration),
        wall_time_s=np.float64(wall_time_s),
    )


def _write_log(
    root: Path,
    *,
    wall_time_1: float,
    wall_time_2: float,
    pmax_offset: float,
    expected_rotation_iteration_2: float,
) -> None:
    pmax_1 = 0.6 + pmax_offset
    pmax_2 = 0.7 + pmax_offset
    lines = [
        "=== RELION Iteration 1/3: current_size=42, healpix_order=1, local_search=False ===",
        f"Class3D optimizer Pmax: value={pmax_1:.9f} numerator=2.4 "
        "half1_mstep_posterior_mass=4.0 half1_particle_count=4",
        "RELION Iteration 1: current_size=42, pixel_res=21.0, res=11.00 A, "
        f"ave_Pmax={pmax_1:.4f}, healpix_order=1, converged=False, "
        f"time={wall_time_1:.1f}s",
        "Strict RELION dynamic dispatch: numbered_iter=2 rank_particle_counts=[2, 2]",
        "RELION current-size decision: iter=2 prev=42 res_shell=11 incr_size=2 "
        f"high_fsc_at_limit=False ave_Pmax={pmax_1:.6f} raw=44 quantized=44",
        "RELION exact expected accuracy: acc_rot=5.000 deg, acc_trans=2.4000 A "
        "(trials=4, first_particle_ids=[10, 11, 12, 13])",
        "=== RELION Iteration 2/3: current_size=44, healpix_order=1, local_search=False ===",
        f"Class3D optimizer Pmax: value={pmax_2:.9f} numerator=2.8 "
        "half1_mstep_posterior_mass=4.0 half1_particle_count=4",
        "RELION Iteration 2: current_size=44, pixel_res=22.0, res=10.00 A, "
        f"ave_Pmax={pmax_2:.4f}, healpix_order=1, converged=False, "
        f"time={wall_time_2:.1f}s",
        "Strict RELION dynamic dispatch: numbered_iter=3 rank_particle_counts=[2, 2]",
        "RELION current-size decision: iter=3 prev=44 res_shell=12 incr_size=2 "
        f"high_fsc_at_limit=False ave_Pmax={pmax_2:.6f} raw=46 quantized=46",
        "RELION exact expected accuracy: "
        f"acc_rot={expected_rotation_iteration_2:.3f} deg, acc_trans=2.2000 A "
        "(trials=4, first_particle_ids=[10, 11, 12, 13])",
        "=== RELION Iteration 3/3: current_size=46, healpix_order=1, local_search=False ===",
    ]
    (root / "run_full_refinement.stderr").write_text("\n".join(lines) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    live = tmp_path / "live"
    sealed = tmp_path / "sealed"
    rotations = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    translations = np.arange(6, dtype=np.float32).reshape(3, 2)
    for root, coarse1, final1, noise1, coarse2, final2, noise2, wall1, wall2 in (
        (
            sealed,
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1.0, 2.0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2.0, 3.0],
            10.0,
            20.0,
        ),
        (
            live,
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1.0, 2.5],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [2.5, 3.0],
            11.0,
            19.0,
        ),
    ):
        _write_iteration(
            root,
            iteration=1,
            coarse=np.asarray(coarse1, dtype=np.int32),
            final=np.asarray(final1, dtype=np.int32),
            noise=np.asarray(noise1, dtype=np.float32),
            rotations=rotations,
            translations=translations,
            wall_time_s=wall1,
        )
        _write_iteration(
            root,
            iteration=2,
            coarse=np.asarray(coarse2, dtype=np.int32),
            final=np.asarray(final2, dtype=np.int32),
            noise=np.asarray(noise2, dtype=np.float32),
            rotations=rotations,
            translations=translations,
            wall_time_s=wall2,
        )
        _write_log(
            root,
            wall_time_1=wall1,
            wall_time_2=wall2,
            pmax_offset=0.0001 if root == live else 0.0,
            expected_rotation_iteration_2=4.5 if root == live else 4.6,
        )
    return live, sealed


@pytest.mark.unit
def test_live_checkpoint_reports_exact_topology_and_mismatch_dynamics(tmp_path):
    live, sealed = _fixture(tmp_path)

    report = auditor.audit(live_control=live, sealed_control=sealed, iteration=2)

    assert report["status"] == "pass"
    assert report["schema"] == "recovar.em_k4_live_checkpoint.v2"
    assert report["topology"]["exact"] is True
    assert report["topology"]["persisted_exact"] is True
    assert report["topology"]["controller_exact"] is True
    assert report["hard_assignments"]["coarse"]["mismatch_count"] == 1
    assert report["hard_assignments"]["coarse"]["dynamics_from_previous"] == {
        "previous": 1,
        "current": 1,
        "persistent": 0,
        "new": 1,
        "resolved": 1,
    }
    assert report["hard_assignments"]["final"]["mismatch_count"] == 2
    assert report["hard_assignments"]["final"]["dynamics_from_previous"] == {
        "previous": 1,
        "current": 2,
        "persistent": 1,
        "new": 1,
        "resolved": 0,
    }
    assert report["noise"]["max_abs_delta"] == pytest.approx(0.5)
    assert report["noise"]["l2_delta"] == pytest.approx(0.5)
    assert report["timing"]["live_minus_sealed_fraction"] == pytest.approx(-0.05)
    assert report["controller_state"]["optimizer_pmax"]["live_minus_sealed"] == pytest.approx(0.0001)
    assert report["controller_state"]["next"]["live"]["dispatch_counts"] == [2, 2]
    assert report["controller_state"]["next"]["expected_accuracy_exact"] is False
    assert "correlation is not computed" in report["quality_metric_policy"]
    assert len(report["live_artifact_sha256"]) == 8
    assert "run_full_refinement.stderr#iteration=2" in report["live_artifact_sha256"]


@pytest.mark.unit
def test_live_checkpoint_fails_topology_when_rotation_grid_differs(tmp_path):
    live, sealed = _fixture(tmp_path)
    path = live / "intermediates" / "it001_rotations.npy"
    rotations = np.load(path)
    rotations[0, 0, 0] += 1.0
    np.save(path, rotations)

    report = auditor.audit(live_control=live, sealed_control=sealed, iteration=2)

    assert report["status"] == "fail"
    assert report["topology"]["exact"] is False
    assert report["topology"]["arrays_exact"]["rotations"] is False


@pytest.mark.unit
def test_live_checkpoint_fails_topology_when_next_dispatch_differs(tmp_path):
    live, sealed = _fixture(tmp_path)
    path = live / "run_full_refinement.stderr"
    text = path.read_text().replace(
        "numbered_iter=3 rank_particle_counts=[2, 2]",
        "numbered_iter=3 rank_particle_counts=[1, 3]",
        1,
    )
    path.write_text(text)

    report = auditor.audit(live_control=live, sealed_control=sealed, iteration=2)

    assert report["status"] == "fail"
    assert report["topology"]["exact"] is False
    assert report["topology"]["persisted_exact"] is True
    assert report["topology"]["controller_exact"] is False


@pytest.mark.unit
def test_live_checkpoint_requires_next_expected_accuracy(tmp_path):
    live, sealed = _fixture(tmp_path)
    path = live / "run_full_refinement.stderr"
    line = (
        "RELION exact expected accuracy: acc_rot=4.500 deg, acc_trans=2.2000 A "
        "(trials=4, first_particle_ids=[10, 11, 12, 13])\n"
    )
    path.write_text(path.read_text().replace(line, "", 1))

    with pytest.raises(auditor.AuditError, match="expected one next expected accuracy"):
        auditor.audit(live_control=live, sealed_control=sealed, iteration=2)


@pytest.mark.unit
def test_live_checkpoint_log_segment_hash_ignores_later_log_growth(tmp_path):
    live, sealed = _fixture(tmp_path)
    report = auditor.audit(live_control=live, sealed_control=sealed, iteration=2)
    key = "run_full_refinement.stderr#iteration=2"

    path = live / "run_full_refinement.stderr"
    with path.open("a") as stream:
        stream.write("later iteration output that must not change iteration 2\n")
    updated = auditor.audit(live_control=live, sealed_control=sealed, iteration=2)

    assert updated["live_artifact_sha256"][key] == report["live_artifact_sha256"][key]


@pytest.mark.unit
def test_live_checkpoint_rejects_nonfinite_noise(tmp_path):
    live, sealed = _fixture(tmp_path)
    path = live / "intermediates" / "it001_noise.npy"
    noise = np.load(path)
    noise[0] = np.nan
    np.save(path, noise)

    with pytest.raises(auditor.AuditError, match="finite numeric"):
        auditor.audit(live_control=live, sealed_control=sealed, iteration=2)
