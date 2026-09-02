import json
from pathlib import Path

import numpy as np
import pytest

from scripts import aggregate_em_real_kclass_halfmap_seeds as aggregate


def _assignments(seed: int, half: int, engine: str) -> np.ndarray:
    base = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    if engine == "relion":
        changes = {42001: (), 42002: (0, 2), 42003: (1, 6)}[seed]
    else:
        changes = {42001: (0,), 42002: (2,), 42003: (1, 6)}[seed]
    result = base.copy()
    for index in changes:
        result[(index + half - 1) % result.size] = (result[(index + half - 1) % result.size] + 1) % 4
    return result


def _write_run(root: Path, seed: int, *, changed_grid: bool = False) -> Path:
    audit_dir = root / f"seed-{seed}" / "audit"
    audit_dir.mkdir(parents=True)
    half_hashes = ["a" * 64, "b" * 64]
    agreements = []
    for half in (1, 2):
        relion = _assignments(seed, half, "relion")
        recovar = _assignments(seed, half, "recovar")
        agreements.append(float(np.mean(relion == recovar)))
        np.savez(
            audit_dir / f"half{half}_particle_state_arrays.npz",
            identity_sha256=np.asarray(half_hashes[half - 1]),
            it008_class_relion_one_based=relion + 1,
            it008_class_recovar_zero_based=recovar,
        )

    classes = []
    for class_id in range(1, 5):
        relion_auc = 0.5 + class_id / 20 + (seed - 42001) / 1000
        recovar_auc = relion_auc - 0.005 * class_id
        classes.append(
            {
                "canonical_class": class_id,
                "prospective_science_metrics": {
                    "common_masked": {
                        "relion_halfmap_band_fsc_auc": relion_auc,
                        "recovar_halfmap_band_fsc_auc": recovar_auc,
                        "recovar_minus_relion_halfmap_band_fsc_auc": recovar_auc - relion_auc,
                        "cross_merged_band_fsc_auc": 0.98 - class_id / 1000,
                    }
                },
            }
        )
    audit = {
        "schema": aggregate.AUDIT_SCHEMA,
        "status": "complete",
        "dataset": "EMPIAR-test",
        "profile": "pilot10k-128",
        "config": {"K": 4, "grid_size": 256 if changed_grid else 128, "max_iter": 8, "seed": seed},
        "particle_split": {
            "selected_count": 16,
            "half_counts": [8, 8],
            "selected_order_sha256": "c" * 64,
            "half_order_sha256": half_hashes,
            "selected_source_indices_sha256": "d" * 64,
            "particle_stack_sha256": "e" * 64,
            "origin_particles_star_sha256": "f" * 64,
            "source_indices_sha256": "0" * 64,
        },
        "analysis_policy": {"schema": "frozen-policy"},
        "thresholds": {"assignment_agreement_min": 0.99},
        "source": {"commit": "1" * 40, "tree": "2" * 40},
        "class_matching": {
            "source_class_for_anchor": {
                "relion_half1": [1, 2, 3, 4],
                "relion_half2": [1, 2, 3, 4],
                "recovar_half1": [1, 2, 3, 4],
                "recovar_half2": [1, 2, 3, 4],
            }
        },
        "assignments_and_support": [
            {"half": half, "agreement": agreements[half - 1]} for half in (1, 2)
        ],
        "classes": classes,
        "performance": {
            engine: [
                {
                    "wall_s": 100.0 + seed - 42000 + half,
                    "peak_hbm_mib": 1000 + half,
                    "max_rss_kib": 2000 + half,
                    "slurm_job_id": str(seed),
                    "physical_gpu_uuid": f"GPU-{seed}",
                }
                for half in (1, 2)
            ]
            for engine in ("relion", "recovar")
        },
        "prospective_science_gate": {
            "accepted": False,
            "failures": ["half1:class_assignment_agreement"],
            "policy": "frozen",
        },
    }
    path = audit_dir / "halfmap_audit.json"
    path.write_text(json.dumps(audit) + "\n")
    return path


def test_aggregate_retains_failed_gates_and_compares_seed_stability(tmp_path: Path) -> None:
    paths = [_write_run(tmp_path, seed) for seed in aggregate.DEFAULT_EXPECTED_SEEDS]

    payload = aggregate.aggregate(paths, expected_seeds=aggregate.DEFAULT_EXPECTED_SEEDS)

    assert payload["schema"] == aggregate.SCHEMA
    assert payload["status"] == "complete"
    assert payload["admission_status"] == "DIAGNOSTIC_ONLY_PER_SEED_GATES_RETAINED"
    assert payload["expected_seeds"] == [42001, 42002, 42003]
    assert all(not row["prospective_science_gate"]["accepted"] for row in payload["runs"])
    assert len(payload["assignment_stability"]["halves"]) == 2
    assert len(payload["assignment_stability"]["halves"][0]["within_engine_cross_seed"]) == 3
    assert payload["quality"]["paired_recovar_minus_relion_halfmap_band_fsc_auc"]["min"] == pytest.approx(
        -0.02
    )
    assert payload["performance_across_six_half_runs"]["recovar"]["peak_hbm_mib"]["max"] == 1002
    assert all(len(row["audit"]["sha256"]) == 64 for row in payload["runs"])


def test_aggregate_rejects_scientific_contract_drift(tmp_path: Path) -> None:
    paths = [
        _write_run(tmp_path, seed, changed_grid=seed == 42003)
        for seed in aggregate.DEFAULT_EXPECTED_SEEDS
    ]

    with pytest.raises(ValueError, match="frozen run contract differs"):
        aggregate.aggregate(paths, expected_seeds=aggregate.DEFAULT_EXPECTED_SEEDS)


def test_aggregate_rejects_particle_identity_drift(tmp_path: Path) -> None:
    paths = [_write_run(tmp_path, seed) for seed in aggregate.DEFAULT_EXPECTED_SEEDS]
    arrays_path = paths[-1].parent / "half1_particle_state_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as arrays:
        payload = {name: arrays[name] for name in arrays.files}
    payload["identity_sha256"] = np.asarray("9" * 64)
    np.savez(arrays_path, **payload)

    with pytest.raises(ValueError, match="particle-state identity hash differs across seeds"):
        aggregate.aggregate(paths, expected_seeds=aggregate.DEFAULT_EXPECTED_SEEDS)


def test_aggregate_reproduces_reported_same_seed_agreement(tmp_path: Path) -> None:
    paths = [_write_run(tmp_path, seed) for seed in aggregate.DEFAULT_EXPECTED_SEEDS]
    path = paths[0]
    payload = json.loads(path.read_text())
    payload["assignments_and_support"][0]["agreement"] = 0.0
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="assignment agreement does not reproduce"):
        aggregate.aggregate(paths, expected_seeds=aggregate.DEFAULT_EXPECTED_SEEDS)


def test_parse_expected_seeds_requires_three_distinct_values() -> None:
    assert aggregate.parse_expected_seeds("42001,42002,42003") == (42001, 42002, 42003)
    with pytest.raises(ValueError, match="three distinct"):
        aggregate.parse_expected_seeds("42001,42001,42003")
