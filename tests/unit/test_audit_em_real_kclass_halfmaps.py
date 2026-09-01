from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

from scripts import audit_em_real_kclass_halfmaps as audit


def _write_particles(path: Path, names: list[str], subsets: list[int], classes=None) -> None:
    particles = pd.DataFrame(
        {
            "rlnImageName": names,
            "rlnRandomSubset": subsets,
        }
    )
    if classes is not None:
        particles["rlnClassNumber"] = classes
    starfile.write({"particles": particles}, path, overwrite=True)


def _split_manifest(tmp_path: Path) -> dict:
    selected = [f"{index}@particles.mrcs" for index in range(1, 7)]
    origin_names = [f"{index}@origin.mrcs" for index in range(1, 7)]
    origin = tmp_path / "origin.star"
    source_indices = tmp_path / "source_indices.npy"
    source = tmp_path / "selected.star"
    half1 = tmp_path / "half1.star"
    half2 = tmp_path / "half2.star"
    _write_particles(origin, origin_names, [1, 2, 1, 2, 1, 2])
    np.save(source_indices, np.arange(6, dtype=np.int64))
    _write_particles(source, selected, [1, 2, 1, 2, 1, 2])
    _write_particles(half1, selected[0::2], [1, 1, 1])
    _write_particles(half2, selected[1::2], [2, 2, 2])
    return {
        "particle_selection": {
            "mode": "full10k",
            "selected_image_names": selected,
            "ordered_image_names_sha256": audit.sha256_strings(selected),
            "source_particles_star": str(source),
            "origin_particles_star": str(origin),
            "origin_particles_star_sha256": audit.sha256_file(origin),
            "source_indices_npy": str(source_indices),
            "source_indices_sha256": audit.sha256_file(source_indices),
            "selected_source_indices": list(range(6)),
            "ordered_source_indices_sha256": audit.sha256_ints(list(range(6))),
            "selection_source_json": None,
            "selection_source_sha256": None,
        },
        "halves": [
            {
                "half": 1,
                "particles_star": str(half1),
                "particle_count": 3,
                "ordered_image_names_sha256": audit.sha256_strings(selected[0::2]),
            },
            {
                "half": 2,
                "particles_star": str(half2),
                "particle_count": 3,
                "ordered_image_names_sha256": audit.sha256_strings(selected[1::2]),
            },
        ],
    }


def test_particle_split_accepts_exact_disjoint_source_order(tmp_path: Path) -> None:
    result = audit.validate_particle_split(_split_manifest(tmp_path))

    assert result["half_counts"] == [3, 3]
    assert result["disjoint"] is True
    assert result["complete_union"] is True


def test_particle_split_rejects_reordered_half_even_if_sets_match(tmp_path: Path) -> None:
    manifest = _split_manifest(tmp_path)
    names = list(reversed(manifest["particle_selection"]["selected_image_names"][0::2]))
    half1 = Path(manifest["halves"][0]["particles_star"])
    _write_particles(half1, names, [1, 1, 1])
    manifest["halves"][0]["ordered_image_names_sha256"] = audit.sha256_strings(names)

    with pytest.raises(audit.AuditError, match="immutable selected-source order"):
        audit.validate_particle_split(manifest)


def test_particle_split_rejects_self_consistent_generated_relabel(tmp_path: Path) -> None:
    manifest = _split_manifest(tmp_path)
    selected = manifest["particle_selection"]["selected_image_names"]
    _write_particles(
        Path(manifest["particle_selection"]["source_particles_star"]),
        selected,
        [2, 1, 2, 1, 2, 1],
    )

    with pytest.raises(audit.AuditError, match="immutable random-subset labels"):
        audit.validate_particle_split(manifest)


def test_particle_split_rejects_changed_selected_source_index_sequence(tmp_path: Path) -> None:
    manifest = _split_manifest(tmp_path)
    manifest["particle_selection"]["selected_source_indices"][2] = 999
    manifest["particle_selection"]["ordered_source_indices_sha256"] = audit.sha256_ints(
        manifest["particle_selection"]["selected_source_indices"]
    )

    with pytest.raises(audit.AuditError, match="selected source-index sequence changed"):
        audit.validate_particle_split(manifest)


def test_particle_split_rejects_origin_source_index_order_disagreement(tmp_path: Path) -> None:
    manifest = _split_manifest(tmp_path)
    path = Path(manifest["particle_selection"]["source_indices_npy"])
    np.save(path, np.asarray([0, 2, 1, 3, 4, 5], dtype=np.int64))
    manifest["particle_selection"]["source_indices_sha256"] = audit.sha256_file(path)

    with pytest.raises(audit.AuditError, match="differ from immutable source-index order"):
        audit.validate_particle_split(manifest)


def test_particle_split_binds_shared_selection_membership_in_origin_order(tmp_path: Path) -> None:
    manifest = _split_manifest(tmp_path)
    all_names = manifest["particle_selection"]["selected_image_names"]
    selected = [all_names[index] for index in (0, 1, 4, 5)]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "same_visited_particle_ids": True,
                "visited_particle_ids": [all_names[index] for index in (5, 0, 4, 1)],
            }
        )
        + "\n"
    )
    selection = manifest["particle_selection"]
    selection.update(
        {
            "mode": "shared200",
            "selection_source_json": str(selection_path),
            "selection_source_sha256": audit.sha256_file(selection_path),
            "selected_image_names": selected,
            "ordered_image_names_sha256": audit.sha256_strings(selected),
            "selected_source_indices": [0, 1, 4, 5],
            "ordered_source_indices_sha256": audit.sha256_ints([0, 1, 4, 5]),
        }
    )
    _write_particles(Path(selection["source_particles_star"]), selected, [1, 2, 1, 2])
    _write_particles(Path(manifest["halves"][0]["particles_star"]), selected[0::2], [1, 1])
    _write_particles(Path(manifest["halves"][1]["particles_star"]), selected[1::2], [2, 2])
    for row, names in zip(manifest["halves"], (selected[0::2], selected[1::2]), strict=True):
        row["particle_count"] = 2
        row["ordered_image_names_sha256"] = audit.sha256_strings(names)

    assert audit.validate_particle_split(manifest)["selection_mode"] == "shared200"
    selection["selected_image_names"] = [all_names[index] for index in (0, 1, 2, 5)]
    selection["ordered_image_names_sha256"] = audit.sha256_strings(selection["selected_image_names"])
    with pytest.raises(audit.AuditError, match="differ from immutable shared200 selection"):
        audit.validate_particle_split(manifest)


def _write_recovar_maps(root: Path, *, mismatch: bool = False) -> None:
    root.mkdir()
    for class_id in range(1, 5):
        payload = f"class-{class_id}".encode()
        (root / f"it007_half1_class{class_id}_reg.mrc").write_bytes(payload)
        second = b"different" if mismatch and class_id == 3 else payload
        (root / f"it007_half2_class{class_id}_reg.mrc").write_bytes(second)


def test_recovar_internal_kclass_half_labels_must_be_identical_replicas(tmp_path: Path) -> None:
    good = tmp_path / "good"
    _write_recovar_maps(good)

    paths, rows = audit._latest_recovar_combined_maps(good, 7)

    assert len(paths) == 4
    assert all(row["byte_identical"] for row in rows)
    assert all("not_independent_halfmap" in row["semantic_role"] for row in rows)

    bad = tmp_path / "bad"
    _write_recovar_maps(bad, mismatch=True)
    with pytest.raises(audit.AuditError, match="refusing ambiguous class 3"):
        audit._latest_recovar_combined_maps(bad, 7)


def test_recovar_final_topology_rejects_extra_class_ids(tmp_path: Path) -> None:
    root = tmp_path / "extra"
    _write_recovar_maps(root)
    for replica in (1, 2):
        (root / f"it007_half{replica}_class5_reg.mrc").write_bytes(b"extra")

    with pytest.raises(audit.AuditError, match="missing or extra class IDs"):
        audit._latest_recovar_combined_maps(root, 7)


def test_cross_process_duplicate_maps_are_rejected(tmp_path: Path) -> None:
    first = [tmp_path / f"first-{index}" for index in range(4)]
    second = [tmp_path / f"second-{index}" for index in range(4)]
    for index, path in enumerate(first):
        path.write_bytes(f"first-{index}".encode())
    for index, path in enumerate(second):
        path.write_bytes(f"second-{index}".encode())

    audit._reject_cross_process_duplicates([first, second], engine="test")
    second[2].write_bytes(first[1].read_bytes())
    with pytest.raises(audit.AuditError, match="false independent-half claim"):
        audit._reject_cross_process_duplicates([first, second], engine="test")


def test_within_process_duplicate_class_maps_are_rejected(tmp_path: Path) -> None:
    paths = [tmp_path / f"class-{index}.mrc" for index in range(4)]
    for index, path in enumerate(paths):
        path.write_bytes(f"class-{index}".encode())
    audit._reject_within_process_duplicates(paths, engine="test", half=1)
    paths[3].write_bytes(paths[0].read_bytes())

    with pytest.raises(audit.AuditError, match="byte-identical class maps"):
        audit._reject_within_process_duplicates(paths, engine="test", half=1)


def test_class_matching_requires_unique_exact_optimum_and_records_margin() -> None:
    scores = np.asarray(
        [
            [0.1, 0.2, 0.9, 0.3],
            [0.8, 0.1, 0.2, 0.3],
            [0.2, 0.3, 0.1, 0.95],
            [0.2, 0.85, 0.1, 0.3],
        ]
    )

    permutation, metadata = audit._hungarian_to_anchor(scores, label="known")

    assert permutation == [1, 3, 0, 2]
    assert metadata["exact_optimum_count"] == 1
    assert metadata["objective_margin"] > 0.0
    assert metadata["permutations_exhaustively_checked"] == 24

    with pytest.raises(audit.AuditError, match="optimum is not unique"):
        audit._hungarian_to_anchor(np.ones((4, 4)), label="tied")


def _analysis_args(tmp_path: Path, *extra: str):
    return audit._parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "audit"),
            "--fit-max-shell",
            "32",
            "--crossing-consecutive-shells",
            "3",
            "--phase-randomization-corrected",
            "false",
            "--absolute-resolution-claim",
            "false",
            "--coarse-healpix-order",
            "1",
            "--refine-healpix-order",
            "2",
            "--interpolation-order",
            "1",
            "--mask-threshold",
            "auto",
            "--mask-lowpass-sigma",
            "2",
            "--mask-extend",
            "4",
            "--mask-soft-edge",
            "4",
            "--mask-cleanup",
            "true",
            *extra,
        ]
    )


def test_analysis_policy_is_frozen_and_cli_bound(tmp_path: Path) -> None:
    expected = audit.expected_analysis_policy(128)
    manifest = {"config": {"grid_size": 128}, "analysis_policy": expected}

    assert audit.validate_analysis_policy(manifest, _analysis_args(tmp_path)) == expected
    changed_manifest = json.loads(json.dumps(manifest))
    changed_manifest["analysis_policy"]["alignment"]["fit_max_shell"] = 31
    with pytest.raises(audit.AuditError, match="manifest analysis_policy changed"):
        audit.validate_analysis_policy(changed_manifest, _analysis_args(tmp_path))

    changed_cli = _analysis_args(tmp_path)
    changed_cli.fit_max_shell = 31
    with pytest.raises(audit.AuditError, match="CLI analysis policy differs"):
        audit.validate_analysis_policy(manifest, changed_cli)


def test_analysis_policy_disclaims_corrected_or_absolute_masked_resolution() -> None:
    fsc = audit.expected_analysis_policy(256)["fsc"]

    assert fsc["phase_randomization_corrected"] is False
    assert fsc["absolute_resolution_claim"] is False


def _identity_curves(class_id: int) -> dict[str, np.ndarray]:
    curves = {}
    half_curve = np.asarray([1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.12, 0.1, 0.05], dtype=np.float64)
    cross_curve = np.ones_like(half_curve)
    prefix = f"class{class_id:03d}_"
    for route in ("unmasked", "common_masked"):
        curves[prefix + f"relion_halfmap_{route}"] = half_curve.copy()
        curves[prefix + f"recovar_halfmap_{route}"] = half_curve.copy()
        for name in ("cross_merged", "cross_half1", "cross_half2"):
            curves[prefix + f"{name}_{route}"] = cross_curve.copy()
    return curves


def test_prospective_class_gate_accepts_identity_and_rejects_halfmap_loss() -> None:
    curves = _identity_curves(1)
    metrics, failures = audit._science_metrics_for_class(
        1,
        curves,
        box_size=128,
        voxel_size=3.0,
        thresholds=audit.EXPECTED_THRESHOLDS,
    )

    assert failures == []
    assert metrics["common_masked"]["recovar_to_relion_resolution_ratio"] == pytest.approx(1.0)

    curves["class001_recovar_halfmap_unmasked"][1:7] -= 0.2
    _, failures = audit._science_metrics_for_class(
        1,
        curves,
        box_size=128,
        voxel_size=3.0,
        thresholds=audit.EXPECTED_THRESHOLDS,
    )
    assert "class001:unmasked:half_band_auc_drop" in failures


def test_resolution_gate_handles_beyond_range_explicitly_and_fails_closed() -> None:
    curves = _identity_curves(1)
    for route in ("unmasked", "common_masked"):
        curves[f"class001_relion_halfmap_{route}"][:] = 0.8
        curves[f"class001_recovar_halfmap_{route}"][:] = 0.8

    metrics, failures = audit._science_metrics_for_class(
        1,
        curves,
        box_size=128,
        voxel_size=3.0,
        thresholds=audit.EXPECTED_THRESHOLDS,
    )

    masked = metrics["common_masked"]
    assert failures == []
    assert masked["resolution_comparison_status"] == "both_beyond_measured_range"
    assert masked["relion_resolution_angstrom"] is None
    assert masked["recovar_resolution_angstrom"] is None
    assert masked["relion_resolution_better_than_angstrom"] == pytest.approx(48.0)
    assert masked["recovar_resolution_better_than_angstrom"] == pytest.approx(48.0)
    assert masked["recovar_to_relion_resolution_ratio"] is None

    curves["class001_recovar_halfmap_common_masked"][3:] = 0.0
    _, failures = audit._science_metrics_for_class(
        1,
        curves,
        box_size=128,
        voxel_size=3.0,
        thresholds=audit.EXPECTED_THRESHOLDS,
    )
    assert "class001:common_masked:resolution" in failures


def test_common_mask_uses_nonnegative_rms_envelope_without_sign_cancellation(monkeypatch) -> None:
    grid = np.indices((8, 8, 8), dtype=np.float32).sum(axis=0)
    source = np.exp(-((grid - 10.5) ** 2) / 8.0).astype(np.float32)
    captured = {}

    def fake_make_mask(volume, **_kwargs):
        captured["volume"] = np.asarray(volume)
        return (volume > np.median(volume)).astype(np.float32)

    monkeypatch.setattr(audit, "make_mask", fake_make_mask)
    maps = [[source, -source, source, -source] for _ in range(4)]

    mask, metadata = audit._common_mask(
        maps,
        policy=audit.expected_analysis_policy(128)["common_mask"],
    )

    assert np.all(captured["volume"] >= 0.0)
    assert np.max(captured["volume"]) > 0.0
    assert np.any(mask == 0.0) and np.any(mask == 1.0)
    assert metadata["nonnegative_envelope"] is True
    assert metadata["sign_cancellation_possible"] is False


def test_assignments_are_joined_by_particle_identity_not_relion_row_order(tmp_path: Path) -> None:
    names = ["1@particles.mrcs", "2@particles.mrcs", "3@particles.mrcs"]
    input_star = tmp_path / "particles.star"
    _write_particles(input_star, names, [1, 1, 1])
    relion_dir = tmp_path / "relion"
    recovar_dir = tmp_path / "recovar"
    relion_dir.mkdir()
    recovar_dir.mkdir()
    _write_particles(
        relion_dir / "run_it008_data.star",
        [names[2], names[0], names[1]],
        [1, 1, 1],
        classes=[3, 1, 2],
    )
    np.savez(
        recovar_dir / "refinement_results.npz",
        class_assignments_by_image_iter_007=np.asarray([0, 1, 2]),
        sig_counts_by_image_iter_007=np.asarray([10, 20, 30]),
        final_all_data_ran=np.asarray(False),
        git_commit=np.asarray("a" * 40),
        symmetry_label=np.asarray("C1"),
    )

    relion, recovar, metadata = audit._read_assignments(relion_dir, recovar_dir, 8, input_star)

    np.testing.assert_array_equal(relion, [0, 1, 2])
    np.testing.assert_array_equal(recovar, [0, 1, 2])
    assert metadata["identity_order"] == "input half STAR rlnImageName order"


def test_gpu_monitor_parser_is_schema_bound(tmp_path: Path) -> None:
    monitor = tmp_path / "gpu.csv"
    monitor.write_text(
        "epoch,gpu_uuid,memory_used_mib\n"
        "100,GPU-deadbeef,2048\n"
        "101,GPU-deadbeef,4096\n"
    )
    assert audit._parse_peak_hbm(monitor) == (4096, "GPU-deadbeef")

    monitor.write_text("timestamp,index,memory.used\n100,0,4096\n")
    with pytest.raises(audit.AuditError, match="schema mismatch"):
        audit._parse_peak_hbm(monitor)


def test_command_records_fail_closed(tmp_path: Path) -> None:
    command = tmp_path / "command.json"
    command.write_text(json.dumps(["engine", "--K", "4"]) + "\n")
    manifest = {
        "halves": [
            {
                "half": half,
                "relion_command_path": str(command),
                "relion_command": ["engine", "--K", "4"],
                "recovar_command_path": str(command),
                "recovar_command": ["engine", "--K", "4"],
            }
            for half in (1, 2)
        ]
    }
    assert audit.validate_commands(manifest)["exact_match"] is True
    command.write_text(json.dumps(["engine", "--K", "3"]) + "\n")
    with pytest.raises(audit.AuditError, match="command changed"):
        audit.validate_commands(manifest)


def _write_slurm_record(path: Path, *, allocated_gpus: int = 1) -> None:
    path.write_text(
        json.dumps(
            {
                "under_slurm": True,
                "job_id": path.stem,
                "ReqTRES": "cpu=8,mem=64G,node=1,billing=8,gres/gpu=1",
                "AllocTRES": "cpu=8,mem=64G,node=1,billing=8,gres/gpu=1",
                "requested_gpus": 1,
                "allocated_gpus": allocated_gpus,
                "OverSubscribe": "OK",
            }
        )
        + "\n"
    )


def test_setup_and_qualification_allocations_are_both_required(tmp_path: Path) -> None:
    setup = tmp_path / "setup.json"
    qualification = tmp_path / "qualification.json"
    _write_slurm_record(setup)
    _write_slurm_record(qualification)
    manifest = {
        "provenance": {
            "setup_slurm_allocation_json": str(setup),
            "slurm_allocation_json": str(qualification),
        }
    }

    result = audit.validate_slurm_allocation(manifest)

    assert result["both_valid"] is True
    assert result["setup"]["valid"] is True
    assert result["qualification"]["valid"] is True
    _write_slurm_record(setup, allocated_gpus=2)
    with pytest.raises(audit.AuditError, match="setup job did not allocate exactly one GPU"):
        audit.validate_slurm_allocation(manifest)
