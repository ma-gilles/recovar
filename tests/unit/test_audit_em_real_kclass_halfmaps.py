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
    source = tmp_path / "selected.star"
    half1 = tmp_path / "half1.star"
    half2 = tmp_path / "half2.star"
    _write_particles(source, selected, [1, 2, 1, 2, 1, 2])
    _write_particles(half1, selected[0::2], [1, 1, 1])
    _write_particles(half2, selected[1::2], [2, 2, 2])
    return {
        "particle_selection": {
            "selected_image_names": selected,
            "ordered_image_names_sha256": audit.sha256_strings(selected),
            "source_particles_star": str(source),
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

    with pytest.raises(audit.AuditError, match="preserve frozen selected-source order"):
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

    mask, metadata = audit._common_mask(maps)

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
