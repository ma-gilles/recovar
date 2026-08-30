from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest

from recovar.utils import helpers

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "prepare_empiar10202_set6_i1.py"
SPEC = importlib.util.spec_from_file_location("prepare_empiar10202_set6_i1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

pytestmark = pytest.mark.unit


def _particle_table() -> pd.DataFrame:
    rows = 4
    values: dict[str, list[str]] = {
        "_rlnRandomSubset": ["1", "2", "1", "2"],
        "_rlnVoltage": ["300.000000"] * rows,
        "_rlnDefocusAngle": ["1.000000", "2.000000", "3.000000", "4.000000"],
        "_rlnSphericalAberration": ["2.700000"] * rows,
        "_rlnDetectorPixelSize": ["0.788000"] * rows,
        "_rlnDefocusU": ["10001.0", "10002.0", "10003.0", "10004.0"],
        "_rlnDefocusV": ["11001.0", "11002.0", "11003.0", "11004.0"],
        "_rlnMagnification": ["10000.000000"] * rows,
        "_rlnPhaseShift": ["0.000000"] * rows,
        "_rlnAmplitudeContrast": ["0.070000"] * rows,
        "_rlnImageName": [f"{index:06d}@particles.mrcs" for index in range(1, rows + 1)],
        "_rlnAngleRot": ["1.25", "2.25", "3.25", "4.25"],
        "_rlnAngleTilt": ["11.25", "12.25", "13.25", "14.25"],
        "_rlnAnglePsi": ["21.25", "22.25", "23.25", "24.25"],
        "_rlnOriginX": ["0.1", "0.2", "0.3", "0.4"],
        "_rlnOriginY": ["-0.1", "-0.2", "-0.3", "-0.4"],
        "_rlnClassNumber": ["1"] * rows,
        "_rlnMicrographName": ["micrograph_a.mrc", "micrograph_a.mrc", "micrograph_b.mrc", "micrograph_b.mrc"],
        "_rlnCoordinateX": ["101", "102", "103", "104"],
        "_rlnCoordinateY": ["201", "202", "203", "204"],
        "_rlnGroupNumber": ["1", "2", "3", "4"],
        "_rlnGroupName": ["group_0000", "group_0000", "group_0001", "group_0001"],
    }
    return pd.DataFrame(values)


def _converted_tables(
    legacy: pd.DataFrame,
    *,
    pixel_size: float = MODULE.VOXEL_SIZE_ANGSTROM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    particles = pd.DataFrame(
        {
            "_rlnRandomSubset": legacy["_rlnRandomSubset"],
            "_rlnDefocusAngle": legacy["_rlnDefocusAngle"],
            "_rlnDefocusU": legacy["_rlnDefocusU"],
            "_rlnDefocusV": legacy["_rlnDefocusV"],
            "_rlnPhaseShift": legacy["_rlnPhaseShift"],
            "_rlnImageName": legacy["_rlnImageName"],
            "_rlnAngleRot": legacy["_rlnAngleRot"],
            "_rlnAngleTilt": legacy["_rlnAngleTilt"],
            "_rlnAnglePsi": legacy["_rlnAnglePsi"],
            "_rlnClassNumber": legacy["_rlnClassNumber"],
            "_rlnMicrographName": legacy["_rlnMicrographName"],
            "_rlnCoordinateX": legacy["_rlnCoordinateX"],
            "_rlnCoordinateY": legacy["_rlnCoordinateY"],
            "_rlnGroupNumber": legacy["_rlnGroupNumber"],
            "_rlnGroupName": legacy["_rlnGroupName"],
            "_rlnOpticsGroup": ["1"] * len(legacy),
            "_rlnOriginXAngst": [f"{float(value) * pixel_size:.6f}" for value in legacy["_rlnOriginX"]],
            "_rlnOriginYAngst": [f"{float(value) * pixel_size:.6f}" for value in legacy["_rlnOriginY"]],
        },
        columns=MODULE.CONVERTED_PARTICLE_COLUMNS,
    )
    optics = pd.DataFrame(
        [
            {
                "_rlnOpticsGroup": "1",
                "_rlnOpticsGroupName": "opticsGroup1",
                "_rlnAmplitudeContrast": "0.070000",
                "_rlnSphericalAberration": "2.700000",
                "_rlnVoltage": "300.000000",
                "_rlnImagePixelSize": f"{pixel_size:.6f}",
                "_rlnImageSize": "800",
                "_rlnImageDimensionality": "2",
            }
        ],
        columns=MODULE.OPTICS_COLUMNS,
    )
    return particles, optics


def _write_relion_map(path: Path, data: np.ndarray, voxel_size: float = 1.5) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        mrc.voxel_size = voxel_size


def test_star_normalization_is_deterministic_and_changes_only_stack_path(tmp_path: Path) -> None:
    source = _particle_table()
    stack = (tmp_path / "particles.mrcs").resolve()
    first = tmp_path / "first.star"
    second = tmp_path / "second.star"

    MODULE.validate_particle_table(
        source,
        expected_count=4,
        expected_half_counts={1: 2, 2: 2},
        expected_stack_basename=stack.name,
    )
    normalized = MODULE.normalized_particle_table(source, stack)
    MODULE.write_deterministic_star(first, normalized)
    MODULE.write_deterministic_star(second, normalized)
    reread = MODULE.verify_prepared_table(source, first, stack)

    assert first.read_bytes() == second.read_bytes()
    assert reread.columns.tolist() == source.columns.tolist()
    for column in source.columns:
        if column != "_rlnImageName":
            np.testing.assert_array_equal(reread[column], source[column])
    assert reread["_rlnImageName"].tolist() == [
        f"{index:06d}@{stack}" for index in range(1, 5)
    ]


@pytest.mark.parametrize("mutation", ["order", "half", "pose", "stack"])
def test_particle_table_validation_fails_closed(mutation: str) -> None:
    table = _particle_table()
    if mutation == "order":
        table.loc[1, "_rlnImageName"] = "000003@particles.mrcs"
    elif mutation == "half":
        table.loc[1, "_rlnRandomSubset"] = "3"
    elif mutation == "pose":
        table.loc[1, "_rlnAngleTilt"] = "nan"
    else:
        table.loc[1, "_rlnImageName"] = "000002@other.mrcs"

    with pytest.raises(ValueError):
        MODULE.validate_particle_table(
            table,
            expected_count=4,
            expected_half_counts={1: 2, 2: 2},
            expected_stack_basename="particles.mrcs",
        )


def test_half_assignment_hash_is_row_order_sensitive() -> None:
    table = _particle_table()
    encoded = MODULE.half_assignment_bytes(table)

    assert encoded == bytes([1, 2, 1, 2])
    assert hashlib.sha256(encoded).hexdigest() != hashlib.sha256(bytes([1, 1, 2, 2])).hexdigest()


def test_relion31_optics_conversion_preserves_particle_metadata(tmp_path: Path) -> None:
    stack = (tmp_path / "particles.mrcs").resolve()
    legacy = MODULE.normalized_particle_table(_particle_table(), stack)
    particles, optics = _converted_tables(legacy)

    audit = MODULE.validate_relion31_conversion(
        legacy,
        particles,
        optics,
        stack,
        expected_count=4,
        expected_half_counts={1: 2, 2: 2},
        box_size=800,
        pixel_size_angstrom=0.788,
    )

    assert audit["particle_count"] == 4
    assert audit["origin_roundtrip_max_error_pixels"] <= 5e-7
    assert audit["optics_group_count"] == 1
    assert audit["optics_fields"] == {
        "OpticsGroup": 1.0,
        "AmplitudeContrast": 0.07,
        "SphericalAberration": 2.7,
        "Voltage": 300.0,
        "ImagePixelSize": 0.788,
        "ImageSize": 800.0,
        "ImageDimensionality": 2.0,
    }


@pytest.mark.parametrize("mutation", ["order", "half", "pose", "ctf", "origin", "optics"])
def test_relion31_optics_conversion_fails_closed(mutation: str, tmp_path: Path) -> None:
    stack = (tmp_path / "particles.mrcs").resolve()
    legacy = MODULE.normalized_particle_table(_particle_table(), stack)
    particles, optics = _converted_tables(legacy)
    if mutation == "order":
        particles.loc[[0, 1], "_rlnImageName"] = particles.loc[[1, 0], "_rlnImageName"].to_numpy()
    elif mutation == "half":
        particles.loc[0, "_rlnRandomSubset"] = "2"
    elif mutation == "pose":
        particles.loc[0, "_rlnAnglePsi"] = "99"
    elif mutation == "ctf":
        particles.loc[0, "_rlnDefocusU"] = "999"
    elif mutation == "origin":
        particles.loc[0, "_rlnOriginXAngst"] = "0.080000"
    else:
        optics.loc[0, "_rlnImagePixelSize"] = "0.789"

    with pytest.raises(ValueError):
        MODULE.validate_relion31_conversion(
            legacy,
            particles,
            optics,
            stack,
            expected_count=4,
            expected_half_counts={1: 2, 2: 2},
            box_size=800,
            pixel_size_angstrom=0.788,
        )


def test_group_numbers_match_relion_stable_first_appearance_semantics(tmp_path: Path) -> None:
    stack = (tmp_path / "particles.mrcs").resolve()
    legacy = MODULE.normalized_particle_table(_particle_table(), stack)
    raw_particles, optics = _converted_tables(legacy)

    particles, group_audit = MODULE.canonicalize_relion_group_numbers(
        raw_particles,
        expected_group_count=2,
    )
    output = tmp_path / "particles_optics.star"
    MODULE.write_deterministic_star31(output, particles, optics)
    reread_particles, reread_optics = MODULE.read_star(str(output))
    metadata_audit = MODULE.validate_relion31_conversion(
        legacy,
        reread_particles,
        reread_optics,
        stack,
        expected_count=4,
        expected_half_counts={1: 2, 2: 2},
        box_size=800,
        pixel_size_angstrom=0.788,
        preserve_legacy_group_numbers=False,
        expected_group_count=2,
    )

    assert particles["_rlnGroupNumber"].tolist() == ["1", "1", "2", "2"]
    assert group_audit["group_name_number_bijection"] is True
    assert group_audit["group_mapping"] == [
        {"group_name": "group_0000", "group_number": 1},
        {"group_name": "group_0001", "group_number": 2},
    ]
    assert metadata_audit["group_mapping_sha256"] == group_audit["group_mapping_sha256"]


def test_group_number_validation_rejects_non_bijection() -> None:
    particles, _ = _converted_tables(_particle_table())
    particles["_rlnGroupNumber"] = ["1", "2", "2", "2"]

    with pytest.raises(ValueError, match="stable first-appearance"):
        MODULE.validate_relion_group_semantics(particles, expected_group_count=2)


def test_build_authoritative_star_seals_optics_and_relion_group_semantics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stack = (tmp_path / "particles.mrcs").resolve()
    legacy = MODULE.normalized_particle_table(_particle_table(), stack)
    raw_particles, optics = _converted_tables(legacy)
    legacy_path = tmp_path / "legacy.star"
    raw_path = tmp_path / "raw.star"
    output_path = tmp_path / "authoritative.star"
    MODULE.write_deterministic_star(legacy_path, legacy)
    MODULE.write_deterministic_star31(raw_path, raw_particles, optics)

    expected_particles, _ = MODULE.canonicalize_relion_group_numbers(
        raw_particles,
        expected_group_count=2,
    )
    expected_path = tmp_path / "expected.star"
    MODULE.write_deterministic_star31(expected_path, expected_particles, optics)
    expected_bytes = expected_path.read_bytes()
    monkeypatch.setattr(MODULE, "PARTICLE_COUNT", 4)
    monkeypatch.setattr(MODULE, "HALF_COUNTS", {1: 2, 2: 2})
    monkeypatch.setattr(MODULE, "PARTICLE_STACK", stack)
    monkeypatch.setattr(MODULE, "RELION_GROUP_COUNT", 2)
    monkeypatch.setitem(
        MODULE.SOURCE_SHA256,
        "half_assignment",
        hashlib.sha256(bytes([1, 2, 1, 2])).hexdigest(),
    )

    audit = MODULE.build_authoritative_star(
        legacy_path,
        raw_path,
        output_path,
        expected_output_size=len(expected_bytes),
        expected_output_sha256=hashlib.sha256(expected_bytes).hexdigest(),
    )

    assert output_path.read_bytes() == expected_bytes
    assert audit["authoritative_refinement_star"] is True
    assert audit["group_count"] == 2
    assert audit["group_name_number_bijection"] is True
    particles, reread_optics = MODULE.read_star(str(output_path))
    assert reread_optics is not None
    assert particles["_rlnGroupNumber"].astype(int).tolist() == [1, 1, 2, 2]


def test_pinned_download_and_decompression_reject_corruption(tmp_path: Path) -> None:
    import gzip

    payload = b"sealed EMD map fixture\n" * 100
    source = tmp_path / "source.map.gz"
    with gzip.GzipFile(filename=source, mode="wb", mtime=0) as stream:
        stream.write(payload)
    gzip_digest = MODULE.sha256_file(source)
    destination = tmp_path / "download.map.gz"

    MODULE.download_pinned(
        source.as_uri(),
        destination,
        size_bytes=source.stat().st_size,
        sha256=gzip_digest,
    )
    output = tmp_path / "download.map"
    MODULE.decompress_pinned_gzip(
        destination,
        output,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert output.read_bytes() == payload

    output.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="unexpected size"):
        MODULE.decompress_pinned_gzip(
            destination,
            output,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_average_relion_maps_is_float64_accumulated_float32_mean(tmp_path: Path) -> None:
    first = np.arange(5**3, dtype=np.float32).reshape(5, 5, 5)
    second = np.flip(first, axis=0).copy()
    first_path = tmp_path / "first.mrc"
    second_path = tmp_path / "second.mrc"
    output = tmp_path / "average.mrc"
    _write_relion_map(first_path, first, voxel_size=MODULE.VOXEL_SIZE_ANGSTROM)
    _write_relion_map(second_path, second, voxel_size=MODULE.VOXEL_SIZE_ANGSTROM)

    MODULE.average_relion_maps((first_path, second_path), output)

    with mrcfile.open(output, permissive=False) as mrc:
        expected = ((first.astype(np.float64) + second.astype(np.float64)) * 0.5).astype(np.float32)
        np.testing.assert_array_equal(mrc.data, expected)
        assert float(mrc.voxel_size.x) == pytest.approx(MODULE.VOXEL_SIZE_ANGSTROM)


def test_reference_twins_are_exact_in_canonical_array_and_hash(tmp_path: Path) -> None:
    relion_raw = np.linspace(-2.0, 3.0, 7**3, dtype=np.float32).reshape(7, 7, 7)
    relion_path = tmp_path / "relion.mrc"
    recovar_path = tmp_path / "recovar.mrc"
    _write_relion_map(relion_path, relion_raw, voxel_size=MODULE.VOXEL_SIZE_ANGSTROM)

    MODULE.write_recovar_frame_twin(relion_path, recovar_path)

    canonical = np.asarray(helpers.load_relion_volume(relion_path), dtype="<f4")
    np.testing.assert_array_equal(helpers.load_mrc(recovar_path), canonical)
    assert MODULE.canonical_array_sha256(relion_path) == hashlib.sha256(
        canonical.tobytes(order="C")
    ).hexdigest()


def test_relion_convert_star_command_and_raw_output_are_sealed(monkeypatch, tmp_path: Path) -> None:
    stack = (tmp_path / "particles.mrcs").resolve()
    legacy_table = MODULE.normalized_particle_table(_particle_table(), stack)
    particles, optics = _converted_tables(legacy_table)
    legacy_path = tmp_path / "legacy.star"
    MODULE.write_deterministic_star(legacy_path, legacy_table)
    fixture_path = tmp_path / "fixture_optics.star"
    MODULE.write_deterministic_star31(fixture_path, particles, optics)
    fixture_bytes = fixture_path.read_bytes()
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    converter = tmp_path / "relion_convert_star"
    converter.write_bytes(b"sealed converter fixture")
    output = tmp_path / "raw_converter_output.star"
    log = tmp_path / "convert.log"
    observed: list[str] = []

    def fake_run(command, **kwargs):
        observed.extend(command)
        Path(command[command.index("--o") + 1]).write_bytes(fixture_bytes)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(MODULE, "PARTICLE_COUNT", 4)
    monkeypatch.setattr(MODULE, "HALF_COUNTS", {1: 2, 2: 2})
    monkeypatch.setattr(MODULE, "PARTICLE_STACK", stack)
    result = MODULE.convert_legacy_star(
        converter,
        legacy_path,
        output,
        log,
        expected_output_size=len(fixture_bytes),
        expected_output_sha256=fixture_sha256,
        expected_half_assignment_sha256=hashlib.sha256(bytes([1, 2, 1, 2])).hexdigest(),
    )

    assert result["sha256"] == fixture_sha256
    assert result["authoritative_refinement_star"] is False
    assert observed == [
        str(converter),
        "--i",
        str(legacy_path),
        "--o",
        str(output.with_suffix(".partial.star")),
        "--box_size",
        "800",
    ]
    assert "--Cs" not in observed
    assert "--Q0" not in observed
    assert output.read_bytes() == fixture_bytes


def test_relion_reference_command_is_explicit_i1_lowpass_and_box(monkeypatch, tmp_path: Path) -> None:
    handler = tmp_path / "relion_image_handler"
    handler.write_bytes(b"fixture")
    average = tmp_path / "average.mrc"
    average.write_bytes(b"fixture")
    output = tmp_path / "reference.mrc"
    log = tmp_path / "handler.log"
    observed: list[str] = []

    def fake_run(command, **kwargs):
        observed.extend(command)
        Path(command[command.index("--o") + 1]).write_bytes(b"reference")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        MODULE,
        "mrc_header",
        lambda path: {
            "shape": [MODULE.BOX_SIZE] * 3,
            "mode": 2,
            "voxel_size_angstrom": [MODULE.VOXEL_SIZE_ANGSTROM] * 3,
        },
    )

    command = MODULE.run_relion_reference_preparation(handler, average, output, log)

    assert command == observed
    assert command[command.index("--sym") + 1] == "I1"
    assert command[command.index("--lowpass") + 1] == "30.0"
    assert command[command.index("--new_box") + 1] == "800"
    assert output.read_bytes() == b"reference"


def test_production_contract_pins_set6_and_excludes_ambiguous_icosahedral_alias() -> None:
    assert MODULE.SOURCE_STAR.parent.name == "06_Final_Stack"
    assert "07_Final_Stack" not in str(MODULE.SOURCE_STAR)
    assert MODULE.PARTICLE_COUNT == 30_515
    assert MODULE.HALF_COUNTS == {1: 15_258, 2: 15_257}
    assert MODULE.NORMALIZED_LEGACY_STAR_SHA256 == (
        "6c5143a5c4f64bd04c51f44f741abd5cfcde7082ad537e256e45c678fe9e2c2f"
    )
    assert MODULE.RAW_CONVERTED_STAR_SHA256 == (
        "d8693e4fb788181bf0f3a3b1e4dfe98470b5f83db4f65978f6700783db12dd09"
    )
    assert MODULE.RAW_CONVERTED_STAR_SIZE_BYTES == 12_440_248
    assert MODULE.PREPARED_STAR_SHA256 == (
        "d66afb3001e6e43463fb699804fe8b50f8cef1f2ccd9730f5275955b6be7b512"
    )
    assert MODULE.PREPARED_STAR_SIZE_BYTES == 10_313_843
    assert MODULE.RELION_CONVERT_STAR_SHA256 == (
        "dc6e1128c1c793d0cd1f6de5380aed1145fd48f9b0d5de3ec94f611dfe05bb4a"
    )
    assert MODULE.RELION_GROUP_COUNT == 176
    assert MODULE.ORIGIN_ROUNDTRIP_ATOL_PIXELS == 5e-7
    assert MODULE.SYMMETRY == "I1"
    assert MODULE.SYMMETRY_OPERATOR_COUNT == 60
    assert all(len(spec["gzip_sha256"]) == 64 for spec in MODULE.EMD_HALF_MAPS)
    assert all(len(spec["map_sha256"]) == 64 for spec in MODULE.EMD_HALF_MAPS)


def test_cli_accepts_explicit_sealed_relion_tools(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    image_handler = tmp_path / "relion_image_handler"
    converter = tmp_path / "relion_convert_star"

    args = MODULE.parse_args(
        [
            "--output-dir",
            str(output),
            "--relion-image-handler",
            str(image_handler),
            "--relion-convert-star",
            str(converter),
        ]
    )

    assert args.output_dir == output
    assert args.relion_image_handler == image_handler
    assert args.relion_convert_star == converter


def test_prepare_refuses_to_overwrite_sealed_output(tmp_path: Path) -> None:
    (tmp_path / "preparation_manifest.json").write_text("sealed\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.prepare(
            tmp_path,
            tmp_path / "relion_image_handler",
            tmp_path / "relion_convert_star",
        )
