"""Independent native RELION5 geometry, pixel and provenance export checks."""

import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from scipy.spatial.transform import Rotation

from recovar.commands.parse_relion5_tomo import convert
from recovar.core.ctf import CTFParamIndex as C
from recovar.data_io.metadata_readers import parse_poses_from_star
from recovar.data_io.starfile import read_star
from recovar.simulation.relion5_export import export_relion5, native_ctf_generator, validate_output_directory

pytestmark = pytest.mark.unit


def _native_euler_matrix(angles):
    """RELION src/euler.cpp Euler_angles2matrix, not the exporter helper."""
    a, b, g = np.deg2rad(angles)
    ca, cb, cg = np.cos([a, b, g])
    sa, sb, sg = np.sin([a, b, g])
    return np.array(
        [
            [cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb],
            [-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb],
            [sb * ca, sb * sa, cb],
        ]
    )


def _native_projection_matrix(row):
    # RELION Tomogram::setProjectionMatrix: Rz(zrot) @ Ry(ytilt) @ Rx(xtilt).
    return Rotation.from_euler(
        "xyz", [row[f"rlnTomo{axis}"] for axis in ("XTilt", "YTilt", "ZRot")], degrees=True
    ).as_matrix()


@pytest.fixture
def simulation():
    n_particles, n_tilts, size = 4, 5, 8
    count = n_particles * n_tilts
    angles = np.array([0.0, 7.0, -7.0, 14.0, -14.0])
    base = Rotation.from_euler(
        "ZYZ", [[31, 48, -22], [-68, 71, 121], [20, 110, 33], [-15, 41, 66]], degrees=True
    ).as_matrix()
    tilts = Rotation.from_euler("x", angles, degrees=True).as_matrix()
    rotations = (base[:, None] @ tilts[None]).reshape(count, 3, 3)
    ctf = np.zeros((count, 11), dtype=np.float64)
    ctf[:, C.DFU] = 15000 + 10 * np.arange(count)
    ctf[:, C.DFV] = 16000 + 10 * np.arange(count)
    ctf[:, C.DFANG] = 23
    ctf[:, C.VOLT] = 300
    ctf[:, C.CS] = 2.7
    ctf[:, C.W] = 0.1
    ctf[:, C.CONTRAST] = np.tile(np.cos(np.deg2rad(angles)), n_particles)
    ctf[:, C.DOSE] = np.tile((np.arange(n_tilts) + 0.5) * 3, n_particles)
    ctf[:, C.BFACTOR] = -4 * ctf[:, C.DOSE]  # Legacy WARP metadata, not a physical envelope.
    return {
        "images": np.arange(count * size * size, dtype=np.float32).reshape(count, size, size) / 100,
        "ctf_params": ctf,
        "rotations": rotations,
        "translations": np.zeros((count, 2)),
        "voxel_size": 4.25,
        "tilt_groups": np.repeat([8, 2, 19, 7], n_tilts),
        "n_tilts": n_tilts,
        "dose_per_tilt": 3,
        "angle_per_tilt": 7,
        "simulation_info": {"forward_ctf_bfactor": np.zeros(count)},
    }


def test_native_pixels_ctf_frame_provenance_and_particle_halves(tmp_path, simulation):
    root = tmp_path / "export"
    original_images = simulation["images"].copy()
    original_ctf = simulation["ctf_params"].copy()
    manifest = export_relion5(root, **simulation)
    documents = starfile.read(root / "particles.star", always_dict=True)
    particles = documents["particles"]
    assert documents["general"]["rlnTomoSubTomosAre2DStacks"] == 1
    assert (documents["optics"]["rlnCtfDataAreCtfPremultiplied"] == 0).all()
    assert (documents["optics"]["rlnImageDimensionality"] == 2).all()
    np.testing.assert_array_equal(np.sort(particles["rlnRandomSubset"]), [1, 1, 2, 2])
    np.testing.assert_array_equal(np.load(root / "particle_groups.npy"), [8, 2, 19, 7])
    frame_rows = np.load(root / "frame_rows.npy")
    expected_rows = np.arange(20).reshape(4, 5)[:, [4, 2, 0, 1, 3]]
    np.testing.assert_array_equal(frame_rows, expected_rows)
    tomograms = starfile.read(root / "tomograms.star", always_dict=True)["global"]
    for p, (_, particle) in enumerate(particles.iterrows()):
        with mrcfile.open(particle["rlnImageName"]) as mrc:
            np.testing.assert_array_equal(mrc.data, -original_images[frame_rows[p]])
            assert float(mrc.voxel_size.x) == pytest.approx(simulation["voxel_size"])
        tomo = tomograms.iloc[p]
        tilt_table = next(iter(starfile.read(tomo["rlnTomoTiltSeriesStarFile"], always_dict=True).values()))
        for column, index in (
            ("rlnDefocusU", C.DFU),
            ("rlnDefocusV", C.DFV),
            ("rlnMicrographPreExposure", C.DOSE),
            ("rlnCtfScalefactor", C.CONTRAST),
        ):
            np.testing.assert_allclose(tilt_table[column], original_ctf[frame_rows[p], index], atol=1e-8)
        np.testing.assert_array_equal(tilt_table["rlnCtfBfactor"], 0)
        assert str(particle["rlnTomoVisibleFrames"]) == "[1,1,1,1,1]"
    np.testing.assert_array_equal(simulation["images"], original_images)
    np.testing.assert_array_equal(simulation["ctf_params"], original_ctf)
    assert manifest["intensity_multiplier"] == -1
    assert "cutoff" in manifest["dose_model_difference"]
    assert json.loads((root / "relion5_export.json").read_text())["status"] == "validated_complete"
    optimisation = starfile.read(root / "optimisation_set.star", always_dict=True)["optimisation_set"]
    assert Path(optimisation["rlnTomoParticlesFile"]) == root / "particles.star"
    assert Path(optimisation["rlnTomoTomogramsFile"]) == root / "tomograms.star"


def test_native_matrix_composition_matches_every_simulated_projection(tmp_path, simulation):
    """Noncommuting orientations detect wrong multiplication order or tilt sign."""
    export_relion5(tmp_path, **simulation)
    particles = starfile.read(tmp_path / "particles.star", always_dict=True)["particles"]
    tomograms = starfile.read(tmp_path / "tomograms.star", always_dict=True)["global"]
    rows = np.load(tmp_path / "frame_rows.npy")
    for p, (_, particle) in enumerate(particles.iterrows()):
        native_particle = _native_euler_matrix(particle[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float))
        tilt_table = next(
            iter(starfile.read(tomograms.iloc[p]["rlnTomoTiltSeriesStarFile"], always_dict=True).values())
        )
        for t, (_, tilt) in enumerate(tilt_table.iterrows()):
            # RELION ml_optimiser uses A = Aproj * A_particle.
            native_effective = _native_projection_matrix(tilt) @ native_particle
            np.testing.assert_allclose(native_effective, simulation["rotations"][rows[p, t]], atol=1e-8)


def test_reverse_converter_recovers_original_pose_dose_order(tmp_path, simulation):
    export_relion5(tmp_path, **simulation)
    flat_path = tmp_path / "back_to_recovar.star"
    convert(str(tmp_path / "tomograms.star"), str(tmp_path / "particles.star"), str(flat_path))
    rotations, translations = parse_poses_from_star(str(flat_path), D=8)
    np.testing.assert_allclose(rotations, simulation["rotations"], atol=1e-5)
    np.testing.assert_allclose(translations, 0, atol=1e-10)
    flat, _ = read_star(str(flat_path))
    np.testing.assert_allclose(flat["_rlnMicrographPreExposure"].astype(float), simulation["ctf_params"][:, C.DOSE])
    for row, image_name in enumerate(flat["_rlnImageName"]):
        number, path = image_name.split("@", 1)
        with mrcfile.open(path) as mrc:
            np.testing.assert_array_equal(mrc.data[int(number) - 1], -simulation["images"][row])


def test_projectable_nonzero_origins_are_in_angstroms(tmp_path, simulation):
    origins = np.array([[1, -2, 3], [3, 4, -1], [-1, 2, 5], [2, -3, 4]], dtype=float)
    rotations = simulation["rotations"].reshape(4, 5, 3, 3)
    base = rotations[:, 0]
    native_projection = rotations @ base[:, None].swapaxes(-1, -2)
    simulation["translations"] = (
        np.einsum("ptij,pj->pti", native_projection[:, :, :2], origins) / simulation["voxel_size"]
    ).reshape(20, 2)
    export_relion5(tmp_path, **simulation)
    particles = starfile.read(tmp_path / "particles.star", always_dict=True)["particles"]
    np.testing.assert_allclose(particles[[f"rlnOrigin{axis}Angst" for axis in "XYZ"]], origins, atol=1e-8)


@pytest.mark.parametrize(
    "problem", ["dose", "rotation", "translation", "partial", "premultiplied", "nan", "voltage", "amplitude"]
)
def test_invalid_native_inputs_fail_before_writing(tmp_path, simulation, problem):
    if problem == "dose":
        simulation["ctf_params"][2, C.DOSE] = -0.1
    elif problem == "rotation":
        simulation["rotations"][2] = Rotation.from_euler("x", 31, degrees=True).as_matrix()
    elif problem == "translation":
        simulation["translations"][1] = [9, 13]
    elif problem == "partial":
        simulation["tilt_groups"][-1] = 999
    elif problem == "premultiplied":
        simulation["simulation_info"]["premultiplied_ctf"] = True
    elif problem == "nan":
        simulation["images"][3, 1, 2] = np.nan
    elif problem == "voltage":
        simulation["ctf_params"][:, C.VOLT] = 200
    elif problem == "amplitude":
        simulation["ctf_params"][:, C.W] = -0.1
    with pytest.raises(ValueError):
        export_relion5(tmp_path, **simulation)
    assert list(tmp_path.iterdir()) == []


def test_actual_zero_dose_not_replaced_by_requested_schedule(tmp_path, simulation):
    """Export the model that generated the pixels, including legacy zero dose."""
    original_images = simulation["images"].copy()
    simulation["ctf_params"][:, C.DOSE] = 0
    manifest = export_relion5(tmp_path, **simulation)
    assert manifest["actual_dose_matches_nominal_schedule"] is False
    assert manifest["actual_dose_range"] == [0, 0]
    assert manifest["nominal_dose_range"] == [1.5, 13.5]
    tomograms = starfile.read(tmp_path / "tomograms.star", always_dict=True)["global"]
    np.testing.assert_array_equal(tomograms["rlnTomoImportFractionalDose"], 0)
    for tilt_path in tomograms["rlnTomoTiltSeriesStarFile"]:
        tilt = next(iter(starfile.read(tilt_path, always_dict=True).values()))
        np.testing.assert_array_equal(tilt["rlnMicrographPreExposure"], 0)
    np.testing.assert_array_equal(simulation["images"], original_images)


def test_native_ctf_wrapper_snapshots_without_changing_generator_arrays(simulation):
    snapshots = []
    parameters = simulation["ctf_params"].copy()
    parameters[:, C.BFACTOR] = np.arange(len(parameters))
    original_bfactor = parameters[:, C.BFACTOR].copy()
    arrays = (parameters, simulation["rotations"], simulation["translations"])

    def original(n_images, size):
        assert n_images == 20 and size == 8
        return arrays

    wrapped = native_ctf_generator(original, snapshots)
    output = wrapped(20, 8)
    assert all(left is right for left, right in zip(output, arrays))
    assert output[0].shape == (20, 11)
    assert len(snapshots) == 1
    np.testing.assert_array_equal(snapshots[0], original_bfactor)
    parameters[:, C.BFACTOR] = -999
    np.testing.assert_array_equal(snapshots[0], original_bfactor)


def test_single_tilt_stack_roundtrip(tmp_path, simulation):
    indices = np.array([0, 5, 10, 15])
    for key in ("images", "ctf_params", "rotations", "translations", "tilt_groups"):
        simulation[key] = simulation[key][indices]
    simulation["n_tilts"] = 1
    simulation["simulation_info"]["forward_ctf_bfactor"] = np.zeros(4)
    export_relion5(tmp_path, **simulation)
    particles = starfile.read(tmp_path / "particles.star", always_dict=True)["particles"]
    for i, path in enumerate(particles["rlnImageName"]):
        with mrcfile.open(path) as mrc:
            np.testing.assert_array_equal(mrc.data.reshape(1, 8, 8), -simulation["images"][i : i + 1])


def test_existing_output_not_overwritten(tmp_path, simulation):
    export_relion5(tmp_path, **simulation)
    original = (tmp_path / "particles.star").read_bytes()
    with pytest.raises(FileExistsError):
        export_relion5(tmp_path, **simulation)
    assert (tmp_path / "particles.star").read_bytes() == original


def test_validate_output_directory_rejects_nonempty_and_symlink(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    assert validate_output_directory(root) == root
    (root / "sentinel.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        validate_output_directory(root)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(FileExistsError):
        validate_output_directory(link)
    assert (root / "sentinel.txt").read_text() == "keep me"
