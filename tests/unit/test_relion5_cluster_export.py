"""Native RELION cluster export must select physical identities, not row offsets."""

import hashlib
import json
import pickle
from argparse import Namespace
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from recovar.data_io.relion5_cluster_export import _write_star, export_clusters

pytestmark = pytest.mark.unit


def _pickle(path, value):
    with Path(path).open("wb") as handle:
        pickle.dump(value, handle)


def _read(path):
    return starfile.read(
        path,
        always_dict=True,
        parse_as_string=[
            "rlnGroupName",
            "rlnTomoParticleName",
            "rlnTomoName",
            "rlnTomoVisibleFrames",
        ],
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def export_case(tmp_path):
    """Native rows, canonical particles and flat image rows deliberately differ."""
    data = tmp_path / "native_project"
    data.mkdir()
    (data / "stacks").mkdir()
    (data / "tilts").mkdir()
    (data / "micrographs").mkdir()
    names = ["tomo/P2", "tomo/P10", "tomo/P1", "tomo/P3", "tomo/P4"]
    native = pd.DataFrame(
        {
            "rlnTomoName": ["tomo"] * 5,
            "rlnTomoParticleName": names,
            "rlnImageName": [f"stacks/p{index}.mrcs" for index in range(5)],
            "rlnOpticsGroup": [1, 2, 1, 2, 1],
            "rlnTomoVisibleFrames": ["[1,0,1]"] * 5,
            "rlnRandomSubset": [2, 1, 1, 2, 2],
            "rlnAngleRot": [11.0, 22.0, 33.0, 44.0, 55.0],
            "rlnAngleTilt": [31.0, 32.0, 33.0, 34.0, 35.0],
            "rlnAnglePsi": [-11.0, -12.0, -13.0, -14.0, -15.0],
            "rlnTomoSubtomogramRot": [1.0, 2.0, 3.0, 4.0, 5.0],
            "rlnTomoSubtomogramTilt": [6.0, 7.0, 8.0, 9.0, 10.0],
            "rlnTomoSubtomogramPsi": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "rlnOriginXAngst": [1.1, 1.2, 1.3, 1.4, 1.5],
            "rlnOriginYAngst": [2.1, 2.2, 2.3, 2.4, 2.5],
            "rlnOriginZAngst": [3.1, 3.2, 3.3, 3.4, 3.5],
            "rlnOriginX": [0.1, 0.2, 0.3, 0.4, 0.5],
            "rlnOriginXPriorAngst": [4.1, 4.2, 4.3, 4.4, 4.5],
            "rlnAngleRotPrior": [61.0, 62.0, 63.0, 64.0, 65.0],
            "rlnCenteredCoordinateXAngst": [100.0, 200.0, 300.0, 400.0, 500.0],
            "rlnCenteredCoordinateYAngst": [-100.0, -200.0, -300.0, -400.0, -500.0],
            "rlnCenteredCoordinateZAngst": [10.0, 20.0, 30.0, 40.0, 50.0],
            "rlnCustomParticleValue": ["keep2", "keep10", "keep1", "keep3", "keep4"],
        }
    )
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1, 2],
            "rlnOpticsGroupName": ["first", "second"],
            "rlnImagePixelSize": [1.5, 1.5],
            "rlnTomoTiltSeriesPixelSize": [1.5, 1.5],
            "rlnImageSize": [128, 128],
            "rlnImageDimensionality": [2, 2],
            "rlnCtfDataAreCtfPremultiplied": [0, 0],
            "rlnVoltage": [300.0, 300.0],
            "rlnSphericalAberration": [2.7, 2.7],
            "rlnAmplitudeContrast": [0.1, 0.1],
        }
    )
    native_path = data / "particles.star"
    starfile.write(
        {
            "general": {"rlnTomoSubTomosAre2DStacks": 1, "rlnCustomFlag": "keep_me"},
            "optics": optics,
            "particles": native,
            "audit": pd.DataFrame({"rlnAuditKey": ["untouched"], "rlnAuditValue": [17]}),
        },
        native_path,
    )
    for index, relative_path in enumerate(native["rlnImageName"]):
        with mrcfile.new(data / relative_path, overwrite=True) as handle:
            handle.set_data(np.full((2, 128, 128), index, dtype=np.float32))
            handle.voxel_size = 1.5
    for index in range(3):
        with mrcfile.new(data / "micrographs" / f"frame{index}.mrc", overwrite=True) as handle:
            handle.set_data(np.full((128, 128), index, dtype=np.float32))
            handle.voxel_size = 1.5
    tilt_path = data / "tilts" / "tomo.star"
    starfile.write(
        {
            "tomo": pd.DataFrame(
                {
                    "rlnMicrographName": [f"micrographs/frame{i}.mrc" for i in range(3)],
                    "rlnTomoXTilt": [0.0, 0.0, 0.0],
                    "rlnTomoYTilt": [-20.0, 0.0, 20.0],
                    "rlnTomoZRot": [0.0, 0.0, 0.0],
                    "rlnTomoXShiftAngst": [0.0, 0.0, 0.0],
                    "rlnTomoYShiftAngst": [0.0, 0.0, 0.0],
                    "rlnMicrographPreExposure": [0.0, 3.0, 6.0],
                    "rlnDefocusU": [15000.0, 15000.0, 15000.0],
                    "rlnDefocusV": [16000.0, 16000.0, 16000.0],
                    "rlnDefocusAngle": [0.0, 0.0, 0.0],
                    "rlnCtfScalefactor": [1.0, 1.0, 1.0],
                    "rlnTomoNominalStageTiltAngle": [-20.0, 0.0, 20.0],
                }
            ),
            "audit_geometry": {"rlnGeometryDescription": "preserve_this_geometry"},
        },
        tilt_path,
    )
    tomograms_path = data / "tomograms.star"
    starfile.write(
        {
            "global": pd.DataFrame(
                {
                    "rlnTomoName": ["tomo"],
                    "rlnTomoTiltSeriesStarFile": ["tilts/tomo.star"],
                    "rlnTomoTiltSeriesPixelSize": [1.5],
                    "rlnVoltage": [300.0],
                    "rlnSphericalAberration": [2.7],
                    "rlnAmplitudeContrast": [0.1],
                    "rlnTomoHand": [-1],
                    "rlnTomoSizeX": [128],
                    "rlnTomoSizeY": [128],
                    "rlnTomoSizeZ": [128],
                }
            )
        },
        tomograms_path,
    )

    # Conversion/downsampling can rearrange images and change stack references.
    # Neither that order nor native row order defines canonical particle IDs.
    flat_path = tmp_path / "particles_D64.star"
    flat_optics = optics.iloc[:1].copy()
    flat_optics["rlnImageSize"] = 64
    flat_optics["rlnImagePixelSize"] = 3.0
    flat_names = names + names
    flat = pd.DataFrame(
        {
            "rlnGroupName": flat_names,
            "rlnImageName": [f"{index + 1:06d}@downsampled.mrcs" for index in range(10)],
            "rlnOpticsGroup": [1] * 10,
            "rlnRandomSubset": [2, 1, 1, 2, 2] * 2,
            "rlnMicrographPreExposure": [0.0] * 5 + [6.0] * 5,
        }
    )
    starfile.write({"optics": flat_optics, "particles": flat}, flat_path)
    pipeline = tmp_path / "pipeline"
    (pipeline / "model").mkdir(parents=True)
    _pickle(
        pipeline / "model" / "params.pkl",
        {
            "version": "0.7",
            "input_args": Namespace(particles=str(flat_path), tilt_series=True, datadir=None, ntilts=-1),
        },
    )
    _pickle(pipeline / "model" / "particles_halfsets.pkl", [np.array([2, 0, 4]), np.array([3, 1])])
    _pickle(pipeline / "model" / "halfsets.pkl", [np.array([0, 2, 4, 5, 7, 9]), np.array([1, 3, 6, 8])])
    analysis = tmp_path / "analysis"
    (analysis / "data").mkdir(parents=True)
    _pickle(
        analysis / "data" / "kmeans_result.pkl",
        {
            "centers": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "labels": np.array([1, 0, 0, 1, 0]),
        },
    )
    return Namespace(
        pipeline=pipeline,
        analysis=analysis,
        particles=native_path,
        tomograms=tomograms_path,
        outdir=tmp_path / "exported",
        datadir=data,
        flat=flat_path,
        native=native,
    )


def _export(case, **kwargs):
    options = {
        key: str(getattr(case, key))
        for key in (
            "pipeline",
            "analysis",
            "particles",
            "tomograms",
            "outdir",
            "datadir",
        )
    }
    options.update(kwargs)
    return export_clusters(**options)


def _labels(case, values):
    _pickle(
        case.analysis / "data" / "kmeans_result.pkl",
        {
            "centers": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "labels": np.asarray(values),
        },
    )


def _halfsets(case, first, second):
    _pickle(case.pipeline / "model" / "particles_halfsets.pkl", [np.asarray(first), np.asarray(second)])
    flat = _read(case.flat)["particles"]
    names = sorted(flat["rlnGroupName"].unique())
    if all(0 <= index < len(names) for index in list(first) + list(second)):
        image_halfsets = [
            np.flatnonzero(flat["rlnGroupName"].isin([names[index] for index in subset]).to_numpy())
            for subset in (first, second)
        ]
        _pickle(case.pipeline / "model" / "halfsets.pkl", image_halfsets)


def _cluster(case, cluster=0):
    return _read(case.outdir / f"cluster{cluster}" / "particles.star")


def test_lexical_identity_mapping_and_complete_outputs(export_case):
    case = export_case
    manifest = _export(case)
    assert isinstance(manifest, dict)
    assert json.loads((case.outdir / "manifest.json").read_text()) == manifest
    assert (case.outdir / "summary.tsv").is_file()
    expected = {0: ["tomo/P2", "tomo/P10", "tomo/P4"], 1: ["tomo/P1", "tomo/P3"]}
    for cluster, names in expected.items():
        folder = case.outdir / f"cluster{cluster}"
        assert _cluster(case, cluster)["particles"]["rlnTomoParticleName"].tolist() == names
        assert (folder / "optimisation_set.star").is_file()
        membership = pd.read_csv(folder / "membership.tsv", sep="\t")
        assert len(membership) == len(names)
        assert membership["rlnTomoParticleName"].tolist() == names
        canonical = sorted(case.native["rlnTomoParticleName"])
        assert membership["recovar_particle_index"].tolist() == [canonical.index(name) for name in names]
        assert membership["native_particle_row"].tolist() == [
            case.native["rlnTomoParticleName"].tolist().index(name) for name in names
        ]
        assert membership["cluster_particle_row"].tolist() == list(range(len(names)))
        assert membership["cluster_label"].tolist() == [cluster] * len(names)
        assert membership["native_visible_tilts"].tolist() == [2] * len(names)
        optimisation = next(iter(_read(folder / "optimisation_set.star").values()))
        assert Path(optimisation["rlnTomoParticlesFile"]).resolve() == (folder / "particles.star").resolve()
        assert Path(optimisation["rlnTomoTomogramsFile"]).resolve() == (case.outdir / "tomograms.star").resolve()
    assert sum(len(_cluster(case, index)["particles"]) for index in (0, 1)) == 5


def test_numeric_looking_particle_names_preserve_leading_zeros(export_case):
    case = export_case
    rename = {"tomo/P2": "002", "tomo/P10": "010", "tomo/P1": "001", "tomo/P3": "003", "tomo/P4": "004"}
    native_blocks = _read(case.particles)
    native_blocks["particles"]["rlnTomoParticleName"] = native_blocks["particles"]["rlnTomoParticleName"].map(rename)
    starfile.write(native_blocks, case.particles, overwrite=True)
    flat_blocks = _read(case.flat)
    flat_blocks["particles"]["rlnGroupName"] = flat_blocks["particles"]["rlnGroupName"].map(rename)
    starfile.write(flat_blocks, case.flat, overwrite=True)
    _halfsets(case, [2, 0, 4], [3, 1])
    _export(case)
    assert _cluster(case)["particles"]["rlnTomoParticleName"].tolist() == ["002", "010", "003"]


@pytest.mark.parametrize("padding", [np.nan, np.iinfo(np.int64).min])
def test_sparse_shuffled_halfsets_gaps_and_excluded_tail(export_case, padding):
    case = export_case
    # Original canonical IDs 0=P1, 1=P10, 2=P2, 3=P3, 4=P4.
    # Keep 0 and 2; omitted label positions 3 and 4 are filtered tail particles.
    _halfsets(case, [2], [0])
    _labels(case, [1, padding, 0])
    _export(case)
    assert _cluster(case, 0)["particles"]["rlnTomoParticleName"].tolist() == ["tomo/P2"]
    assert _cluster(case, 1)["particles"]["rlnTomoParticleName"].tolist() == ["tomo/P1"]


def test_positive_label_outside_retained_particles_rejected(export_case):
    _halfsets(export_case, [2], [0])
    _labels(export_case, [1, 0, 0])
    with pytest.raises(ValueError):
        _export(export_case)


@pytest.mark.parametrize(
    "image_halves",
    [
        ([1, 3, 6, 8], [0, 2, 4, 5, 7, 9]),
        ([0, 2, 5, 7], [1, 3, 6, 8]),
        ([0, 2, 4, 5, 7], [1, 3, 6, 8, 9]),
    ],
)
def test_image_halfsets_must_match_corresponding_physical_halfsets(export_case, image_halves):
    _pickle(export_case.pipeline / "model" / "halfsets.pkl", [np.asarray(half) for half in image_halves])
    with pytest.raises(ValueError):
        _export(export_case)
    assert not export_case.outdir.exists()


def test_image_subset_retains_full_native_visibility_for_selected_particles(export_case):
    # One retained image per original physical group is enough for membership;
    # native export still keeps both visible frames, not just the RECOVAR subset.
    _pickle(export_case.pipeline / "model" / "halfsets.pkl", [np.array([0, 2, 4]), np.array([1, 3])])
    _export(export_case)
    for cluster in (0, 1):
        particles = _cluster(export_case, cluster)["particles"]
        assert particles["rlnTomoVisibleFrames"].eq("[1,0,1]").all()
        membership = pd.read_csv(export_case.outdir / f"cluster{cluster}" / "membership.tsv", sep="\t")
        assert membership["native_visible_tilts"].eq(2).all()


@pytest.mark.parametrize("first,second", [([0, 0], [1]), ([0, 2], [2]), ([-1, 0], [1]), ([0, 5], [1])])
def test_invalid_original_halfset_ids_rejected(export_case, first, second):
    _halfsets(export_case, first, second)
    with pytest.raises(ValueError):
        _export(export_case)


@pytest.mark.parametrize("bad_label", [np.nan, np.inf, -1, 0.5, 2])
def test_invalid_retained_cluster_label_rejected(export_case, bad_label):
    labels = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    labels[2] = bad_label
    _labels(export_case, labels)
    with pytest.raises(ValueError):
        _export(export_case)


def test_missing_retained_label_rejected(export_case):
    _labels(export_case, [1, 0, 0, 1])
    with pytest.raises(ValueError):
        _export(export_case)


@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_missing_or_ambiguous_native_identity_rejected(export_case, failure):
    case = export_case
    blocks = _read(case.particles)
    if failure == "missing":
        blocks["particles"] = blocks["particles"].iloc[1:].copy()
    else:
        blocks["particles"].loc[1, "rlnTomoParticleName"] = "tomo/P2"
    starfile.write(blocks, case.particles, overwrite=True)
    with pytest.raises(ValueError):
        _export(case)


@pytest.mark.parametrize("visible", ["[1,2,0]", "[1,1]", "[0,0,0]"])
def test_invalid_native_visible_frames_rejected(export_case, visible):
    case = export_case
    blocks = _read(case.particles)
    blocks["particles"].loc[0, "rlnTomoVisibleFrames"] = visible
    starfile.write(blocks, case.particles, overwrite=True)
    with pytest.raises(ValueError):
        _export(case)


def test_missing_selected_native_stack_rejected(export_case):
    case = export_case
    blocks = _read(case.particles)
    blocks["particles"].loc[0, "rlnImageName"] = "stacks/nonexistent.mrcs"
    starfile.write(blocks, case.particles, overwrite=True)
    with pytest.raises((FileNotFoundError, ValueError)):
        _export(case)


@pytest.mark.parametrize("mismatch", [False, True])
def test_optional_flattened_tomogram_identity_checked(export_case, mismatch):
    documents = _read(export_case.flat)
    documents["particles"]["rlnTomoName"] = "tomo"
    if mismatch:
        documents["particles"].loc[documents["particles"]["rlnGroupName"] == "tomo/P2", "rlnTomoName"] = (
            "another_tomogram"
        )
    starfile.write(documents, export_case.flat, overwrite=True)
    if mismatch:
        with pytest.raises(ValueError):
            _export(export_case)
        assert not export_case.outdir.exists()
    else:
        _export(export_case)
        assert _cluster(export_case)["particles"]["rlnTomoParticleName"].tolist() == ["tomo/P2", "tomo/P10", "tomo/P4"]


@pytest.mark.parametrize(
    "source,block,column",
    [
        ("particles", "optics", "rlnImageSize"),
        ("particles", "optics", "rlnImagePixelSize"),
        ("particles", "optics", "rlnTomoTiltSeriesPixelSize"),
        ("particles", "optics", "rlnVoltage"),
        ("particles", "optics", "rlnSphericalAberration"),
        ("particles", "optics", "rlnAmplitudeContrast"),
        ("tomograms", "global", "rlnTomoTiltSeriesPixelSize"),
        ("tomograms", "global", "rlnVoltage"),
        ("tomograms", "global", "rlnSphericalAberration"),
        ("tomograms", "global", "rlnAmplitudeContrast"),
    ],
)
def test_missing_required_relion_native_fields_rejected(export_case, source, block, column):
    path = getattr(export_case, source)
    documents = _read(path)
    documents[block] = documents[block].drop(columns=column)
    starfile.write(documents, path, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)
    assert not export_case.outdir.exists()


@pytest.mark.parametrize(
    "source,block,column",
    [
        ("particles", "optics", "rlnTomoTiltSeriesPixelSize"),
        ("particles", "optics", "rlnVoltage"),
        ("particles", "optics", "rlnSphericalAberration"),
        ("particles", "optics", "rlnAmplitudeContrast"),
        ("tomograms", "global", "rlnTomoTiltSeriesPixelSize"),
        ("tomograms", "global", "rlnVoltage"),
        ("tomograms", "global", "rlnSphericalAberration"),
        ("tomograms", "global", "rlnAmplitudeContrast"),
    ],
)
def test_nonfinite_required_relion_native_fields_rejected(export_case, source, block, column):
    path = getattr(export_case, source)
    documents = _read(path)
    documents[block].loc[0, column] = np.inf
    starfile.write(documents, path, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)
    assert not export_case.outdir.exists()


def test_native_metadata_and_D128_stacks_preserved_from_D64_analysis(export_case):
    case = export_case
    source_blocks = _read(case.particles)
    hashes = {path: _sha256(path) for path in case.datadir.rglob("*") if path.is_file()}
    _export(case)
    exported = _cluster(case)
    assert exported["general"] == source_blocks["general"]
    # STAR has no numeric dtype declaration; 300 and 300.0 are equivalent.
    pd.testing.assert_frame_equal(exported["optics"], source_blocks["optics"], check_dtype=False)
    pd.testing.assert_frame_equal(exported["audit"], source_blocks["audit"], check_dtype=False)
    names = exported["particles"]["rlnTomoParticleName"]
    expected = (
        source_blocks["particles"]
        .loc[source_blocks["particles"]["rlnTomoParticleName"].isin(names)]
        .reset_index(drop=True)
    )
    # Path rebasing is allowed; all scientific particle fields must be untouched.
    pd.testing.assert_frame_equal(
        exported["particles"].drop(columns="rlnImageName"),
        expected.drop(columns="rlnImageName"),
        check_dtype=False,
    )
    assert exported["optics"]["rlnImageSize"].tolist() == [128, 128]
    assert exported["optics"]["rlnImagePixelSize"].tolist() == [1.5, 1.5]
    for source_ref, exported_ref in zip(expected["rlnImageName"], exported["particles"]["rlnImageName"]):
        path = Path(exported_ref)
        if not path.is_absolute():
            path = case.datadir / path
        assert path.resolve() == (case.datadir / source_ref).resolve()
        with mrcfile.open(path) as handle:
            assert handle.data.shape == (2, 128, 128)
    assert hashes == {path: _sha256(path) for path in hashes}


def test_relative_paths_use_explicit_datadir_not_current_directory(export_case, tmp_path, monkeypatch):
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    _export(export_case, clusters=[0])
    assert _cluster(export_case)["particles"]["rlnTomoParticleName"].tolist() == ["tomo/P2", "tomo/P10", "tomo/P4"]
    assert not (export_case.outdir / "cluster1").exists()


@pytest.mark.parametrize("with_datadir", [False, True])
def test_copied_nested_geometry_rebases_micrographs_without_changing_values(export_case, with_datadir):
    case = export_case
    source_path = case.datadir / "tilts" / "tomo.star"
    source = _read(source_path)
    if not with_datadir:
        source["tomo"]["rlnMicrographName"] = [f"../micrographs/frame{index}.mrc" for index in range(3)]
        starfile.write(source, source_path, overwrite=True)
    source_hash = _sha256(source_path)
    _export(case, datadir=str(case.datadir) if with_datadir else None)
    tomograms = _read(case.outdir / "tomograms.star")["global"]
    exported_path = Path(tomograms.loc[0, "rlnTomoTiltSeriesStarFile"])
    assert exported_path.is_absolute()
    assert exported_path.parent == case.outdir / "tilt_series"
    assert exported_path != source_path
    exported = _read(exported_path)
    assert exported["audit_geometry"] == source["audit_geometry"]
    pd.testing.assert_frame_equal(
        exported["tomo"].drop(columns="rlnMicrographName"),
        source["tomo"].drop(columns="rlnMicrographName"),
        check_dtype=False,
    )
    assert exported["tomo"]["rlnMicrographName"].tolist() == [
        str(case.datadir / "micrographs" / f"frame{index}.mrc") for index in range(3)
    ]
    for path in exported["tomo"]["rlnMicrographName"]:
        with mrcfile.open(path) as handle:
            assert handle.data.shape == (128, 128)
    assert _sha256(source_path) == source_hash


def test_missing_first_full_micrograph_rejected(export_case):
    geometry_path = export_case.datadir / "tilts" / "tomo.star"
    documents = _read(geometry_path)
    documents["tomo"].loc[0, "rlnMicrographName"] = "micrographs/missing_first.mrc"
    starfile.write(documents, geometry_path, overwrite=True)
    with pytest.raises((FileNotFoundError, ValueError)):
        _export(export_case)
    assert not export_case.outdir.exists()


def test_nested_index_at_stack_references_keep_frame_indices(export_case):
    case = export_case
    stack = case.datadir / "micrographs" / "all_frames.mrcs"
    with mrcfile.new(stack, overwrite=True) as handle:
        handle.set_data(np.arange(3 * 128 * 128, dtype=np.float32).reshape(3, 128, 128))
        handle.voxel_size = 1.5
    stack_hash = _sha256(stack)
    geometry_path = case.datadir / "tilts" / "tomo.star"
    documents = _read(geometry_path)
    prefixes = ["000003", "000001", "000002"]
    documents["tomo"]["rlnMicrographName"] = [f"{prefix}@micrographs/all_frames.mrcs" for prefix in prefixes]
    starfile.write(documents, geometry_path, overwrite=True)
    _export(case)
    tomograms = _read(case.outdir / "tomograms.star")["global"]
    copied = _read(tomograms.loc[0, "rlnTomoTiltSeriesStarFile"])
    assert copied["tomo"]["rlnMicrographName"].tolist() == [f"{prefix}@{stack}" for prefix in prefixes]
    assert _sha256(stack) == stack_hash


@pytest.mark.parametrize(
    "column",
    [
        "rlnTomoYTilt",
        "rlnTomoZRot",
        "rlnTomoXShiftAngst",
        "rlnTomoYShiftAngst",
        "rlnDefocusU",
        "rlnDefocusV",
        "rlnDefocusAngle",
        "rlnMicrographPreExposure",
    ],
)
def test_missing_required_tilt_geometry_fields_rejected(export_case, column):
    geometry_path = export_case.datadir / "tilts" / "tomo.star"
    documents = _read(geometry_path)
    documents["tomo"] = documents["tomo"].drop(columns=column)
    starfile.write(documents, geometry_path, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)


def test_incomplete_native_coordinates_rejected(export_case):
    documents = _read(export_case.particles)
    documents["particles"] = documents["particles"].drop(columns="rlnCenteredCoordinateZAngst")
    starfile.write(documents, export_case.particles, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)


@pytest.mark.parametrize(
    "column",
    ["rlnCenteredCoordinateXAngst", "rlnAngleRot", "rlnOriginXAngst", "rlnTomoSubtomogramRot"],
)
def test_nonfinite_native_coordinate_and_pose_fields_rejected(export_case, column):
    documents = _read(export_case.particles)
    documents["particles"].loc[0, column] = np.inf
    starfile.write(documents, export_case.particles, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)
    assert not export_case.outdir.exists()


def test_partial_subtomogram_orientation_rejected(export_case):
    documents = _read(export_case.particles)
    documents["particles"] = documents["particles"].drop(columns="rlnTomoSubtomogramPsi")
    starfile.write(documents, export_case.particles, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)


def test_nonfinite_tilt_angle_rejected(export_case):
    geometry_path = export_case.datadir / "tilts" / "tomo.star"
    documents = _read(geometry_path)
    documents["tomo"].loc[0, "rlnTomoYTilt"] = np.inf
    starfile.write(documents, geometry_path, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)


@pytest.mark.parametrize("bad_vector", ["[1,0,0]", "[1,0,0,1e309]"])
def test_malformed_legacy_projection_vector_rejected(export_case, bad_vector):
    geometry_path = export_case.datadir / "tilts" / "tomo.star"
    documents = _read(geometry_path)
    documents["tomo"] = documents["tomo"].drop(
        columns=["rlnTomoYTilt", "rlnTomoZRot", "rlnTomoXShiftAngst", "rlnTomoYShiftAngst"],
    )
    for index, axis in enumerate("XYZW"):
        vector = [int(component == index) for component in range(4)]
        documents["tomo"][f"rlnTomoProj{axis}"] = str(vector)
    documents["tomo"].loc[0, "rlnTomoProjX"] = bad_vector
    starfile.write(documents, geometry_path, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)


def test_zero_tomogram_dimensions_preserved_for_centered_coordinates(export_case):
    documents = _read(export_case.tomograms)
    columns = [f"rlnTomoSize{axis}" for axis in "XYZ"]
    documents["global"][columns] = 0
    starfile.write(documents, export_case.tomograms, overwrite=True)
    _export(export_case)
    np.testing.assert_array_equal(_read(export_case.outdir / "tomograms.star")["global"][columns], 0)


def test_missing_tomogram_dimension_rejected(export_case):
    documents = _read(export_case.tomograms)
    documents["global"] = documents["global"].drop(columns="rlnTomoSizeZ")
    starfile.write(documents, export_case.tomograms, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)


def test_complete_native_pixel_coordinates_supported(export_case):
    documents = _read(export_case.particles)
    columns = [f"rlnCenteredCoordinate{axis}Angst" for axis in "XYZ"]
    documents["particles"] = documents["particles"].drop(columns=columns)
    for index, axis in enumerate("XYZ"):
        documents["particles"][f"rlnCoordinate{axis}"] = np.arange(5) + index
    starfile.write(documents, export_case.particles, overwrite=True)
    _export(export_case)
    particles = _cluster(export_case)["particles"]
    np.testing.assert_array_equal(particles["rlnCoordinateX"], [0, 1, 4])
    np.testing.assert_array_equal(particles["rlnCoordinateY"], [1, 2, 5])
    np.testing.assert_array_equal(particles["rlnCoordinateZ"], [2, 3, 6])


@pytest.mark.parametrize(
    "column",
    [
        "rlnVoltage",
        "rlnSphericalAberration",
        "rlnAmplitudeContrast",
        "rlnTomoTiltSeriesPixelSize",
    ],
)
def test_native_optics_and_tomogram_values_must_agree(export_case, column):
    documents = _read(export_case.tomograms)
    documents["global"].loc[0, column] += 0.05
    starfile.write(documents, export_case.tomograms, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)


def test_native_optics_agreement_allows_relion_absolute_tolerance(export_case):
    documents = _read(export_case.tomograms)
    documents["global"].loc[0, "rlnAmplitudeContrast"] += 0.0005
    starfile.write(documents, export_case.tomograms, overwrite=True)
    _export(export_case)
    assert len(_cluster(export_case)["particles"]) == 3


def test_nonfinite_tomogram_hand_rejected(export_case):
    documents = _read(export_case.tomograms)
    documents["global"]["rlnTomoHand"] = np.inf
    starfile.write(documents, export_case.tomograms, overwrite=True)
    with pytest.raises(ValueError):
        _export(export_case)


def test_explicit_flat_source_override_for_migrated_pipeline(export_case):
    case = export_case
    params_path = case.pipeline / "model" / "params.pkl"
    with params_path.open("rb") as handle:
        params = pickle.load(handle)
    params["input_args"].particles = "/missing/old_machine/particles.star"
    _pickle(params_path, params)
    _export(case, flat_particles=str(case.flat))
    assert _cluster(case)["particles"]["rlnTomoParticleName"].tolist() == ["tomo/P2", "tomo/P10", "tomo/P4"]


@pytest.mark.parametrize("field", ["parameters", "provenance"])
def test_analysis_from_another_pipeline_rejected(export_case, field):
    key = "result_dir" if field == "parameters" else "pipeline_result_dir"
    (export_case.analysis / "job.json").write_text(
        json.dumps(
            {
                "status": "completed",
                field: {key: str(export_case.pipeline.parent / "other_pipeline")},
            }
        )
    )
    with pytest.raises(ValueError):
        _export(export_case)
    assert not export_case.outdir.exists()


def test_matching_analysis_provenance_accepted(export_case):
    (export_case.analysis / "job.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "parameters": {"result_dir": str(export_case.pipeline)},
                "provenance": {"pipeline_result_dir": str(export_case.pipeline)},
            }
        )
    )
    _export(export_case)
    assert len(_cluster(export_case)["particles"]) == 3


@pytest.mark.parametrize("reset_poses,reset_halfsets", [(True, False), (False, True), (True, True)])
def test_reset_modes_are_independent(export_case, reset_poses, reset_halfsets):
    case = export_case
    _export(case, reset_poses=reset_poses, reset_halfsets=reset_halfsets)
    pose_columns = [
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
        "rlnOriginXAngst",
        "rlnOriginYAngst",
        "rlnOriginZAngst",
        "rlnOriginX",
        "rlnOriginXPriorAngst",
        "rlnAngleRotPrior",
    ]
    extraction_columns = ["rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"]
    original = case.native.set_index("rlnTomoParticleName")
    for index in (0, 1):
        particles = _cluster(case, index)["particles"].set_index("rlnTomoParticleName")
        expected = original.loc[particles.index]
        if reset_poses:
            np.testing.assert_array_equal(particles[pose_columns].to_numpy(), 0.0)
        else:
            np.testing.assert_allclose(particles[pose_columns], expected[pose_columns])
        np.testing.assert_allclose(particles[extraction_columns], expected[extraction_columns])
        if reset_halfsets:
            assert "rlnRandomSubset" not in particles.columns
        else:
            np.testing.assert_array_equal(particles["rlnRandomSubset"], expected["rlnRandomSubset"])
        np.testing.assert_array_equal(particles["rlnCenteredCoordinateXAngst"], expected["rlnCenteredCoordinateXAngst"])
        np.testing.assert_array_equal(particles["rlnTomoVisibleFrames"], expected["rlnTomoVisibleFrames"])


def test_existing_output_directory_is_not_overwritten(export_case):
    case = export_case
    case.outdir.mkdir()
    marker = case.outdir / "user_data.txt"
    marker.write_text("keep this existing output\n")
    with pytest.raises((FileExistsError, ValueError)):
        _export(case)
    assert marker.read_text() == "keep this existing output\n"
    assert list(case.outdir.iterdir()) == [marker]


def test_repeat_export_cannot_overwrite_completed_output(export_case):
    case = export_case
    _export(case)
    hashes = {path: _sha256(path) for path in case.outdir.rglob("*") if path.is_file()}
    with pytest.raises((FileExistsError, ValueError)):
        _export(case)
    assert hashes == {path: _sha256(path) for path in hashes}


def test_write_star_refuses_existing_file(tmp_path):
    path = tmp_path / "particles.star"
    path.write_text("keep this existing file\n")
    with pytest.raises(FileExistsError):
        _write_star(path, {"particles": pd.DataFrame({"rlnAngleRot": [0.0]})})
    assert path.read_text() == "keep this existing file\n"
