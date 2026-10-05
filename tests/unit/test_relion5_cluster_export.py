"""The RELION 5 cluster export selects native particles by name, not by row position."""

import pickle
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

from recovar.data_io.relion5_cluster_export import export_clusters

pytestmark = pytest.mark.unit

# Native row order differs from both RECOVAR order (sorted names) and flat image order.
# "0010" and "10" collide if the names are ever parsed as numbers.
NATIVE_NAMES = ["10", "0010", "007", "2", "0001", "unretained"]
# RECOVAR order is sorted(NATIVE_NAMES): 0001, 0010, 007, 10, 2, unretained.
LABELS = {"0001": 1, "0010": 0, "007": 2, "10": 1, "2": 0}


def _pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _read(path):
    return starfile.read(path, always_dict=True, parse_as_string=["rlnTomoParticleName", "rlnGroupName"])


def _case(tmp_path, labels=None, native_extra=None):
    data = tmp_path / "project"
    (data / "stacks").mkdir(parents=True)
    native = pd.DataFrame(
        {
            "rlnTomoName": ["tomo"] * len(NATIVE_NAMES),
            "rlnTomoParticleName": NATIVE_NAMES,
            "rlnImageName": [f"stacks/{name}.mrcs" for name in NATIVE_NAMES],
            "rlnOpticsGroup": 1,
            "rlnAngleRot": np.arange(len(NATIVE_NAMES)) + 0.1,
            "rlnRandomSubset": [1, 2] * (len(NATIVE_NAMES) // 2),
            **(native_extra or {}),
        }
    )
    for name in NATIVE_NAMES:
        (data / "stacks" / f"{name}.mrcs").touch()
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [1.5]})
    starfile.write(
        {"general": {"rlnTomoSubTomosAre2DStacks": 1}, "optics": optics, "particles": native}, data / "p.star"
    )
    (data / "tomograms.star").touch()

    flat_names = [name for name in reversed(NATIVE_NAMES) for _ in range(2)]
    flat = pd.DataFrame({"rlnGroupName": flat_names, "rlnImageName": [f"{i + 1}@flat.mrcs" for i in range(12)]})
    starfile.write({"optics": optics, "particles": flat}, tmp_path / "flat.star")
    _pickle(
        tmp_path / "pipeline" / "model" / "params.pkl", {"input_args": Namespace(particles=str(tmp_path / "flat.star"))}
    )
    if labels is None:
        labels = np.array([LABELS[name] for name in sorted(LABELS)])  # unretained tail is cut off
    _pickle(tmp_path / "analysis" / "data" / "kmeans_result.pkl", {"labels": labels, "centers": np.zeros((3, 2))})
    return {
        "pipeline": tmp_path / "pipeline",
        "analysis": tmp_path / "analysis",
        "particles": data / "p.star",
        "tomograms": data / "tomograms.star",
        "outdir": tmp_path / "out",
    }


def _expected(case, names):
    native = _read(case["particles"])["particles"]
    return native.loc[native.rlnTomoParticleName.isin(names)].reset_index(drop=True)


def test_clusters_select_native_rows_by_name(tmp_path):
    case = _case(tmp_path)
    summary = export_clusters(**case)
    assert summary.particles.tolist() == [2, 2, 1]
    for cluster in range(3):
        documents = _read(case["outdir"] / f"cluster{cluster}" / "particles.star")
        assert list(documents) == ["general", "optics", "particles"]
        written = documents["particles"]
        expected = _expected(case, [name for name, label in LABELS.items() if label == cluster])
        expected["rlnImageName"] = [str(case["particles"].parent / path) for path in expected.rlnImageName]
        pd.testing.assert_frame_equal(written, expected)
        optimisation = _read(case["outdir"] / f"cluster{cluster}" / "optimisation_set.star")[""]
        assert optimisation["rlnTomoTomogramsFile"] == str(case["tomograms"])
        assert Path(optimisation["rlnTomoParticlesFile"]) == case["outdir"] / f"cluster{cluster}" / "particles.star"


def test_class_star_numbers_every_labelled_particle(tmp_path, caplog):
    case = _case(tmp_path, native_extra={"rlnClassNumber": [9] * len(NATIVE_NAMES)})
    export_clusters(**case, clusters=[2])
    assert [path.name for path in case["outdir"].glob("cluster*")] == ["cluster2"]
    classes = _read(case["outdir"] / "particles_classes.star")["particles"]
    assert classes.rlnTomoParticleName.tolist() == [name for name in NATIVE_NAMES if name in LABELS]
    assert classes.rlnClassNumber.tolist() == [LABELS[name] + 1 for name in classes.rlnTomoParticleName]
    assert "replaces it" in caplog.text


@pytest.mark.parametrize("sentinel", [np.nan, -1, np.iinfo(np.int64).min])
def test_unretained_sentinel_is_excluded(tmp_path, sentinel):
    labels = np.array([LABELS[name] for name in sorted(LABELS)] + [sentinel])
    case = _case(tmp_path, labels=labels if np.isnan(sentinel) else labels.astype(np.int64))
    export_clusters(**case)
    classes = _read(case["outdir"] / "particles_classes.star")["particles"]
    assert "unretained" not in classes.rlnTomoParticleName.tolist()


def test_refuses_non_empty_outdir(tmp_path):
    case = _case(tmp_path)
    case["outdir"].mkdir()
    (case["outdir"] / "old").touch()
    with pytest.raises(FileExistsError):
        export_clusters(**case)


@pytest.mark.parametrize(("renamed", "error"), [("0010", "not unique"), ("0002", "missing")])
def test_refuses_duplicate_or_missing_native_names(tmp_path, renamed, error):
    """Rename native particle 0001 to an existing name (duplicate) or a new one (0001 missing)."""
    case = _case(tmp_path)
    star = _read(case["particles"])
    star["particles"] = star["particles"].replace({"rlnTomoParticleName": {"0001": renamed}})
    case["particles"].unlink()
    starfile.write(star, case["particles"])
    with pytest.raises(ValueError, match=error):
        export_clusters(**case)
