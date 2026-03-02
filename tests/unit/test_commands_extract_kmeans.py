"""
Unit tests for recovar.commands.extract_image_subset_from_kmeans.

Covers:
  extract_image_subset_from_kmeans – label filtering, inverse flag, output writing
"""
import os
import pytest
import numpy as np

from recovar.commands import extract_image_subset_from_kmeans as ekm_cmd

pytestmark = pytest.mark.unit


def _fake_centers(labels: np.ndarray) -> dict:
    """Build a minimal fake centers pkl dict with the given label array."""
    return {"labels": labels.astype(float)}


def _touch(path: str) -> str:
    """Create an empty file at *path* so os.path.exists() passes, return path."""
    open(path, "wb").close()
    return path


# ---------------------------------------------------------------------------
# extract_image_subset_from_kmeans – core logic
# ---------------------------------------------------------------------------

def test_extracts_correct_indices_for_single_cluster(monkeypatch, tmp_path):
    """Selecting cluster 1 must return only the images labelled 1."""
    labels = np.array([0, 1, 2, 1, 0, 2], dtype=float)
    centers_pkl = _touch(str(tmp_path / "centers.pkl"))
    output_pkl = str(tmp_path / "indices.pkl")

    saved = {}
    monkeypatch.setattr(ekm_cmd.utils, "pickle_load", lambda _: _fake_centers(labels))
    monkeypatch.setattr(ekm_cmd.utils, "pickle_dump",
                        lambda obj, path: saved.update({"obj": obj, "path": path}))

    ekm_cmd.extract_image_subset_from_kmeans(
        path_to_centers=centers_pkl,
        kmeans_indices=[1],
        inverse=False,
        output_path=output_pkl,
    )

    expected = np.where(labels == 1)[0]
    np.testing.assert_array_equal(saved["obj"], expected)
    assert saved["path"] == output_pkl


def test_extracts_correct_indices_for_multiple_clusters(monkeypatch, tmp_path):
    """Selecting clusters [0, 2] returns images labelled 0 or 2."""
    labels = np.array([0, 1, 2, 0, 2, 1, 0], dtype=float)
    centers_pkl = _touch(str(tmp_path / "centers.pkl"))
    output_pkl = str(tmp_path / "indices.pkl")

    saved = {}
    monkeypatch.setattr(ekm_cmd.utils, "pickle_load", lambda _: _fake_centers(labels))
    monkeypatch.setattr(ekm_cmd.utils, "pickle_dump",
                        lambda obj, path: saved.update({"obj": obj, "path": path}))

    ekm_cmd.extract_image_subset_from_kmeans(
        path_to_centers=centers_pkl,
        kmeans_indices=[0, 2],
        inverse=False,
        output_path=output_pkl,
    )

    expected = np.where((labels == 0) | (labels == 2))[0]
    np.testing.assert_array_equal(np.sort(saved["obj"]), expected)


def test_inverse_flag_returns_complement(monkeypatch, tmp_path):
    """When inverse=True the function must return images NOT in the selected clusters."""
    labels = np.array([0, 1, 2, 1, 0, 2], dtype=float)
    centers_pkl = _touch(str(tmp_path / "centers.pkl"))
    output_pkl = str(tmp_path / "indices.pkl")

    saved = {}
    monkeypatch.setattr(ekm_cmd.utils, "pickle_load", lambda _: _fake_centers(labels))
    monkeypatch.setattr(ekm_cmd.utils, "pickle_dump",
                        lambda obj, path: saved.update({"obj": obj, "path": path}))

    ekm_cmd.extract_image_subset_from_kmeans(
        path_to_centers=centers_pkl,
        kmeans_indices=[1],
        inverse=True,
        output_path=output_pkl,
    )

    # Everything that is not cluster 1
    expected = np.where(labels != 1)[0]
    np.testing.assert_array_equal(np.sort(saved["obj"]), expected)


def test_all_images_selected_when_all_clusters_given(monkeypatch, tmp_path):
    """Selecting every cluster must return all image indices."""
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=float)
    centers_pkl = _touch(str(tmp_path / "centers.pkl"))
    output_pkl = str(tmp_path / "indices.pkl")

    saved = {}
    monkeypatch.setattr(ekm_cmd.utils, "pickle_load", lambda _: _fake_centers(labels))
    monkeypatch.setattr(ekm_cmd.utils, "pickle_dump",
                        lambda obj, path: saved.update({"obj": obj, "path": path}))

    ekm_cmd.extract_image_subset_from_kmeans(
        path_to_centers=centers_pkl,
        kmeans_indices=[0, 1, 2],
        inverse=False,
        output_path=output_pkl,
    )

    np.testing.assert_array_equal(np.sort(saved["obj"]), np.arange(len(labels)))


def test_empty_result_when_no_cluster_matches(monkeypatch, tmp_path):
    """Selecting a cluster that doesn't exist must return an empty index array."""
    labels = np.array([0, 1, 0, 1], dtype=float)
    centers_pkl = _touch(str(tmp_path / "centers.pkl"))
    output_pkl = str(tmp_path / "indices.pkl")

    saved = {}
    monkeypatch.setattr(ekm_cmd.utils, "pickle_load", lambda _: _fake_centers(labels))
    monkeypatch.setattr(ekm_cmd.utils, "pickle_dump",
                        lambda obj, path: saved.update({"obj": obj, "path": path}))

    ekm_cmd.extract_image_subset_from_kmeans(
        path_to_centers=centers_pkl,
        kmeans_indices=[99],   # no image is labelled 99
        inverse=False,
        output_path=output_pkl,
    )

    assert saved["obj"].size == 0


def test_path_validation_rejects_missing_centers_file(tmp_path):
    """Passing a non-existent path_to_centers must raise FileNotFoundError."""
    missing = str(tmp_path / "does_not_exist.pkl")
    out = str(tmp_path / "out.pkl")
    with pytest.raises(FileNotFoundError):
        ekm_cmd.extract_image_subset_from_kmeans(missing, [0], False, out)


def test_path_validation_requires_pkl_extension_for_centers(tmp_path):
    """path_to_centers must end with .pkl; otherwise ValueError is raised."""
    wrong_ext = _touch(str(tmp_path / "centers.txt"))
    out = str(tmp_path / "out.pkl")
    with pytest.raises(ValueError):
        ekm_cmd.extract_image_subset_from_kmeans(wrong_ext, [0], False, out)


def test_path_validation_requires_pkl_extension_for_output(monkeypatch, tmp_path):
    """output_path must end with .pkl; otherwise ValueError is raised."""
    centers = _touch(str(tmp_path / "centers.pkl"))
    wrong_out = str(tmp_path / "output.csv")

    monkeypatch.setattr(ekm_cmd.utils, "pickle_load",
                        lambda _: _fake_centers(np.array([0, 1])))

    with pytest.raises(ValueError):
        ekm_cmd.extract_image_subset_from_kmeans(centers, [0], False, wrong_out)
