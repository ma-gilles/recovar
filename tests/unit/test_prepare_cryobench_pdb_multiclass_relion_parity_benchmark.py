import sys

import numpy as np
import pytest


def test_class_distribution_parser_supports_matrix_options():
    from scripts import prepare_cryobench_pdb_multiclass_relion_parity_benchmark as prep

    assert np.allclose(prep._class_distribution("uniform", 3), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(prep._class_distribution("linear", 3), [3 / 6, 2 / 6, 1 / 6])
    assert np.allclose(prep._class_distribution("head-heavy", 3), [3 / 5, 1 / 5, 1 / 5])
    assert np.allclose(prep._class_distribution("custom:2,1,1", 3), [0.5, 0.25, 0.25])

    with pytest.raises(ValueError, match="expected 3"):
        prep._class_distribution("custom:1,1", 3)
    with pytest.raises(ValueError, match="non-negative"):
        prep._class_distribution("custom:1,-1,1", 3)


def test_real_volume_symmetry_preserves_c1_and_averages_ordered_c4():
    from scripts import prepare_cryobench_pdb_multiclass_relion_parity_benchmark as prep

    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[1, 2, 3] = 4.0
    c4 = np.asarray(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        ],
        dtype=np.float64,
    )

    c1_result = prep._symmetrize_real_volume(volume, "C1")
    c4_result = prep._symmetrize_real_volume(volume, "C4", operators=c4)

    np.testing.assert_array_equal(c1_result, volume)
    np.testing.assert_array_equal(c4_result, np.rot90(c4_result, axes=(0, 1)))
    assert c4_result.dtype == np.float32
    assert float(c4_result.sum()) == pytest.approx(float(volume.sum()))


def test_real_volume_symmetry_rejects_wrong_operator_count():
    from scripts import prepare_cryobench_pdb_multiclass_relion_parity_benchmark as prep

    with pytest.raises(ValueError, match="requires operators with shape"):
        prep._symmetrize_real_volume(
            np.ones((3, 3, 3), dtype=np.float32),
            "C4",
            operators=np.eye(3, dtype=np.float64)[None, :, :],
        )


def test_c1_symmetry_contract_does_not_require_optional_relion_binding():
    from scripts import prepare_cryobench_pdb_multiclass_relion_parity_benchmark as prep

    contract = prep._symmetry_contract("c1")

    assert contract["canonical_label"] == "C1"
    assert contract["operator_count"] == 1
    assert len(contract["operators_sha256"]) == 64
    assert contract["operator_source"] == "analytic identity (RELION C1 convention)"


def test_multiclass_pdb_cli_forwards_robustness_matrix_options(monkeypatch, tmp_path):
    from scripts import prepare_cryobench_pdb_multiclass_relion_parity_benchmark as prep

    captured = {}

    def fake_prepare_benchmark(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)

    monkeypatch.setattr(prep, "prepare_benchmark", fake_prepare_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_cryobench_pdb_multiclass_relion_parity_benchmark.py",
            "--pdb-dir",
            str(tmp_path / "pdbs"),
            "--output-dir",
            str(tmp_path / "out"),
            "--n-images",
            "10000",
            "--grid-size",
            "128",
            "--noise-level",
            "3",
            "--noise-model",
            "radial1",
            "--dataset-params-option",
            "kent",
            "--class-distribution",
            "linear",
            "--pdb-bfactor",
            "80",
            "--init-radius",
            "9",
            "--noise-scale-std",
            "0.2",
            "--contrast-std",
            "0.3",
            "--volume-radius",
            "0.65",
            "--image-offset-n-std",
            "0.5",
            "--percent-outliers",
            "0.25",
            "--outlier-pdb-path",
            str(tmp_path / "outlier.pdb"),
            "--noise-rng-batch-size",
            "256",
            "--symmetry",
            "I1",
            "--no-streaming-mmap",
        ],
    )

    prep.main()

    assert captured["output_dir"] == tmp_path / "out"
    assert captured["pdb_dir"] == tmp_path / "pdbs"
    assert captured["n_images"] == 10000
    assert captured["grid_size"] == 128
    assert captured["noise_level"] == 3
    assert captured["noise_model"] == "radial1"
    assert captured["dataset_params_option"] == "kent"
    assert captured["class_distribution"] == "linear"
    assert captured["pdb_bfactor"] == 80
    assert captured["init_radius"] == 9
    assert captured["noise_scale_std"] == 0.2
    assert captured["contrast_std"] == 0.3
    assert captured["volume_radius"] == 0.65
    assert captured["image_offset_n_std"] == 0.5
    assert captured["percent_outliers"] == 0.25
    assert captured["outlier_pdb_path"] == tmp_path / "outlier.pdb"
    assert captured["noise_rng_batch_size"] == 256
    assert captured["symmetry"] == "I1"
    assert captured["streaming_mmap"] is False


def test_k4_reference_files_use_each_consumers_coordinate_frame(tmp_path, monkeypatch):
    """Keep native class inputs distinct from the RELION STAR references."""
    from types import SimpleNamespace

    import starfile

    from recovar.utils import helpers
    from scripts import prepare_cryobench_pdb_multiclass_relion_parity_benchmark as prep

    shape = (4, 4, 4)
    # Asymmetric integer volumes expose axis swaps as well as sign mistakes.
    volumes = np.stack([
        np.arange(64, dtype=np.float32).reshape(shape) + 64 * class_id
        for class_id in range(4)
    ])
    fourier = np.stack([np.asarray(prep.ftu.get_dft3(v)).reshape(-1) for v in volumes])
    dataset = SimpleNamespace(
        volume_shape=shape, voxel_size=2.0,
        get_valid_frequency_indices=lambda *, rad: np.ones(64, dtype=np.float32),
    )
    heterogeneous = SimpleNamespace(volumes=fourier, get_mean=lambda: fourier.mean(axis=0))
    monkeypatch.setattr(prep, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(prep.utils, "pickle_load", lambda path: {})
    monkeypatch.setattr(prep.synthetic_dataset, "load_heterogeneous_reconstruction", lambda info: heterogeneous)

    # Exercise real Fourier conversion and MRC/STAR writers and readers.
    prep._write_class_references(tmp_path, 4, 4, 2)
    classes = starfile.read(tmp_path / "reference_init_classes_relion.star", always_dict=True)["model_classes"]
    assert list(classes["rlnReferenceImage"]) == [
        f"reference_init_class{k:03d}_relion.mrc" for k in range(1, 5)
    ]
    np.testing.assert_array_equal(classes["rlnClassDistribution"], np.full(4, 0.25))
    for k, expected in enumerate(volumes, start=1):
        native, voxel = helpers.load_mrc(tmp_path / f"reference_init_class{k:03d}.mrc", return_voxel_size=True)
        relion_path = tmp_path / classes["rlnReferenceImage"].iloc[k - 1]
        relion, relion_voxel = helpers.load_relion_volume(relion_path, return_voxel_size=True)
        np.testing.assert_array_equal(native, expected)
        np.testing.assert_array_equal(relion, expected)
        np.testing.assert_array_equal(helpers.load_mrc(relion_path), -expected)
        assert float(voxel.x) == 2.0
        assert float(relion_voxel.x) == 2.0
