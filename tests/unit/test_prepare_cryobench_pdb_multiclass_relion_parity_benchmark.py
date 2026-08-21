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
    assert captured["streaming_mmap"] is False
