import sys

import numpy as np


def test_prepare_pdb_k1_cli_forwards_matrix_generation_options(monkeypatch, tmp_path):
    from scripts import prepare_pdb_k1_relion_sanity_benchmark as prep

    captured = {}

    def fake_prepare_benchmark(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)

    monkeypatch.setattr(prep, "prepare_benchmark", fake_prepare_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_pdb_k1_relion_sanity_benchmark.py",
            "--output-dir",
            str(tmp_path),
            "--n-images",
            "3000",
            "--grid-size",
            "128",
            "--noise-level",
            "5",
            "--noise-model",
            "radial1",
            "--dataset-params-option",
            "nonuniform",
            "--pdb-bfactor",
            "80",
            "--noise-scale-std",
            "0.7",
            "--contrast-std",
            "0.6",
            "--volume-radius",
            "0.65",
            "--percent-outliers",
            "0.5",
            "--put-extra-particles",
            "--image-offset-n-std",
            "1.5",
            "--relion-bg-radius-px",
            "42",
            "--noise-rng-batch-size",
            "256",
            "--no-streaming-mmap",
        ],
    )

    prep.main()

    assert captured["output_dir"] == tmp_path
    assert captured["n_images"] == 3000
    assert captured["grid_size"] == 128
    assert captured["noise_level"] == 5
    assert captured["noise_model"] == "radial1"
    assert captured["dataset_params_option"] == "nonuniform"
    assert captured["pdb_bfactor"] == 80
    assert captured["noise_scale_std"] == 0.7
    assert captured["contrast_std"] == 0.6
    assert captured["volume_radius"] == 0.65
    assert captured["percent_outliers"] == 0.5
    assert captured["put_extra_particles"] is True
    assert captured["image_offset_n_std"] == 1.5
    assert captured["relion_bg_radius_px"] == 42
    assert captured["noise_rng_batch_size"] == 256
    assert captured["streaming_mmap"] is False


def test_prepare_pdb_k1_outlier_cases_pass_generated_outlier_volume(monkeypatch, tmp_path):
    from scripts import prepare_pdb_k1_relion_sanity_benchmark as prep

    trajectory_calls = []
    captured = {}

    def fake_generate_trajectory_volumes(**kwargs):
        trajectory_calls.append(kwargs)
        prefix = kwargs["output_prefix"]
        for idx in range(kwargs["n_volumes"]):
            path = prep.Path(f"{prefix}{idx:04d}.mrc")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake mrc")

    def fake_generate_synthetic_dataset(output_folder, *_args, **kwargs):
        captured["outlier_file_input"] = kwargs["outlier_file_input"]
        output_dir = prep.Path(output_folder)
        (output_dir / "particles.star").write_text("data_particles\n")
        (output_dir / "particles.16.mrcs").write_bytes(b"fake stack")
        prep.utils.pickle_dump((np.eye(3, dtype=np.float32)[None], np.zeros((1, 2), dtype=np.float32)), output_dir / "poses.pkl")
        prep.utils.pickle_dump(np.zeros((1, 10), dtype=np.float32), output_dir / "ctf.pkl")
        prep.utils.pickle_dump({"image_assignment": np.array([0, -1], dtype=int)}, output_dir / "simulation_info.pkl")

    monkeypatch.setattr(prep, "generate_trajectory_volumes", fake_generate_trajectory_volumes)
    monkeypatch.setattr(prep.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)
    monkeypatch.setattr(prep, "_write_references", lambda *_args, **_kwargs: None)

    prep.prepare_benchmark(
        tmp_path,
        n_images=1,
        grid_size=16,
        voxel_size=34.0,
        noise_level=1.0,
        noise_model="white",
        dataset_params_option="uniform",
        init_resolution_ang=30.0,
        pdb_path=None,
        pdb_bfactor=80.0,
        noise_scale_std=0.0,
        contrast_std=0.0,
        volume_radius=0.7,
        percent_outliers=0.2,
        put_extra_particles=False,
        image_offset_n_std=0.0,
        relion_bg_radius_px=None,
        noise_rng_batch_size=None,
        relion_normalize=True,
        streaming_mmap=False,
        streaming_chunk_size=500,
        disc_type="cubic",
        seed=17,
    )

    assert len(trajectory_calls) == 2
    assert trajectory_calls[1]["n_volumes"] == 2
    assert captured["outlier_file_input"] == str(tmp_path / "pdb_outlier_state" / "outlier0001.mrc")
