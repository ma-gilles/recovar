import argparse
import os
import pickle
import numpy as np
import pytest

from recovar.commands import pipeline_with_outliers

pytestmark = pytest.mark.unit


def _make_args(tmp_path, *, tilt_series=False):
    return argparse.Namespace(
        outdir=str(tmp_path / "pipeline"),
        zdim=[4],
        k_rounds=2,
        no_z_regularization=False,
        delete_rounds=False,
        use_contrast_detection=False,
        use_junk_detection=False,
        no_plots=True,
        low_contrast_threshold=0.1,
        high_contrast_threshold=3.5,
        max_contrast=4.0,
        particle_bad_fraction_threshold=0.7,
        micrograph_bad_fraction_threshold=0.7,
        junk_threshold=0.5,
        particles_per_cluster=None,
        save_pipeline_indices=True,
        output_format="both",
        tilt_series=tilt_series,
        tilt_ind=None,
        ind=None,
        particles="particles.mrcs",
        poses="poses.pkl",
        ctf="ctf.pkl",
        mask="mask.mrc",
        halfsets=None,
        datadir=None,
        strip_prefix=None,
        n_images=-1,
        lazy=True,
        uninvert_data="false",
        downsample=None,
        no_downsample=False,
        focus_mask=None,
        keep_input_mask=False,
        use_complement_mask=False,
        mask_dilate_iter=0,
        dilated_mask_dilation_iters=None,
        dont_use_image_mask=False,
        correct_contrast=False,
        only_mean=False,
        tilt_series_ctf="cryoem",
        dose_per_tilt=None,
        angle_per_tilt=None,
        ntilts=None,
        shared_contrast_across_tilts=False,
        premultiplied_ctf=False,
        gpu_memory=None,
        multi_gpu=False,
        n_gpus=None,
        accept_cpu=True,
        low_memory_option=False,
        very_low_memory_option=False,
        noise_model="radial",
        mean_fn="triangular",
        do_over_with_contrast=False,
        ignore_zero_frequency=False,
        padding=0,
        new_noise_est=False,
        use_reg_mean_in_contrast=True,
        multi_zdim_embedding=False,
        keep_intermediate=False,
        test_covar_options=False,
    )


def _write_round_artifacts(round_outdir):
    model_dir = round_outdir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {"latent_coords": {4: np.zeros((1, 4), dtype=np.float32)}}
    with open(model_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    with open(round_outdir / "command.txt", "w") as f:
        f.write("test")


def test_pipeline_with_outliers_uses_original_image_ids_between_rounds(tmp_path, monkeypatch):
    args = _make_args(tmp_path, tilt_series=False)
    combined_round_1 = np.array([10, 20, 35], dtype=np.int32)
    combined_round_2 = np.array([10, 35], dtype=np.int32)

    monkeypatch.setattr(pipeline_with_outliers, "add_args", lambda parser: None)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(pipeline_with_outliers.output, "mkdir_safe", os.makedirs)
    monkeypatch.setattr(pipeline_with_outliers.output, "PipelineOutput", lambda outdir: object())
    monkeypatch.setattr(
        pipeline_with_outliers.output,
        "standard_pipeline_plots",
        lambda *unused, **unused_kwargs: None,
    )

    def fake_standard_recovar_pipeline(pipeline_args):
        _write_round_artifacts(tmp_path / "pipeline" / f"round_{len(created_rounds) + 1}")
        if len(created_rounds) == 1:
            with open(pipeline_args.ind, "rb") as f:
                np.testing.assert_array_equal(pickle.load(f), combined_round_1)
        created_rounds.append(pipeline_args.outdir)

    def fake_outlier_main():
        round_outdir = tmp_path / "pipeline" / f"round_{len(outlier_rounds) + 1}"
        combined_dir = round_outdir / "outlier_detection" / "data" / "combined_results"
        combined_dir.mkdir(parents=True, exist_ok=True)
        current = combined_round_1 if not outlier_rounds else combined_round_2
        with open(combined_dir / "combined_image_inliers_4.pkl", "wb") as f:
            pickle.dump(current, f)
        with open(combined_dir / "combined_image_outliers_4.pkl", "wb") as f:
            pickle.dump(np.array([], dtype=np.int32), f)
        outlier_rounds.append(str(round_outdir))

    created_rounds = []
    outlier_rounds = []
    monkeypatch.setattr(
        pipeline_with_outliers,
        "standard_recovar_pipeline",
        fake_standard_recovar_pipeline,
    )
    import recovar.commands.outlier_detection as outlier_detection

    monkeypatch.setattr(outlier_detection, "main", fake_outlier_main)

    pipeline_with_outliers._run_pipeline_with_outlier_removal_impl(args)

    with open(tmp_path / "pipeline" / "inliers_round_1.pkl", "rb") as f:
        np.testing.assert_array_equal(pickle.load(f), combined_round_1)
    with open(tmp_path / "pipeline" / "round_2" / "inliers_round_1.pkl", "rb") as f:
        np.testing.assert_array_equal(pickle.load(f), combined_round_1)


def test_pipeline_with_outliers_uses_original_particle_ids_between_rounds(tmp_path, monkeypatch):
    args = _make_args(tmp_path, tilt_series=True)
    combined_images_round_1 = np.array([100, 110, 120], dtype=np.int32)
    combined_particles_round_1 = np.array([7, 11], dtype=np.int32)
    combined_images_round_2 = np.array([100, 120], dtype=np.int32)
    combined_particles_round_2 = np.array([11], dtype=np.int32)

    monkeypatch.setattr(pipeline_with_outliers, "add_args", lambda parser: None)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(pipeline_with_outliers.output, "mkdir_safe", os.makedirs)
    monkeypatch.setattr(pipeline_with_outliers.output, "PipelineOutput", lambda outdir: object())
    monkeypatch.setattr(
        pipeline_with_outliers.output,
        "standard_pipeline_plots",
        lambda *unused, **unused_kwargs: None,
    )

    def fake_standard_recovar_pipeline(pipeline_args):
        _write_round_artifacts(tmp_path / "pipeline" / f"round_{len(created_rounds) + 1}")
        if len(created_rounds) == 1:
            with open(pipeline_args.ind, "rb") as f:
                np.testing.assert_array_equal(pickle.load(f), combined_images_round_1)
            with open(pipeline_args.tilt_ind, "rb") as f:
                np.testing.assert_array_equal(pickle.load(f), combined_particles_round_1)
        created_rounds.append(pipeline_args.outdir)

    def fake_outlier_main():
        round_outdir = tmp_path / "pipeline" / f"round_{len(outlier_rounds) + 1}"
        combined_dir = round_outdir / "outlier_detection" / "data" / "combined_results"
        combined_dir.mkdir(parents=True, exist_ok=True)
        image_ids = combined_images_round_1 if not outlier_rounds else combined_images_round_2
        particle_ids = combined_particles_round_1 if not outlier_rounds else combined_particles_round_2
        with open(combined_dir / "combined_image_inliers_4.pkl", "wb") as f:
            pickle.dump(image_ids, f)
        with open(combined_dir / "combined_image_outliers_4.pkl", "wb") as f:
            pickle.dump(np.array([], dtype=np.int32), f)
        with open(combined_dir / "combined_particle_inliers_4.pkl", "wb") as f:
            pickle.dump(particle_ids, f)
        with open(combined_dir / "combined_particle_outliers_4.pkl", "wb") as f:
            pickle.dump(np.array([], dtype=np.int32), f)
        outlier_rounds.append(str(round_outdir))

    created_rounds = []
    outlier_rounds = []
    monkeypatch.setattr(
        pipeline_with_outliers,
        "standard_recovar_pipeline",
        fake_standard_recovar_pipeline,
    )
    import recovar.commands.outlier_detection as outlier_detection

    monkeypatch.setattr(outlier_detection, "main", fake_outlier_main)

    pipeline_with_outliers._run_pipeline_with_outlier_removal_impl(args)

    with open(tmp_path / "pipeline" / "particle_inliers_round_1.pkl", "rb") as f:
        np.testing.assert_array_equal(pickle.load(f), combined_particles_round_1)
    with open(tmp_path / "pipeline" / "round_2" / "particle_inliers_round_1.pkl", "rb") as f:
        np.testing.assert_array_equal(pickle.load(f), combined_particles_round_1)
