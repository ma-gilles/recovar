import numpy as np
import pandas as pd
import pytest
import starfile

from scripts.analyze_vdam_cutoff_particle_panel import analyze


@pytest.mark.unit
def test_cutoff_particle_panel_closes_components_and_identity(tmp_path):
    image_size = 4
    n4 = float(image_size**4)
    relion_star = tmp_path / "run_it001_data.star"
    starfile.write(
        {"particles": pd.DataFrame({"rlnImageName": ["8@particles.mrcs", "3@particles.mrcs"]})},
        relion_star,
    )
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    rows = []
    for part_id, original_index, xa, aa, image_power in (
        (0, 7, -2.0, 3.0, 11.0),
        (1, 2, -5.0, 7.0, 13.0),
    ):
        direct = aa - 2.0 * xa + image_power
        np.savez(
            score_dir / f"local_score_it001_image_{original_index}_single_class.npz",
            selected_global_image_indices=np.asarray([original_index]),
            current_size=np.asarray([4]),
            debug_wavg_cutoff_triplet_xa_aa_diff2=np.asarray([xa, aa, direct]) * n4,
            posterior=np.asarray([[[0.25, 0.75]]]),
            reconstruction_probs=np.asarray([[[0.125, 0.0]]], dtype=np.float32),
            reconstruction_sample_mask=np.asarray([[[True, False]]]),
        )
        rows.append(
            "acc_components\titer=1"
            f"\tpart_id={part_id}\thalfset=-1\trandom_subset=-1\toptics_group=0\tshell=2"
            f"\tdirect_residual={direct}\taa={aa}\txa={xa}"
            f"\tinferred_image_power={image_power}\tsumw_group=0.125\tNpix_per_shell=9\n"
        )
    native_tsv = tmp_path / "sigma2_noise_components.tsv"
    native_tsv.write_text("".join(rows))

    report = analyze(
        native_tsv,
        score_dir,
        relion_star,
        iteration=1,
        half=-1,
        image_size=image_size,
        expected_particle_count=2,
    )

    assert report["identity"]["particle_count"] == 2
    assert report["identity"]["cutoff_shell"] == 2
    assert [row["original_index"] for row in report["per_particle"]] == [7, 2]
    for comparison in report["comparisons"].values():
        assert comparison["max_abs_error"] == pytest.approx(0.0)
        assert comparison["signed_sum_error"] == pytest.approx(0.0)


@pytest.mark.unit
def test_cutoff_particle_panel_rejects_incomplete_candidate_capture(tmp_path):
    relion_star = tmp_path / "run_it001_data.star"
    starfile.write(
        {"particles": pd.DataFrame({"rlnImageName": ["1@particles.mrcs"]})},
        relion_star,
    )
    native_tsv = tmp_path / "sigma2_noise_components.tsv"
    native_tsv.write_text(
        "acc_components\titer=1\tpart_id=0\thalfset=-1\trandom_subset=-1"
        "\toptics_group=0\tshell=2\tdirect_residual=1\taa=1\txa=1"
        "\tinferred_image_power=2\tsumw_group=1\tNpix_per_shell=9\n"
    )
    score_dir = tmp_path / "scores"
    score_dir.mkdir()

    with pytest.raises(ValueError, match="no RECOVAR score dumps"):
        analyze(
            native_tsv,
            score_dir,
            relion_star,
            iteration=1,
            half=-1,
            image_size=4,
        )
