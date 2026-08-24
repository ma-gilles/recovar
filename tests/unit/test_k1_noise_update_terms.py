import numpy as np
import pytest

from scripts.analyze_k1_noise_update_terms import _positive_ratio, analyze


@pytest.mark.unit
def test_positive_ratio_records_empty_zero_component():
    report = _positive_ratio(np.zeros(3), np.zeros(3))

    assert report == {
        "count": 0,
        "median": None,
        "p05": None,
        "p95": None,
        "min": None,
        "max": None,
    }


@pytest.mark.unit
def test_noise_update_terms_replay_equal_sufficient_statistics(tmp_path):
    image_size = 4
    n4 = float(image_size**4)
    raw = np.asarray([8.0, 20.0, 36.0], dtype=np.float64)
    npix = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    sumw = 2.0
    new = raw / (2.0 * sumw * npix)
    residual = np.asarray([3.0, 7.0, 11.0], dtype=np.float64) * n4
    image_power = raw * n4 - residual
    native_path = tmp_path / "sigma2_noise_raw.tsv"
    native_path.write_text(
        "".join(
            "acc_particle\titer=2\tpart_id=17\thalfset=1\trandom_subset=1"
            f"\toptics_group=0\tshell={shell}\traw_sigma2_accum={raw[shell]}"
            f"\tsumw_group={sumw}\tNpix_per_shell={npix[shell]}"
            "\told_sigma2=1.0\tnew_sigma2=0.0\tmy_mu=0.0"
            f"\timage_power={raw[shell] - residual[shell] / n4}"
            f"\tdirect_residual={residual[shell] / n4}\n"
            for shell in range(3)
        )
        + "".join(
            "final\titer=2\tpart_id=-1\thalfset=1\trandom_subset=-1"
            f"\toptics_group=0\tshell={shell}\traw_sigma2_accum={raw[shell]}"
            f"\tsumw_group={sumw}\tNpix_per_shell={npix[shell]}"
            f"\told_sigma2=1.0\tnew_sigma2={new[shell]}\tmy_mu=0.0"
            "\timage_power=0.0\tdirect_residual=0.0\n"
            for shell in range(3)
        )
    )
    recovar_path = tmp_path / "recovar_noise_update_it002.npz"
    np.savez(
        recovar_path,
        current_size=np.asarray([2], dtype=np.int32),
        relion_half_plane_shell_counts=npix,
        half1_wsum_sigma2_noise=residual,
        half1_wsum_img_power=image_power,
        half1_wsum_total=raw * n4,
        half1_sumw=np.asarray([sumw]),
        half1_previous_sigma2_noise=np.ones(3) * n4,
        half1_sigma2_noise=new * n4,
    )

    report = analyze(
        native_path,
        recovar_path,
        iteration=2,
        half=1,
        image_size=image_size,
        recovar_prefix="half1",
    )

    assert report["identity"]["native_halfset"] == 1
    assert report["identity"]["recovar_prefix"] == "half1"
    assert report["denominator"]["recovar_over_native_sumw"] == 1.0
    assert report["comparisons"]["raw_total_recovar_vs_native"]["max_abs"] == 0.0
    assert report["comparisons"]["residual_recovar_vs_native_particles"]["max_abs"] == 0.0
    assert report["comparisons"]["image_power_recovar_vs_native_particles"]["max_abs"] == 0.0
    assert report["comparisons"]["new_noise_recovar_vs_native"]["max_abs"] == 0.0
    assert report["comparisons"]["native_formula_replay"]["max_abs"] == 0.0
    assert report["comparisons"]["recovar_formula_replay"]["max_abs"] == 0.0
