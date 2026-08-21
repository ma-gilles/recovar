import numpy as np

from scripts.analyze_k1_scale_update_terms import analyze


def _model_text(*, scales, counts, dvp) -> str:
    class_rows = "\n".join(
        f"{shell} 0 0 {value} 0 0 0 0" for shell, value in enumerate(dvp)
    )
    group_rows = "\n".join(
        f"{index + 1} {index + 1} {count} {scale:.12g}"
        for index, (count, scale) in enumerate(zip(counts, scales, strict=True))
    )
    return f"""
data_model_class_1

loop_
_rlnSpectralIndex #1
_rlnResolution #2
_rlnAngstromResolution #3
_rlnSsnrMap #4
_rlnGoldStandardFsc #5
_rlnFourierCompleteness #6
_rlnReferenceSigma2 #7
_rlnReferenceTau2 #8
{class_rows}

data_model_groups

loop_
_rlnGroupNumber #1
_rlnGroupName #2
_rlnGroupNrParticles #3
_rlnGroupScaleCorrection #4
{group_rows}
"""


def test_scale_update_terms_replays_clipping_and_global_normalization(tmp_path):
    counts = np.asarray([1, 0, 1, 0], dtype=np.int64)
    xa = np.asarray([0.1, 0.0, 2.0, 0.0], dtype=np.float64)
    aa = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    clipped = np.asarray([0.2, 1.0, 2.0, 1.0], dtype=np.float64)
    final = clipped / 1.1

    input_model = tmp_path / "run_it001_half1_model.star"
    output_model = tmp_path / "run_it002_half1_model.star"
    input_model.write_text(_model_text(scales=np.ones(4), counts=counts, dvp=[4.0, 2.0]))
    output_model.write_text(_model_text(scales=final, counts=counts, dvp=[0.0, 0.0]))

    components = tmp_path / "sigma2_noise_components.tsv"
    components.write_text(
        "acc_components\titer=2\tpart_id=0\thalfset=1\tshell=0\taa=1\txa=0.1\n"
        "acc_components\titer=2\tpart_id=0\thalfset=1\tshell=1\taa=99\txa=99\n"
        "acc_components\titer=2\tpart_id=2\thalfset=1\tshell=0\taa=1\txa=2\n"
        "acc_components\titer=2\tpart_id=2\thalfset=1\tshell=1\taa=99\txa=99\n"
    )
    parity = tmp_path / "iter_002.npz"
    np.savez(
        parity,
        half1_wsum_scale_correction_xa=xa,
        half1_wsum_scale_correction_aa=aa,
        half1_group_particle_counts=counts,
        half1_group_scale_corrections=final,
        scale_correction_data_vs_prior=np.asarray([4.0, 2.0]),
    )

    report = analyze(
        components,
        input_model,
        output_model,
        parity,
        iteration=2,
        half=1,
        target_group_index=0,
    )

    assert report["identity"]["selected_shells"] == [0]
    assert report["update"]["native_normalization_average"] == 1.1
    assert report["update"]["recovar_normalization_average"] == 1.1
    assert report["target"]["native_raw"] == 0.1
    assert report["target"]["native_clipped"] == 0.2
    assert report["comparisons"]["native_replay_vs_model_scale"]["max_abs"] < 1e-11
    assert report["comparisons"]["recovar_replay_vs_dump_scale"]["max_abs"] == 0.0

    scaled_report = analyze(
        components,
        input_model,
        output_model,
        parity,
        iteration=2,
        half=1,
        target_group_index=0,
        recovar_term_divisor=2.0,
    )
    assert scaled_report["target"]["recovar_xa_native_units"] == 0.05
    assert scaled_report["target"]["recovar_aa_native_units"] == 0.5
