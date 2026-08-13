from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sparse
from scripts.analyze_k1_scale_aa_boundary import analyze


def test_scale_aa_boundary_replays_operands_and_matches_native_shells(tmp_path: Path):
    capture = tmp_path / "capture.npz"
    native = tmp_path / "native.tsv"
    proj_abs2 = np.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    ctf_probs = np.asarray([[0.25, 0.5], [0.75, 0.25]], dtype=np.float32)
    noise = np.asarray([4.0, 8.0], dtype=np.float32)
    mask = np.asarray([True, True])
    raw = (ctf_probs * noise[None, :]).astype(np.float32)
    before_scale = (proj_abs2 * raw).astype(np.float32)
    terms = (before_scale / np.float32(4.0)).astype(np.float32)
    per_pixel = np.sum(terms, axis=0, dtype=np.float32)
    per_shell = per_pixel.astype(np.float64)
    np.savez_compressed(
        capture,
        schema=np.asarray("recovar-k1-norm-residual-inputs-v2"),
        iteration=np.int64(2),
        half=np.int64(1),
        original_index=np.int64(1096),
        group_id=np.int64(109),
        scale_for_stats=np.float32(2.0),
        scale_correction_pixel_mask=mask,
        scale_shell_indices=np.asarray([0, 1], dtype=np.int32),
        proj_abs2_for_noise=proj_abs2,
        ctf_probs=ctf_probs,
        noise_variance_for_noise=noise,
        scale_ctf_probs_raw=raw,
        scale_aa_terms_before_scale=before_scale,
        scale_aa_terms=terms,
        scale_aa_per_pixel=per_pixel,
        scale_aa_per_shell=per_shell,
        scale_aa_per_image=np.float32(np.sum(terms, dtype=np.float32)),
    )
    native.write_text(
        "acc_components\titer=2\tpart_id=109\thalfset=1\trandom_subset=-1"
        f"\toptics_group=0\tshell=0\taa={per_shell[0] / 16.0}\n"
        "acc_components\titer=2\tpart_id=109\thalfset=1\trandom_subset=-1"
        f"\toptics_group=0\tshell=1\taa={per_shell[1] / 16.0}\n"
    )

    report = analyze(
        capture,
        native,
        expected_iteration=2,
        expected_half=1,
        expected_part_id=109,
        expected_original_index=1096,
        recovar_term_divisor=16.0,
    )

    assert all(
        report["local_replay"][key]
        for key in (
            "ctf_probs_raw_bit_exact",
            "aa_products_bit_exact",
            "scale_adjusted_aa_terms_bit_exact",
            "per_pixel_reduction_bit_exact",
            "per_image_reduction_bit_exact",
        )
    )
    assert report["native_shell_comparison"]["relative_l2"] == 0.0
    assert report["native_shell_comparison"]["ratio_median"] == 1.0


def test_chunked_scale_aa_writer_preserves_float32_chunk_order(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sparse._bpref_contribution_context, "iteration", 2)
    monkeypatch.setitem(sparse._bpref_contribution_context, "half", 1)
    aa_chunks = [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0]], dtype=np.float32),
    ]
    norm_a2_chunks = [
        np.asarray([[5.0, 6.0]], dtype=np.float32),
        np.asarray([[7.0, 8.0]], dtype=np.float32),
    ]
    norm_xa_chunks = [
        np.asarray([[0.25, 0.5]], dtype=np.float32),
        np.asarray([[0.75, 1.0]], dtype=np.float32),
    ]
    count = sparse._write_chunked_scale_aa_dump(
        dump_dir=tmp_path,
        experiment_dataset=object(),
        image_indices=np.asarray([1096], dtype=np.int64),
        target_rows=np.asarray([0], dtype=np.int64),
        current_size=60,
        bucket_group_ids=np.asarray([109], dtype=np.int32),
        bucket_scale_for_stats=np.asarray([2.0], dtype=np.float32),
        scale_correction_pixel_mask=np.asarray([True, True]),
        scale_shell_indices=np.asarray([0, 1], dtype=np.int32),
        chunk_ranges=[(0, 2), (2, 4)],
        posterior_mass_chunks=[np.asarray([0.4], dtype=np.float32), np.asarray([0.6], dtype=np.float32)],
        proj_abs2_sum_chunks=[np.asarray([[2.0, 3.0]], dtype=np.float32)] * 2,
        ctf_probs_raw_sum_chunks=[np.asarray([[4.0, 5.0]], dtype=np.float32)] * 2,
        xa_per_pixel_chunks=[
            np.asarray([[0.5, 1.0]], dtype=np.float32),
            np.asarray([[1.5, 2.0]], dtype=np.float32),
        ],
        xa_per_image_chunks=[np.asarray([1.5], dtype=np.float32), np.asarray([3.5], dtype=np.float32)],
        norm_a2_per_pixel_chunks=norm_a2_chunks,
        norm_a2_per_image_chunks=[np.asarray([11.0], dtype=np.float32), np.asarray([15.0], dtype=np.float32)],
        norm_xa_per_pixel_chunks=norm_xa_chunks,
        norm_xa_per_image_chunks=[np.asarray([0.75], dtype=np.float32), np.asarray([1.75], dtype=np.float32)],
        aa_before_scale_per_pixel_chunks=[value * np.float32(4.0) for value in aa_chunks],
        aa_per_pixel_chunks=aa_chunks,
        aa_per_image_chunks=[np.asarray([3.0], dtype=np.float32), np.asarray([7.0], dtype=np.float32)],
        posterior_probs_chunks=[
            np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
            np.asarray([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
        ],
        rotation_matrix_chunks=[
            np.broadcast_to(np.eye(3, dtype=np.float32), (1, 2, 3, 3)),
            np.broadcast_to(2.0 * np.eye(3, dtype=np.float32), (1, 2, 3, 3)),
        ],
        fine_translations=np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
        aa_feature_per_shell_chunks=[
            np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            np.asarray([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        ],
        aa_feature_shell_ids=np.asarray([0, 1], dtype=np.int32),
        atomic_xa_per_pixel=np.asarray([[9.0, 10.0]], dtype=np.float32),
        atomic_aa_per_pixel=np.asarray([[11.0, 12.0]], dtype=np.float32),
        noise_variance_for_noise=np.asarray([2.0, 3.0], dtype=np.float32),
        weighted_img_per_image=np.asarray([31.0], dtype=np.float64),
        relion_norm_high_shell=np.asarray([17.0], dtype=np.float64),
        norm_shifted_images=np.asarray(
            [[[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]]],
            dtype=np.complex64,
        ),
    )
    assert count == 1
    capture_path = tmp_path / "scale_aa_chunked_orig001096_half1_cs060.npz"
    with np.load(capture_path, allow_pickle=False) as capture:
        assert capture["schema"].item() == "recovar-k1-scale-xa-aa-chunked-v4"
        np.testing.assert_array_equal(capture["scale_xa_per_pixel"], [2.0, 3.0])
        assert float(capture["scale_xa_per_image"]) == 5.0
        np.testing.assert_array_equal(capture["scale_xa_atomic_per_pixel"], [9.0, 10.0])
        np.testing.assert_array_equal(capture["scale_aa_atomic_per_pixel"], [11.0, 12.0])
        np.testing.assert_array_equal(capture["scale_aa_per_pixel"], [4.0, 6.0])
        np.testing.assert_array_equal(capture["scale_aa_per_shell"], [4.0, 6.0])
        assert capture["candidate_posterior_probs"].shape == (4, 2)
        assert capture["candidate_rotation_matrices"].shape == (4, 3, 3)
        np.testing.assert_array_equal(capture["fine_translations"], [[0.0, 0.0], [1.0, -1.0]])
        assert capture["candidate_aa_feature_per_shell"].shape == (4, 2)
        np.testing.assert_array_equal(capture["candidate_aa_feature_shell_ids"], [0, 1])
        assert float(capture["scale_aa_per_image"]) == 10.0
        assert float(capture["pixel_sum_minus_production_total"]) == 0.0
        np.testing.assert_array_equal(capture["norm_a2_per_pixel_by_chunk"], [[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_array_equal(capture["norm_xa_per_pixel_by_chunk"], [[0.25, 0.5], [0.75, 1.0]])
        np.testing.assert_array_equal(capture["norm_a2_per_image_by_chunk"], [11.0, 15.0])
        np.testing.assert_array_equal(capture["norm_xa_per_image_by_chunk"], [0.75, 1.75])
        assert float(capture["norm_a2_per_image"]) == 26.0
        assert float(capture["norm_xa_per_image"]) == 2.5
        assert float(capture["norm_residual_per_image"]) == 21.0
        np.testing.assert_array_equal(capture["noise_variance_for_noise"], [2.0, 3.0])
        assert float(capture["weighted_img_per_image"]) == 31.0
        assert float(capture["relion_norm_high_shell"]) == 17.0
        np.testing.assert_array_equal(
            capture["norm_shifted_images"],
            [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
        )

    native = tmp_path / "native_chunked.tsv"
    native.write_text(
        "acc_components\titer=2\tpart_id=109\thalfset=1\trandom_subset=-1\toptics_group=0\tshell=0\taa=0.25\n"
        "acc_components\titer=2\tpart_id=109\thalfset=1\trandom_subset=-1\toptics_group=0\tshell=1\taa=0.375\n"
    )
    report = analyze(
        capture_path,
        native,
        expected_iteration=2,
        expected_half=1,
        expected_part_id=109,
        expected_original_index=1096,
        recovar_term_divisor=16.0,
    )
    assert report["identity"]["rotation_chunk_count"] == 2
    assert report["local_replay"]["scale_adjusted_aa_terms_bit_exact"]
    assert report["local_replay"]["per_pixel_reduction_bit_exact"]
    assert report["native_shell_comparison"]["relative_l2"] == 0.0
