"""Unit tests for the pass-1/pass-2 top-2 candidate-margin debug helpers.

These are diagnostic-only, env-gated helpers (default off, zero-cost when
unset) added to answer: for a specific particle whose final pose disagrees
with RELION, how close was recovar's own winning candidate to its runner-up
at (a) pass-1's coarse RELION-firstiter_cc winner-take-all scoring
(``k_class._log_pass1_top2_debug``) and (b) pass-2's fine/oversampled
per-particle scoring (``sparse_pass2_bucketed._log_pass2_top2_debug``)? A
small margin (comparable to the ~1e-4 GPU arithmetic-parity scale in
``recovar/em/CLAUDE.md``) indicates an ordinary near-tie flip; a large one
would indicate a real implementational gap.

See ``docs/math/relion_parity_agent_notes.md`` for the investigation this
tooling was built for.
"""

import logging

import numpy as np
import pytest

from recovar.em.dense_single_volume import k_class as k_class_module
from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sparse_pass2_module


class TestPass1Top2DebugIndices:
    def test_empty_by_default(self, monkeypatch):
        monkeypatch.delenv("RECOVAR_PASS1_TOP2_DEBUG_INDICES", raising=False)
        assert k_class_module._pass1_top2_debug_target_indices() == ()

    def test_parses_comma_separated_ints(self, monkeypatch):
        monkeypatch.setenv("RECOVAR_PASS1_TOP2_DEBUG_INDICES", "220,292")
        assert k_class_module._pass1_top2_debug_target_indices() == (220, 292)

    def test_blank_env_value_is_empty(self, monkeypatch):
        monkeypatch.setenv("RECOVAR_PASS1_TOP2_DEBUG_INDICES", "")
        assert k_class_module._pass1_top2_debug_target_indices() == ()


class TestLogPass1Top2Debug:
    def _full_coarse_stats(self, n_images=3):
        # (n_classes=1, n_images) layout, matching
        # significance.py's _compute_k_class_significance_batched.
        best = np.zeros((1, n_images), dtype=np.float32)
        second = np.zeros((1, n_images), dtype=np.float32)
        best_assign = np.zeros((1, n_images), dtype=np.int32)
        second_assign = np.zeros((1, n_images), dtype=np.int32)
        best[0, 1] = 0.51974893
        second[0, 1] = 0.51689810
        best_assign[0, 1] = 675656
        second_assign[0, 1] = 697964
        return {
            "class_best_offset_free_log_score_per_image": best,
            "class_second_best_offset_free_log_score_per_image": second,
            "class_hard_assignments": best_assign,
            "class_second_hard_assignments": second_assign,
        }

    def test_logs_margin_and_pose_ids(self, caplog):
        with caplog.at_level(logging.WARNING, logger=k_class_module.logger.name):
            k_class_module._log_pass1_top2_debug(
                self._full_coarse_stats(), (1,), dataset_tag="h2"
            )
        [record] = [r for r in caplog.records if "PASS1_TOP2_DEBUG" in r.message]
        assert "dataset=h2" in record.message
        assert "idx=1" in record.message
        assert "best_pose_id=675656" in record.message
        assert "second_pose_id=697964" in record.message
        # margin = 0.51974893 - 0.51689810
        assert "margin=0.0028508" in record.message

    def test_out_of_range_index_warns_without_crashing(self, caplog):
        with caplog.at_level(logging.WARNING, logger=k_class_module.logger.name):
            k_class_module._log_pass1_top2_debug(self._full_coarse_stats(), (99,), dataset_tag="h2")
        assert any("out of range" in r.message for r in caplog.records)

    def test_missing_second_best_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger=k_class_module.logger.name):
            k_class_module._log_pass1_top2_debug({}, (0,), dataset_tag="h2")
        assert any("were not returned" in r.message for r in caplog.records)

    def test_decodes_euler_free_rotation_matrices_when_supplied(self, tmp_path, monkeypatch):
        n_rot = 4
        rotations = np.stack([np.eye(3, dtype=np.float32) * (i + 1) for i in range(n_rot)])
        n_translations = 5
        stats = self._full_coarse_stats()
        # best_id=675656, second_id=697964 -> rot ids far outside this tiny
        # synthetic grid, so use small deliberate ids instead for this check.
        stats["class_hard_assignments"][0, 1] = 2 * n_translations + 3  # rot=2
        stats["class_second_hard_assignments"][0, 1] = 1 * n_translations + 0  # rot=1
        dump_path = tmp_path / "{dataset_tag}_{idx}.npz"
        monkeypatch.setenv("RECOVAR_PASS1_TOP2_DEBUG_DUMP_PATH", str(dump_path))
        k_class_module._log_pass1_top2_debug(
            stats,
            (1,),
            dataset_tag="h2",
            rotations=rotations,
            n_translations=n_translations,
        )
        dumped = np.load(str(tmp_path / "h2_1.npz"))
        np.testing.assert_array_equal(dumped["best_rot_matrix"], rotations[2])
        np.testing.assert_array_equal(dumped["second_rot_matrix"], rotations[1])
        assert int(dumped["best_pose_id"]) == 2 * n_translations + 3
        assert int(dumped["second_pose_id"]) == 1 * n_translations + 0


class TestPass2Top2DebugIndices:
    def test_empty_by_default(self, monkeypatch):
        monkeypatch.delenv("RECOVAR_PASS2_TOP2_DEBUG_INDICES", raising=False)
        assert sparse_pass2_module._pass2_top2_debug_target_indices() == ()

    def test_parses_comma_separated_ints(self, monkeypatch):
        monkeypatch.setenv("RECOVAR_PASS2_TOP2_DEBUG_INDICES", "220,292")
        assert sparse_pass2_module._pass2_top2_debug_target_indices() == (220, 292)


class TestLogPass2Top2Debug:
    def test_logs_margin_and_decodes_flat_rot_trans_ids(self, caplog):
        # scores: (batch=2, R=8, T=4); target row=1 (image_idx=220).
        rng = np.random.default_rng(0)
        scores = rng.uniform(0.0, 0.5, size=(2, 8, 4)).astype(np.float32)
        scores[1].flat[:] = 0.1
        scores[1, 3, 2] = 0.52413278  # flat id 3*4+2=14 -> best
        scores[1, 3, 3] = 0.52410786  # flat id 15 -> second
        image_indices = np.array([999, 220])

        with caplog.at_level(logging.WARNING, logger=sparse_pass2_module.logger.name):
            sparse_pass2_module._log_pass2_top2_debug(scores, image_indices, (220,), dataset_tag="h2")

        [record] = [r for r in caplog.records if "PASS2_TOP2_DEBUG" in r.message]
        assert "dataset=h2" in record.message
        assert "image_idx=220" in record.message
        assert "n_candidates=32" in record.message
        assert "best_flat_id=14(rot=3,trans=2)" in record.message
        assert "second_flat_id=15(rot=3,trans=3)" in record.message
        assert "margin=2.4914e-05" in record.message or "margin=2.49" in record.message

    def test_index_not_in_bucket_is_silently_skipped(self, caplog):
        scores = np.zeros((1, 2, 2), dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger=sparse_pass2_module.logger.name):
            sparse_pass2_module._log_pass2_top2_debug(scores, np.array([5]), (999,), dataset_tag="h2")
        assert not [r for r in caplog.records if "PASS2_TOP2_DEBUG" in r.message]
