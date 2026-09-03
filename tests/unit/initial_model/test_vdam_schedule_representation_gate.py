from __future__ import annotations

import json

import numpy as np

from scripts import analyze_vdam_schedule_representation_gate as analyzer


def test_exact_panel_checks_every_pair() -> None:
    result = analyzer._exact_panel(
        {
            "dynamic_1": np.asarray([1, 2]),
            "dynamic_2": np.asarray([1, 2]),
            "oracle_1": np.asarray([1, 2]),
            "oracle_2": np.asarray([1, 2]),
        }
    )

    assert result["pair_count"] == 6
    assert result["all_exact"] is True


def test_multi_repeat_envelope_accepts_cross_delta_inside_repeat_noise() -> None:
    values = {
        "dynamic_1": np.asarray([0.0]),
        "dynamic_2": np.asarray([2.0]),
        "oracle_1": np.asarray([0.5]),
        "oracle_2": np.asarray([1.5]),
    }
    result = analyzer._multi_repeat_envelope(
        values,
        dynamic_labels=("dynamic_1", "dynamic_2"),
        oracle_labels=("oracle_1", "oracle_2"),
    )

    assert result["within_observed_repeat_envelope"] is True
    assert result["dynamic_pair_count"] == 1
    assert result["cross_pair_count"] == 4


def test_multi_repeat_envelope_rejects_unseen_cross_shift() -> None:
    values = {
        "dynamic_1": np.asarray([1.0]),
        "dynamic_2": np.asarray([1.0]),
        "oracle_1": np.asarray([2.0]),
        "oracle_2": np.asarray([2.0]),
    }
    result = analyzer._multi_repeat_envelope(
        values,
        dynamic_labels=("dynamic_1", "dynamic_2"),
        oracle_labels=("oracle_1", "oracle_2"),
    )

    assert result["within_observed_repeat_envelope"] is False
    assert result["repeat_envelope_normalized_l2"] == 0.0


def test_single_process_panel_passes_exact_and_zero_noise_contract(
    tmp_path, monkeypatch
) -> None:
    arms = {}
    for label in analyzer.PANEL_ARM_ORDER:
        dynamic = label.startswith("dynamic_")
        capacity = 64 if dynamic else 288
        arms[label] = {
            "label": label,
            "block_capacity": capacity,
            "wall_s": 1.0 if dynamic else 2.0,
            "output_prefix": str(tmp_path / label / "run_it026"),
        }
    panel = {
        "schema": analyzer.PANEL_SCHEMA,
        "arm_order": list(analyzer.PANEL_ARM_ORDER),
        "arms": arms,
        "checkpoint_optimiser_sha256": "a" * 64,
        "input_star_sha256": "b" * 64,
        "environment_without_block_capacity": {"EXACT_PROFILE": "1"},
    }
    (tmp_path / "panel.json").write_text(json.dumps(panel))

    def fake_artifacts(prefix):
        label = prefix.parent.name
        dynamic = label.startswith("dynamic_")
        representation = (
            "dense_full_direct_dynamic_fallback"
            if dynamic
            else "dense_full_direct_static_capacity"
        )
        meta = {
            field: np.asarray([1.0])
            for field in (*analyzer.REQUIRED_EXACT_META, *analyzer.ATOMIC_META)
        }
        meta.update(
            {
                field: 1.0 for field in analyzer.REQUIRED_EXACT_SCALARS
            }
        )
        meta.update(
            current_size=60,
            healpix_order=2,
            n_rotations=36864,
            n_translations=148,
            subset_size=200,
            random_perturbation=0.25,
        )
        return {
            "meta": meta,
            "hybrid": {
                "score_representation_batch_counts": {representation: 1},
                "fallback_batch_count": 1 if dynamic else 0,
                "fallback_reasons": {"block_capacity_overflow": 1}
                if dynamic
                else {},
            },
            "volume": np.ones((2, 2, 2), dtype=np.float32),
            "data_star": np.asarray([1.0]),
            "model_star": np.asarray([1.0]),
        }

    monkeypatch.setattr(analyzer, "_load_output_prefix", fake_artifacts)

    result = analyzer.analyze_panel(tmp_path)

    assert result["pass"] is True
    assert result["dynamic_repeat_count"] == 4
    assert result["oracle_repeat_count"] == 4
    assert result["wall_time"]["dynamic_speedup_over_oracle"] == 2.0
    assert result["provenance"]["capacity_contract_exact"] is True
