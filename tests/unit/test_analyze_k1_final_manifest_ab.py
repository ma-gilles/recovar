from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_k1_final_manifest_ab import ORDERED_FIELDS, _json_safe, analyze


def _write_manifest(path: Path, *, half: int, changed_field: str | None = None) -> None:
    values: dict[str, np.ndarray] = {
        "half_index": np.asarray(half, dtype=np.int32),
        "iteration": np.asarray(16, dtype=np.int32),
    }
    boolean_fields = {
        "half_spectrum_scoring",
        "use_float64_scoring",
        "use_float64_projections",
        "score_with_masked_images",
        "perturbation_applied",
        "local_search",
    }
    integer_fields = {
        "current_size",
        "projection_padding_factor",
        "reconstruction_padding_factor",
        "perturbation_relion_iteration",
    }
    complex_fields = {"mean_vol_ft"}
    for index, field in enumerate(ORDERED_FIELDS):
        if field in boolean_fields:
            value = np.asarray(index % 2 == 0, dtype=bool)
        elif field in integer_fields:
            value = np.asarray(index + 1, dtype=np.int32)
        elif field in complex_fields:
            value = np.asarray([index + 1j, index + 2j], dtype=np.complex128)
        else:
            value = np.asarray([index + 0.25, index + 0.5], dtype=np.float32)
        if field == changed_field:
            value = value.copy()
            if value.dtype == bool:
                value[...] = ~value
            else:
                value.reshape(-1)[0] += 1
        values[field] = value
    np.savez(path, **values)


def test_final_manifest_ab_reports_first_difference_in_declared_order(tmp_path):
    control_h1 = tmp_path / "control_h1.npz"
    control_h2 = tmp_path / "control_h2.npz"
    candidate_h1 = tmp_path / "candidate_h1.npz"
    candidate_h2 = tmp_path / "candidate_h2.npz"
    _write_manifest(control_h1, half=0)
    _write_manifest(control_h2, half=1)
    _write_manifest(candidate_h1, half=0, changed_field="image_corrections")
    _write_manifest(candidate_h2, half=1)

    report = analyze(
        control_half1=control_h1,
        control_half2=control_h2,
        candidate_half1=candidate_h1,
        candidate_half2=candidate_h2,
    )

    assert report["first_non_bit_exact"] == {
        "half_index": 0,
        "field": "image_corrections",
    }
    image_metrics = report["halves"][0]["fields"]["image_corrections"]
    assert image_metrics["bit_unequal_count"] == 1
    assert image_metrics["max_abs_delta"] == 1.0


def test_final_manifest_ab_is_exact_for_identical_inputs(tmp_path):
    paths = [tmp_path / f"manifest_{index}.npz" for index in range(4)]
    _write_manifest(paths[0], half=0)
    _write_manifest(paths[1], half=1)
    _write_manifest(paths[2], half=0)
    _write_manifest(paths[3], half=1)

    report = analyze(
        control_half1=paths[0],
        control_half2=paths[1],
        candidate_half1=paths[2],
        candidate_half2=paths[3],
    )

    assert report["first_non_bit_exact"] is None
    assert all(
        metrics["bit_equal_fraction"] == 1.0
        for half in report["halves"]
        for metrics in half["fields"].values()
    )


def test_final_manifest_ab_fails_closed_on_wrong_half(tmp_path):
    paths = [tmp_path / f"manifest_{index}.npz" for index in range(4)]
    _write_manifest(paths[0], half=0)
    _write_manifest(paths[1], half=1)
    _write_manifest(paths[2], half=1)
    _write_manifest(paths[3], half=1)

    with pytest.raises(ValueError, match="candidate manifest half_index=1, expected 0"):
        analyze(
            control_half1=paths[0],
            control_half2=paths[1],
            candidate_half1=paths[2],
            candidate_half2=paths[3],
        )


def test_final_manifest_ab_json_safe_uses_null_for_undefined_metrics():
    assert _json_safe({"finite": 1.0, "nan": float("nan"), "inf": float("inf")}) == {
        "finite": 1.0,
        "nan": None,
        "inf": None,
    }


def test_final_manifest_ab_treats_matching_empty_fields_as_bit_exact(tmp_path):
    paths = [tmp_path / f"empty_manifest_{index}.npz" for index in range(4)]
    for index, path in enumerate(paths):
        _write_manifest(path, half=index % 2)
        with np.load(path, allow_pickle=False) as archive:
            values = {field: archive[field] for field in archive.files}
        values["rotation_log_prior"] = np.array([], dtype=np.float64)
        np.savez(path, **values)

    report = analyze(
        control_half1=paths[0],
        control_half2=paths[1],
        candidate_half1=paths[2],
        candidate_half2=paths[3],
    )

    assert report["first_non_bit_exact"] is None
    assert report["halves"][0]["fields"]["rotation_log_prior"]["bit_equal_fraction"] == 1.0


def test_final_manifest_ab_aligns_candidate_particle_rows(tmp_path):
    paths = [tmp_path / f"ordered_manifest_{index}.npz" for index in range(4)]
    for index, path in enumerate(paths):
        _write_manifest(path, half=index % 2)
    gather = np.asarray([1, 0], dtype=np.int64)
    for path in paths[2:]:
        with np.load(path, allow_pickle=False) as archive:
            values = {field: archive[field] for field in archive.files}
        for field in (
            "rotation_log_prior",
            "translation_log_prior",
            "translation_prior_centers",
            "image_corrections",
            "scale_corrections",
            "image_pre_shifts",
            "absolute_previous_translations",
        ):
            values[field] = values[field][gather]
        np.savez(path, **values)

    report = analyze(
        control_half1=paths[0],
        control_half2=paths[1],
        candidate_half1=paths[2],
        candidate_half2=paths[3],
        candidate_row_gathers=(gather, gather),
    )

    assert report["first_non_bit_exact"] is None
    assert report["halves"][0]["candidate_row_alignment"]["row_count"] == 2
