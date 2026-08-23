import numpy as np

from recovar.utils import helpers
from scripts import analyze_bpref_reduction_precision_ab as analyzer
from scripts.analyze_bpref_reduction_precision_ab import (
    _factor_comparisons,
    _load_recovar_mrc,
    _load_relion_mrc,
)


def test_factor_ab_requires_only_production_numerator_to_change(tmp_path):
    default = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
    highest = np.array([[1.001 + 2j, 3 + 4.001j]], dtype=np.complex64)
    common = {
        "numerator_highest_f32": highest,
        "numerator_sequential_f32": highest,
        "term_f32": np.ones((1, 2), dtype=np.complex64),
    }
    control = tmp_path / "control.npz"
    fixed = tmp_path / "fixed.npz"
    np.savez(control, numerator_f32=default, **common)
    np.savez(fixed, numerator_f32=highest, **common)

    result = _factor_comparisons(control, fixed)

    assert result["changed_arrays"] == ["numerator_f32"]
    assert result["fixed_matches_highest_exactly"] is True


def test_map_loaders_return_common_recovar_frame(tmp_path):
    volume = np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)
    recovar_path = tmp_path / "recovar.mrc"
    relion_path = tmp_path / "relion.mrc"
    helpers.write_mrc(str(recovar_path), volume)
    helpers.write_relion_mrc(str(relion_path), volume)

    recovar_loaded = _load_recovar_mrc(recovar_path)
    relion_loaded = _load_relion_mrc(relion_path)

    np.testing.assert_array_equal(recovar_loaded, volume)
    np.testing.assert_array_equal(relion_loaded, volume)
    np.testing.assert_array_equal(recovar_loaded, relion_loaded)


def test_analysis_schema_uses_canonical_signed_relion_comparisons(tmp_path, monkeypatch):
    control = tmp_path / "control"
    fixed = tmp_path / "fixed"
    relion = tmp_path / "relion"
    volume = np.random.default_rng(0).standard_normal((8, 8, 8)).astype(np.float32)
    for root in (control, fixed):
        (root / "recovar").mkdir(parents=True)
        (root / "intermediates").mkdir()
        for name in ("final_half1.mrc", "final_half2.mrc", "final_merged.mrc"):
            helpers.write_mrc(str(root / "recovar" / name), volume)
        for half in (0, 1):
            for field in ("Ft_y", "Ft_ctf"):
                np.save(root / "intermediates" / f"it000_{field}_{half}.npy", np.ones(2, dtype=np.float32))
    relion.mkdir()
    for name in ("run_it001_half1_class001.mrc", "run_it001_half2_class001.mrc"):
        helpers.write_relion_mrc(str(relion / name), volume)

    monkeypatch.setattr(
        analyzer,
        "_timing",
        lambda _root: {
            "gpu_uuid": "GPU-test",
            "production_wall_s": 1,
            "factor_extraction_wall_s": 1,
            "pass2_half_wall_s": [1.0, 1.0],
            "pass2_median_wall_s": 1.0,
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_factor_comparisons",
        lambda _control, _fixed: {"fixed_matches_highest_exactly": True},
    )

    report = analyzer.analyze(control, fixed, relion, expected_gpu_uuid="GPU-test")

    assert report["schema"] == "recovar-bpref-reduction-precision-ab-v2"
    assert report["relion_map_frame_alignment"]["ad_hoc_map_multiplier"] is None
    comparisons = report["map_fsc_comparisons"]
    assert "control_vs_relion_canonical" in comparisons
    assert "fixed_vs_relion_canonical" in comparisons
    assert "control_vs_relion_sign_aligned" not in comparisons
    assert comparisons["control_vs_relion_canonical"]["merged"]["fsc_auc_non_dc"] > 0.0
