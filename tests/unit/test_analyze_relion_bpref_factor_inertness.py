import json

import mrcfile
import numpy as np

from scripts import analyze_relion_bpref_factor_inertness as inertness


def _write_bpref(path, values):
    with path.open("wb") as stream:
        np.asarray(values.shape, dtype=np.int64).tofile(stream)
        values.tofile(stream)


def test_inertness_supports_nondefault_capture_iteration(tmp_path):
    control = tmp_path / "control"
    capture = tmp_path / "capture"
    for root in (control, capture):
        (root / "dumps").mkdir(parents=True)
        (root / "relion").mkdir()

    complex_values = np.arange(12, dtype=np.float64).reshape(3, 2, 2).astype(np.complex128)
    real_values = np.arange(1, 13, dtype=np.float64).reshape(3, 2, 2)
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(8, 8, 8)).astype(np.float32)
    for root in (control, capture):
        for rank in (1, 2):
            prefix = f"mstep_it004_rank{rank}_half{rank}_c0_pre_lowres_join_bpref"
            _write_bpref(root / "dumps" / f"{prefix}_data.bin", complex_values)
            _write_bpref(root / "dumps" / f"{prefix}_weight.bin", real_values)
        for half in (1, 2):
            with mrcfile.new(root / "relion" / f"run_it004_half{half}_class001.mrc") as handle:
                handle.set_data(volume)

    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "capture_inertness_qualified": True,
                "array_comparisons": {
                    f"rank{rank}_half{rank}_{field}": {
                        "control_a_vs_control_b": {"relative_l2": 1e-12}
                    }
                    for rank in (1, 2)
                    for field in ("data", "weight")
                },
                "map_fsc_comparisons": {
                    f"half{half}": {"control_a_vs_control_b": {"fsc_auc_non_dc": 0.999999}}
                    for half in (1, 2)
                },
            }
        )
    )

    report = inertness.analyze(control, capture, reference, iteration=4, multiplier=2.0)

    assert report["capture_iteration"] == 4
    assert report["capture_inertness_qualified"] is True
    assert all(item["exact_equal"] for item in report["array_comparisons"].values())
    assert all(item["fsc_auc_non_dc"] == 1.0 for item in report["map_fsc_comparisons"].values())
