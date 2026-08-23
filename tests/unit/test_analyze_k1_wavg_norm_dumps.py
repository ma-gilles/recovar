import numpy as np
import pytest

from scripts.analyze_k1_wavg_norm_dumps import analyze


@pytest.mark.unit
def test_wavg_norm_dump_analysis_joins_original_rows_across_halves(tmp_path):
    image_size = 4
    divisor = float(image_size**4)
    native_path = tmp_path / "native.npz"
    np.savez(
        native_path,
        input_row=np.asarray([0, 1, 2, 3]),
        half=np.asarray([1, 2, 1, 2]),
        direct_current_size=np.asarray([1.0, 2.0, 3.0, 4.0]),
        powerclass_high_shell=np.asarray([0.5, 0.6, 0.7, 0.8]),
        total=np.asarray([1.5, 2.6, 3.7, 4.8]),
    )
    recovar_paths = []
    for half, rows in ((1, np.asarray([2, 0])), (2, np.asarray([3, 1]))):
        direct = np.asarray([3.0, 1.0]) if half == 1 else np.asarray([4.0, 2.0])
        high = np.asarray([0.7, 0.5]) if half == 1 else np.asarray([0.8, 0.6])
        path = tmp_path / f"recovar_half{half}.npz"
        np.savez(
            path,
            half=np.asarray(half),
            original_row=rows,
            direct_current_size=direct * divisor,
            powerclass_high_shell=high * divisor,
            total=(direct + high) * divisor,
        )
        recovar_paths.append(path)

    report = analyze(native_path, recovar_paths, image_size=image_size)

    assert report["particle_count"] == 4
    assert report["metrics"]["direct_current_size"]["max_abs"] == 0.0
    assert report["metrics"]["powerclass_high_shell"]["max_abs"] == 0.0
    assert report["metrics"]["total"]["max_abs"] == 0.0
    assert [record["half"] for record in report["halves"]] == [1, 2]
