import json

import numpy as np
import pytest

from recovar.em.ppca_refinement.fixture_validation import validate_kclass_to_ppca_initialization
from recovar.utils import helpers as utils


pytestmark = pytest.mark.unit


def _write_fixture(tmp_path):
    base = np.arange(64, dtype=np.float32).reshape(4, 4, 4) / 100.0
    pc1 = np.zeros_like(base)
    pc1[0, 0, 0] = 1.0
    pc2 = np.zeros_like(base)
    pc2[3, 3, 3] = -0.5
    volumes = np.stack(
        [
            base - pc1,
            base + 0.25 * pc1 + pc2,
            base + 0.75 * pc1 - pc2,
        ],
        axis=0,
    )
    weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)

    kclass_paths = []
    gt_rows = []
    for idx, vol in enumerate(volumes):
        kclass_path = tmp_path / f"recovar_class{idx + 1:03d}.mrc"
        gt_path = tmp_path / f"gt_class{idx + 1:03d}.mrc"
        utils.write_relion_mrc(kclass_path, vol)
        utils.write_mrc(str(gt_path), vol)
        kclass_paths.append(str(kclass_path))
        gt_rows.append({"class_index": idx, "class_number": idx + 1, "volume_path": str(gt_path)})

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "output_maps": kclass_paths,
                "recovar_class_weights": weights.tolist(),
                "relion_class_weights_in_recovar_order": weights.tolist(),
                "best_permutation": {"recovar_to_relion": [0, 1, 2]},
            }
        )
    )
    manifest_path = tmp_path / "class_manifest.json"
    manifest_path.write_text(json.dumps(gt_rows))
    return summary_path, manifest_path


def test_validate_kclass_to_ppca_initialization_real_mrc_fixture(tmp_path):
    summary_path, manifest_path = _write_fixture(tmp_path)

    validation, init, gt_init = validate_kclass_to_ppca_initialization(
        summary_path,
        class_manifest_json=manifest_path,
        q=2,
        min_gt_mean_correlation=0.999999,
        min_gt_subspace_agreement=0.999999,
    )

    assert validation.passed, validation.failures
    assert init.mu.shape == (4, 4, 4)
    assert init.W.shape == (2, 4, 4, 4)
    assert gt_init is not None
    assert validation.summary["covariance_trace"]["relative_error"] < 1e-6
    assert validation.summary["loader_frame_check"]["mu_relative_error"] < 1e-6
    assert validation.summary["loader_frame_check"]["W_subspace_agreement"] > 0.999999
    assert validation.summary["gt_comparison"]["mean_correlation"] > 0.999999
