import numpy as np

from recovar import utils
from recovar.commands import spike_kernel_report


def _write_compute_state_candidates(root, grid, volumes, *, deconvolved=False):
    state_dir = root / "diagnostics" / "state000"
    estimates = state_dir / "estimates_filt"
    estimates.mkdir(parents=True)
    params = {"voxel_size": 1.5, "heterogeneity_bins": np.asarray(grid, dtype=np.float32)}
    if deconvolved:
        params["lambda_grid"] = np.asarray(grid, dtype=np.float32)
        params["sigma_ref"] = 1.0
    utils.pickle_dump(params, state_dir / "params.pkl")
    for idx, volume in enumerate(volumes):
        utils.write_mrc(str(estimates / f"{idx:04d}.mrc"), np.asarray(volume, dtype=np.float32), voxel_size=1.5)


def test_spike_kernel_report_generates_core_outputs(tmp_path):
    shape = (8, 8, 8)
    rng = np.random.default_rng(0)
    target = rng.normal(size=shape).astype(np.float32)
    mask = np.ones(shape, dtype=np.float32)
    standard_volumes = [target + 0.05 * (idx + 1) for idx in range(3)]
    deconv_volumes = [target + 0.04 * (idx + 1) for idx in range(3)]

    standard_root = tmp_path / "standard"
    deconvolved_root = tmp_path / "deconvolved"
    _write_compute_state_candidates(standard_root, [0.5, 1.0, 2.0], standard_volumes)
    _write_compute_state_candidates(deconvolved_root, [0.3, 0.6, 1.2], deconv_volumes, deconvolved=True)

    target_path = tmp_path / "gt.mrc"
    mask_path = tmp_path / "mask.mrc"
    utils.write_mrc(str(target_path), target, voxel_size=1.5)
    utils.write_mrc(str(mask_path), mask, voxel_size=1.5)

    cfg = spike_kernel_report.SpikeKernelReportConfig(
        standard_root=standard_root,
        deconvolved_root=deconvolved_root,
        target_volume=target_path,
        mask=mask_path,
        out_dir=tmp_path / "08_kernel_report",
        expected_candidates=3,
        report_title="unit test",
    )
    summary = spike_kernel_report.generate_report(cfg)

    assert (cfg.out_dir / "summary.json").exists()
    assert (cfg.out_dir / "README.md").exists()
    assert (cfg.out_dir / "plots" / "all_candidates_true_gt_fsc_error.png").exists()
    assert (cfg.out_dir / "plots" / "selected_standard_vs_deconvolved_true_gt_overlay.png").exists()
    assert (cfg.out_dir / "oracle_estimators" / "combined_standard_deconvolved_shell_oracle_estimator.mrc").exists()
    assert summary["all_candidate_report"]["standard_best_index_0based"] in {0, 1, 2}
    assert summary["all_candidate_report"]["deconv_best_index_0based"] in {0, 1, 2}
