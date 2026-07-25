import pytest

from scripts.compare_k4_backend_trajectories import compare


def _fsc_report(values_by_iteration):
    iterations = []
    for iteration, values in enumerate(values_by_iteration, start=1):
        iterations.append(
            {
                "relion_iteration": iteration,
                "classes": [
                    {
                        "cross_engine": {"fsc_auc": value},
                        "gt_fsc_auc_delta": -0.0001 * class_index,
                    }
                    for class_index, value in enumerate(values)
                ],
                "class_agreement": {
                    "status": "available",
                    "agreement": 1.0 - iteration * 0.001,
                },
            }
        )
    return {
        "schema": "em_k4_fsc_trajectory_audit_v2",
        "status": "fail",
        "earliest_failure": "synthetic",
        "numbered_iteration_count": len(iterations),
        "numbered_iterations": iterations,
    }


def _topology_report(status="pass"):
    return {
        "schema": "em_k4_control_topology_audit_v1",
        "status": status,
        "combined_control_pass": status == "pass",
    }


def _walltime(wall_s, gpu_uuid="GPU-fixed"):
    return {"wall_s": wall_s, "gpu_uuid": gpu_uuid}


def test_compare_k4_backend_trajectories_counts_fixed_gate_improvement():
    report = compare(
        _fsc_report([[0.996, 0.994], [0.993, 0.992]]),
        _fsc_report([[0.997, 0.996], [0.996, 0.992]]),
        _topology_report(),
        _topology_report(),
        _walltime(100),
        _walltime(80),
        baseline_label="host_numpy",
        candidate_label="relion_cuda",
    )

    assert (
        report["classification"]
        == "candidate_improves_fixed_direct_fsc_auc_gate_count"
    )
    assert report["backends"]["host_numpy"]["direct_fsc_auc_checks_passed"] == 1
    assert report["backends"]["relion_cuda"]["direct_fsc_auc_checks_passed"] == 3
    assert report["candidate_minus_baseline"]["direct_fsc_auc_checks_passed"] == 2
    assert report["candidate_minus_baseline"]["wall_s"] == -20


def test_compare_k4_backend_trajectories_records_exact_topology_status():
    report = compare(
        _fsc_report([[0.999, 0.998]]),
        _fsc_report([[0.999, 0.998]]),
        _topology_report(),
        _topology_report("fail"),
        _walltime(10),
        _walltime(10),
        baseline_label="host_numpy",
        candidate_label="relion_cuda",
    )

    assert report["backends"]["host_numpy"]["exact_control_topology"] is True
    assert report["backends"]["relion_cuda"]["exact_control_topology"] is False
    assert (
        report["classification"]
        == "candidate_preserves_fixed_direct_fsc_auc_gate_count"
    )


def test_compare_k4_backend_trajectories_rejects_cross_gpu_pair():
    with pytest.raises(ValueError, match="same physical GPU"):
        compare(
            _fsc_report([[0.999]]),
            _fsc_report([[0.999]]),
            _topology_report(),
            _topology_report(),
            _walltime(10, "GPU-one"),
            _walltime(10, "GPU-two"),
            baseline_label="host_numpy",
            candidate_label="relion_cuda",
        )
