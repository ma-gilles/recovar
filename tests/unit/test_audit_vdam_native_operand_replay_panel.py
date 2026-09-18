import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_vdam_native_operand_replay_panel as audit
from scripts.audit_vdam_repeat_panel import RepeatPanelError


def _write_json(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values))


def _panel(weights: list[float]) -> dict[str, object]:
    return {
        "weights": np.asarray([weights], dtype=np.float32),
        "eulers": np.eye(3, dtype=np.float32).reshape(1, 9),
    }


def _tree(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "submission_provenance.json",
        {"native_repeat_operands": True},
    )
    for repeat in (1, 2):
        paths = audit._expected_repeat_paths(tmp_path, repeat)
        paths["native_topology"].parent.mkdir(parents=True, exist_ok=True)
        paths["native_topology"].touch()
        paths["native_topology_data_star"].parent.mkdir(parents=True, exist_ok=True)
        paths["native_topology_data_star"].touch()
        paths["native_panel_directory"].mkdir(parents=True, exist_ok=True)
    for kind, start in (("private", 1), ("shared", 3)):
        for repeat in (1, 2):
            paths = audit._expected_repeat_paths(tmp_path, repeat)
            _write_json(
                tmp_path / f"arm-{start + repeat - 1:02d}-{kind}" / "provenance.json",
                {
                    **{name: str(path) for name, path in paths.items()},
                    "native_panel_weights": True,
                },
            )
            _write_json(
                tmp_path
                / f"arm-{start + repeat - 1:02d}-{kind}"
                / "replay"
                / "worker_private_report.json",
                {"native_panel_weights_replayed": True},
            )
    return tmp_path


def test_reports_native_posterior_and_replay_width(monkeypatch, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    monkeypatch.setattr(
        audit,
        "audit_native_bpref_repeat_panel",
        lambda *args, **kwargs: {"schema": "base", "result": "complete"},
    )
    panels = [
        {10: _panel([0.8, 0.2]), 20: _panel([0.5, 0.5])},
        {10: _panel([0.7, 0.3]), 20: _panel([0.4, 0.6])},
    ]

    def load_panels(path, *_args, **_kwargs):
        return panels[0 if "repeat-01" in str(path) else 1]

    monkeypatch.setattr(audit, "_load_native_panels", load_panels)
    monkeypatch.setattr(
        audit,
        "_load_native_topology",
        lambda path, *_args, **_kwargs: {
            10: (0 if "repeat-01" in str(path) else 1, 1),
            20: (2, 1),
        },
    )
    report = audit.audit_native_operand_replay_panel(root, repeat_count=2)
    assert report["schema"] == audit.SCHEMA
    metrics = report["native_repeat_operands"]
    assert metrics["particle_count"] == 2
    assert metrics["pooled_normalized_posterior"][
        "maximum_pairwise_relative_l2"
    ] > 0.0
    assert metrics["worker_owner_repeat_mismatch_count"] == 1
    assert metrics["support_mismatch_coordinate_count"] == 0


def test_rejects_arm_borrowing_wrong_native_repeat(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    provenance_path = root / "arm-04-shared" / "provenance.json"
    values = json.loads(provenance_path.read_text())
    values["native_panel_directory"] = str(root / "repeat-01" / "native_panels")
    _write_json(provenance_path, values)
    with pytest.raises(RepeatPanelError, match="does not come from native repeat 2"):
        audit._validate_replay_paths(root, repeat_count=2)
