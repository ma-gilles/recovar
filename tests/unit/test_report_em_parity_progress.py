from scripts.report_em_parity_progress import build_progress, render_markdown


def test_reports_current_scientific_panels() -> None:
    progress = build_progress()

    assert progress["schema"] == "recovar.em_parity_progress.v26"
    assert [
        (panel["label"], panel["passed"], panel["evaluated"], panel["denominator"])
        for panel in progress["panels"]
    ] == [
        ("K=1 strict FSC/FSC-AUC", 31, 34, 34),
        ("K=4 per-class FSC-AUC", 41, 60, 60),
        ("K=4 all-class iterations", 9, 15, 15),
        ("VDAM fixed RELION parity", 12, 12, 12),
    ]
    assert progress["realdata"] == {
        "calibration_cases": 3,
        "target": "empiar-10202-set06-k1-I1",
        "target_status": "relion_complete_recovar_pending",
    }
    assert progress["remaining"]["k1"] == ["k1-04", "k1-05", "k1-10"]


def test_markdown_distinguishes_current_and_historical_evidence() -> None:
    markdown = render_markdown(build_progress())

    assert "| K=1 strict FSC/FSC-AUC | **31** | 34 | 34 | 91.2% |" in markdown
    assert "recovar-experiments/tree/58574b1" in markdown
