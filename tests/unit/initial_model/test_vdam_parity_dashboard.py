from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD = REPO_ROOT / "docs" / "math" / "vdam_relion_parity_dashboard.md"


@pytest.fixture(scope="module")
def dashboard() -> str:
    return DASHBOARD.read_text()


@pytest.mark.unit
def test_dashboard_keeps_frozen_and_secondary_tracks_separate(dashboard: str) -> None:
    assert "**Authoritative release status: NOT READY.**" in dashboard
    assert re.findall(
        r"\| Frozen v3 K=1 correctness \| \*\*(\d+) / (\d+)\*\* \|",
        dashboard,
    ) == [("2", "20")]
    assert re.findall(
        r"\| Frozen v3 runtime \| \*\*(\d+) / (\d+)\*\* \|",
        dashboard,
    ) == [("0", "20")]
    assert (
        "| `legacy_parameter_expansion_v2` | **6 / 15** | `regression` | **none** |"
        in dashboard
    )
    assert (
        "neither a subtotal nor evidence that the v3 score is 8/35"
        in dashboard.replace("\n", " ")
    )

    for heading in (
        "## Frozen v3 K=1 scorecard",
        "## Non-scoring v2 expansion",
        "## Correctness: current hybrid-GEMM blocker",
        "## Runtime and performance lanes",
        "### Retained or accepted for narrower use",
        "### Rejected or do not promote",
        "## K>1",
        "## Real data",
        "## Next gates",
        "## Evidence and reproducibility",
        "## Code references",
    ):
        assert heading in dashboard


@pytest.mark.unit
def test_dashboard_records_hybrid_evidence_without_score_inflation(dashboard: str) -> None:
    expected_facts = (
        "roughly **4.5x faster**",
        "Explicit componentwise `abs2`, job `13327200` | **NO GO**",
        "Full promoted operands / FP64, job `13327874` | **NO GO**",
        "Selected 16-rotation-block direct primitive, job `13328717` | **PRIMITIVE PASS**",
        "Dual raw/post streaming certificate at `6e4e0ae65cba3cd0f86febdfe319e747ced07d97` | **INTEGRATED**",
        "All-1000 GF46 diagnostic, job `13329608` | **COMPLETE / NON-SCORING**",
        "**1,000 / 1,000** particles over **1,069,056,000** finite pairs",
        "`Emax=3.3125`, RMS `0.05108`, signed mean `+0.00789`",
        "Winner mismatch **1 / 1,000** (`particle 1933`, tiny margins)",
        "one extra and **0 false negatives** (`particle 636`)",
        "**727 / 1,000** covered",
        "median / p95 / max = `6 / 6 / 6`",
        "The next selector is **per-source-block maxima**",
        "Score impact: **none**.",
    )
    for fact in expected_facts:
        assert fact in dashboard

    assert "standalone GEMM arms changed downstream discrete state" in dashboard
    assert "diagnostic-only and timing-ineligible" in dashboard
    assert "frozen correctness **2 / 20**" in dashboard
    assert "frozen runtime **0 / 20**" in dashboard
