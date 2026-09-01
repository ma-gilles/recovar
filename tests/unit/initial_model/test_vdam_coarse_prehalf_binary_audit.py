import pytest

from scripts import audit_vdam_coarse_prehalf_binary as audit


def _sass(header: str, half_count: int, *, mutate_encoding: bool = False) -> str:
    instructions = []
    address = 0
    for index in range(half_count):
        encoding = 0x100 + index + int(mutate_encoding and index == 0)
        instructions.append(
            f"        /*{address:04x}*/ FMUL R1, R1, 0.5 ; /* 0x{encoding:016x} */\n"
            "                                      /* 0x0000000000000001 */"
        )
        address += 16
    instructions.append(
        f"        /*{address:04x}*/ FFMA R2, R1, R3, R2 ; /* 0x0000000000000200 */\n"
        "                                      /* 0x0000000000000002 */"
    )
    return f"Function : {header}\n" + "\n".join(instructions) + "\n"


def _resource(header: str, *, registers: int = 48, shared: int = 7040) -> str:
    return f" Function {header}:\n  REG:{registers} STACK:0 SHARED:{shared} LOCAL:0 CONSTANT[0]:644\n"


def _fixtures(*, mutate_default_encoding: bool = False, prehalf_registers: int = 48):
    baseline_header = f"prefix_{audit.KERNEL}{audit.BASELINE_TOKEN}_suffix"
    default_header = f"prefix_{audit.KERNEL}{audit.DEFAULT_TOKEN}_suffix"
    prehalf_header = f"prefix_{audit.KERNEL}{audit.PREHALF_TOKEN}_suffix"
    baseline_sass = _sass(baseline_header, 16)
    candidate_sass = _sass(
        default_header,
        16,
        mutate_encoding=mutate_default_encoding,
    ) + _sass(prehalf_header, 1)
    baseline_resources = _resource(baseline_header)
    candidate_resources = _resource(default_header) + _resource(
        prehalf_header,
        registers=prehalf_registers,
    )
    return {
        "baseline_sass": baseline_sass,
        "candidate_sass": candidate_sass,
        "baseline_resources": baseline_resources,
        "candidate_resources": candidate_resources,
    }


@pytest.mark.unit
def test_binary_audit_accepts_exact_default_and_16_to_1_prehalf() -> None:
    report = audit._analyze_dumps(**_fixtures())

    assert report["pass"] is True
    assert all(report["gates"].values())
    assert report["sass"]["baseline"]["half_fmul_count"] == 16
    assert report["sass"]["candidate_default"]["half_fmul_count"] == 16
    assert report["sass"]["candidate_prehalf"]["half_fmul_count"] == 1
    assert report["default_atomic_sass_comparison"]["exact"] is True
    assert report["default_atomic_sass_comparison"]["instruction_text_mismatch_count"] == 0
    assert report["default_atomic_sass_comparison"]["encoding_mismatch_count"] == 0
    assert report["default_enablement_allowed"] is False
    assert report["production_wiring_evaluated"] is False


@pytest.mark.unit
def test_binary_audit_rejects_default_sass_change() -> None:
    report = audit._analyze_dumps(**_fixtures(mutate_default_encoding=True))

    assert report["gates"]["default_atomic_sass_exact"] is False
    assert report["default_atomic_sass_comparison"]["instruction_text_mismatch_count"] == 0
    assert report["default_atomic_sass_comparison"]["encoding_mismatch_count"] == 1
    assert (
        report["default_atomic_sass_comparison"]["baseline_instruction_sequence_sha256"]
        != report["default_atomic_sass_comparison"]["candidate_instruction_sequence_sha256"]
    )
    assert report["pass"] is False


@pytest.mark.unit
def test_binary_audit_rejects_prehalf_register_growth() -> None:
    report = audit._analyze_dumps(**_fixtures(prehalf_registers=49))

    assert report["gates"]["prehalf_resource_nonregression"] is False
    assert report["pass"] is False


@pytest.mark.unit
def test_binary_audit_rejects_missing_prehalf_specialization() -> None:
    fixtures = _fixtures()
    prehalf_header = f"Function : prefix_{audit.KERNEL}{audit.PREHALF_TOKEN}_suffix"
    fixtures["candidate_sass"] = fixtures["candidate_sass"].split(prehalf_header, 1)[0]

    with pytest.raises(
        audit.BinaryAuditError,
        match="candidate pre-half atomic kernel matched 0 kernel blocks",
    ):
        audit._analyze_dumps(**fixtures)
