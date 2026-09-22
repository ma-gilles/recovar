"""Exact Slurm allocation checks shared by EM qualification launchers."""
from __future__ import annotations

import os
import re
import subprocess
from typing import Any

class PairRunError(RuntimeError):
    """Raised when paired execution or provenance validation fails."""


def _parse_scontrol_fields(text: str) -> dict[str, str]:
    # Slurm field names are not restricted to alphanumerics.  In particular,
    # the one-line ``scontrol show job`` output contains keys such as
    # ``Socks/Node`` and ``NtasksPerN:B:S:C``.  If those delimiters are not
    # recognised, their text is accidentally appended to the preceding value
    # (most dangerously ``AllocTRES``), making an exact allocation look like a
    # resource mismatch.
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:.-]*)=", text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[match.group(1)] = text[start:end].strip()
    return fields


def _gpu_count_from_tres(value: str) -> int | None:
    generic = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", value)
    if generic is not None:
        return int(generic.group(1))
    typed = re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)(?:,|$)", value)
    return sum(int(item) for item in typed) if typed else None


def _slurm_allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return {"under_slurm": False, "job_id": None}
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = _parse_scontrol_fields(result.stdout.strip())
    requested = fields.get("ReqTRES", "")
    allocated = fields.get("AllocTRES", "")
    if requested != allocated:
        raise PairRunError(f"Slurm ReqTRES != AllocTRES: {requested!r} != {allocated!r}")
    if _gpu_count_from_tres(requested) != 1:
        raise PairRunError(f"paired run requires exactly one requested and allocated GPU: {requested}")
    if fields.get("OverSubscribe") != "OK":
        raise PairRunError(
            f"paired run must be nonexclusive (expected OverSubscribe=OK): {fields.get('OverSubscribe')}"
        )
    return {
        "under_slurm": True,
        "job_id": job_id,
        "ReqTRES": requested,
        "AllocTRES": allocated,
        "TresPerNode": fields.get("TresPerNode"),
        "OverSubscribe": fields.get("OverSubscribe"),
        "NumNodes": fields.get("NumNodes"),
        "NodeList": fields.get("NodeList"),
        "raw_scontrol": result.stdout.strip(),
    }
