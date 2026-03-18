#!/usr/bin/env python3
"""Parse JUnit XML results from parallel test groups and print a summary table.

Usage:
    python scripts/summarize_test_results.py results/*.xml

Exits 0 if all groups passed, 1 if any failures.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit_xml(path: Path) -> dict:
    """Extract top-level stats from a JUnit XML file."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return {"name": path.stem, "tests": 0, "fail": 0, "error": 0,
                "time": 0.0, "status": "PARSE_ERROR"}

    root = tree.getroot()
    # Handle both <testsuites> and <testsuite> as root
    if root.tag == "testsuites":
        suites = list(root)
    else:
        suites = [root]

    tests = fail = error = 0
    total_time = 0.0
    for ts in suites:
        tests += int(ts.get("tests", 0))
        fail += int(ts.get("failures", 0))
        error += int(ts.get("errors", 0))
        total_time += float(ts.get("time", 0))

    status = "PASS" if (fail + error) == 0 and tests > 0 else "FAIL"
    if tests == 0:
        status = "NO_TESTS"

    return {"name": path.stem, "tests": tests, "fail": fail + error,
            "time": total_time, "status": status}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: summarize_test_results.py <junit_xml> [<junit_xml> ...]",
              file=sys.stderr)
        sys.exit(2)

    results = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            results.append({"name": p.stem, "tests": 0, "fail": 0,
                            "time": 0.0, "status": "MISSING"})
        else:
            results.append(parse_junit_xml(p))

    total_tests = sum(r["tests"] for r in results)
    total_fail = sum(r["fail"] for r in results)
    total_time = sum(r["time"] for r in results)
    all_pass = all(r["status"] == "PASS" for r in results)

    print("\n===== PARALLEL TEST SUMMARY =====")
    for r in results:
        # Extract group name: strip the tag prefix (parallel_YYYYMMDD_HHMMSS_RAND_)
        name = r["name"]
        parts = name.split("_")
        # Find where the group name starts (after parallel_date_time_rand)
        if len(parts) > 4 and parts[0] == "parallel":
            group = "_".join(parts[4:])
        else:
            group = name
        print(f"  {group:20s} {r['status']:5s}  tests={r['tests']:4d}  "
              f"fail={r['fail']:2d}  time={r['time']:.1f}s")

    overall = "PASS" if all_pass else "FAIL"
    print(f"  {'TOTAL':20s} {overall:5s}  tests={total_tests:4d}  "
          f"fail={total_fail:2d}  time={total_time:.1f}s")
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
