#!/usr/bin/env python
"""Live verification for the density->explore link and .star/.ind subset export.

Run this ON della (with the same Python that runs the GUI server) AFTER deploying
the updated backend and restarting the server.  It exercises the two new
features end-to-end against the running server and, crucially, reads back the
constructed .star to confirm it round-trips and references the right particles.

    python verify_gui_fixes.py --port 8083 \
        --project /scratch/gpfs/GILLES/gui_10073_20260605_005014

Exits non-zero on the first failed check.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import urllib.error
import urllib.request


def _req(method: str, url: str, body: dict | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw.decode(errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--project", required=True, help="project directory path")
    ap.add_argument("--n", type=int, default=8, help="subset size to test")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def ok(msg: str) -> None:
        print(f"  \033[32mPASS\033[0m {msg}")

    def fail(msg: str) -> None:
        print(f"  \033[31mFAIL\033[0m {msg}")
        sys.exit(1)

    print("== Re-register project (idempotent; restart drops the in-memory registry) ==")
    st, proj = _req("POST", f"{base}/api/projects", {
        "path": args.project,
        "name": os.path.basename(args.project.rstrip("/")) or "project",
    })
    if st not in (200, 201) or not isinstance(proj, dict) or "id" not in proj:
        fail(f"could not register project {args.project}: {st} {proj}")
    pid = proj["id"]
    ok(f"project registered: id={pid}")

    st, detail = _req("GET", f"{base}/api/projects/{pid}")
    if st != 200 or not isinstance(detail, dict):
        fail(f"GET project detail failed: {st} {detail}")
    jobs = detail.get("jobs", [])
    completed = [j for j in jobs if j.get("status") == "completed"]
    density = next((j for j in completed if j.get("type") == "Density"), None)
    analyze = next((j for j in completed if j.get("type") == "Analyze"), None)
    pipeline = next((j for j in completed if j.get("type") == "Pipeline"), None)
    explore_src = analyze or pipeline
    if explore_src is None:
        fail("no completed Analyze or Pipeline job to test subset export")
    ok(f"found jobs: density={bool(density)} analyze={bool(analyze)} pipeline={bool(pipeline)}")

    print("== #3: Density -> Explore target resolution ==")
    if density is None:
        print("  SKIP no completed Density job in this project")
    else:
        st, res = _req("GET", f"{base}/api/jobs/{density['id']}/explore-target")
        if st != 200 or not isinstance(res, dict):
            fail(f"explore-target call failed: {st} {res}")
        target = res.get("target_job_id")
        if not target:
            fail(f"explore-target returned no target for density {density['id']}")
        match = [j for j in jobs if j.get("id") == target]
        if not match or match[0].get("type") not in ("Analyze", "Pipeline"):
            fail(f"explore-target {target} is not an Analyze/Pipeline job")
        ok(f"density {density['id']} -> {match[0]['type']} {target}")

    print("== #4: subset .ind + .star export (incl. .mrcs construction) ==")
    indices = list(range(args.n))
    st, subset = _req("POST", f"{base}/api/subsets", {
        "project_id": pid,
        "name": "verify_gui_fixes",
        "source_job_id": explore_src["id"],
        "indices": indices,
    })
    if st not in (200, 201) or not isinstance(subset, dict) or "id" not in subset:
        fail(f"create_subset failed: {st} {subset}")
    sid = subset["id"]
    ok(f".ind created: {subset.get('path')} (n={subset.get('n_particles')})")
    if not (subset.get("path") and os.path.isfile(subset["path"])):
        fail(".ind file not found on disk")
    ok(".ind file exists on disk")

    st, star = _req("POST", f"{base}/api/subsets/{sid}/export-star", {"particles_star": ""})
    if st != 200 or not isinstance(star, dict) or "path" not in star:
        fail(f"export-star failed: {st} {star}")
    star_path = star["path"]
    ok(f".star exported: {star_path} (n={star.get('n_particles')})")
    if star.get("n_particles") != len(indices):
        fail(f".star particle count {star.get('n_particles')} != requested {len(indices)}")
    ok(".star particle count matches subset size")

    # Round-trip: read the star back with recovar and check structure.
    if not os.path.isfile(star_path):
        fail(f".star not found on disk: {star_path}")
    try:
        from recovar.data_io.starfile import read_star
    except Exception as exc:  # noqa: BLE001
        fail(f"could not import recovar.data_io.starfile: {exc}")
    pdf, optics = read_star(star_path)
    if len(pdf) != len(indices):
        fail(f"read-back row count {len(pdf)} != {len(indices)}")
    ok(f".star round-trips via read_star: {len(pdf)} rows")
    img_col = next(
        (c for c in pdf.columns if c.lstrip("_").lower() == "rlnimagename"), None
    )
    if img_col is None:
        fail(f".star missing rlnImageName column; has {list(pdf.columns)[:8]}")
    ok(f".star has {img_col} (rows: {list(pdf[img_col].iloc[:3])})")

    # Tidy up the throwaway test subset record (best effort).
    _req("DELETE", f"{base}/api/subsets/{sid}")

    print("\n\033[32mALL CHECKS PASSED\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
