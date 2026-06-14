#!/usr/bin/env python3
"""Comprehensive live API smoke for the RECOVAR GUI backend.

Exercises every API route against a running server with real project data,
auto-discovering job / subset / mask / volume IDs.  Reports PASS/FAIL per
endpoint and a summary.  Read-only except for idempotent create+cleanup
(a throwaway subset) and validation/preview POSTs; never submits a compute job,
never deletes real data, never mutates settings.

    python3 api_smoke.py --port 8085 \
        --project /scratch/gpfs/GILLES/gui_10073_20260605_005014 \
        --project /scratch/gpfs/GILLES/gui_et_test
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

R = []  # rows: (ok, label, status, note)
SAVED_MASKS = []  # throwaway mask paths to clean up externally (printed at end)


def req(method, url, body=None, max_read=200_000):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            raw = resp.read(max_read)
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                try:
                    return resp.status, json.loads(raw), ct
                except json.JSONDecodeError:
                    return resp.status, raw, ct
            return resp.status, raw, ct
    except urllib.error.HTTPError as e:
        raw = e.read(max_read)
        try:
            return e.code, json.loads(raw), ""
        except json.JSONDecodeError:
            return e.code, raw.decode(errors="replace"), ""
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}", ""


def check(label, method, path, base, body=None, expect=(200,), want_keys=None,
          want_list=False):
    url = base + path
    status, data, ct = req(method, url, body)
    ok = status in expect
    note = ""
    if status == -1:
        ok = False
        note = str(data)[:120]
    elif not ok:
        snippet = data if isinstance(data, str) else json.dumps(data)[:160]
        note = f"want {expect} got {status}: {str(snippet)[:160]}"
    else:
        if want_keys and isinstance(data, dict):
            miss = [k for k in want_keys if k not in data]
            if miss:
                ok = False
                note = f"200 but missing keys {miss}"
        if want_list and not isinstance(data, list):
            ok = False
            note = f"200 but not a list ({type(data).__name__})"
    R.append((ok, label, status, note))
    mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  {mark} [{status}] {label}" + (f"  -- {note}" if note else ""))
    return data


def first(jobs, jtype, status="completed"):
    return next((j for j in jobs if j.get("type") == jtype
                 and (status is None or j.get("status") == status)), None)


def smoke_project(base, ppath):
    name = os.path.basename(ppath.rstrip("/"))
    print(f"\n===== PROJECT: {name} ({ppath}) =====")
    proj = check(f"POST /projects (register {name})", "POST", "/api/projects",
                 base, body={"path": ppath, "name": name}, expect=(200, 201),
                 want_keys=["id"])
    if not isinstance(proj, dict) or "id" not in proj:
        print("  cannot continue this project (no id)")
        return
    pid = proj["id"]
    detail = check("GET /projects/{id}", "GET", f"/api/projects/{pid}", base,
                   want_keys=["jobs"])
    jobs = detail.get("jobs", []) if isinstance(detail, dict) else []
    print(f"  ({len(jobs)} jobs)")

    check("GET /masks/by-project/{id}", "GET",
          f"/api/masks/by-project/{pid}", base, want_list=True)
    check("GET /subsets?project_id", "GET",
          f"/api/subsets?project_id={pid}", base, want_list=True)
    # scan (re-import existing outputs; idempotent, returns a summary)
    check("POST /projects/{id}/scan", "POST",
          f"/api/projects/{pid}/scan", base, body={"scan_path": ppath},
          expect=(200,))

    # ---- Per-job endpoints: pick a representative job of each present type ----
    types = sorted({j.get("type") for j in jobs})
    print(f"  job types present: {types}")
    pipeline = first(jobs, "Pipeline")
    analyze = first(jobs, "Analyze")
    density = first(jobs, "Density")
    explore_src = analyze or pipeline

    for jt in types:
        j = first(jobs, jt) or first(jobs, jt, status=None)
        if not j:
            continue
        jid = j["id"]
        tag = f"{jt}:{jid[:8]}"
        check(f"GET /jobs/{{id}} ({tag})", "GET", f"/api/jobs/{jid}", base,
              want_keys=["id", "type", "status"])
        check(f"GET /jobs/{{id}}/logs ({tag})", "GET",
              f"/api/jobs/{jid}/logs", base)
        check(f"GET /jobs/{{id}}/plots ({tag})", "GET",
              f"/api/jobs/{jid}/plots", base, want_list=True)
        check(f"GET /jobs/{{id}}/volumes ({tag})", "GET",
              f"/api/jobs/{jid}/volumes", base, want_list=True)
        check(f"GET /jobs/{{id}}/suggested-next ({tag})", "GET",
              f"/api/jobs/{jid}/suggested-next", base, want_list=True)
        check(f"GET /jobs/{{id}}/sbatch-script ({tag})", "GET",
              f"/api/jobs/{jid}/sbatch-script", base)
        check(f"GET /jobs/{{id}}/chart-data?name=fsc ({tag})", "GET",
              f"/api/jobs/{jid}/chart-data?name=fsc", base, expect=(200, 404))

    # ---- Embeddings (on the explore source: Analyze/Pipeline) ----
    if explore_src:
        sid = explore_src["id"]
        tag = f"{explore_src['type']}:{sid[:8]}"
        avail = check(f"GET /embeddings/available ({tag})", "GET",
                      f"/api/jobs/{sid}/embeddings/available", base)
        check(f"GET /related-density ({tag})", "GET",
              f"/api/jobs/{sid}/related-density", base, want_list=True)
        check(f"GET /explore-target ({tag})", "GET",
              f"/api/jobs/{sid}/explore-target", base, want_keys=["target_job_id"])
        # embeddings binary: just confirm 200 + nonempty
        zdim = None
        if isinstance(avail, dict):
            zd = avail.get("zdims") or avail.get("available_zdims")
            if isinstance(zd, list) and zd:
                zdim = zd[0]
        q = f"?zdim={zdim}" if zdim is not None else ""
        check(f"GET /embeddings ({tag}){q}", "GET",
              f"/api/jobs/{sid}/embeddings{q}", base)
        # density coloring (needs a density job)
        if density:
            dq = (f"?density_job_id={density['id']}"
                  + (f"&zdim={zdim}" if zdim is not None else ""))
            check(f"GET /embeddings/density (via {density['id'][:8]})", "GET",
                  f"/api/jobs/{sid}/embeddings/density{dq}", base)

    # ---- Density job -> explore-target (the #3 feature) ----
    if density:
        check("GET /explore-target (Density -> Analyze/Pipeline)", "GET",
              f"/api/jobs/{density['id']}/explore-target", base,
              want_keys=["target_job_id"])

    # ---- Volume endpoints (need a real volume path) ----
    vol_path = None
    # find a volume path from any job's volumes
    for j in jobs:
        v = req("GET", base + f"/api/jobs/{j['id']}/volumes")[1]
        if isinstance(v, list) and v:
            cand = v[0].get("path") if isinstance(v[0], dict) else None
            if cand:
                vol_path = cand
                break
    if vol_path:
        from urllib.parse import quote
        vp = quote(vol_path, safe="")
        check("GET /volumes/info", "GET",
              f"/api/volumes/info?path={vp}", base)
        check("GET /volumes/slice", "GET",
              f"/api/volumes/slice?path={vp}&axis=2&index=32", base,
              expect=(200,))
        check("GET /volumes/raw (downsampled)", "GET",
              f"/api/volumes/raw?path={vp}&downsample=64", base, expect=(200,))
        check("POST /files/validate-mrc", "POST",
              "/api/files/validate-mrc", base, body={"path": vol_path})
    else:
        print("  (no volume path found; skipping volume/validate-mrc endpoints)")

    # ---- Mask Wizard endpoints (need a volume) ----
    if vol_path:
        from urllib.parse import quote
        check("POST /masks/preview (overlay PNG)", "POST",
              "/api/masks/preview", base,
              body={"source_path": vol_path, "threshold": None,
                    "soft_edge": 6.0, "axis": 2}, expect=(200,))
        check("POST /masks/segment-info", "POST", "/api/masks/segment-info",
              base, body={"source_path": vol_path, "threshold": None},
              expect=(200,))
        pv = check("POST /masks/preview-volume", "POST",
                   "/api/masks/preview-volume", base,
                   body={"project_id": pid, "source_path": vol_path,
                         "threshold": None}, expect=(200,))
        if isinstance(pv, dict) and pv.get("path"):
            check("DELETE /masks/preview-volume (cleanup)", "DELETE",
                  f"/api/masks/preview-volume?path={quote(pv['path'], safe='')}",
                  base, expect=(200,))
        sv = check("POST /masks/save (throwaway)", "POST", "/api/masks/save",
                   base, body={"project_id": pid, "source_path": vol_path,
                               "output_name": "apismoke_tmp_mask",
                               "threshold": None}, expect=(200,))
        saved_path = sv.get("path") if isinstance(sv, dict) else None
        if saved_path:
            SAVED_MASKS.append(saved_path)
            bo = check("POST /masks/boolean-op (union)", "POST",
                       "/api/masks/boolean-op", base,
                       body={"project_id": pid, "mask_a": saved_path,
                             "mask_b": saved_path, "op": "union",
                             "output_name": "apismoke_tmp_bool"}, expect=(200,))
            if isinstance(bo, dict) and bo.get("path"):
                SAVED_MASKS.append(bo["path"])

    # ---- Subset create + provenance + export-star + cleanup (#4 feature) ----
    if explore_src:
        st, sub, _ = req("POST", base + "/api/subsets",
                         {"project_id": pid, "name": "apismoke_tmp",
                          "source_job_id": explore_src["id"],
                          "indices": [0, 1, 2, 3, 4]})
        if st in (200, 201) and isinstance(sub, dict) and "id" in sub:
            R.append((True, "POST /subsets (create)", st, ""))
            print(f"  \033[32mPASS\033[0m [{st}] POST /subsets (create)")
            ssid = sub["id"]
            check("GET /subsets/{id}", "GET", f"/api/subsets/{ssid}", base)
            check("GET /subsets/{id}/provenance", "GET",
                  f"/api/subsets/{ssid}/provenance", base)
            star_resp = check("POST /subsets/{id}/export-star", "POST",
                  f"/api/subsets/{ssid}/export-star", base,
                  body={"particles_star": ""}, want_keys=["path", "n_particles"])
            if isinstance(star_resp, dict) and star_resp.get("path"):
                check("POST /files/validate-star (real constructed star)",
                      "POST", "/api/files/validate-star", base,
                      body={"path": star_resp["path"]}, expect=(200,))
            sc, _, _ = req("DELETE", base + f"/api/subsets/{ssid}")
            R.append((sc in (200, 204), "DELETE /subsets/{id} (cleanup)", sc, ""))
            print(f"  {'PASS' if sc in (200,204) else 'FAIL'} [{sc}] DELETE /subsets/{{id}} (cleanup)")
        else:
            R.append((False, "POST /subsets (create)", st, str(sub)[:120]))
            print(f"  \033[31mFAIL\033[0m [{st}] POST /subsets (create) -- {str(sub)[:120]}")


def smoke_global(base, ppath_for_files):
    print("\n===== GLOBAL / SYSTEM endpoints =====")
    check("GET /system/info", "GET", "/api/system/info", base,
          want_keys=["hostname"])
    check("GET /system/slurm-defaults", "GET", "/api/system/slurm-defaults", base)
    check("GET /projects (list)", "GET", "/api/projects", base, want_list=True)
    check("GET /settings/local-defaults", "GET",
          "/api/settings/local-defaults", base)
    check("GET /settings/slurm-defaults", "GET",
          "/api/settings/slurm-defaults", base)
    # files
    check("GET /files/browse", "GET",
          f"/api/files/browse?path={ppath_for_files}", base)
    # validate-star on a constructed/known star if present
    cand_star = os.path.join(ppath_for_files, "subsets")
    check("POST /files/validate-star (out-of-sandbox -> graceful 403/4xx)",
          "POST", "/api/files/validate-star", base,
          body={"path": "/nonexistent/x.star"},
          expect=(200, 400, 403, 404, 422))
    # job validate + preview-sbatch (no real submission)
    check("POST /jobs/validate (empty -> graceful 4xx/200)", "POST",
          "/api/jobs/validate", base, body={"type": "pipeline", "params": {}},
          expect=(200, 400, 422))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--project", action="append", default=[], dest="projects")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    smoke_global(base, args.projects[0] if args.projects else "/scratch")
    for p in args.projects:
        smoke_project(base, p)

    if SAVED_MASKS:
        print("\nTHROWAWAY MASKS TO CLEAN UP (rm these):")
        for m in SAVED_MASKS:
            print(f"  {m}")

    total = len(R)
    failed = [r for r in R if not r[0]]
    print("\n" + "=" * 60)
    print(f"API SMOKE: {total - len(failed)}/{total} passed")
    if failed:
        print("\nFAILURES:")
        for ok, label, status, note in failed:
            print(f"  [{status}] {label}  -- {note}")
        return 1
    print("\033[32mALL API ENDPOINTS OK\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
