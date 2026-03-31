/**
 * Shared helpers for Playwright E2E tests.
 *
 * These tests run against the real backend (FastAPI + SQLite) serving
 * both the API and the built frontend.  The backend must be started
 * before the test run with:
 *
 *   TEST_PORT=8099 pixi run python -m recovar.gui_v2.backend.main --port 8099
 */
import { expect } from "@playwright/test";

export const BASE_URL = `http://localhost:${process.env.TEST_PORT ?? "8099"}`;
export const PIPELINE_OUTPUT =
  "/scratch/gpfs/GILLES/mg6942/old_regression_scores_v2/spa/test_dataset/pipeline_output_old";
export const TEST_PROJECT_DIR =
  "/scratch/gpfs/GILLES/mg6942/gui_e2e_test_pw";

// ── API helpers (bypass the UI for setup/teardown) ──────────────────────

/** POST JSON to the backend API. */
export async function apiPost(path, body) {
  const resp = await fetch(`${BASE_URL}/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  if (resp.status === 204) return undefined;
  return resp.json();
}

/** GET JSON from the backend API. */
export async function apiGet(path) {
  const resp = await fetch(`${BASE_URL}/api${path}`);
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

/** Create a project via API and scan pipeline_output_old into it.
 *
 *  Idempotent: if the pipeline output was already imported in a previous
 *  test run (same project path), we look up the existing Pipeline job
 *  from the project's job list instead of failing.
 */
export async function setupProjectWithPipelineJob() {
  const project = await apiPost(
    "/projects",
    { path: TEST_PROJECT_DIR, name: "E2E Test Project" }
  );

  const scan = await apiPost(
    `/projects/${project.id}/scan`, { scan_path: PIPELINE_OUTPUT }
  );

  // Try newly imported jobs first
  let pipelineJob = scan.imported.find((j) => j.type === "Pipeline");

  if (!pipelineJob) {
    // Scan imported nothing new — the job may already exist from a
    // previous run.  Look it up from the project's job list.
    const projectData = await apiGet(`/projects/${project.id}`);
    const jobs = projectData.jobs || [];
    pipelineJob = jobs.find((j) => j.type === "Pipeline");
  }

  if (!pipelineJob) throw new Error("No pipeline job found after scan");

  return {
    projectId: project.id,
    projectPath: project.path,
    jobId: pipelineJob.id,
  };
}

/** Set the active project in localStorage so the UI picks it up. */
export async function setProjectInStorage(page, project) {
  await page.evaluate((p) => {
    localStorage.setItem("recovar_active_project", JSON.stringify(p));
  }, project);
}

/** Wait for the app to be fully loaded (sidebar rendered). */
export async function waitForApp(page) {
  // The sidebar always renders the "recovar" link
  await page.locator("aside").waitFor({ state: "visible", timeout: 10_000 });
}
