#!/usr/bin/env node
/**
 * Run E2E tests using Playwright directly (bypasses the CLI test runner
 * which hangs on Node 25 + HPC).
 *
 * Usage: node e2e/run-tests.mjs [filter]
 *
 * Env: TEST_PORT (default 8099)
 */
import { chromium } from "playwright";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.TEST_PORT ?? "8099";
const BASE_URL = `http://localhost:${PORT}`;
const PIPELINE_OUTPUT =
  "/scratch/gpfs/GILLES/mg6942/old_regression_scores_v2/spa/test_dataset/pipeline_output_old";
const TEST_PROJECT_DIR = "/scratch/gpfs/GILLES/mg6942/gui_e2e_test_pw";
const SCREENSHOT_DIR = path.join(__dirname, "..", "e2e-results");

fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

// ── Result tracking ────────────────────────────────────────────────────
let passed = 0;
let failed = 0;
let skipped = 0;
const failures = [];
const filter = process.argv[2] || "";

async function apiPost(apiPath, body) {
  const resp = await fetch(`${BASE_URL}/api${apiPath}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  if (resp.status === 204) return undefined;
  return resp.json();
}

async function apiGet(apiPath) {
  const resp = await fetch(`${BASE_URL}/api${apiPath}`);
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function setupProject() {
  const project = await apiPost("/projects", {
    path: TEST_PROJECT_DIR,
    name: "E2E Test Project",
  });

  // Try scanning — may find no new jobs if project was already set up
  const scan = await apiPost(`/projects/${project.id}/scan`, {
    scan_path: PIPELINE_OUTPUT,
  });
  let pipelineJob = scan.imported.find((j) => j.type === "Pipeline");

  if (!pipelineJob) {
    // Jobs already imported — get them from the project detail
    const detail = await apiGet(`/projects/${project.id}`);
    const existing = detail.jobs.find((j) => j.type === "Pipeline");
    if (existing) pipelineJob = existing;
  }

  return { project, jobId: pipelineJob?.id };
}

async function runTest(name, fn) {
  if (filter && !name.toLowerCase().includes(filter.toLowerCase())) {
    skipped++;
    return;
  }
  process.stdout.write(`  ${name} ... `);
  try {
    await fn();
    passed++;
    console.log("\x1b[32mPASSED\x1b[0m");
  } catch (e) {
    failed++;
    console.log(`\x1b[31mFAILED\x1b[0m: ${e.message}`);
    failures.push({ name, error: e.message });
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || "Assertion failed");
}

function assertGt(a, b, msg) {
  if (!(a > b)) throw new Error(msg || `Expected ${a} > ${b}`);
}

function assertIncludes(str, sub, msg) {
  if (!str.includes(sub)) throw new Error(msg || `Expected "${str}" to include "${sub}"`);
}

// ── Main ───────────────────────────────────────────────────────────────

async function main() {
  console.log(`\nrecovar GUI v2 — E2E Tests`);
  console.log(`Backend: ${BASE_URL}`);
  console.log(`Pipeline output: ${PIPELINE_OUTPUT}`);
  console.log(`Test project dir: ${TEST_PROJECT_DIR}\n`);

  // Clean up any previous test project
  fs.rmSync(TEST_PROJECT_DIR, { recursive: true, force: true });

  // Launch browser
  const browser = await chromium.launch({ headless: true });

  try {
    // ── Setup: create project & scan pipeline output ────────────────
    console.log("Setting up project...");
    const { project, jobId } = await setupProject();
    assert(project.id, "Project creation failed");
    assert(jobId, "No pipeline job imported from scan");
    console.log(`  Project: ${project.id} (${project.path})`);
    console.log(`  Pipeline job: ${jobId}\n`);

    // ── AC-1: Create project & configure pipeline ───────────────────
    console.log("\x1b[1mAC-1: Create project & configure pipeline\x1b[0m");
    {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
      const page = await ctx.newPage();

      await runTest("dashboard shows Create/Open when no project", async () => {
        await page.goto(BASE_URL);
        await page.evaluate(() => localStorage.clear());
        await page.goto(BASE_URL);
        await page.waitForSelector("aside", { timeout: 10000 });

        // Check for Create Project button in main panel
        const createBtn = await page.$("text=Create Project");
        assert(createBtn, "Create Project button not found");

        const openBtn = await page.$("text=Open Project");
        assert(openBtn, "Open Project button not found");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac1-no-project.png"),
        });
      });

      await runTest("sidebar shows Create/Open when no project", async () => {
        const sidebar = page.locator("aside");
        const createText = await sidebar.locator("text=Create Project").count();
        assertGt(createText, 0, "Sidebar should show Create Project");

        const openText = await sidebar.locator("text=Open Project").count();
        assertGt(openText, 0, "Sidebar should show Open Project");
      });

      await runTest("New Job page shows pipeline form with project", async () => {
        // Navigate to root first to set localStorage in the app's origin
        await page.goto(BASE_URL);
        await page.evaluate(
          (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
          { id: project.id, path: project.path, name: project.name }
        );
        // Now navigate to jobs/new — the app reads localStorage on mount
        await page.goto(`${BASE_URL}/jobs/new`);
        await page.waitForSelector("aside", { timeout: 10000 });

        // Wait for React to hydrate and render the form
        await page.waitForTimeout(2000);

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac1-pipeline-form.png"),
        });

        const heading = await page.$("text=New Job");
        assert(heading, "New Job heading not found");

        // The Pipeline form has a "Particles" label — look for it
        const bodyHtml = await page.evaluate(() => document.body.innerText);
        assert(
          bodyHtml.includes("Particles") || bodyHtml.includes("Pipeline"),
          `Pipeline form not found. Page text: ${bodyHtml.slice(0, 300)}`
        );
      });

      await runTest("New Job redirects when no project", async () => {
        await page.evaluate(() => localStorage.clear());
        await page.goto(`${BASE_URL}/jobs/new`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(1000);

        const msg = await page.$("text=create or open a project");
        assert(msg, "Should show 'create or open a project' message");
      });

      await ctx.close();
    }

    // ── AC-2: Real-time logs ────────────────────────────────────────
    console.log("\n\x1b[1mAC-2: Real-time logs\x1b[0m");
    {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
      const page = await ctx.newPage();

      await runTest("job detail has Logs tab", async () => {
        await page.goto(BASE_URL);
        await page.evaluate(
          (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
          { id: project.id, path: project.path, name: project.name }
        );
        await page.goto(`${BASE_URL}/jobs/${jobId}`);
        await page.waitForSelector("aside", { timeout: 10000 });

        // Wait for job detail to load
        await page.waitForTimeout(2000);

        const logsBtn = await page.$("button:has-text('Logs')");
        assert(logsBtn, "Logs tab button not found");

        // Click it
        await logsBtn.click();
        await page.waitForTimeout(1000);

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac2-logs-tab.png"),
        });
      });

      await runTest("job detail shows status badge", async () => {
        await page.goto(`${BASE_URL}/jobs/${jobId}`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        const statusBadge = await page.$("text=completed");
        assert(statusBadge, "Status badge showing 'completed' not found");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac2-job-detail.png"),
        });
      });

      await ctx.close();
    }

    // ── AC-3: View volumes and plots ────────────────────────────────
    console.log("\n\x1b[1mAC-3: View volumes and plots\x1b[0m");
    {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
      const page = await ctx.newPage();

      await runTest("Volumes tab shows MRC files by category", async () => {
        await page.goto(BASE_URL);
        await page.evaluate(
          (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
          { id: project.id, path: project.path, name: project.name }
        );
        await page.goto(`${BASE_URL}/jobs/${jobId}`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        const volumesBtn = await page.$("button:has-text('Volumes')");
        assert(volumesBtn, "Volumes tab not found");
        await volumesBtn.click();
        await page.waitForTimeout(2000);

        // Should show category headers
        const meanHeader = await page.$("text=mean");
        assert(meanHeader, "Mean category not found");

        const eigenHeader = await page.$("text=eigen");
        assert(eigenHeader, "Eigen category not found");

        // Should show .mrc files
        const mrcFiles = await page.$$("text=.mrc");
        assertGt(mrcFiles.length, 0, "No .mrc files shown");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac3-volumes.png"),
        });
      });

      await runTest("volumes API returns correct categories", async () => {
        const volumes = await apiGet(`/jobs/${jobId}/volumes`);
        assertGt(volumes.length, 0, "No volumes returned");

        const meanVols = volumes.filter((v) => v.category === "mean");
        assertGt(meanVols.length, 0, "No mean volumes");

        const eigenVols = volumes.filter((v) => v.category === "eigen");
        assertGt(eigenVols.length, 0, "No eigen volumes");
      });

      await runTest("volume slice API returns PNG", async () => {
        const volumes = await apiGet(`/jobs/${jobId}/volumes`);
        const meanVol = volumes.find((v) => v.category === "mean");
        assert(meanVol, "No mean volume found");

        const resp = await fetch(
          `${BASE_URL}/api/volumes/slice?path=${encodeURIComponent(meanVol.path)}&axis=0&idx=64`
        );
        assert(resp.status === 200, `Slice returned ${resp.status}`);
        assert(
          resp.headers.get("content-type") === "image/png",
          `Expected image/png, got ${resp.headers.get("content-type")}`
        );

        const body = await resp.arrayBuffer();
        assertGt(body.byteLength, 100, "PNG too small");
      });

      await runTest("volume info API returns shape", async () => {
        const volumes = await apiGet(`/jobs/${jobId}/volumes`);
        const meanVol = volumes.find((v) => v.category === "mean");

        const info = await apiGet(
          `/volumes/info?path=${encodeURIComponent(meanVol.path)}`
        );
        assert(info.shape.length === 3, `Expected 3D shape, got ${info.shape}`);
        assertGt(info.shape[0], 0, "Shape[0] should be > 0");
        assertGt(info.voxel_size, 0, "Voxel size should be > 0");
      });

      await runTest("Plots tab shows diagnostic images", async () => {
        await page.goto(`${BASE_URL}/jobs/${jobId}`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        const plotsBtn = await page.$("button:has-text('Plots')");
        assert(plotsBtn, "Plots tab not found");
        await plotsBtn.click();
        await page.waitForTimeout(3000);

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac3-plots.png"),
        });
      });

      await runTest("plots API returns file list", async () => {
        const plots = await apiGet(`/jobs/${jobId}/plots`);
        assertGt(plots.length, 0, "No plots returned");
        for (const p of plots) {
          assert(
            p.name.endsWith(".png") || p.name.endsWith(".pdf"),
            `Unexpected plot extension: ${p.name}`
          );
        }
      });

      await ctx.close();
    }

    // ── AC-4: Submit analyze (suggested next) ───────────────────────
    console.log("\n\x1b[1mAC-4: Submit analyze job (suggested next)\x1b[0m");
    {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
      const page = await ctx.newPage();

      await runTest("suggested-next API returns Analyze", async () => {
        const suggestions = await apiGet(`/jobs/${jobId}/suggested-next`);
        assertGt(suggestions.length, 0, "No suggestions");

        const analyze = suggestions.find((s) => s.type === "Analyze");
        assert(analyze, "No Analyze suggestion");
        assertIncludes(analyze.label, "Analyze", "Label should mention Analyze");
        assert(analyze.prefilled_params.result_dir, "Should have result_dir");
      });

      await runTest("job detail shows Suggested Next Steps", async () => {
        await page.goto(BASE_URL);
        await page.evaluate(
          (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
          { id: project.id, path: project.path, name: project.name }
        );
        await page.goto(`${BASE_URL}/jobs/${jobId}`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        const nextSteps = await page.$("text=Suggested Next");
        assert(nextSteps, "Suggested Next Steps section not found");

        const analyzeLink = await page.$("a:has-text('Analyze')");
        assert(analyzeLink, "Analyze link not found in suggestions");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac4-suggested-next.png"),
        });
      });

      await runTest("Analyze form accessible from New Job", async () => {
        // Ensure project is in localStorage
        await page.goto(BASE_URL);
        await page.evaluate(
          (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
          { id: project.id, path: project.path, name: project.name }
        );
        await page.goto(`${BASE_URL}/jobs/new`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac4-new-job-before-select.png"),
        });

        // Find job type select and change to Analyze
        const jobTypeSelect = await page.$("select");
        if (!jobTypeSelect) {
          // Debug: show page content
          const text = await page.evaluate(() => document.body.innerText);
          assert(false, `Job type select not found. Page: ${text.slice(0, 300)}`);
        }
        await jobTypeSelect.selectOption("analyze");
        await page.waitForTimeout(1000);

        // Should show result directory field
        const bodyText = await page.evaluate(() => document.body.innerText);
        assert(
          bodyText.includes("Result") || bodyText.includes("Analyze"),
          `Analyze form not shown. Page: ${bodyText.slice(0, 300)}`
        );

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac4-analyze-form.png"),
        });
      });

      await runTest("Parameters tab has CLI command toggle", async () => {
        await page.goto(`${BASE_URL}/jobs/${jobId}`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        const paramsBtn = await page.$("button:has-text('Parameters')");
        assert(paramsBtn, "Parameters tab not found");
        await paramsBtn.click();
        await page.waitForTimeout(1000);

        const cliBtn = await page.$("button:has-text('CLI Command')");
        assert(cliBtn, "Show CLI Command button not found");

        const cloneBtn = await page.$("button:has-text('Clone')");
        assert(cloneBtn, "Clone Job button not found");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac4-params-tab.png"),
        });
      });

      await ctx.close();
    }

    // ── AC-5: Latent explorer ───────────────────────────────────────
    console.log("\n\x1b[1mAC-5: Latent explorer\x1b[0m");
    {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
      const page = await ctx.newPage();

      await runTest("explore page loads for pipeline job", async () => {
        await page.goto(BASE_URL);
        await page.evaluate(
          (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
          { id: project.id, path: project.path, name: project.name }
        );
        await page.goto(`${BASE_URL}/explore/${jobId}`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        const exploreHeading = await page.$("text=Explore");
        assert(exploreHeading, "Explore heading not found");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac5-explore.png"),
        });
      });

      await runTest("explore has Latent Space / Volumes toggle", async () => {
        const latentBtn = await page.$("text=Latent Space");
        assert(latentBtn, "Latent Space button not found");

        const volumesBtn = await page.$("text=Volumes");
        assert(volumesBtn, "Volumes button not found");
      });

      await runTest("Explore button on completed pipeline detail", async () => {
        await page.goto(`${BASE_URL}/jobs/${jobId}`);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        const exploreLink = await page.$("a:has-text('Explore')");
        assert(exploreLink, "Explore button/link not found on job detail");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac5-explore-button.png"),
        });
      });

      await ctx.close();
    }

    // ── AC-6: Lasso export / subset API ─────────────────────────────
    console.log("\n\x1b[1mAC-6: Lasso export / subset API\x1b[0m");
    {
      await runTest("create subset via API", async () => {
        const subset = await apiPost("/subsets", {
          project_id: project.id,
          name: "test_lasso_selection",
          source_job_id: null,
          method: { type: "lasso", description: "E2E test" },
          indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        });
        assert(subset.id, "Subset creation returned no id");
        assert(subset.name === "test_lasso_selection", "Wrong subset name");
        assert(subset.n_particles === 10, `Expected 10 particles, got ${subset.n_particles}`);
        assertIncludes(subset.path, ".ind", "Path should contain .ind");
      });

      await runTest("list subsets for project", async () => {
        const subsets = await apiGet(`/subsets?project_id=${project.id}`);
        assertGt(subsets.length, 0, "No subsets returned");
        const testSubset = subsets.find((s) => s.name === "test_lasso_selection");
        assert(testSubset, "Test subset not found in list");
        assert(testSubset.n_particles === 10, "Wrong particle count");
      });

      await runTest("delete subset", async () => {
        const subset = await apiPost("/subsets", {
          project_id: project.id,
          name: "to_delete",
          indices: [0, 1, 2],
        });
        const resp = await fetch(`${BASE_URL}/api/subsets/${subset.id}`, {
          method: "DELETE",
        });
        assert(resp.status === 204, `Delete returned ${resp.status}`);

        const subsets = await apiGet(`/subsets?project_id=${project.id}`);
        const found = subsets.find((s) => s.id === subset.id);
        assert(!found, "Deleted subset should not appear in list");
      });
    }

    // ── AC-7: Local/cluster mode ────────────────────────────────────
    console.log("\n\x1b[1mAC-7: Local/cluster mode transparency\x1b[0m");
    {
      const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
      const page = await ctx.newPage();

      await runTest("system info API returns expected fields", async () => {
        const info = await apiGet("/system/info");
        assert(info.hostname, "No hostname");
        assert(
          ["slurm", "local"].includes(info.executor_mode),
          `Invalid executor_mode: ${info.executor_mode}`
        );
        assert(info.recovar_version, "No recovar_version");
        assert(typeof info.gpu_count === "number", "gpu_count should be number");
      });

      await runTest("dashboard shows system info bar", async () => {
        await page.goto(BASE_URL);
        await page.evaluate(
          (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
          { id: project.id, path: project.path, name: project.name }
        );
        await page.goto(BASE_URL);
        await page.waitForSelector("aside", { timeout: 10000 });
        await page.waitForTimeout(2000);

        // Check for hostname display
        const hostname = await page.$("text=della");
        assert(hostname, "Hostname 'della' not shown in system info bar");

        // Check for mode display
        const mode = await page.$("text=mode");
        assert(mode, "Execution mode not shown");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac7-dashboard-full.png"),
        });
      });

      await runTest("file browser API works", async () => {
        const entries = await apiGet(
          `/files/browse?path=${encodeURIComponent("/scratch/gpfs/GILLES/mg6942")}`
        );
        assertGt(entries.length, 0, "No entries returned");
        assert(entries.some((e) => e.is_dir), "No directories in result");
      });

      await runTest("sidebar visible with project info", async () => {
        const sidebar = await page.$("aside");
        assert(sidebar, "Sidebar not found");

        // Should show project name or recovar branding
        const recovarLink = await page.$("text=recovar");
        assert(recovarLink, "recovar link not in sidebar");

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, "ac7-sidebar.png"),
        });
      });

      await ctx.close();
    }
  } finally {
    await browser.close();
  }

  // ── Summary ──────────────────────────────────────────────────────────
  console.log("\n" + "═".repeat(60));
  console.log(
    `\x1b[1mResults:\x1b[0m ${passed} passed, ${failed} failed, ${skipped} skipped`
  );

  if (failures.length > 0) {
    console.log("\n\x1b[31mFailures:\x1b[0m");
    for (const f of failures) {
      console.log(`  ✗ ${f.name}: ${f.error}`);
    }
  }

  console.log(`\nScreenshots saved to: ${SCREENSHOT_DIR}`);
  console.log("═".repeat(60));

  // Cleanup
  fs.rmSync(TEST_PROJECT_DIR, { recursive: true, force: true });

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(2);
});
