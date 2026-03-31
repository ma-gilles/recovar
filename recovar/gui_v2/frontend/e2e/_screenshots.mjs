#!/usr/bin/env node
/**
 * Take 4 screenshots for visual verification of the GUI.
 */
import { chromium } from "playwright";
import fs from "fs";

const PORT = process.env.TEST_PORT ?? "8099";
const BASE = `http://localhost:${PORT}`;
const OUT = "/tmp/gui_screenshots";
const PIPELINE_OUTPUT =
  "/scratch/gpfs/GILLES/mg6942/old_regression_scores_v2/spa/test_dataset/pipeline_output_old";
const PROJECT_DIR = "/scratch/gpfs/GILLES/mg6942/gui_screenshot_project";

fs.mkdirSync(OUT, { recursive: true });

async function apiPost(path, body) {
  const r = await fetch(`${BASE}/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

async function apiGet(path) {
  const r = await fetch(`${BASE}/api${path}`);
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

async function main() {
  // Clean up previous project dir
  fs.rmSync(PROJECT_DIR, { recursive: true, force: true });

  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();

  // ── 1. Dashboard (no project) ─────────────────────────────────────
  console.log("1. Dashboard (no project open)...");
  await page.goto(BASE);
  await page.evaluate(() => localStorage.clear());
  await page.goto(BASE);
  await page.waitForSelector("aside", { timeout: 10000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT}/1_dashboard_no_project.png`, fullPage: true });
  console.log("   saved: 1_dashboard_no_project.png");

  // ── 2. After creating project and scanning ────────────────────────
  console.log("2. Creating project and scanning pipeline_output_old...");
  const project = await apiPost("/projects", {
    path: PROJECT_DIR,
    name: "Demo Project",
  });
  const scan = await apiPost(`/projects/${project.id}/scan`, {
    scan_path: PIPELINE_OUTPUT,
  });

  let pipelineJob = scan.imported.find((j) => j.type === "Pipeline");
  if (!pipelineJob) {
    const detail = await apiGet(`/projects/${project.id}`);
    pipelineJob = detail.jobs.find((j) => j.type === "Pipeline");
  }
  const jobId = pipelineJob.id;
  console.log(`   Project: ${project.id}, Pipeline job: ${jobId}`);

  // Set project in localStorage and reload dashboard
  await page.evaluate(
    (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
    { id: project.id, path: project.path, name: project.name }
  );
  await page.goto(BASE);
  await page.waitForSelector("aside", { timeout: 10000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT}/2_dashboard_with_project.png`, fullPage: true });
  console.log("   saved: 2_dashboard_with_project.png");

  // ── 3. Job detail page ────────────────────────────────────────────
  console.log("3. Job detail page for imported pipeline job...");
  await page.goto(`${BASE}/jobs/${jobId}`);
  await page.waitForSelector("aside", { timeout: 10000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT}/3_job_detail_overview.png`, fullPage: true });
  console.log("   saved: 3_job_detail_overview.png");

  // Also capture the Volumes tab
  const volumesTab = await page.$("button:has-text('Volumes')");
  if (volumesTab) {
    await volumesTab.click();
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${OUT}/3b_job_detail_volumes.png`, fullPage: true });
    console.log("   saved: 3b_job_detail_volumes.png");
  }

  // ── 4. Volume viewer with a volume loaded ─────────────────────────
  console.log("4. Volume viewer...");

  // Get a mean volume path
  const volumes = await apiGet(`/jobs/${jobId}/volumes`);
  const meanVol = volumes.find((v) => v.category === "mean");
  if (!meanVol) {
    console.log("   ERROR: No mean volume found!");
  } else {
    // Navigate to explore page and switch to Volumes view
    await page.goto(`${BASE}/explore/${jobId}`);
    await page.waitForSelector("aside", { timeout: 10000 });
    await page.waitForTimeout(2000);

    // Click Volumes toggle
    const volBtn = await page.$("button:has-text('Volumes')");
    if (volBtn) {
      await volBtn.click();
      await page.waitForTimeout(2000);
    }
    await page.screenshot({ path: `${OUT}/4_explore_volumes.png`, fullPage: true });
    console.log("   saved: 4_explore_volumes.png");

    // Also capture a volume slice via API and save as PNG
    const sliceResp = await fetch(
      `${BASE}/api/volumes/slice?path=${encodeURIComponent(meanVol.path)}&axis=2&idx=64`
    );
    if (sliceResp.ok) {
      const sliceBuffer = Buffer.from(await sliceResp.arrayBuffer());
      fs.writeFileSync(`${OUT}/4b_volume_slice_z64.png`, sliceBuffer);
      console.log("   saved: 4b_volume_slice_z64.png (API slice render)");
    }

    // Get volume info for display
    const info = await apiGet(
      `/volumes/info?path=${encodeURIComponent(meanVol.path)}`
    );
    console.log(`   Volume: ${meanVol.name}, shape=${info.shape}, voxel=${info.voxel_size}`);
  }

  // ── Bonus: Plots tab ──────────────────────────────────────────────
  console.log("Bonus: Plots tab...");
  await page.goto(`${BASE}/jobs/${jobId}`);
  await page.waitForSelector("aside", { timeout: 10000 });
  await page.waitForTimeout(2000);
  const plotsTab = await page.$("button:has-text('Plots')");
  if (plotsTab) {
    await plotsTab.click();
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `${OUT}/5_job_detail_plots.png`, fullPage: true });
    console.log("   saved: 5_job_detail_plots.png");
  }

  // ── Bonus: Latent space explorer ──────────────────────────────────
  console.log("Bonus: Latent space explorer...");
  await page.goto(`${BASE}/explore/${jobId}`);
  await page.waitForSelector("aside", { timeout: 10000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT}/6_latent_explorer.png`, fullPage: true });
  console.log("   saved: 6_latent_explorer.png");

  // ── Bonus: Analyze form ───────────────────────────────────────────
  console.log("Bonus: Analyze form...");
  await page.goto(`${BASE}/jobs/new`);
  await page.waitForSelector("aside", { timeout: 10000 });
  await page.waitForTimeout(1000);
  const sel = await page.$("select");
  if (sel) {
    await sel.selectOption("analyze");
    await page.waitForTimeout(1000);
  }
  await page.screenshot({ path: `${OUT}/7_analyze_form.png`, fullPage: true });
  console.log("   saved: 7_analyze_form.png");

  await ctx.close();
  await browser.close();

  // Cleanup
  fs.rmSync(PROJECT_DIR, { recursive: true, force: true });

  console.log(`\nAll screenshots saved to ${OUT}/`);
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
