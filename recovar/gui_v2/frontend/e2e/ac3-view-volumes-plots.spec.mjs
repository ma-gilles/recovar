/**
 * AC-3: Open the completed pipeline output — view the mean volume,
 *       browse eigenvolumes, view diagnostic plots.
 *
 * Uses the scanned pipeline_output_old which has ~60 MRC files and ~5 PNGs.
 */
import { test, expect } from "@playwright/test";
import {
  BASE_URL,
  setupProjectWithPipelineJob,
  setProjectInStorage,
  waitForApp,
  apiGet,
} from "./helpers.mjs";

test.describe("AC-3: View volumes and plots", () => {
  let projectId;
  let projectPath;
  let jobId;

  test.beforeAll(async () => {
    const setup = await setupProjectWithPipelineJob();
    projectId = setup.projectId;
    projectPath = setup.projectPath;
    jobId = setup.jobId;
  });

  test("Volumes tab shows MRC files grouped by category", async ({ page }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/jobs/${jobId}`);
    await waitForApp(page);

    // Wait for job to load
    await expect(page.getByText(/Pipeline/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Click the Volumes tab
    await page.getByRole("button", { name: /Volumes/i }).click();

    // Should show volume categories
    await expect(page.getByText(/mean/i).first()).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByText(/eigen/i).first()).toBeVisible();

    // Should show individual MRC files
    await expect(page.getByText(/\.mrc/i).first()).toBeVisible();
  });

  test("volumes API returns correct categories", async () => {
    const volumes = await apiGet(`/jobs/${jobId}/volumes`);

    expect(volumes.length).toBeGreaterThan(0);

    // Should have mean volumes
    const meanVols = volumes.filter((v) => v.category === "mean");
    expect(meanVols.length).toBeGreaterThan(0);

    // Should have eigen volumes
    const eigenVols = volumes.filter((v) => v.category === "eigen");
    expect(eigenVols.length).toBeGreaterThan(0);

    // Each volume should have a valid path
    for (const v of volumes) {
      expect(v.path).toContain(".mrc");
      expect(v.size_bytes).toBeGreaterThan(0);
    }
  });

  test("volume slice API returns a PNG image", async () => {
    // Get the first mean volume
    const volumes = await apiGet(`/jobs/${jobId}/volumes`);
    const meanVol = volumes.find((v) => v.category === "mean");
    expect(meanVol).toBeTruthy();

    // Request a slice
    const resp = await fetch(
      `${BASE_URL}/api/volumes/slice?path=${encodeURIComponent(meanVol.path)}&axis=0&idx=64`
    );
    expect(resp.status).toBe(200);
    expect(resp.headers.get("content-type")).toBe("image/png");

    const body = await resp.arrayBuffer();
    expect(body.byteLength).toBeGreaterThan(100); // Non-trivial PNG
  });

  test("volume info API returns shape and stats", async () => {
    const volumes = await apiGet(
      `/jobs/${jobId}/volumes`
    );
    const meanVol = volumes.find((v) => v.category === "mean");
    expect(meanVol).toBeTruthy();

    const info = await apiGet(`/volumes/info?path=${encodeURIComponent(meanVol.path)}`);

    expect(info.shape).toHaveLength(3);
    expect(info.shape[0]).toBeGreaterThan(0);
    expect(info.voxel_size).toBeGreaterThan(0);
  });

  test("Plots tab shows diagnostic images", async ({ page }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/jobs/${jobId}`);
    await waitForApp(page);

    // Wait for job to load
    await expect(page.getByText(/Pipeline/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Click the Plots tab
    await page.getByRole("button", { name: /Plots/i }).click();

    // Should show plot images (PNG thumbnails)
    // The plots tab uses img elements to display them
    const plotImages = page.locator("img");
    await expect(plotImages.first()).toBeVisible({ timeout: 10_000 });
  });

  test("plots API returns PNG file list", async () => {
    const plots = await apiGet(
      `/jobs/${jobId}/plots`
    );

    expect(plots.length).toBeGreaterThan(0);
    for (const p of plots) {
      expect(p.name).toMatch(/\.(png|pdf)$/);
    }
  });
});
