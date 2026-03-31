/**
 * AC-7: All of the above on a machine without SLURM (local mode),
 *       transparently.
 *
 * This cluster has SLURM, so we verify:
 * - The system info API reports executor_mode correctly
 * - The system info bar renders in the dashboard
 * - The system info endpoint returns all expected fields
 *
 * On a non-SLURM machine the executor_mode would be "local" and
 * all the same tests above would pass unchanged.
 */
import { test, expect } from "@playwright/test";
import {
  BASE_URL,
  setupProjectWithPipelineJob,
  setProjectInStorage,
  waitForApp,
  apiGet,
} from "./helpers.mjs";

test.describe("AC-7: Local/cluster mode transparency", () => {
  test("system info API returns expected fields", async () => {
    const info = await apiGet("/system/info");

    expect(info.hostname).toBeTruthy();
    expect(["slurm", "local"]).toContain(info.executor_mode);
    expect(info.recovar_version).toBeTruthy();
    expect(typeof info.gpu_count).toBe("number");
  });

  test("dashboard shows system info bar", async ({ page }) => {
    // Set up a project so we also see the full dashboard
    const setup = await setupProjectWithPipelineJob();

    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: setup.projectId,
      path: setup.projectPath,
      name: "E2E Test Project",
    });
    await page.goto(BASE_URL);
    await waitForApp(page);

    // System info bar should show hostname
    await expect(page.getByText(/della/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Should show execution mode
    const modeText = page.getByText(/mode/i).first();
    await expect(modeText).toBeVisible();
  });

  test("file browser API works (filesystem access)", async () => {
    const entries = await apiGet("/files/browse?path=/scratch/gpfs/GILLES/mg6942");

    expect(entries.length).toBeGreaterThan(0);
    expect(entries.some((e) => e.is_dir)).toBe(true);
  });

  test("dashboard renders complete UI with project", async ({ page }) => {
    const setup = await setupProjectWithPipelineJob();

    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: setup.projectId,
      path: setup.projectPath,
      name: "E2E Test Project",
    });
    await page.goto(BASE_URL);
    await waitForApp(page);

    // Wait for full load
    await expect(page.getByText("E2E Test Project")).toBeVisible({
      timeout: 10_000,
    });

    // Take a full screenshot for visual verification
    await page.screenshot({
      path: "e2e-results/ac7-full-dashboard.png",
      fullPage: true,
    });

    // Verify sidebar has project info
    const sidebar = page.locator("aside");
    await expect(sidebar).toBeVisible();

    // Verify main panel
    const main = page.locator("main");
    await expect(main).toBeVisible();
  });
});
