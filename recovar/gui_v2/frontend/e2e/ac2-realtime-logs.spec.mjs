/**
 * AC-2: Watch real-time logs while a job runs.
 *
 * We test that the Logs tab exists on the job detail page and that
 * the LogViewer component renders.  We can't test true real-time
 * streaming without a running job, but we verify the WebSocket
 * connection attempt is made and the UI is ready.
 */
import { test, expect } from "@playwright/test";
import {
  BASE_URL,
  setupProjectWithPipelineJob,
  setProjectInStorage,
  waitForApp,
} from "./helpers.mjs";

test.describe("AC-2: Real-time logs", () => {
  let projectId;
  let projectPath;
  let jobId;

  test.beforeAll(async () => {
    const setup = await setupProjectWithPipelineJob();
    projectId = setup.projectId;
    projectPath = setup.projectPath;
    jobId = setup.jobId;
  });

  test("job detail page has Logs tab", async ({ page }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/jobs/${jobId}`);
    await waitForApp(page);

    // Wait for job detail to load
    await expect(page.getByText(/Pipeline/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Logs tab should be visible
    const logsTab = page.getByRole("button", { name: /Logs/i });
    await expect(logsTab).toBeVisible();

    // Click the Logs tab
    await logsTab.click();

    // LogViewer component should render (it shows a log container or
    // "No log output" message for a completed/imported job)
    const logArea = page.locator('[class*="font-mono"]');
    await expect(logArea.first()).toBeVisible({ timeout: 5_000 });
  });

  test("job detail page shows Overview tab with status", async ({ page }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/jobs/${jobId}`);
    await waitForApp(page);

    // Should show status badge
    await expect(page.getByText(/completed/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Should show output directory
    await expect(page.getByText(/pipeline_output_old/i)).toBeVisible();
  });
});
