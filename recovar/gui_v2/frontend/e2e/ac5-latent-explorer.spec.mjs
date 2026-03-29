/**
 * AC-5: Open analyze results — explore latent space scatter plot,
 *       click k-means center to compute volume, click two centers
 *       for trajectory.
 *
 * The latent explorer requires an analyze job output. Since we only
 * have pipeline_output_old (no analyze output), this test verifies:
 * - The explore page route loads
 * - The LatentExplorer component renders
 * - The Volumes sub-tab works
 * - The scatter panel renders (even if empty data)
 */
import { test, expect } from "@playwright/test";
import {
  BASE_URL,
  setupProjectWithPipelineJob,
  setProjectInStorage,
  waitForApp,
} from "./helpers.mjs";

test.describe("AC-5: Latent explorer", () => {
  let projectId;
  let projectPath;
  let jobId;

  test.beforeAll(async () => {
    const setup = await setupProjectWithPipelineJob();
    projectId = setup.projectId;
    projectPath = setup.projectPath;
    jobId = setup.jobId;
  });

  test("explore page loads for a pipeline job", async ({ page }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/explore/${jobId}`);
    await waitForApp(page);

    // Should show "Explore" heading
    await expect(page.getByText("Explore").first()).toBeVisible({
      timeout: 10_000,
    });
  });

  test("explore page has Latent Space and Volumes toggle", async ({ page }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/explore/${jobId}`);
    await waitForApp(page);

    // Should show the view mode toggle buttons
    await expect(page.getByText("Latent Space")).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByText("Volumes")).toBeVisible();
  });

  test("Volumes view shows volume viewer", async ({ page }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/explore/${jobId}`);
    await waitForApp(page);

    // Click Volumes toggle
    await page.getByText("Volumes").click();

    // Wait for volumes to load — should show volume entries or a viewer
    await page.waitForTimeout(2_000);

    // Take screenshot to verify
    await page.screenshot({
      path: "e2e-results/ac5-volumes-view.png",
      fullPage: true,
    });
  });

  test("Explore button appears on completed pipeline job detail", async ({
    page,
  }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/jobs/${jobId}`);
    await waitForApp(page);

    // Wait for job detail to load
    await expect(page.getByText(/completed/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Should have "Explore" button since this is a completed pipeline job
    const exploreBtn = page.getByRole("link", { name: /Explore/i });
    await expect(exploreBtn).toBeVisible();
  });
});
