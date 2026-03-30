/**
 * AC-4: Submit an analyze job on the pipeline output
 *       (suggested as next step).
 *
 * Tests that the "suggested next steps" mechanism works and that
 * the Analyze form is accessible.
 */
import { test, expect } from "@playwright/test";
import {
  BASE_URL,
  setupProjectWithPipelineJob,
  setProjectInStorage,
  waitForApp,
  apiGet,
} from "./helpers.mjs";

test.describe("AC-4: Submit analyze job (suggested next)", () => {
  let projectId;
  let projectPath;
  let jobId;

  test.beforeAll(async () => {
    const setup = await setupProjectWithPipelineJob();
    projectId = setup.projectId;
    projectPath = setup.projectPath;
    jobId = setup.jobId;
  });

  test("suggested-next API returns Analyze suggestion for completed pipeline", async () => {
    const suggestions = await apiGet(`/jobs/${jobId}/suggested-next`);

    expect(suggestions.length).toBeGreaterThan(0);

    const analyzeSuggestion = suggestions.find((s) => s.type === "Analyze");
    expect(analyzeSuggestion).toBeTruthy();
    expect(analyzeSuggestion.label).toContain("Analyze");
    expect(analyzeSuggestion.prefilled_params.result_dir).toBeTruthy();
  });

  test("job detail Overview tab shows Suggested Next Steps", async ({
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

    // Wait for the job to load
    await expect(page.getByText(/completed/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Should show "Suggested Next Steps" section
    await expect(page.getByText(/Suggested Next/i)).toBeVisible({
      timeout: 10_000,
    });

    // Should have an "Analyze" link
    await expect(
      page.getByRole("link", { name: /Analyze/i }).first()
    ).toBeVisible();
  });

  test("New Job page Analyze form has required fields", async ({ page }) => {
    await page.goto(BASE_URL);
    await setProjectInStorage(page, {
      id: projectId,
      path: projectPath,
      name: "E2E Test Project",
    });
    await page.goto(`${BASE_URL}/jobs/new`);
    await waitForApp(page);

    // Select Analyze from the job type dropdown
    await page.locator("select").first().selectOption("analyze");

    // Should see Analyze form fields
    await expect(page.getByText(/Result Directory/i).first()).toBeVisible({
      timeout: 5_000,
    });
  });

  test("Parameters tab shows job params and CLI command toggle", async ({
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

    // Wait for job to load
    await expect(page.getByText(/Pipeline/i).first()).toBeVisible({
      timeout: 10_000,
    });

    // Click the Parameters tab
    await page.getByRole("button", { name: /Parameters/i }).click();

    // Should show "Show CLI Command" toggle
    await expect(
      page.getByRole("button", { name: /CLI Command/i })
    ).toBeVisible({ timeout: 5_000 });

    // Should show "Clone Job" button
    await expect(page.getByRole("button", { name: /Clone/i })).toBeVisible();
  });
});
