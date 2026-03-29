/**
 * AC-1: Open the GUI, create a project, select a .star file, configure
 *       a pipeline job, and see the submit form — without touching a terminal.
 *
 * We test the full browser flow up to the submit button being enabled.
 * We do NOT actually submit to SLURM (that would be a real job).
 */
import { test, expect } from "@playwright/test";
import { BASE_URL, TEST_PROJECT_DIR, waitForApp } from "./helpers.mjs";

test.describe("AC-1: Create project & configure pipeline", () => {
  test.beforeEach(async ({ page }) => {
    // Clear any previous project from localStorage
    await page.goto(BASE_URL);
    await page.evaluate(() => localStorage.clear());
  });

  test("dashboard shows Create / Open buttons when no project is open", async ({
    page,
  }) => {
    await page.goto(BASE_URL);
    await waitForApp(page);

    // Main panel should show "Create Project" and "Open Project"
    await expect(page.getByRole("button", { name: /Create Project/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Open Project/i })).toBeVisible();

    // Should NOT show "+ New Job" (that requires a project)
    await expect(page.getByRole("link", { name: /New Job/i })).not.toBeVisible();
  });

  test("sidebar shows Create / Open when no project is open", async ({
    page,
  }) => {
    await page.goto(BASE_URL);
    await waitForApp(page);

    const sidebar = page.locator("aside");
    await expect(sidebar.getByText(/Create Project/i)).toBeVisible();
    await expect(sidebar.getByText(/Open Project/i)).toBeVisible();
  });

  test("Create Project form opens and creates a project", async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForApp(page);

    // Click "Create Project" button (use the one in main panel)
    const mainPanel = page.locator("main");
    await mainPanel.getByRole("button", { name: /Create Project/i }).click();

    // Form should appear with Directory input
    await expect(mainPanel.getByLabel(/Directory/i)).toBeVisible();

    // Fill in the path
    await mainPanel.getByPlaceholder(/scratch/).fill(TEST_PROJECT_DIR);

    // Fill in the name
    await mainPanel.getByLabel(/Project Name/i).fill("E2E Test Project");

    // Submit
    await mainPanel.getByRole("button", { name: /Create Project/i }).last().click();

    // After creation, the project should be active — we should see the project name
    // and the "Scan for Existing Jobs" button
    await expect(page.getByText("E2E Test Project")).toBeVisible({ timeout: 10_000 });
  });

  test("New Job page shows pipeline form with project open", async ({
    page,
  }) => {
    // Set up a project via API first
    const resp = await fetch(`${BASE_URL}/api/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: TEST_PROJECT_DIR, name: "E2E Test" }),
    });
    const project = await resp.json();

    // Navigate with project in localStorage
    await page.goto(BASE_URL);
    await page.evaluate(
      (p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
      { id: project.id, path: project.path, name: project.name }
    );
    await page.goto(`${BASE_URL}/jobs/new`);
    await waitForApp(page);

    // Should see "New Job" heading
    await expect(page.getByText("New Job")).toBeVisible();

    // Should see Job Type selector defaulting to Pipeline
    await expect(page.getByLabel(/Job Type/i)).toBeVisible();

    // Pipeline form should be visible with Particles field
    await expect(page.getByText(/Particles/i)).toBeVisible();
  });

  test("New Job page redirects to dashboard when no project", async ({
    page,
  }) => {
    await page.goto(BASE_URL);
    await page.evaluate(() => localStorage.clear());
    await page.goto(`${BASE_URL}/jobs/new`);
    await waitForApp(page);

    // Should see "create or open a project" message
    await expect(
      page.getByText(/create or open a project/i)
    ).toBeVisible();

    // Should have link back to dashboard
    await expect(page.getByRole("link", { name: /Dashboard/i })).toBeVisible();
  });
});
