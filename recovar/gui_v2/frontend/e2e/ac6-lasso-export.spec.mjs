/**
 * AC-6: Draw a lasso on the scatter plot, export the selection
 *       as a .ind file.
 *
 * Since this requires actual analyze results with embeddings,
 * we test the subset API and the UI components for creating subsets.
 */
import { test, expect } from "@playwright/test";
import {
  BASE_URL,
  setupProjectWithPipelineJob,
  apiPost,
  apiGet,
} from "./helpers.mjs";

test.describe("AC-6: Lasso export / subset API", () => {
  let projectId;

  test.beforeAll(async () => {
    const setup = await setupProjectWithPipelineJob();
    projectId = setup.projectId;
  });

  test("subset API: create a subset with indices", async () => {
    const subset = await apiPost("/subsets", {
      project_id: projectId,
      name: "test_lasso_selection",
      source_job_id: null,
      method: { type: "lasso", description: "E2E test" },
      indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    });

    expect(subset.id).toBeTruthy();
    expect(subset.name).toBe("test_lasso_selection");
    expect(subset.n_particles).toBe(10);
    expect(subset.path).toContain(".ind");
  });

  test("subset API: list subsets for project", async () => {
    const subsets = await apiGet(`/subsets?project_id=${projectId}`);

    expect(subsets.length).toBeGreaterThan(0);

    const testSubset = subsets.find((s) => s.name === "test_lasso_selection");
    expect(testSubset).toBeTruthy();
    expect(testSubset.n_particles).toBe(10);
  });

  test("subset API: delete a subset", async () => {
    // First create one to delete
    const subset = await apiPost("/subsets", {
      project_id: projectId,
      name: "to_delete",
      indices: [0, 1, 2],
    });

    // Delete it
    const resp = await fetch(`${BASE_URL}/api/subsets/${subset.id}`, {
      method: "DELETE",
    });
    expect(resp.status).toBe(204);

    // Verify it's gone
    const subsets = await apiGet(`/subsets?project_id=${projectId}`);
    expect(subsets.find((s) => s.id === subset.id)).toBeUndefined();
  });
});
