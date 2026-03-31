import { defineConfig } from "@playwright/test";

const PORT = process.env.TEST_PORT ?? "8099";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  workers: 1, // serial: tests share backend state (project/jobs)
  use: {
    baseURL: `http://localhost:${PORT}`,
    screenshot: "on",
    trace: "retain-on-failure",
    headless: true,
    viewport: { width: 1280, height: 800 },
  },
  outputDir: "./e2e-results",
  reporter: [["list"], ["html", { open: "never", outputFolder: "e2e-report" }]],
});
