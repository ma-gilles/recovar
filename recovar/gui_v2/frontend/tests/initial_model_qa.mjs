import { chromium } from "playwright";

const base = process.env.RECOVAR_GUI_QA_BASE;
const projectId = process.env.RECOVAR_GUI_QA_PROJECT_ID;
const projectPath = process.env.RECOVAR_GUI_QA_PROJECT_PATH;
const screenshot = process.env.RECOVAR_GUI_QA_SCREENSHOT;

if (!base || !projectId || !projectPath) {
  throw new Error("RECOVAR_GUI_QA_BASE, RECOVAR_GUI_QA_PROJECT_ID, and RECOVAR_GUI_QA_PROJECT_PATH are required");
}

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  await page.addInitScript(({ id, path }) => {
    localStorage.setItem("recovar_active_project", JSON.stringify({ id, path, name: "InitialModel QA" }));
  }, { id: projectId, path: projectPath });

  await page.goto(`${base}/jobs/new`);
  await page.locator("select").first().selectOption("initial_model");
  await page.getByText("Submit InitialModel Job", { exact: true }).waitFor();

  const defaults = await page.evaluate(async () => {
    const response = await fetch("/api/jobs/initial-model/defaults");
    if (!response.ok) throw new Error(`defaults endpoint returned ${response.status}`);
    return response.json();
  });
  if (defaults.nr_iter !== 200 || defaults.nr_classes !== 1 || defaults.require_custom_cuda !== true) {
    throw new Error(`unexpected native defaults: ${JSON.stringify(defaults)}`);
  }

  async function fieldValue(label) {
    return page.getByText(label, { exact: true }).locator("../..").locator("input").inputValue();
  }
  const core = {
    iterations: await fieldValue("Iterations"),
    classes: await fieldValue("Classes (K)"),
    tau2: await fieldValue("Tau2 Fudge"),
    diameter: await fieldValue("Particle Diameter (Å)"),
  };
  if (core.iterations !== "200" || core.classes !== "1" || core.tau2 !== "4" || core.diameter !== "200") {
    throw new Error(`form defaults differ from API: ${JSON.stringify(core)}`);
  }

  await page.getByText("Advanced InitialModel controls", { exact: true }).click();
  await page.getByText("Fourier Backend", { exact: true }).waitFor();
  const requireCuda = page.getByRole("checkbox", { name: /Require custom CUDA/ });
  if (!(await requireCuda.isChecked())) throw new Error("custom CUDA gate is not enabled by default");
  if (screenshot) await page.screenshot({ path: screenshot, fullPage: true });

  console.log(JSON.stringify({ status: "PASS", defaults: core, screenshot: screenshot ?? null }));
} finally {
  await browser.close();
}
