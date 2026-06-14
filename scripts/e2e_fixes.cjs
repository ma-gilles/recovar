/* Verify the two new fixes: project switcher + Mask Wizard 3D default. */
const { chromium } = require("playwright");
const BASE = process.env.BASE || "http://127.0.0.1:8080";
const SHOT = "/tmp/e2e_shots";

(async () => {
  const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
  const projects = await (await fetch(BASE + "/api/projects")).json();
  const spa = projects.find((p) => p.path.includes("10073")) || projects[0];
  const errors = [];
  const ctx = await browser.newContext({ viewport: { width: 1500, height: 950 } });
  await ctx.addInitScript((p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
    { id: spa.id, path: spa.path, name: "10073" });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message.slice(0, 160)));
  page.on("console", (m) => { if (m.type() === "error" && !/Failed to load resource/i.test(m.text())) errors.push("CONSOLE: " + m.text().slice(0, 160)); });

  // --- Project switcher ---
  await page.goto(BASE + "/", { waitUntil: "domcontentloaded", timeout: 45000 });
  await page.waitForTimeout(2500);
  const switchBtn = page.getByTitle("Switch project").first();
  const hasSwitcher = await switchBtn.count();
  console.log(`project switcher button present: ${hasSwitcher > 0}`);
  if (hasSwitcher) {
    await switchBtn.click();
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${SHOT}/13_switcher_open.png` });
    // count project options + try switching to et_test
    const etOpt = page.getByRole("button", { name: /et_test|ET test/ }).first();
    const etCount = await etOpt.count();
    console.log(`et_test option in dropdown: ${etCount > 0}`);
    if (etCount > 0) {
      await etOpt.click();
      await page.waitForTimeout(2500);
      // after switch: sidebar should show et_test; Trajectory group should have jobs
      const traj = await page.getByText(/Trajectory/i).count();
      const switched = await page.getByTitle("Switch project").first().innerText().catch(() => "");
      console.log(`after switch -> header shows: "${switched.replace(/\n/g, " ").trim()}" ; Trajectory label present: ${traj > 0}`);
      await page.screenshot({ path: `${SHOT}/14_switched_to_et.png` });
    }
  }

  // --- Mask Wizard 3D default --- (use a job with volumes)
  const detail = await (await fetch(BASE + `/api/projects/${spa.id}`)).json();
  let jobWithVols = null;
  for (const j of (detail.jobs || []).filter((x) => x.status === "completed")) {
    const v = await (await fetch(BASE + `/api/jobs/${j.id}/volumes`)).json();
    if (Array.isArray(v) && v.length) { jobWithVols = j; break; }
  }
  if (jobWithVols) {
    await page.goto(BASE + `/jobs/${jobWithVols.id}`, { waitUntil: "domcontentloaded", timeout: 45000 });
    await page.getByRole("tab", { name: "Volumes" }).or(page.getByRole("button", { name: "Volumes" })).first().click({ timeout: 10000 });
    await page.waitForTimeout(2000);
    await page.getByTitle("Create mask from this volume").first().click({ timeout: 10000 });
    await page.waitForTimeout(3500);
    // 3D button should be active (aria-pressed=true)
    const threeD = page.getByRole("button", { name: "3D" }).first();
    const pressed = await threeD.getAttribute("aria-pressed").catch(() => null);
    console.log(`Mask Wizard 3D button aria-pressed: ${pressed} (expect "true")`);
    await page.screenshot({ path: `${SHOT}/15_wizard_3d_default.png` });
  }

  await browser.close().catch(() => {});
  console.log(`\nJS errors: ${errors.length}`);
  errors.forEach((e) => console.log("  " + e));
  process.exit(errors.length ? 1 : 0);
})().catch((e) => { console.error("harness:", e); process.exit(2); });
