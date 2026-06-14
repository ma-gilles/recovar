/* Focused deep E2E: 3D volume viewer (VTK) + Mask Wizard modal. */
const { chromium } = require("playwright");
const BASE = process.env.BASE || "http://127.0.0.1:8085";
const SHOT = "/tmp/e2e_shots";

(async () => {
  const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
  const projects = await (await fetch(BASE + "/api/projects")).json();
  const spa = projects.find((p) => p.path.includes("10073")) || projects[0];
  const detail = await (await fetch(BASE + `/api/projects/${spa.id}`)).json();
  const jobs = detail.jobs || [];
  // pick a job that actually has volumes
  let job = null;
  for (const j of jobs.filter((x) => x.status === "completed")) {
    const v = await (await fetch(BASE + `/api/jobs/${j.id}/volumes`)).json();
    if (Array.isArray(v) && v.length) { job = j; break; }
  }
  if (!job) { console.log("no job with volumes found"); process.exit(2); }

  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  await ctx.addInitScript((p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
    { id: spa.id, path: spa.path, name: spa.name });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message.slice(0, 200)));
  page.on("console", (m) => { if (m.type() === "error" && !/Failed to load resource/i.test(m.text())) errors.push("CONSOLE: " + m.text().slice(0, 200)); });
  page.on("response", (r) => { if (r.status() >= 500) errors.push("HTTP" + r.status() + ": " + r.url()); });

  console.log(`Deep test on ${job.type} ${job.id.slice(0, 8)}`);
  await page.goto(BASE + `/jobs/${job.id}`, { waitUntil: "domcontentloaded", timeout: 45000 });

  // --- Volumes tab / 3D viewer ---
  const volTab = page.getByRole("tab", { name: "Volumes" }).or(page.getByRole("button", { name: "Volumes" })).first();
  await volTab.click({ timeout: 10000 });
  await page.waitForTimeout(2000);
  const wand = await page.getByTitle("Create mask from this volume").count();
  console.log(`  Volumes tab: wand-buttons=${wand}`);
  await page.waitForTimeout(5000); // let VTK fetch+render the default volume over the tunnel
  const canvas = await page.locator("canvas").count();
  console.log(`  3D viewer: <canvas> elements=${canvas}`);
  await page.screenshot({ path: `${SHOT}/10_volumes_tab.png` });

  // --- Click a volume -> VTK 3D viewer should render a canvas ---
  try {
    await page.getByText(/output\/volumes\/mean\.mrc|mean\.mrc/).first().click({ timeout: 8000 });
    await page.waitForSelector("canvas", { timeout: 45000 }); // 67MB volume over tunnel
    await page.waitForTimeout(3000);
    const c2 = await page.locator("canvas").count();
    console.log(`  3D viewer after volume click: <canvas>=${c2}`);
    await page.screenshot({ path: `${SHOT}/10b_volume_loaded.png` });
  } catch (e) { console.log("  volume-click/canvas: " + e.message.slice(0, 140)); }

  // --- Mask Wizard modal ---
  if (wand > 0) {
    await page.getByTitle("Create mask from this volume").first().click({ timeout: 10000 });
    await page.waitForTimeout(4000); // wizard loads + builds initial preview
    const markers = await page.getByText(/Threshold|Soft edge|Dilation|Save Mask|Eraser|Segments/i).count();
    console.log(`  Mask Wizard: control markers visible=${markers}`);
    await page.screenshot({ path: `${SHOT}/11_mask_wizard.png` });
  }

  await browser.close().catch(() => {});
  console.log(`\nJS errors / 5xx: ${errors.length}`);
  errors.forEach((e) => console.log("  \x1b[31m" + e + "\x1b[0m"));
  console.log(errors.length ? "\x1b[31mDEEP TEST: issues found\x1b[0m" : "\x1b[32mDEEP TEST: clean\x1b[0m");
  process.exit(errors.length ? 1 : 0);
})().catch((e) => { console.error("harness:", e); process.exit(2); });
