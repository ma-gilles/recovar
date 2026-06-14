/* Focused E2E: a FAILED job's detail page (error + logs render path). */
const { chromium } = require("playwright");
const BASE = process.env.BASE || "http://127.0.0.1:8085";
(async () => {
  const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
  const projects = await (await fetch(BASE + "/api/projects")).json();
  const spa = projects.find((p) => p.path.includes("10073")) || projects[0];
  const detail = await (await fetch(BASE + `/api/projects/${spa.id}`)).json();
  const failed = (detail.jobs || []).find((j) => j.status === "failed");
  if (!failed) { console.log("no failed job to test"); process.exit(0); }
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  await ctx.addInitScript((p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
    { id: spa.id, path: spa.path, name: spa.name });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message.slice(0, 200)));
  page.on("console", (m) => { if (m.type() === "error" && !/Failed to load resource/i.test(m.text())) errors.push("CONSOLE: " + m.text().slice(0, 200)); });
  page.on("response", (r) => { if (r.status() >= 500) errors.push("HTTP" + r.status() + ": " + r.url()); });
  console.log(`Failed job: ${failed.id.slice(0, 8)} (${failed.type})`);
  await page.goto(BASE + `/jobs/${failed.id}`, { waitUntil: "domcontentloaded", timeout: 45000 });
  await page.waitForTimeout(1500);
  // click Logs (where the error/traceback shows)
  try {
    await page.getByRole("tab", { name: "Logs" }).or(page.getByRole("button", { name: "Logs" })).first().click({ timeout: 6000 });
    await page.waitForTimeout(1500);
  } catch {}
  const failedBadge = await page.getByText(/failed/i).count();
  console.log(`  'failed' status text present: ${failedBadge > 0}`);
  await page.screenshot({ path: "/tmp/e2e_shots/12_failed_job.png" });
  await browser.close().catch(() => {});
  console.log(`\nJS errors / 5xx: ${errors.length}`);
  errors.forEach((e) => console.log("  \x1b[31m" + e + "\x1b[0m"));
  console.log(errors.length ? "\x1b[31mFAILED-JOB PAGE: issues\x1b[0m" : "\x1b[32mFAILED-JOB PAGE: clean\x1b[0m");
  process.exit(errors.length ? 1 : 0);
})().catch((e) => { console.error("harness:", e); process.exit(2); });
