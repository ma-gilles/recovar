/* Browser E2E for the RECOVAR GUI.
 *
 * Drives every frontend route in a real browser against the live server,
 * capturing console errors, uncaught exceptions, failed requests, and 5xx
 * responses.  Also clicks through job-detail tabs, explorer color modes, and
 * every job-form type.  A route "fails" if it throws a JS error/exception or
 * gets a 5xx; 4xx are logged as warnings (the app probes some endpoints).
 *
 *   node scripts/e2e_gui.cjs            # uses BASE=http://127.0.0.1:8085
 */
const { chromium } = require("playwright");

const BASE = process.env.BASE || "http://127.0.0.1:8085";
const SHOT_DIR = "/tmp/e2e_shots";
const results = [];

async function api(path) {
  const r = await fetch(BASE + path);
  return r.json();
}

function attach(page, bucket) {
  const onConsole = (m) => {
    if (m.type() !== "error") return;
    const t = m.text();
    // Browsers auto-log "Failed to load resource" for every 4xx/5xx; those are
    // already tracked by the response handler.  Only count genuine JS errors.
    if (/Failed to load resource/i.test(t)) return;
    bucket.errors.push("CONSOLE: " + t.slice(0, 300));
  };
  const onPageErr = (e) => bucket.errors.push("PAGEERROR: " + (e.message || e).toString().slice(0, 300));
  const onReqFail = (r) => {
    const u = r.url();
    if (u.startsWith("data:") || u.endsWith("favicon.ico")) return;
    const err = r.failure() ? r.failure().errorText : "";
    if (/ERR_ABORTED/i.test(err)) return; // aborted on navigation/close — benign
    bucket.errors.push("REQFAIL: " + u + " " + err);
  };
  const onResp = (r) => {
    const s = r.status();
    if (s >= 500) bucket.errors.push("HTTP" + s + ": " + r.url());
    else if (s >= 400 && r.url().includes("/api/")) bucket.warns.push("HTTP" + s + ": " + r.url().replace(BASE, ""));
  };
  page.on("console", onConsole);
  page.on("pageerror", onPageErr);
  page.on("requestfailed", onReqFail);
  page.on("response", onResp);
  return () => {
    page.off("console", onConsole);
    page.off("pageerror", onPageErr);
    page.off("requestfailed", onReqFail);
    page.off("response", onResp);
  };
}

async function route(page, name, path, opts = {}) {
  const bucket = { errors: [], warns: [] };
  const detach = attach(page, bucket);
  try {
    // domcontentloaded (not networkidle): some pages stream large volumes /
    // poll job status over the slow tunnel and never go fully idle.
    await page.goto(BASE + path, { waitUntil: "domcontentloaded", timeout: 45000 });
  } catch (e) {
    bucket.errors.push("GOTO: " + e.message.slice(0, 200));
  }
  if (opts.waitFor) {
    try { await page.waitForSelector(opts.waitFor, { timeout: 15000 }); }
    catch (e) { bucket.errors.push("waitFor(" + opts.waitFor + "): not found"); }
  }
  if (opts.action) {
    try { await opts.action(page); }
    catch (e) { bucket.errors.push("action: " + e.message.slice(0, 200)); }
  }
  await page.waitForTimeout(900); // let late async errors surface
  detach();
  try { await page.screenshot({ path: `${SHOT_DIR}/${name}.png`, fullPage: false }); } catch {}
  const ok = bucket.errors.length === 0;
  results.push({ name, path, ok, errors: bucket.errors, warns: bucket.warns });
  const mark = ok ? "\x1b[32mPASS\x1b[0m" : "\x1b[31mFAIL\x1b[0m";
  console.log(`  ${mark} ${name}  (${path})`);
  bucket.errors.forEach((e) => console.log("       \x1b[31m" + e + "\x1b[0m"));
  if (bucket.warns.length) console.log("       warns: " + bucket.warns.slice(0, 4).join(" | "));
}

async function clickTabs(page, names) {
  for (const t of names) {
    try {
      const el = page.getByRole("tab", { name: t }).or(page.getByRole("button", { name: t })).first();
      if (await el.count()) { await el.click({ timeout: 4000 }); await page.waitForTimeout(700); }
    } catch {}
  }
}

(async () => {
  require("fs").mkdirSync(SHOT_DIR, { recursive: true });
  console.log("Connecting to Chrome on :9222 ...");
  const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");

  // discover real ids
  const projects = await api("/api/projects");
  const spa = projects.find((p) => p.path.includes("10073")) || projects[0];
  const et = projects.find((p) => p.path.includes("et_test"));
  const detail = await api(`/api/projects/${spa.id}`);
  const jobs = detail.jobs || [];
  const analyze = jobs.find((j) => j.type === "Analyze" && j.status === "completed");
  const density = jobs.find((j) => j.type === "Density" && j.status === "completed");
  const anyJob = analyze || jobs.find((j) => j.status === "completed") || jobs[0];
  const twoJobs = jobs.filter((j) => j.status === "completed").slice(0, 2).map((j) => j.id);
  console.log(`SPA project ${spa.name} (${spa.id})  jobs=${jobs.length}`);
  console.log(`  analyze=${analyze && analyze.id.slice(0,8)} density=${density && density.id.slice(0,8)} detail=${anyJob && anyJob.id.slice(0,8)}`);

  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  await ctx.addInitScript((p) => {
    localStorage.setItem("recovar_active_project", JSON.stringify(p));
  }, { id: spa.id, path: spa.path, name: spa.name });
  const page = await ctx.newPage();

  console.log("\n=== ROUTES ===");
  await route(page, "01_dashboard", "/", { waitFor: "text=/Recent|Jobs|Pipeline/i" });
  await route(page, "02_settings", "/settings", { waitFor: "text=/SLURM|Local|Settings/i",
    action: (p) => clickTabs(p, ["Local GPU", "SLURM Cluster", "Local", "SLURM"]) });
  await route(page, "03_masks", "/masks", {});

  // job forms — every type
  const types = ["pipeline", "analyze", "downsample", "compute_state",
                 "compute_trajectory", "density", "stable_states", "postprocess"];
  for (const t of types) {
    await route(page, `04_form_${t}`, `/jobs/new?type=${t}`, { waitFor: "form, input, button" });
  }

  // job detail + tabs
  if (anyJob) {
    await route(page, "05_job_detail", `/jobs/${anyJob.id}`, {
      waitFor: "text=/Overview|Logs|Parameters|Volumes/i",
      action: (p) => clickTabs(p, ["Logs", "Parameters", "Plots", "Volumes", "Overview"]),
    });
  }
  // density job detail (has the new Explore Latent Space button)
  if (density) {
    await route(page, "06_density_detail", `/jobs/${density.id}`, {
      waitFor: "text=/Overview|Logs|Parameters/i",
    });
  }

  // latent explorer (the big interactive page)
  if (analyze) {
    await route(page, "07_explore", `/explore/${analyze.id}`, {
      waitFor: ".js-plotly-plot, canvas, svg",
      action: async (p) => {
        await p.waitForTimeout(2500); // plotly + embeddings load
        for (const lbl of ["Deconvolved", "K-means", "Point density", "None"]) {
          try {
            const b = p.getByRole("button", { name: lbl }).first();
            if (await b.count()) { await b.click({ timeout: 3000 }); await p.waitForTimeout(1200); }
          } catch {}
        }
      },
    });
  }

  // compare
  if (twoJobs.length === 2) {
    await route(page, "08_compare", `/compare?jobs=${twoJobs.join(",")}`, {});
  }

  // cryo-ET project dashboard (seed et project context)
  if (et) {
    const etPage = await (async () => {
      const c = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
      await c.addInitScript((p) => localStorage.setItem("recovar_active_project", JSON.stringify(p)),
        { id: et.id, path: et.path, name: et.name });
      return c.newPage();
    })();
    const bucket = { errors: [], warns: [] };
    const detach = attach(etPage, bucket);
    try { await etPage.goto(BASE + "/", { waitUntil: "domcontentloaded", timeout: 45000 }); } catch (e) { bucket.errors.push("GOTO " + e.message); }
    await etPage.waitForTimeout(1500);
    detach();
    await etPage.screenshot({ path: `${SHOT_DIR}/09_et_dashboard.png` });
    const ok = bucket.errors.length === 0;
    results.push({ name: "09_et_dashboard", path: "/ (et project)", ok, errors: bucket.errors, warns: bucket.warns });
    console.log(`  ${ok ? "\x1b[32mPASS\x1b[0m" : "\x1b[31mFAIL\x1b[0m"} 09_et_dashboard`);
    bucket.errors.forEach((e) => console.log("       \x1b[31m" + e + "\x1b[0m"));
  }

  await browser.close().catch(() => {});

  const failed = results.filter((r) => !r.ok);
  console.log("\n" + "=".repeat(60));
  console.log(`E2E: ${results.length - failed.length}/${results.length} routes clean`);
  console.log(`screenshots in ${SHOT_DIR}`);
  if (failed.length) {
    console.log("\nFAILURES:");
    failed.forEach((r) => { console.log(`  ${r.name} (${r.path})`); r.errors.forEach((e) => console.log("    " + e)); });
    process.exit(1);
  }
  console.log("\x1b[32mALL ROUTES CLEAN\x1b[0m");
})().catch((e) => { console.error("E2E harness error:", e); process.exit(2); });
