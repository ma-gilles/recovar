#!/usr/bin/env bash
# GUI QA: build, start server, run interaction tests, take screenshots.
# Usage: ./scripts/gui_qa.sh [--scan-dir /path/to/pipeline/output] [--port 8091]
#
# Output: /tmp/gui_qa/screenshots/*.png + /tmp/gui_qa/results.txt
# Exit code: 0 if all tests pass, 1 if any fail
#
# This script is the single entry point for both:
# 1. Automated QA after implementation (run by building agent or QA agent)
# 2. Manual verification (developer runs it, reads screenshots)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GUI_DIR="$REPO_DIR/recovar/gui_v2"
QA_DIR="/tmp/gui_qa"
PORT=8091
SCAN_DIR="/scratch/gpfs/GILLES/mg6942/old_regression_scores_v2/spa/test_dataset/pipeline_output_old"

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --scan-dir) SCAN_DIR="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=== RECOVAR GUI QA ==="
echo "Repo:     $REPO_DIR"
echo "Port:     $PORT"
echo "Scan dir: $SCAN_DIR"
echo "Output:   $QA_DIR"
echo ""

# Clean
rm -rf "$QA_DIR"
mkdir -p "$QA_DIR/screenshots"

# Step 1: Build frontend
echo ">>> Building frontend..."
cd "$GUI_DIR/frontend"
npm ci --silent 2>/dev/null || npm install --silent 2>/dev/null
npx vite build 2>&1 | tail -3
echo ""

# Step 2: Ensure Playwright Firefox
echo ">>> Ensuring Firefox for Playwright..."
npx playwright install firefox 2>/dev/null || true
echo ""

# Step 3: Start server
echo ">>> Starting server on port $PORT..."
cd "$REPO_DIR"

# Kill any existing server on this port
lsof -ti:$PORT 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

pixi run python -m recovar.gui_v2.backend.main --port $PORT 2>"$QA_DIR/server.log" &
SERVER_PID=$!
sleep 4

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start. Log:"
    cat "$QA_DIR/server.log"
    exit 1
fi

# Verify server responds
if ! curl -s "http://localhost:$PORT/api/system/info" > /dev/null 2>&1; then
    # Server may ignore --port, check what it's actually on
    ACTUAL_PORT=$(grep "running on" "$QA_DIR/server.log" | grep -oP ':\K\d+' | head -1)
    if [[ -n "$ACTUAL_PORT" && "$ACTUAL_PORT" != "$PORT" ]]; then
        echo "WARNING: Server started on port $ACTUAL_PORT instead of $PORT"
        PORT=$ACTUAL_PORT
    else
        echo "ERROR: Server not responding"
        cat "$QA_DIR/server.log"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
fi
echo "    Server running on port $PORT (PID $SERVER_PID)"

cleanup() {
    echo ">>> Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Step 4: Create project and scan via API
echo ">>> Creating test project and scanning..."
PROJECT_DIR="/tmp/gui_qa_project_$$"
mkdir -p "$PROJECT_DIR"

PROJECT_ID=$(curl -s -X POST "http://localhost:$PORT/api/projects" \
    -H "Content-Type: application/json" \
    -d "{\"path\": \"$PROJECT_DIR\", \"name\": \"QA Test\"}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

JOB_ID=$(curl -s -X POST "http://localhost:$PORT/api/projects/$PROJECT_ID/scan" \
    -H "Content-Type: application/json" \
    -d "{\"scan_path\": \"$SCAN_DIR\"}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['imported'][0]['id'])")

echo "    Project: $PROJECT_ID"
echo "    Job: $JOB_ID"
echo ""

# Step 5: Run interaction tests
echo ">>> Running interaction tests..."

cat > "$QA_DIR/qa_test.mjs" << TESTEOF
import { firefox } from 'playwright';

const BASE = 'http://localhost:${PORT}';
const DIR = '${QA_DIR}/screenshots';
const PROJECT_ID = '${PROJECT_ID}';
const JOB_ID = '${JOB_ID}';

const results = [];
function log(status, test, detail) {
    results.push({ status, test, detail: detail || '' });
    const icon = status === 'PASS' ? '\u2713' : status === 'FAIL' ? '\u2717' : '\u26a0';
    console.log('  ' + icon + ' ' + status + ': ' + test + (detail ? ' -- ' + detail : ''));
}

async function main() {
    const browser = await firefox.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

    // Set active project
    await page.goto(BASE);
    await page.evaluate((pid) => { localStorage.setItem('recovar_active_project', pid); }, PROJECT_ID);
    await page.reload();
    await page.waitForTimeout(2000);
    await page.screenshot({ path: DIR + '/01_dashboard.png' });
    log('PASS', 'Dashboard loads');

    // Check sidebar
    const bodyText = await page.textContent('body') || '';
    if (bodyText.includes('PIPELINE')) log('PASS', 'Sidebar shows PIPELINE category');
    else if (bodyText.includes('OTHER')) log('FAIL', 'Sidebar shows OTHER instead of PIPELINE');
    else log('WARN', 'Sidebar category unclear');

    // Job overview
    await page.goto(BASE + '/jobs/' + JOB_ID);
    await page.waitForTimeout(2000);
    await page.screenshot({ path: DIR + '/02_job_overview.png' });

    const overviewText = await page.textContent('body') || '';
    if (overviewText.includes('Completed') || overviewText.includes('completed')) log('PASS', 'Job shows Completed status');
    else log('FAIL', 'Job status not showing Completed');

    if (overviewText.includes('Analyze')) log('PASS', 'Suggested next step visible');
    else log('FAIL', 'No suggested next step');

    // TEST: Suggested next pre-fill
    try {
        const btn = page.locator('text=Analyze this pipeline').first();
        if (await btn.isVisible({ timeout: 3000 })) {
            await btn.click();
            await page.waitForTimeout(2000);
            await page.screenshot({ path: DIR + '/03_suggested_analyze.png' });
            const t = await page.textContent('body') || '';
            const url = page.url();
            const hasType = t.toLowerCase().includes('analyze') && (url.includes('type=') || t.includes('Result Directory'));
            const hasPath = t.includes('pipeline_output');
            if (hasType && hasPath) log('PASS', 'Analyze form pre-filled with type and result_dir');
            else if (hasType) log('FAIL', 'Analyze form opens but result_dir NOT pre-filled');
            else log('FAIL', 'Did not navigate to Analyze form', 'URL: ' + url);
        } else { log('FAIL', 'Suggested next button not found'); }
    } catch (e) { log('FAIL', 'Suggested next click', e.message); }

    // TEST: Volumes tab click interaction
    await page.goto(BASE + '/jobs/' + JOB_ID);
    await page.waitForTimeout(1500);
    try {
        await page.locator('button:has-text("Volumes")').click();
        await page.waitForTimeout(1500);
        await page.screenshot({ path: DIR + '/04_volumes_tab.png' });

        const mrcCount = await page.locator('text=.mrc').count();
        if (mrcCount > 0) log('PASS', 'Volumes tab lists ' + mrcCount + ' MRC files');
        else log('FAIL', 'Volumes tab: no MRC files listed');

        // Click first volume
        const mrc = page.locator('text=mean.mrc').first();
        if (await mrc.isVisible({ timeout: 2000 })) {
            await mrc.click();
            await page.waitForTimeout(3000);
            await page.screenshot({ path: DIR + '/05_volume_viewer.png' });
            const hasViewer = await page.locator('img[src*="slice"], canvas, [class*="viewer"], [class*="slice"]').first().isVisible({ timeout: 2000 }).catch(() => false);
            if (hasViewer) log('PASS', 'Volume click: slice viewer loaded');
            else log('FAIL', 'Volume click: no viewer appeared after clicking mean.mrc');
        } else {
            // Try any mrc
            const any = page.locator('text=.mrc').first();
            await any.click();
            await page.waitForTimeout(3000);
            await page.screenshot({ path: DIR + '/05_volume_viewer.png' });
            log('WARN', 'Clicked .mrc but mean.mrc not found');
        }
    } catch (e) { log('FAIL', 'Volumes interaction', e.message); }

    // TEST: Plots tab images
    await page.goto(BASE + '/jobs/' + JOB_ID);
    await page.waitForTimeout(1500);
    try {
        await page.locator('button:has-text("Plots")').click();
        await page.waitForTimeout(3000);
        await page.screenshot({ path: DIR + '/06_plots.png' });
        const imgs = await page.locator('img').all();
        let broken = 0, loaded = 0;
        for (const img of imgs) { if (await img.evaluate(el => el.naturalWidth) === 0) broken++; else loaded++; }
        if (broken > 0) log('FAIL', 'Plots: ' + broken + '/' + imgs.length + ' broken images');
        else if (loaded > 0) log('PASS', 'Plots: all ' + loaded + ' images loaded');
        else log('WARN', 'Plots: no images found');
    } catch (e) { log('FAIL', 'Plots tab', e.message); }

    // TEST: Parameters tab real params
    await page.goto(BASE + '/jobs/' + JOB_ID);
    await page.waitForTimeout(1500);
    try {
        await page.locator('button:has-text("Parameters")').click();
        await page.waitForTimeout(1000);
        await page.screenshot({ path: DIR + '/07_params.png' });
        const t = await page.textContent('body') || '';
        if (t.includes('legacy_import') && !t.includes('particles') && !t.includes('zdim'))
            log('FAIL', 'Params: only legacy_import, no real pipeline params');
        else if (t.includes('particles') || t.includes('zdim') || t.includes('volume_shape'))
            log('PASS', 'Params: real pipeline parameters shown');
        else log('WARN', 'Params: content unclear');
    } catch (e) { log('FAIL', 'Params tab', e.message); }

    // TEST: Job submission (type case)
    try {
        await page.goto(BASE + '/jobs/new');
        await page.waitForTimeout(1500);
        await page.screenshot({ path: DIR + '/08_new_job.png' });
        const submitBtn = page.locator('button:has-text("Submit")');
        if (await submitBtn.isVisible({ timeout: 2000 })) {
            await submitBtn.click();
            await page.waitForTimeout(2000);
            await page.screenshot({ path: DIR + '/09_submit_result.png' });
            const t = await page.textContent('body') || '';
            if (t.includes('Unknown job type')) log('FAIL', 'Submit: "Unknown job type" case mismatch error');
            else log('PASS', 'Submit: no job type error');
        }
    } catch (e) { log('FAIL', 'Job submission', e.message); }

    // TEST: Explore page
    try {
        await page.goto(BASE + '/jobs/' + JOB_ID);
        await page.waitForTimeout(1500);
        const explore = page.locator('button:has-text("Explore")');
        if (await explore.isVisible({ timeout: 2000 })) {
            await explore.click();
            await page.waitForTimeout(2000);
            await page.screenshot({ path: DIR + '/10_explore.png' });
            const t = await page.textContent('body') || '';
            if (t.includes('Run Analyze') || t.includes('No analysis'))
                log('PASS', 'Explore: shows "Run Analyze first" empty state');
            else if (t.includes('zdim'))
                log('PASS', 'Explore: latent explorer controls visible');
            else log('WARN', 'Explore: page content unclear');
        }
    } catch (e) { log('FAIL', 'Explore page', e.message); }

    await browser.close();

    // Summary
    const pass = results.filter(r => r.status === 'PASS').length;
    const fail = results.filter(r => r.status === 'FAIL').length;
    const warn = results.filter(r => r.status === 'WARN').length;
    console.log('');
    console.log('=== QA RESULTS: ' + pass + ' PASS, ' + fail + ' FAIL, ' + warn + ' WARN ===');
    if (fail > 0) {
        console.log('');
        console.log('FAILURES:');
        results.filter(r => r.status === 'FAIL').forEach((r, i) => {
            console.log('  ' + (i+1) + '. ' + r.test + (r.detail ? ' -- ' + r.detail : ''));
        });
    }
    console.log('');
    console.log('Screenshots: ${QA_DIR}/screenshots/');

    // Write machine-readable results
    const fs = await import('fs');
    fs.writeFileSync('${QA_DIR}/results.json', JSON.stringify({ pass, fail, warn, results }, null, 2));

    process.exit(fail > 0 ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
TESTEOF

cd "$GUI_DIR/frontend"
node "$QA_DIR/qa_test.mjs" 2>&1
QA_EXIT=$?

echo ""
echo "=== QA COMPLETE ==="
echo "Screenshots: $QA_DIR/screenshots/"
echo "Results:     $QA_DIR/results.json"
echo "Server log:  $QA_DIR/server.log"
ls -1 "$QA_DIR/screenshots/"*.png 2>/dev/null | while read f; do echo "  $(basename $f)"; done

# Cleanup temp project
rm -rf "$PROJECT_DIR"

exit $QA_EXIT
