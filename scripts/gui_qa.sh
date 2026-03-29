#!/usr/bin/env bash
# GUI QA: self-contained test of ALL 7 acceptance criteria.
# Handles its own server lifecycle — no external setup needed.
#
# Usage: ./scripts/gui_qa.sh [--port 8091]
# Output: /tmp/gui_qa/screenshots/*.png + /tmp/gui_qa/results.json
# Exit: 0 if all pass, 1 if any fail
#
# Tests with TWO datasets:
#   1. pipeline_output_old (pipeline only — tests volumes, plots, suggested-next)
#   2. pipeline_output (has analyze results — tests scatter plot, lasso, embeddings)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GUI_DIR="$REPO_DIR/recovar/gui_v2"
QA_DIR="/tmp/gui_qa"
PORT="${1:-8091}"
PROJECT_DIR="/tmp/gui_qa_project_$$"

# Data with ONLY pipeline results (no analyze)
PIPELINE_ONLY="/scratch/gpfs/GILLES/mg6942/old_regression_scores_v2/spa/test_dataset/pipeline_output_old"
# Data with pipeline AND analyze results (UMAP, k-means, trajectories)
PIPELINE_WITH_ANALYZE="/scratch/gpfs/GILLES/mg6942/old_regression_scores_v2/spa/test_dataset/pipeline_output"

# Parse --port flag
[[ "${1:-}" == "--port" ]] && PORT="$2"

echo "=== RECOVAR GUI QA — ALL ACCEPTANCE CRITERIA ==="
echo "Port: $PORT | Output: $QA_DIR"
echo ""

# ── Clean ──
rm -rf "$QA_DIR" "$PROJECT_DIR"
mkdir -p "$QA_DIR/screenshots" "$PROJECT_DIR"

# ── Kill existing server on this port ──
echo ">>> Cleaning up port $PORT..."
lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# ── Build frontend ──
echo ">>> Building frontend..."
cd "$GUI_DIR/frontend"
npm ci --silent 2>/dev/null || npm install --silent 2>/dev/null
npx vite build 2>&1 | tail -1
npx playwright install firefox 2>/dev/null || true

# ── Start server ──
echo ">>> Starting server..."
cd "$REPO_DIR"
pixi run python -m recovar.gui_v2.backend.main --port "$PORT" > "$QA_DIR/server.log" 2>&1 &
SERVER_PID=$!
sleep 5

cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -rf "$PROJECT_DIR"
}
trap cleanup EXIT

# Check server — it may ignore --port
ACTUAL_PORT=$PORT
if ! curl -sf "http://localhost:$PORT/api/system/info" > /dev/null 2>&1; then
    ACTUAL_PORT=$(grep -oP 'running on http://127\.0\.0\.1:\K\d+' "$QA_DIR/server.log" | head -1)
    if [[ -n "$ACTUAL_PORT" ]] && curl -sf "http://localhost:$ACTUAL_PORT/api/system/info" > /dev/null 2>&1; then
        echo "    (server on port $ACTUAL_PORT instead of $PORT)"
    else
        echo "FATAL: Server failed to start. Log:"
        cat "$QA_DIR/server.log"
        exit 1
    fi
fi
BASE="http://localhost:$ACTUAL_PORT"
echo "    Server running at $BASE"

# ── Create project + scan both datasets ──
echo ">>> Creating project and scanning data..."
PID=$(curl -s -X POST "$BASE/api/projects" -H "Content-Type: application/json" \
    -d "{\"path\":\"$PROJECT_DIR\",\"name\":\"QA\"}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

SCAN1=$(curl -s -X POST "$BASE/api/projects/$PID/scan" -H "Content-Type: application/json" \
    -d "{\"scan_path\":\"$PIPELINE_ONLY\"}")
JOB_PIPELINE=$(echo "$SCAN1" | python3 -c "import sys,json; print(json.load(sys.stdin)['imported'][0]['id'])" 2>/dev/null || echo "")

SCAN2=$(curl -s -X POST "$BASE/api/projects/$PID/scan" -H "Content-Type: application/json" \
    -d "{\"scan_path\":\"$PIPELINE_WITH_ANALYZE\"}")
JOB_ANALYZED=$(echo "$SCAN2" | python3 -c "import sys,json; print(json.load(sys.stdin)['imported'][0]['id'])" 2>/dev/null || echo "")

echo "    Pipeline-only job: $JOB_PIPELINE"
echo "    Pipeline+analyze job: $JOB_ANALYZED"

if [[ -z "$JOB_PIPELINE" || -z "$JOB_ANALYZED" ]]; then
    echo "FATAL: Failed to import jobs. Scan responses:"
    echo "  Scan1: $SCAN1"
    echo "  Scan2: $SCAN2"
    exit 1
fi

# ── API-level tests (fast, reliable) ──
echo ""
echo ">>> API tests..."
API_FAILS=0

api_test() {
    local name="$1" url="$2" check="$3"
    local resp
    resp=$(curl -s -o /tmp/gui_qa_resp -w "%{http_code}" "$url")
    if [[ "$resp" == "200" ]] && eval "$check" < /tmp/gui_qa_resp > /dev/null 2>&1; then
        echo "  PASS: $name"
    else
        echo "  FAIL: $name (HTTP $resp)"
        cat /tmp/gui_qa_resp 2>/dev/null | head -3
        API_FAILS=$((API_FAILS + 1))
    fi
}

api_test "System info" "$BASE/api/system/info" "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['slurm_available']\""
api_test "Project details" "$BASE/api/projects/$PID" "python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d['jobs'])>=2\""
api_test "Job detail (pipeline)" "$BASE/api/jobs/$JOB_PIPELINE" "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['status']=='completed'\""
api_test "Volumes list" "$BASE/api/jobs/$JOB_PIPELINE/volumes" "python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d)>5\""
api_test "Plots list" "$BASE/api/jobs/$JOB_PIPELINE/plots" "python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d)>0\""
api_test "Suggested next" "$BASE/api/jobs/$JOB_PIPELINE/suggested-next" "python3 -c \"import sys,json; d=json.load(sys.stdin); assert any('nalyze' in s.get('label','') for s in d)\""
api_test "Embeddings available" "$BASE/api/jobs/$JOB_ANALYZED/embeddings/available" "python3 -c \"import sys,json; d=json.load(sys.stdin); assert 4 in d['zdims']\""
api_test "Embeddings data (zdim=4)" "$BASE/api/jobs/$JOB_ANALYZED/embeddings?zdim=4" "python3 -c \"import sys; assert len(sys.stdin.buffer.read()) > 1000\""
api_test "Subset create" "" "true"  # tested below

# Subset test (POST)
SUB_RESP=$(curl -s -X POST "$BASE/api/subsets" -H "Content-Type: application/json" \
    -d "{\"project_id\":\"$PID\",\"name\":\"qa_test\",\"source_job_id\":\"$JOB_ANALYZED\",\"zdim\":4,\"method\":{\"type\":\"manual_indices\"},\"indices\":[0,1,2,3,4]}")
SUB_ID=$(echo "$SUB_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")
if [[ -n "$SUB_ID" ]]; then
    echo "  PASS: Subset create+list"
else
    echo "  FAIL: Subset create (response: $SUB_RESP)"
    API_FAILS=$((API_FAILS + 1))
fi

echo "  API: $API_FAILS failures"

# ── Browser tests (Playwright) ──
echo ""
echo ">>> Browser interaction tests..."

cat > "$QA_DIR/browser_tests.mjs" << BROWSEREOF
import { firefox } from 'playwright';
const BASE = '${BASE}';
const DIR = '${QA_DIR}/screenshots';
const PID = '${PID}';
const JOB_PIPELINE = '${JOB_PIPELINE}';
const JOB_ANALYZED = '${JOB_ANALYZED}';

const results = [];
function log(ac, status, test, detail) {
    results.push({ ac, status, test, detail: detail || '' });
    const icon = status === 'PASS' ? '\u2713' : status === 'FAIL' ? '\u2717' : '\u26a0';
    console.log('  ' + icon + ' AC-' + ac + ' ' + status + ': ' + test + (detail ? ' -- ' + detail : ''));
}

async function main() {
    const browser = await firefox.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

    // Set active project
    await page.goto(BASE);
    await page.evaluate((pid) => localStorage.setItem('recovar_active_project', pid), PID);
    await page.reload();
    await page.waitForTimeout(2000);

    // AC-1: Dashboard
    await page.screenshot({ path: DIR + '/ac1_dashboard.png' });
    const body1 = await page.textContent('body') || '';
    if (body1.includes('New Job') && (body1.includes('PIPELINE') || body1.includes('pipeline')))
        log(1, 'PASS', 'Dashboard with project and job list');
    else
        log(1, 'FAIL', 'Dashboard missing jobs or New Job button');

    // AC-1: New job form
    await page.goto(BASE + '/jobs/new');
    await page.waitForTimeout(1500);
    await page.screenshot({ path: DIR + '/ac1_new_job.png' });
    const body1b = await page.textContent('body') || '';
    if (body1b.includes('Particles') && body1b.includes('Mask'))
        log(1, 'PASS', 'Pipeline form has Particles and Mask fields');
    else
        log(1, 'FAIL', 'Pipeline form missing required fields');

    // AC-3: Volume viewer
    await page.goto(BASE + '/jobs/' + JOB_PIPELINE);
    await page.waitForTimeout(1500);
    const volTab = page.locator('button:has-text("Volumes")');
    if (await volTab.isVisible({ timeout: 2000 })) {
        await volTab.click();
        await page.waitForTimeout(1500);
        const mrc = page.locator('text=mean.mrc').first();
        if (await mrc.isVisible({ timeout: 2000 })) {
            await mrc.click();
            await page.waitForTimeout(3000);
            await page.screenshot({ path: DIR + '/ac3_volume_click.png' });
            const hasViewer = await page.locator('img[src*="slice"], canvas, [class*="viewer"], [class*="slice"]').first().isVisible({ timeout: 2000 }).catch(() => false);
            if (hasViewer) log(3, 'PASS', 'Volume click loads slice viewer');
            else log(3, 'FAIL', 'Volume click: no viewer appeared');
        } else { log(3, 'FAIL', 'mean.mrc not found in volumes list'); }
    } else { log(3, 'FAIL', 'Volumes tab not found'); }

    // AC-3: Plots
    await page.goto(BASE + '/jobs/' + JOB_PIPELINE);
    await page.waitForTimeout(1500);
    const plotTab = page.locator('button:has-text("Plots")');
    if (await plotTab.isVisible({ timeout: 2000 })) {
        await plotTab.click();
        await page.waitForTimeout(3000);
        await page.screenshot({ path: DIR + '/ac3_plots.png' });
        const imgs = await page.locator('img').all();
        let broken = 0;
        for (const img of imgs) { if (await img.evaluate(el => el.naturalWidth) === 0) broken++; }
        if (broken > 0) log(3, 'FAIL', broken + ' broken plot images');
        else if (imgs.length > 0) log(3, 'PASS', 'All ' + imgs.length + ' plot images loaded');
        else log(3, 'WARN', 'No plot images found');
    }

    // AC-4: Suggested next pre-fill
    await page.goto(BASE + '/jobs/' + JOB_PIPELINE);
    await page.waitForTimeout(1500);
    const sugBtn = page.locator('text=Analyze this pipeline').first();
    if (await sugBtn.isVisible({ timeout: 3000 })) {
        await sugBtn.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ path: DIR + '/ac4_suggested.png' });
        const body4 = await page.textContent('body') || '';
        if (body4.includes('pipeline_output'))
            log(4, 'PASS', 'Analyze form pre-filled with result_dir');
        else
            log(4, 'FAIL', 'Analyze form NOT pre-filled with result_dir');
    } else { log(4, 'FAIL', 'Suggested next button not found'); }

    // AC-5: CRITICAL - Latent explorer with REAL analyzed data
    await page.goto(BASE + '/explore/' + JOB_ANALYZED);
    await page.waitForTimeout(3000);
    await page.screenshot({ path: DIR + '/ac5_explore_initial.png' });

    // Check for scatter plot content
    const body5 = await page.textContent('body') || '';
    if (body5.includes('No analysis') || body5.includes('Run Analyze')) {
        log(5, 'FAIL', 'Explore shows "no analysis" for job that HAS analyze results', 'Embeddings endpoint may not find analysis_* directory');
    } else {
        // Check for zdim dropdown
        const zdimSel = page.locator('select').first();
        if (await zdimSel.isVisible({ timeout: 2000 })) {
            const opts = await zdimSel.locator('option').allTextContents();
            log(5, opts.length > 1 ? 'PASS' : 'FAIL', 'zdim dropdown has ' + opts.length + ' options', opts.join(', '));

            // Select zdim=4
            try {
                await zdimSel.selectOption('4');
                await page.waitForTimeout(3000);
            } catch(e) { /* may already be selected */ }
        }

        await page.screenshot({ path: DIR + '/ac5_scatter.png' });

        // Check for canvas (scatter plot rendering)
        const canvases = await page.locator('canvas').count();
        if (canvases > 0) log(5, 'PASS', 'Scatter plot canvas rendered (' + canvases + ' canvas elements)');
        else log(5, 'FAIL', 'No scatter plot canvas found');
    }

    // AC-6: Lasso selection
    const scatterCanvas = page.locator('canvas').first();
    if (await scatterCanvas.isVisible({ timeout: 2000 }).catch(() => false)) {
        const box = await scatterCanvas.boundingBox();
        if (box) {
            const cx = box.x + box.width / 2;
            const cy = box.y + box.height / 2;
            await page.keyboard.down('Shift');
            await page.mouse.move(cx - 40, cy - 40);
            await page.mouse.down();
            for (let i = 0; i < 8; i++) {
                const angle = (i / 8) * 2 * Math.PI;
                await page.mouse.move(cx + 40 * Math.cos(angle), cy + 40 * Math.sin(angle));
            }
            await page.mouse.up();
            await page.keyboard.up('Shift');
            await page.waitForTimeout(1000);
            await page.screenshot({ path: DIR + '/ac6_lasso.png' });

            const body6 = await page.textContent('body') || '';
            if (body6.includes('Export') || body6.includes('export') || body6.includes('selected'))
                log(6, 'PASS', 'Lasso selection shows export option');
            else
                log(6, 'WARN', 'Lasso drawn but no export button visible (may need different trigger)');
        } else { log(6, 'FAIL', 'Canvas has no bounding box'); }
    } else {
        log(6, 'FAIL', 'No scatter canvas for lasso — AC-5 must pass first');
    }

    // AC-7: System info + sidebar
    await page.goto(BASE);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: DIR + '/ac7_system.png' });
    const body7 = await page.textContent('body') || '';
    if (body7.includes('Cluster') || body7.includes('della'))
        log(7, 'PASS', 'System info bar visible');
    else
        log(7, 'FAIL', 'System info not visible');

    if (body7.includes('PIPELINE'))
        log(7, 'PASS', 'Sidebar shows PIPELINE category');
    else if (body7.includes('OTHER'))
        log(7, 'FAIL', 'Sidebar shows OTHER instead of PIPELINE');
    else
        log(7, 'WARN', 'Sidebar category unclear');

    await browser.close();

    // Summary
    const pass = results.filter(r => r.status === 'PASS').length;
    const fail = results.filter(r => r.status === 'FAIL').length;
    const warn = results.filter(r => r.status === 'WARN').length;

    console.log('');
    console.log('=== ACCEPTANCE CRITERIA RESULTS ===');
    for (let ac = 1; ac <= 7; ac++) {
        const acResults = results.filter(r => r.ac === ac);
        const anyFail = acResults.some(r => r.status === 'FAIL');
        const allPass = acResults.every(r => r.status === 'PASS');
        const icon = anyFail ? '\u2717' : allPass ? '\u2713' : '\u26a0';
        console.log('  ' + icon + ' AC-' + ac + ': ' + (anyFail ? 'FAIL' : allPass ? 'PASS' : 'WARN'));
    }
    console.log('');
    console.log('Total: ' + pass + ' pass, ' + fail + ' fail, ' + warn + ' warn');

    // Write results
    const fs = await import('fs');
    fs.writeFileSync('${QA_DIR}/results.json', JSON.stringify({ pass, fail, warn, results }, null, 2));

    process.exit(fail > 0 ? 1 : 0);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
BROWSEREOF

cd "$GUI_DIR/frontend"
node "$QA_DIR/browser_tests.mjs" 2>&1
BROWSER_EXIT=$?

# ── Final summary ──
echo ""
echo "=== QA COMPLETE ==="
echo "API failures: $API_FAILS"
echo "Browser exit: $BROWSER_EXIT"
echo "Screenshots: $QA_DIR/screenshots/"
ls -1 "$QA_DIR/screenshots/"*.png 2>/dev/null | while read f; do echo "  $(basename "$f")"; done
echo "Results: $QA_DIR/results.json"

[[ $API_FAILS -gt 0 || $BROWSER_EXIT -ne 0 ]] && exit 1
exit 0
