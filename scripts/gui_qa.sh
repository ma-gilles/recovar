#!/usr/bin/env bash
# GUI QA script: build, start server, take screenshots, run E2E tests.
# Usage: ./scripts/gui_qa.sh [--scan-dir /path/to/pipeline/output]
#
# Requires: Firefox, Xvfb (available on Della), Node.js
# Output: screenshots in /tmp/gui_qa/, test results on stdout

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GUI_DIR="$REPO_DIR/recovar/gui_v2"
QA_DIR="/tmp/gui_qa"
PORT=8091
SCAN_DIR="${1:---scan-dir /scratch/gpfs/GILLES/mg6942/old_regression_scores_v2/spa/test_dataset/pipeline_output_old}"

# Parse --scan-dir flag
if [[ "${1:-}" == "--scan-dir" ]]; then
    SCAN_DIR="$2"
    shift 2
else
    SCAN_DIR="/scratch/gpfs/GILLES/mg6942/old_regression_scores_v2/spa/test_dataset/pipeline_output_old"
fi

echo "=== RECOVAR GUI QA ==="
echo "Repo: $REPO_DIR"
echo "Port: $PORT"
echo "Scan dir: $SCAN_DIR"
echo "Output: $QA_DIR"
echo ""

# Clean previous run
rm -rf "$QA_DIR"
mkdir -p "$QA_DIR"

# Step 1: Build frontend
echo ">>> Step 1: Building frontend..."
cd "$GUI_DIR/frontend"
npm ci --silent 2>/dev/null || npm install --silent
npm run build 2>&1 | tail -3
echo "    Frontend built."

# Step 2: Install playwright browser
echo ">>> Step 2: Ensuring Firefox for Playwright..."
npx playwright install firefox 2>/dev/null || true

# Step 3: Start server
echo ">>> Step 3: Starting server on port $PORT..."
cd "$REPO_DIR"
pixi run python -m recovar.gui_v2.backend.main --port $PORT &
SERVER_PID=$!
sleep 3

# Check server started
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi
echo "    Server running (PID $SERVER_PID)"

# Cleanup function
cleanup() {
    echo ">>> Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Step 4: Create project and scan
echo ">>> Step 4: Creating test project and scanning..."
PROJECT_DIR="/tmp/gui_qa_project_$$"
mkdir -p "$PROJECT_DIR"

# Create project via API
PROJECT_RESP=$(curl -s -X POST "http://localhost:$PORT/api/projects" \
    -H "Content-Type: application/json" \
    -d "{\"path\": \"$PROJECT_DIR\", \"name\": \"QA Test Project\"}")
PROJECT_ID=$(echo "$PROJECT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")

if [[ -z "$PROJECT_ID" ]]; then
    echo "ERROR: Failed to create project"
    echo "Response: $PROJECT_RESP"
    exit 1
fi
echo "    Project created: $PROJECT_ID"

# Scan existing pipeline output
SCAN_RESP=$(curl -s -X POST "http://localhost:$PORT/api/projects/$PROJECT_ID/scan" \
    -H "Content-Type: application/json" \
    -d "{\"scan_path\": \"$SCAN_DIR\"}")
echo "    Scan response: $SCAN_RESP"

# Step 5: Take screenshots
echo ">>> Step 5: Taking screenshots..."

# Write screenshot script
cat > "$QA_DIR/take_screenshots.mjs" << 'SCREENSHOTEOF'
import { firefox } from 'playwright';

const PORT = process.env.QA_PORT || '8091';
const BASE = `http://localhost:${PORT}`;
const DIR = process.env.QA_DIR || '/tmp/gui_qa';

async function main() {
    const browser = await firefox.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

    const screenshots = [
        { name: '01_dashboard', url: '/', wait: 2000 },
        { name: '02_dashboard_with_project', url: '/', wait: 2000 },
    ];

    // Take dashboard screenshot
    await page.goto(BASE);
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/01_dashboard.png`, fullPage: false });
    console.log('  01_dashboard.png');

    // Find first job link and click it
    try {
        const jobLink = await page.locator('a[href*="/jobs/"]').first();
        if (await jobLink.isVisible({ timeout: 3000 })) {
            await jobLink.click();
            await page.waitForTimeout(2000);
            await page.screenshot({ path: `${DIR}/02_job_overview.png`, fullPage: false });
            console.log('  02_job_overview.png');

            // Click Volumes tab
            const volTab = page.locator('button:has-text("Volumes"), a:has-text("Volumes")');
            if (await volTab.isVisible({ timeout: 2000 })) {
                await volTab.click();
                await page.waitForTimeout(1000);
                await page.screenshot({ path: `${DIR}/03_job_volumes.png`, fullPage: false });
                console.log('  03_job_volumes.png');
            }

            // Click Plots tab
            const plotTab = page.locator('button:has-text("Plots"), a:has-text("Plots")');
            if (await plotTab.isVisible({ timeout: 2000 })) {
                await plotTab.click();
                await page.waitForTimeout(2000);
                await page.screenshot({ path: `${DIR}/04_job_plots.png`, fullPage: false });
                console.log('  04_job_plots.png');
            }

            // Click Explore button
            const exploreBtn = page.locator('button:has-text("Explore"), a:has-text("Explore")');
            if (await exploreBtn.isVisible({ timeout: 2000 })) {
                await exploreBtn.click();
                await page.waitForTimeout(2000);
                await page.screenshot({ path: `${DIR}/05_explore.png`, fullPage: false });
                console.log('  05_explore.png');

                // Click Volumes toggle on explore page
                const volToggle = page.locator('button:has-text("Volumes")');
                if (await volToggle.isVisible({ timeout: 2000 })) {
                    await volToggle.click();
                    await page.waitForTimeout(1000);
                    await page.screenshot({ path: `${DIR}/06_explore_volumes.png`, fullPage: false });
                    console.log('  06_explore_volumes.png');
                }
            }
        }
    } catch (e) {
        console.log('  Warning: Could not navigate to job detail:', e.message);
    }

    // New Job page
    try {
        await page.goto(`${BASE}/jobs/new`);
        await page.waitForTimeout(1500);
        await page.screenshot({ path: `${DIR}/07_new_job.png`, fullPage: false });
        console.log('  07_new_job.png');
    } catch (e) {
        console.log('  Warning: Could not navigate to new job:', e.message);
    }

    await browser.close();
    console.log(`\n  All screenshots saved to ${DIR}/`);
}

main().catch(e => { console.error(e); process.exit(1); });
SCREENSHOTEOF

cd "$GUI_DIR/frontend"
QA_PORT=$PORT QA_DIR=$QA_DIR node "$QA_DIR/take_screenshots.mjs" 2>&1

# Step 6: Run E2E tests
echo ""
echo ">>> Step 6: Running E2E tests..."
if [[ -f "$GUI_DIR/tests/e2e/run-tests.mjs" ]]; then
    cd "$GUI_DIR/frontend"
    QA_PORT=$PORT node "$GUI_DIR/tests/e2e/run-tests.mjs" 2>&1 || true
else
    echo "    No E2E test runner found at tests/e2e/run-tests.mjs"
fi

# Step 7: Summary
echo ""
echo "=== QA COMPLETE ==="
echo "Screenshots: $QA_DIR/"
ls -1 "$QA_DIR"/*.png 2>/dev/null | while read f; do echo "  $(basename $f)"; done
echo ""
echo "To review visually, read the PNG files."
echo "Cleanup: rm -rf $PROJECT_DIR"
