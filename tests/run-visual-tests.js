/**
 * Visual Comparison Test Runner
 *
 * Uses the existing visual-validation.html page to:
 * 1. Render each STEP file with our tessellator and occt-import-js
 * 2. Take screenshots and compare using pixelmatch
 * 3. Report pass/fail for all 119+ STEP files
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join, relative } from 'path';
import fs from 'fs';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

// Configuration
const CONFIG = {
    vitePort: 5175,
    timeout: 180000,  // 3 minutes for complex files
    headless: true,
    screenshotDir: join(__dirname, 'visual-results'),
    // Pixel difference threshold (0-1, where 0 = identical, 1 = completely different)
    // 5% allows for minor differences in triangle tessellation
    pixelThreshold: 0.05,
    // Pixelmatch color sensitivity (0-1, lower = more sensitive)
    colorThreshold: 0.1,
};

// Colors
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    dim: '\x1b[2m',
    cyan: '\x1b[36m',
    magenta: '\x1b[35m',
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

/**
 * Find all STEP files
 */
function findStepFiles() {
    const stepDir = join(PROJECT_ROOT, 'step-examples');
    const files = [];

    function walk(dir) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = join(dir, entry.name);
            if (entry.isDirectory()) {
                walk(fullPath);
            } else if (entry.name.endsWith('.step') || entry.name.endsWith('.stp')) {
                files.push(fullPath);
            }
        }
    }

    walk(stepDir);
    return files.sort();
}

/**
 * Start Vite dev server
 */
async function startViteServer() {
    return new Promise((resolve, reject) => {
        log('Starting Vite dev server...', 'blue');

        const vite = spawn('npx', ['vite', '--port', CONFIG.vitePort.toString()], {
            cwd: PROJECT_ROOT,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let started = false;
        const timeout = setTimeout(() => {
            if (!started) {
                vite.kill();
                reject(new Error('Vite server startup timeout'));
            }
        }, 30000);

        vite.stdout.on('data', (data) => {
            const output = data.toString();
            if (output.includes('Local:') && !started) {
                started = true;
                clearTimeout(timeout);
                log(`Vite server started on port ${CONFIG.vitePort}`, 'green');
                resolve(vite);
            }
        });

        vite.stderr.on('data', (data) => {
            const output = data.toString();
            if (output.includes('error')) {
                console.error('Vite error:', output);
            }
        });

        vite.on('error', (err) => {
            clearTimeout(timeout);
            reject(err);
        });
    });
}

/**
 * Launch browser with WebGPU support
 */
async function launchBrowser() {
    log('Launching browser with WebGPU...', 'blue');

    const browser = await puppeteer.launch({
        headless: CONFIG.headless,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=swiftshader',
            '--enable-gpu-rasterization',
            '--disable-gpu-sandbox',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--window-size=1600,900',
        ],
    });

    return browser;
}

/**
 * Convert data URL to PNG buffer
 */
function dataUrlToBuffer(dataUrl) {
    const base64 = dataUrl.replace(/^data:image\/png;base64,/, '');
    return Buffer.from(base64, 'base64');
}

/**
 * Compare two PNG buffers
 */
function compareImages(buffer1, buffer2) {
    const img1 = PNG.sync.read(buffer1);
    const img2 = PNG.sync.read(buffer2);

    // Check dimensions match
    if (img1.width !== img2.width || img1.height !== img2.height) {
        return {
            match: false,
            diffPercent: 1.0,
            error: `Dimension mismatch: ${img1.width}x${img1.height} vs ${img2.width}x${img2.height}`
        };
    }

    const width = img1.width;
    const height = img1.height;
    const diff = new PNG({ width, height });

    const numDiffPixels = pixelmatch(
        img1.data,
        img2.data,
        diff.data,
        width,
        height,
        { threshold: CONFIG.colorThreshold }
    );

    const totalPixels = width * height;
    const diffPercent = numDiffPixels / totalPixels;

    return {
        match: diffPercent <= CONFIG.pixelThreshold,
        diffPercent,
        diffPixels: numDiffPixels,
        totalPixels,
        diffBuffer: PNG.sync.write(diff)
    };
}

/**
 * Test a single STEP file using visual-validation.html
 */
async function testFile(page, filePath, saveScreenshots = false) {
    const relativePath = relative(PROJECT_ROOT, filePath);
    const fileName = relativePath.replace(/\//g, '_').replace(/\.(step|stp)$/i, '');

    try {
        // Run the test using the exposed API
        await page.evaluate(async (testPath) => {
            await window.visualValidation.runTest(testPath);
        }, relativePath);

        // Wait for rendering to complete
        await new Promise(r => setTimeout(r, 500));

        // Get stats
        const stats = await page.evaluate(() => {
            return window.visualValidation.getStats();
        });

        // Parse triangle counts from stats (format: "123 tris, 45ms")
        const parseTriCount = (str) => {
            if (!str || str === '-' || str === 'Failed') return null;
            const match = str.match(/(\d+)\s*tris/);
            return match ? parseInt(match[1], 10) : null;
        };

        const oursTriangles = parseTriCount(stats.occ);
        const refTriangles = parseTriCount(stats.occt);

        // Check if either failed to produce triangles
        if (oursTriangles === null || refTriangles === null) {
            return {
                file: relativePath,
                status: 'error',
                error: oursTriangles === null ? 'Our pipeline failed' : 'Reference failed',
                ours: { triangles: oursTriangles },
                reference: { triangles: refTriangles },
                diffPercent: null
            };
        }

        // Get screenshots - compare our output (canvas-occ) with reference (canvas-occt)
        const [oursDataUrl, refDataUrl] = await page.evaluate(() => {
            return [
                window.visualValidation.getCanvasDataUrl('canvas-occ'),
                window.visualValidation.getCanvasDataUrl('canvas-occt')
            ];
        });

        const oursBuffer = dataUrlToBuffer(oursDataUrl);
        const refBuffer = dataUrlToBuffer(refDataUrl);

        // Compare images
        const comparison = compareImages(oursBuffer, refBuffer);

        // Save screenshots if requested
        if (saveScreenshots) {
            const dir = join(CONFIG.screenshotDir, fileName);
            fs.mkdirSync(dir, { recursive: true });
            fs.writeFileSync(join(dir, 'ours.png'), oursBuffer);
            fs.writeFileSync(join(dir, 'reference.png'), refBuffer);
            if (comparison.diffBuffer) {
                fs.writeFileSync(join(dir, 'diff.png'), comparison.diffBuffer);
            }
        }

        return {
            file: relativePath,
            status: comparison.match ? 'pass' : 'fail',
            diffPercent: comparison.diffPercent,
            ours: { triangles: oursTriangles },
            reference: { triangles: refTriangles },
            triangleDiff: Math.abs(oursTriangles - refTriangles) / Math.max(oursTriangles, refTriangles)
        };

    } catch (e) {
        return {
            file: relativePath,
            status: 'error',
            error: e.message,
            ours: null,
            reference: null,
            diffPercent: null
        };
    }
}

/**
 * Main test runner
 */
async function main() {
    const args = process.argv.slice(2);
    const saveScreenshots = args.includes('--save-screenshots') || args.includes('--save');
    const failedOnly = args.includes('--failed-only');
    const filterPattern = args.find(a => !a.startsWith('--'));

    log('\n========================================', 'blue');
    log('  Visual Comparison Test Suite', 'blue');
    log('========================================\n', 'blue');

    log(`Pixel threshold: ${(CONFIG.pixelThreshold * 100).toFixed(0)}% difference allowed`, 'dim');

    // Create results directory
    if (saveScreenshots) {
        fs.mkdirSync(CONFIG.screenshotDir, { recursive: true });
        log(`Screenshots will be saved to: ${CONFIG.screenshotDir}`, 'dim');
    }

    let viteProcess = null;
    let browser = null;
    const results = [];

    try {
        // Start Vite
        viteProcess = await startViteServer();

        // Launch browser
        browser = await launchBrowser();
        const page = await browser.newPage();

        // Set viewport
        await page.setViewport({ width: 1600, height: 900 });

        // Enable console logging for debugging
        page.on('console', (msg) => {
            if (msg.type() === 'error' && !msg.text().includes('404')) {
                log(`[Browser] ${msg.text()}`, 'dim');
            }
        });

        // Navigate to visual-validation.html
        log('Loading visual validation page...', 'blue');
        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/visual-validation.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        // Wait for initialization
        await page.waitForFunction(
            () => window.visualValidationReady === true,
            { timeout: CONFIG.timeout }
        );
        log('Visual validation harness ready\n', 'green');

        // Find STEP files
        let stepFiles = findStepFiles();
        log(`Found ${stepFiles.length} STEP files\n`, 'blue');

        // Filter if pattern provided
        if (filterPattern) {
            stepFiles = stepFiles.filter(f => f.toLowerCase().includes(filterPattern.toLowerCase()));
            log(`Filtered to ${stepFiles.length} files matching "${filterPattern}"\n`, 'yellow');
        }

        // Test each file
        let passed = 0;
        let failed = 0;
        let errors = 0;

        for (let i = 0; i < stepFiles.length; i++) {
            const file = stepFiles[i];
            const relativePath = relative(PROJECT_ROOT, file);
            const progress = `[${i + 1}/${stepFiles.length}]`;

            process.stdout.write(`${colors.dim}${progress}${colors.reset} ${relativePath} `);

            const result = await testFile(page, file, saveScreenshots);
            results.push(result);

            if (result.status === 'pass') {
                passed++;
                const diff = (result.diffPercent * 100).toFixed(1);
                log(`PASS (${diff}% diff, ${result.ours.triangles} vs ${result.reference.triangles} tris)`, 'green');
            } else if (result.status === 'fail') {
                failed++;
                const diff = (result.diffPercent * 100).toFixed(1);
                log(`FAIL (${diff}% diff, ${result.ours.triangles} vs ${result.reference.triangles} tris)`, 'red');
            } else {
                errors++;
                log(`ERROR: ${result.error}`, 'yellow');
            }
        }

        // Summary
        log('\n========================================', 'blue');
        log('  SUMMARY', 'blue');
        log('========================================', 'blue');
        log(`  Passed: ${passed}`, 'green');
        log(`  Failed: ${failed}`, failed > 0 ? 'red' : 'green');
        log(`  Errors: ${errors}`, errors > 0 ? 'yellow' : 'green');
        log(`  Total:  ${stepFiles.length}`, 'blue');
        log(`  Pass Rate: ${((passed / stepFiles.length) * 100).toFixed(1)}%`, passed === stepFiles.length ? 'green' : 'yellow');
        log('========================================\n', 'blue');

        // Show failed files
        if (failed > 0) {
            log('Failed files (visual difference > threshold):', 'red');
            const failedResults = results.filter(r => r.status === 'fail').sort((a, b) => b.diffPercent - a.diffPercent);
            for (const result of failedResults) {
                const diff = (result.diffPercent * 100).toFixed(1);
                const triDiff = (result.triangleDiff * 100).toFixed(0);
                log(`  ${result.file}`, 'red');
                log(`    Visual: ${diff}% diff | Triangles: ${result.ours.triangles} vs ${result.reference.triangles} (${triDiff}% diff)`, 'dim');
            }
            log('');
        }

        // Show errors
        if (errors > 0) {
            log('Error files (failed to process):', 'yellow');
            for (const result of results.filter(r => r.status === 'error')) {
                log(`  ${result.file}: ${result.error}`, 'yellow');
            }
            log('');
        }

        // Save JSON report
        const reportPath = join(CONFIG.screenshotDir, 'visual-test-report.json');
        fs.mkdirSync(CONFIG.screenshotDir, { recursive: true });
        fs.writeFileSync(reportPath, JSON.stringify({
            timestamp: new Date().toISOString(),
            config: {
                pixelThreshold: CONFIG.pixelThreshold,
                colorThreshold: CONFIG.colorThreshold,
            },
            summary: { passed, failed, errors, total: stepFiles.length },
            results: results.sort((a, b) => {
                // Sort by status (errors first, then fails, then passes)
                const order = { error: 0, fail: 1, pass: 2 };
                return order[a.status] - order[b.status];
            })
        }, null, 2));
        log(`Report saved to: ${reportPath}`, 'dim');

        // Exit with appropriate code
        process.exit(failed > 0 || errors > 0 ? 1 : 0);

    } catch (e) {
        log(`\nTest runner error: ${e.message}`, 'red');
        console.error(e.stack);
        process.exit(1);
    } finally {
        if (browser) {
            await browser.close();
            log('Browser closed', 'dim');
        }
        if (viteProcess) {
            viteProcess.kill();
            log('Vite server stopped', 'dim');
        }
    }
}

main();
