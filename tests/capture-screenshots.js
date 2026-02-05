/**
 * Automated Screenshot Capture for Visual Validation
 *
 * Captures renders from all 3 algorithms (ear-clipping, CDT, occt-import-js)
 * and saves them for visual comparison.
 *
 * Usage:
 *   node tests/capture-screenshots.js                     # Capture baseline
 *   node tests/capture-screenshots.js --checkpoint=1     # Capture for checkpoint 1
 *   node tests/capture-screenshots.js --test=simple      # Only simple tests
 *   node tests/capture-screenshots.js --test=holes       # Only hole tests
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

// Parse command line arguments
const args = process.argv.slice(2);
const getArg = (name) => {
    const arg = args.find(a => a.startsWith(`--${name}=`));
    return arg ? arg.split('=')[1] : null;
};

const checkpoint = getArg('checkpoint') || 'baseline';
const testFilter = getArg('test') || 'all';
const port = parseInt(getArg('port') || '5177', 10);

// Test files organized by category
const TEST_FILES = {
    simple: [
        { name: 'simple-square', path: 'step-examples/benchmark/simple-square.step' },
    ],
    holes: [
        { name: 'plate-small-2x2', path: 'step-examples/benchmark/plate-small-2x2.step' },
        { name: 'plate-medium-5x5', path: 'step-examples/benchmark/plate-medium-5x5.step' },
    ],
    holes_large: [
        { name: 'plate-large-10x10', path: 'step-examples/benchmark/plate-large-10x10.step' },
    ],
    multiface: [
        { name: 'unit-box', path: 'step-examples/c4-multiface/unit-box.step' },
        { name: 'tetrahedron', path: 'step-examples/c4-multiface/tetrahedron.step' },
    ],
    curved: [
        { name: 'cylinder', path: 'step-examples/c4-surfaces/cylinder.step' },
        { name: 'sphere', path: 'step-examples/c4-surfaces/sphere.step' },
        { name: 'torus', path: 'step-examples/c4-surfaces/torus.step' },
    ],
};

// Colors for terminal output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m',
    dim: '\x1b[2m',
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

/**
 * Start Vite dev server
 */
async function startViteServer() {
    return new Promise((resolve, reject) => {
        log(`Starting Vite dev server on port ${port}...`, 'blue');

        const vite = spawn('npx', ['vite', '--port', port.toString()], {
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
                log(`Vite server started on port ${port}`, 'green');
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
        headless: true,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=swiftshader',
            '--enable-gpu-rasterization',
            '--disable-gpu-sandbox',
            '--no-sandbox',
            '--disable-setuid-sandbox',
        ],
    });

    return browser;
}

/**
 * Capture screenshots for a single test file
 */
async function captureTest(page, testFile, outputDir) {
    log(`  Testing: ${testFile.name}`, 'cyan');

    try {
        // Run the test
        await page.evaluate(async (path) => {
            await window.visualValidation.runTest(path);
        }, testFile.path);

        // Wait for rendering to stabilize
        await new Promise(r => setTimeout(r, 1000));

        // Capture each canvas
        const canvases = [
            { id: 'canvas-occ', label: 'ear-clip' },
            { id: 'canvas-cdt', label: 'cdt' },
            { id: 'canvas-occt', label: 'occt-import' },
        ];

        for (const canvas of canvases) {
            const dataUrl = await page.evaluate((canvasId) => {
                return window.visualValidation.getCanvasDataUrl(canvasId);
            }, canvas.id);

            // Convert data URL to buffer and save
            const base64Data = dataUrl.replace(/^data:image\/png;base64,/, '');
            const buffer = Buffer.from(base64Data, 'base64');

            const filename = `${testFile.name}_${canvas.label}.png`;
            const filepath = join(outputDir, filename);
            fs.writeFileSync(filepath, buffer);

            log(`    Saved: ${filename}`, 'dim');
        }

        // Get stats
        const stats = await page.evaluate(() => {
            return window.visualValidation.getStats();
        });

        log(`    Stats: OCC=${stats.occ}, OCCT=${stats.occt}`, 'green');

        return { success: true, stats };

    } catch (error) {
        log(`    Error: ${error.message}`, 'red');
        return { success: false, error: error.message };
    }
}

/**
 * Main entry point
 */
async function main() {
    log('\n' + '='.repeat(60), 'blue');
    log('  Visual Validation Screenshot Capture', 'blue');
    log('='.repeat(60) + '\n', 'blue');

    log(`Checkpoint: ${checkpoint}`, 'cyan');
    log(`Test filter: ${testFilter}`, 'cyan');

    // Create output directory
    const outputDir = join(PROJECT_ROOT, 'tests', 'validation-screenshots', checkpoint);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    log(`Output directory: ${outputDir}\n`, 'cyan');

    // Determine which tests to run
    let testsToRun = [];
    if (testFilter === 'all') {
        for (const category of Object.values(TEST_FILES)) {
            testsToRun = testsToRun.concat(category);
        }
    } else if (TEST_FILES[testFilter]) {
        testsToRun = TEST_FILES[testFilter];
    } else {
        log(`Unknown test filter: ${testFilter}`, 'red');
        log(`Available filters: all, ${Object.keys(TEST_FILES).join(', ')}`, 'yellow');
        process.exit(1);
    }

    log(`Running ${testsToRun.length} tests\n`, 'cyan');

    let viteProcess = null;
    let browser = null;
    const results = [];

    try {
        // Start Vite server
        viteProcess = await startViteServer();

        // Launch browser
        browser = await launchBrowser();
        const page = await browser.newPage();

        // Set viewport
        await page.setViewport({ width: 1400, height: 800 });

        // Enable console logging
        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                log(`[Browser Error] ${msg.text()}`, 'red');
            }
        });

        // Navigate to visual validation page
        log('Navigating to visual validation page...', 'blue');
        await page.goto(`http://localhost:${port}/tests/visual-validation.html`, {
            waitUntil: 'networkidle0',
            timeout: 60000,
        });

        // Wait for initialization
        await page.waitForFunction(
            () => window.visualValidationReady === true,
            { timeout: 60000 }
        );
        log('Visual validation ready\n', 'green');

        // Run tests
        for (const testFile of testsToRun) {
            const result = await captureTest(page, testFile, outputDir);
            results.push({ name: testFile.name, ...result });
        }

        // Print summary
        log('\n' + '='.repeat(60), 'blue');
        log('  Summary', 'blue');
        log('='.repeat(60), 'blue');

        const successful = results.filter(r => r.success).length;
        log(`Captured: ${successful}/${results.length} tests`, successful === results.length ? 'green' : 'yellow');
        log(`Output: ${outputDir}`, 'cyan');

        // Save results metadata
        const metadata = {
            checkpoint,
            timestamp: new Date().toISOString(),
            tests: results,
        };
        fs.writeFileSync(
            join(outputDir, 'metadata.json'),
            JSON.stringify(metadata, null, 2)
        );
        log('Metadata saved to metadata.json', 'dim');

    } catch (error) {
        log(`\nCapture error: ${error.message}`, 'red');
        console.error(error.stack);
    } finally {
        if (browser) {
            await browser.close();
            log('\nBrowser closed', 'dim');
        }
        if (viteProcess) {
            viteProcess.kill();
            log('Vite server stopped', 'dim');
        }
    }
}

main();
