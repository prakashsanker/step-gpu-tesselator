/**
 * Automated Visual Comparison: CDT vs occt-import-js
 *
 * Runs all STEP files through both pipelines and compares results.
 * Produces a summary report of triangle counts, timing, and any mismatches.
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

const CONFIG = {
    vitePort: 5176,
    timeout: 120000,
    headless: true,
};

// Colors for terminal output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    dim: '\x1b[2m',
    cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

/**
 * Find all STEP files in the step-examples directory
 */
function findAllStepFiles() {
    const stepDir = join(PROJECT_ROOT, 'step-examples');
    const files = [];

    function walkDir(dir, prefix = '') {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = join(dir, entry.name);
            const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;

            if (entry.isDirectory()) {
                walkDir(fullPath, relativePath);
            } else if (entry.name.endsWith('.step') || entry.name.endsWith('.STEP')) {
                files.push({
                    name: relativePath,
                    path: `step-examples/${relativePath}`,
                    fullPath: fullPath,
                });
            }
        }
    }

    walkDir(stepDir);
    return files;
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
        ],
    });

    return browser;
}

/**
 * Main comparison function
 */
async function main() {
    log('\n' + '='.repeat(80), 'blue');
    log('  CDT vs occt-import-js: Full STEP File Comparison', 'blue');
    log('='.repeat(80) + '\n', 'blue');

    // Find all STEP files
    const stepFiles = findAllStepFiles();
    log(`Found ${stepFiles.length} STEP files to compare\n`, 'cyan');

    let viteProcess = null;
    let browser = null;

    const results = [];
    let matched = 0;
    let mismatched = 0;
    let errors = 0;

    try {
        // Start server and browser
        viteProcess = await startViteServer();
        browser = await launchBrowser();
        const page = await browser.newPage();

        // Navigate to test harness
        log('Loading test harness...', 'blue');
        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/test-harness.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        // Wait for harness to be ready
        await page.waitForFunction(
            () => window.testHarnessReady === true,
            { timeout: CONFIG.timeout }
        );
        log('Test harness ready\n', 'green');

        // Process each file
        for (let i = 0; i < stepFiles.length; i++) {
            const file = stepFiles[i];
            const progress = `[${i + 1}/${stepFiles.length}]`;

            try {
                // Load file content
                const stepContent = fs.readFileSync(file.fullPath, 'utf8');

                // Run comparison in browser
                const result = await page.evaluate(async (stepText, filePath) => {
                    const output = {
                        file: filePath,
                        cdt: { triangles: 0, time: 0, error: null },
                        occt: { triangles: 0, time: 0, error: null },
                    };

                    // Run CDT
                    try {
                        const cdtStart = performance.now();
                        const cdtResult = await window.testHarness.parseStepCDT(stepText);
                        output.cdt.time = performance.now() - cdtStart;

                        if (cdtResult.success) {
                            output.cdt.triangles = cdtResult.mesh.triangleCount;
                        } else {
                            output.cdt.error = cdtResult.error;
                        }
                    } catch (e) {
                        output.cdt.error = e.message;
                    }

                    // Run occt-import-js
                    try {
                        const occtStart = performance.now();
                        const occtResult = await window.testHarness.parseStepOcct(stepText);
                        output.occt.time = performance.now() - occtStart;

                        if (occtResult.success) {
                            output.occt.triangles = occtResult.triangleCount;
                        } else {
                            output.occt.error = occtResult.error;
                        }
                    } catch (e) {
                        output.occt.error = e.message;
                    }

                    return output;
                }, stepContent, file.name);

                results.push(result);

                // Determine match status
                const cdtOk = result.cdt.error === null;
                const occtOk = result.occt.error === null;

                if (!cdtOk || !occtOk) {
                    errors++;
                    log(`${progress} ${file.name}: ERROR`, 'red');
                    if (result.cdt.error) log(`    CDT: ${result.cdt.error}`, 'red');
                    if (result.occt.error) log(`    OCCT: ${result.occt.error}`, 'red');
                } else {
                    // Both succeeded - compare triangles
                    // Allow some tolerance for curved surfaces (different tessellation settings)
                    const diff = Math.abs(result.cdt.triangles - result.occt.triangles);
                    const maxTris = Math.max(result.cdt.triangles, result.occt.triangles);
                    const diffPercent = maxTris > 0 ? (diff / maxTris) * 100 : 0;

                    // Consider it a match if within 50% (curved surfaces vary a lot)
                    // or if the absolute difference is small (< 10 triangles)
                    const isMatch = diff < 10 || diffPercent < 50;

                    if (isMatch) {
                        matched++;
                        const speedup = result.occt.time / result.cdt.time;
                        const speedupStr = speedup >= 1
                            ? `${speedup.toFixed(1)}x faster`
                            : `${(1/speedup).toFixed(1)}x slower`;
                        log(`${progress} ${file.name}: ${colors.green}MATCH${colors.reset} (CDT: ${result.cdt.triangles}, OCCT: ${result.occt.triangles}, ${speedupStr})`, 'reset');
                    } else {
                        mismatched++;
                        log(`${progress} ${file.name}: ${colors.yellow}MISMATCH${colors.reset} (CDT: ${result.cdt.triangles}, OCCT: ${result.occt.triangles})`, 'reset');
                    }
                }

            } catch (e) {
                errors++;
                results.push({
                    file: file.name,
                    cdt: { triangles: 0, time: 0, error: e.message },
                    occt: { triangles: 0, time: 0, error: e.message },
                });
                log(`${progress} ${file.name}: ERROR - ${e.message}`, 'red');
            }
        }

        // Print summary
        log('\n' + '='.repeat(80), 'blue');
        log('  SUMMARY', 'blue');
        log('='.repeat(80), 'blue');
        log(`  Total files:  ${stepFiles.length}`, 'cyan');
        log(`  Matched:      ${matched}`, 'green');
        log(`  Mismatched:   ${mismatched}`, mismatched > 0 ? 'yellow' : 'green');
        log(`  Errors:       ${errors}`, errors > 0 ? 'red' : 'green');
        log('='.repeat(80) + '\n', 'blue');

        // Save detailed results to JSON
        const reportPath = join(PROJECT_ROOT, 'tests', 'comparison-results.json');
        fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
        log(`Detailed results saved to: ${reportPath}`, 'dim');

        // Print timing comparison for successful files
        const successfulResults = results.filter(r => !r.cdt.error && !r.occt.error);
        if (successfulResults.length > 0) {
            const totalCdtTime = successfulResults.reduce((sum, r) => sum + r.cdt.time, 0);
            const totalOcctTime = successfulResults.reduce((sum, r) => sum + r.occt.time, 0);
            const avgSpeedup = totalOcctTime / totalCdtTime;

            log('\n  TIMING COMPARISON', 'cyan');
            log(`  Total CDT time:   ${(totalCdtTime / 1000).toFixed(2)}s`, 'reset');
            log(`  Total OCCT time:  ${(totalOcctTime / 1000).toFixed(2)}s`, 'reset');
            log(`  Average speedup:  ${avgSpeedup.toFixed(2)}x`, avgSpeedup >= 1 ? 'green' : 'yellow');
        }

    } catch (e) {
        log(`\nFatal error: ${e.message}`, 'red');
        console.error(e.stack);
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

    process.exit(errors > 0 || mismatched > 0 ? 1 : 0);
}

main();
