/**
 * OCC Parser Benchmark
 *
 * Profiles the OpenCascade.js-based STEP parser (occ-test.ts)
 * to identify performance bottlenecks for optimization.
 *
 * Compares against occt-import-js as baseline.
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
    timeout: 180000,
    headless: true,
    warmupRuns: 1,
    benchmarkRuns: 3,
};

const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m',
    dim: '\x1b[2m',
    bold: '\x1b[1m',
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

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
                log(`Vite error: ${output}`, 'red');
            }
        });

        vite.on('error', reject);
    });
}

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

function loadStepFile(relativePath) {
    const fullPath = join(PROJECT_ROOT, relativePath);
    return fs.readFileSync(fullPath, 'utf-8');
}

function formatTime(ms) {
    if (ms < 1) return `${(ms * 1000).toFixed(1)}µs`;
    if (ms < 1000) return `${ms.toFixed(2)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

function formatSpeedup(ratio) {
    if (ratio >= 1) {
        return `${colors.green}${ratio.toFixed(2)}x faster${colors.reset}`;
    } else {
        return `${colors.red}${(1/ratio).toFixed(2)}x slower${colors.reset}`;
    }
}

async function runBenchmarks(page) {
    const benchmarkFiles = [
        { name: 'Simple Square', path: 'step-examples/benchmark/simple-square.step' },
        { name: 'Small (4 holes)', path: 'step-examples/benchmark/plate-small-2x2.step' },
        { name: 'Medium (25 holes)', path: 'step-examples/benchmark/plate-medium-5x5.step' },
        { name: 'Large (100 holes)', path: 'step-examples/benchmark/plate-large-10x10.step' },
        { name: 'Unit Box', path: 'step-examples/c4-multiface/unit-box.step' },
        { name: 'Cylinder', path: 'step-examples/c4-surfaces/cylinder.step' },
        { name: 'Sphere', path: 'step-examples/c4-surfaces/sphere.step' },
        { name: 'Rounded Cube', path: 'step-examples/c3-curves/rounded-cube.step' },
        // Complex models - skip for now (too slow)
        // { name: 'Rocky House', path: 'step-examples/complex/rocky_house.step' },
    ];

    const results = [];

    for (const file of benchmarkFiles) {
        // Check if file exists
        const fullPath = join(PROJECT_ROOT, file.path);
        if (!fs.existsSync(fullPath)) {
            log(`Skipping ${file.name}: file not found`, 'yellow');
            continue;
        }

        log(`\n${'─'.repeat(60)}`, 'dim');
        log(`Benchmarking: ${file.name}`, 'cyan');
        log(`${'─'.repeat(60)}`, 'dim');

        const stepContent = loadStepFile(file.path);
        const stepBase64 = Buffer.from(stepContent).toString('base64');

        // Warmup runs
        log(`  Warming up (${CONFIG.warmupRuns} runs)...`, 'dim');
        for (let i = 0; i < CONFIG.warmupRuns; i++) {
            try {
                await page.evaluate(async (text) => {
                    await window.benchmark.runOCC(text);
                }, stepContent);
            } catch (e) {
                log(`  Warmup error: ${e.message}`, 'yellow');
            }
        }

        // Benchmark runs
        const occTimes = [];
        const occtImportTimes = [];
        let occResult, occtImportResult;

        for (let i = 0; i < CONFIG.benchmarkRuns; i++) {
            log(`  Run ${i + 1}/${CONFIG.benchmarkRuns}...`, 'dim');

            // OCC implementation (OpenCascade.js)
            occResult = await page.evaluate(async (text) => {
                return await window.benchmark.runOCC(text);
            }, stepContent);

            if (occResult.success) {
                occTimes.push(occResult.totalTime);
            }

            // occt-import-js baseline
            occtImportResult = await page.evaluate(async (base64) => {
                return await window.benchmark.runOCCTImport(base64);
            }, stepBase64);

            if (occtImportResult.success) {
                occtImportTimes.push(occtImportResult.totalTime);
            }
        }

        // Calculate averages
        const occAvg = occTimes.length > 0 ? occTimes.reduce((a, b) => a + b) / occTimes.length : null;
        const occtImportAvg = occtImportTimes.length > 0 ? occtImportTimes.reduce((a, b) => a + b) / occtImportTimes.length : null;

        // Output results
        log(`\n  Results:`, 'bold');

        if (occAvg !== null) {
            log(`    OCC (OpenCascade.js):`, 'green');
            log(`      Total time: ${formatTime(occAvg)}`);
            log(`      Vertices: ${occResult.vertexCount}`);
            log(`      Triangles: ${occResult.triangleCount}`);
            log(`      Has normals: ${occResult.hasNormals}`);
            log(`      Has colors: ${occResult.hasColors}`);
        } else {
            log(`    OCC: FAILED - ${occResult?.error}`, 'red');
        }

        if (occtImportAvg !== null) {
            log(`    occt-import-js (baseline):`, 'yellow');
            log(`      Total time: ${formatTime(occtImportAvg)}`);
            log(`      Vertices: ${occtImportResult.vertexCount}`);
            log(`      Triangles: ${occtImportResult.triangleCount}`);
        } else {
            log(`    occt-import-js: FAILED - ${occtImportResult?.error}`, 'red');
        }

        // Comparison
        if (occAvg !== null && occtImportAvg !== null) {
            const speedup = occtImportAvg / occAvg;
            log(`\n    OCC vs occt-import-js: ${formatSpeedup(speedup)}`);
        }

        results.push({
            name: file.name,
            occ: occAvg !== null ? {
                totalTime: occAvg,
                vertexCount: occResult.vertexCount,
                triangleCount: occResult.triangleCount,
            } : null,
            occtImport: occtImportAvg !== null ? {
                totalTime: occtImportAvg,
                vertexCount: occtImportResult.vertexCount,
                triangleCount: occtImportResult.triangleCount,
            } : null,
        });
    }

    return results;
}

function printSummary(results) {
    log(`\n${'═'.repeat(100)}`, 'blue');
    log(`  OCC BENCHMARK SUMMARY`, 'bold');
    log(`${'═'.repeat(100)}`, 'blue');

    console.log('\n');
    console.log('  File                    │ OCC (ours)  │ occt-import │ Comparison      │ Triangles');
    console.log('  ────────────────────────┼─────────────┼─────────────┼─────────────────┼───────────');

    for (const r of results) {
        const name = r.name.padEnd(22);
        const occTime = r.occ ? formatTime(r.occ.totalTime).padEnd(11) : 'N/A'.padEnd(11);
        const occtTime = r.occtImport ? formatTime(r.occtImport.totalTime).padEnd(11) : 'N/A'.padEnd(11);

        let comparison = 'N/A'.padEnd(15);
        if (r.occ && r.occtImport) {
            const ratio = r.occtImport.totalTime / r.occ.totalTime;
            comparison = (ratio >= 1 ? `${ratio.toFixed(2)}x faster` : `${(1/ratio).toFixed(2)}x slower`).padEnd(15);
        }

        const triangles = r.occ ? r.occ.triangleCount.toString().padEnd(9) : 'N/A';

        console.log(`  ${name} │ ${occTime} │ ${occtTime} │ ${comparison} │ ${triangles}`);
    }

    console.log('\n');

    // Summary stats
    const validResults = results.filter(r => r.occ && r.occtImport);
    if (validResults.length > 0) {
        const avgRatio = validResults.reduce((sum, r) => sum + r.occtImport.totalTime / r.occ.totalTime, 0) / validResults.length;
        log(`  Average OCC vs occt-import-js: ${formatSpeedup(avgRatio)}`, 'bold');

        const totalOccTime = validResults.reduce((sum, r) => sum + r.occ.totalTime, 0);
        const totalOcctTime = validResults.reduce((sum, r) => sum + r.occtImport.totalTime, 0);
        log(`  Total OCC time: ${formatTime(totalOccTime)}`, 'dim');
        log(`  Total occt-import time: ${formatTime(totalOcctTime)}`, 'dim');
    }

    log(`${'═'.repeat(100)}\n`, 'blue');
}

async function main() {
    log('\n' + '═'.repeat(80), 'blue');
    log('  OCC Parser Benchmark (OpenCascade.js)', 'bold');
    log('═'.repeat(80) + '\n', 'blue');

    let viteProcess = null;
    let browser = null;

    try {
        viteProcess = await startViteServer();

        browser = await launchBrowser();
        const page = await browser.newPage();

        page.on('console', (msg) => {
            const text = msg.text();
            if (msg.type() === 'error') {
                log(`[Browser Error] ${text}`, 'red');
            } else if (text.includes('[OCC')) {
                log(`  ${text}`, 'dim');
            }
        });

        log('Navigating to benchmark page...', 'blue');
        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/occ-benchmark.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        await page.waitForFunction(
            () => window.benchmarkReady === true,
            { timeout: CONFIG.timeout }
        );
        log('Benchmark harness ready\n', 'green');

        const results = await runBenchmarks(page);

        printSummary(results);

    } catch (e) {
        log(`\nBenchmark error: ${e.message}`, 'red');
        console.error(e.stack);
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
