/**
 * Comprehensive Benchmark Runner for OCC Tessellator
 *
 * Compares our pipeline (OpenCascade.js + GPU tessellation) against occt-import-js
 *
 * Usage:
 *   node tests/benchmark-comprehensive.js           # Run fast benchmarks only
 *   node tests/benchmark-comprehensive.js --all     # Run all benchmarks including complex
 *   node tests/benchmark-comprehensive.js --complex-only  # Run only complex models
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
const runAll = args.includes('--all');
const complexOnly = args.includes('--complex-only');

// Configuration
const CONFIG = {
    vitePort: 5175,
    timeout: 300000, // 5 minutes for complex models
    headless: true,
    warmupRuns: 1,
    benchmarkRuns: 3,
};

// Benchmark files
const FAST_BENCHMARKS = [
    { name: 'Simple Square', path: 'step-examples/benchmark/simple-square.step', category: 'simple' },
    { name: 'Small Plate (4 holes)', path: 'step-examples/benchmark/plate-small-2x2.step', category: 'holes' },
    { name: 'Medium Plate (25 holes)', path: 'step-examples/benchmark/plate-medium-5x5.step', category: 'holes' },
    { name: 'Unit Box', path: 'step-examples/c4-multiface/unit-box.step', category: 'multiface' },
    { name: 'Tetrahedron', path: 'step-examples/c4-multiface/tetrahedron.step', category: 'multiface' },
    { name: 'Cylinder', path: 'step-examples/c4-surfaces/cylinder.step', category: 'curved' },
    { name: 'Sphere', path: 'step-examples/c4-surfaces/sphere.step', category: 'curved' },
    { name: 'Torus', path: 'step-examples/c4-surfaces/torus.step', category: 'curved' },
];

const COMPLEX_BENCHMARKS = [
    { name: 'Large Plate (100 holes)', path: 'step-examples/benchmark/plate-large-10x10.step', category: 'holes' },
    { name: 'XLarge Plate (400 holes)', path: 'step-examples/benchmark/plate-xlarge-20x20.step', category: 'holes' },
    { name: 'XXLarge Plate (900 holes)', path: 'step-examples/benchmark/plate-xxlarge-30x30.step', category: 'holes' },
    // Valid complex models
    { name: 'Rotor (141K tris)', path: 'step-examples/complex/rotor-201nal.step', category: 'complex' },
    { name: 'Raw Material', path: 'step-examples/complex/raw-material.step', category: 'complex' },
    { name: 'Cube', path: 'step-examples/complex/cube.step', category: 'complex' },
    { name: 'Conical Surface', path: 'step-examples/complex/conical-surface.step', category: 'complex' },
    { name: 'VM-002', path: 'step-examples/VM-002.STEP', category: 'industry' },
    { name: 'VM-001', path: 'step-examples/VM-001.STEP', category: 'industry' },
    // BSpline surfaces (untested category)
    { name: 'BSpline Bowl', path: 'step-examples/c5-bspline/bspline-bowl.step', category: 'bspline' },
    { name: 'BSpline Dome', path: 'step-examples/c5-bspline/bspline-dome.step', category: 'bspline' },
    // Trimmed surfaces (untested category)
    { name: 'Cylinder with Hole', path: 'step-examples/c6-trimmed/cylinder-with-hole.step', category: 'trimmed' },
    { name: 'Pipe with Porthole', path: 'step-examples/c6-trimmed/pipe-with-porthole.step', category: 'trimmed' },
    // NOTE: rocky_house_*.step and air.step are INVALID (unresolved references/truncated)
];

// Colors for terminal output
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
 * Load a STEP file from disk
 */
function loadStepFile(relativePath) {
    const fullPath = join(PROJECT_ROOT, relativePath);
    const content = fs.readFileSync(fullPath, 'utf8');
    const stats = fs.statSync(fullPath);
    return { content, size: stats.size };
}

/**
 * Run benchmark for a single file
 */
async function runBenchmark(page, benchmark) {
    let stepData;
    try {
        stepData = loadStepFile(benchmark.path);
    } catch (e) {
        return { success: false, error: `Failed to load: ${e.message}` };
    }

    const results = {
        name: benchmark.name,
        category: benchmark.category,
        fileSize: stepData.size,
        occ: { times: [], vertices: 0, triangles: 0 },
        occtImport: { times: [], vertices: 0, triangles: 0 },
    };

    // Warmup runs
    for (let i = 0; i < CONFIG.warmupRuns; i++) {
        await page.evaluate(async (content) => {
            try { await window.benchmark.runOCC(content); } catch (e) {}
            try { await window.benchmark.runOcctImport(content); } catch (e) {}
        }, stepData.content);
    }

    // Benchmark runs
    for (let i = 0; i < CONFIG.benchmarkRuns; i++) {
        const runResult = await page.evaluate(async (content) => {
            const occResult = await window.benchmark.runOCC(content);
            const occtResult = await window.benchmark.runOcctImport(content);
            return { occResult, occtResult };
        }, stepData.content);

        if (runResult.occResult.success) {
            results.occ.times.push(runResult.occResult.totalTime);
            results.occ.vertices = runResult.occResult.vertexCount;
            results.occ.triangles = runResult.occResult.triangleCount;
            results.occ.phases = runResult.occResult.phases;
        }

        if (runResult.occtResult.success) {
            results.occtImport.times.push(runResult.occtResult.totalTime);
            results.occtImport.vertices = runResult.occtResult.vertexCount;
            results.occtImport.triangles = runResult.occtResult.triangleCount;
        }
    }

    // Calculate averages
    if (results.occ.times.length > 0) {
        results.occ.avgTime = results.occ.times.reduce((a, b) => a + b, 0) / results.occ.times.length;
    }
    if (results.occtImport.times.length > 0) {
        results.occtImport.avgTime = results.occtImport.times.reduce((a, b) => a + b, 0) / results.occtImport.times.length;
    }

    // Calculate speedup
    if (results.occ.avgTime && results.occtImport.avgTime) {
        results.speedup = results.occtImport.avgTime / results.occ.avgTime;
    }

    results.success = results.occ.times.length > 0 && results.occtImport.times.length > 0;
    return results;
}

/**
 * Print results table
 */
function printResultsTable(results) {
    log('\n' + '='.repeat(120), 'blue');
    log('  BENCHMARK RESULTS', 'bold');
    log('='.repeat(120), 'blue');

    // Header
    const header = [
        'Model'.padEnd(30),
        'Size'.padStart(10),
        'OCC Time'.padStart(12),
        'OCC Tris'.padStart(12),
        'occt Time'.padStart(12),
        'occt Tris'.padStart(12),
        'Speedup'.padStart(10),
    ].join(' | ');
    log(header, 'cyan');
    log('-'.repeat(120), 'dim');

    // Rows
    for (const r of results) {
        if (!r.success) {
            log(`${r.name.padEnd(30)} | FAILED: ${r.error || 'Unknown error'}`, 'red');
            continue;
        }

        const sizeStr = (r.fileSize / 1024).toFixed(1) + ' KB';
        const occTimeStr = r.occ.avgTime.toFixed(0) + ' ms';
        const occTrisStr = r.occ.triangles.toLocaleString();
        const occtTimeStr = r.occtImport.avgTime.toFixed(0) + ' ms';
        const occtTrisStr = r.occtImport.triangles.toLocaleString();

        let speedupStr = r.speedup.toFixed(2) + 'x';
        let speedupColor = r.speedup >= 1 ? 'green' : 'red';

        const row = [
            r.name.padEnd(30),
            sizeStr.padStart(10),
            occTimeStr.padStart(12),
            occTrisStr.padStart(12),
            occtTimeStr.padStart(12),
            occtTrisStr.padStart(12),
            speedupStr.padStart(10),
        ].join(' | ');

        // Color the entire row based on speedup
        log(row, speedupColor);
    }

    log('='.repeat(120) + '\n', 'blue');

    // Summary statistics
    const successResults = results.filter(r => r.success);
    if (successResults.length > 0) {
        const avgSpeedup = successResults.reduce((a, r) => a + r.speedup, 0) / successResults.length;
        const fasterCount = successResults.filter(r => r.speedup >= 1).length;

        log('SUMMARY:', 'bold');
        log(`  Total benchmarks: ${results.length}`, 'cyan');
        log(`  Successful: ${successResults.length}`, 'cyan');
        log(`  Average speedup: ${avgSpeedup.toFixed(2)}x`, avgSpeedup >= 1 ? 'green' : 'red');
        log(`  Faster than occt-import-js: ${fasterCount}/${successResults.length}`, 'cyan');
    }
}

/**
 * Save results to JSON
 */
function saveResultsJson(results) {
    const outputPath = join(PROJECT_ROOT, 'tests', 'benchmark-results.json');
    const output = {
        timestamp: new Date().toISOString(),
        config: CONFIG,
        results: results,
    };
    fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
    log(`\nResults saved to: ${outputPath}`, 'dim');
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
    log('\n' + '='.repeat(60), 'blue');
    log('  OCC Tessellator Comprehensive Benchmark', 'blue');
    log('='.repeat(60) + '\n', 'blue');

    // Determine which benchmarks to run
    let benchmarks = [];
    if (complexOnly) {
        benchmarks = COMPLEX_BENCHMARKS;
        log('Running COMPLEX benchmarks only', 'yellow');
    } else if (runAll) {
        benchmarks = [...FAST_BENCHMARKS, ...COMPLEX_BENCHMARKS];
        log('Running ALL benchmarks (fast + complex)', 'yellow');
    } else {
        benchmarks = FAST_BENCHMARKS;
        log('Running FAST benchmarks only (use --all for complex models)', 'yellow');
    }

    log(`Total benchmarks: ${benchmarks.length}\n`, 'cyan');

    let viteProcess = null;
    let browser = null;
    const results = [];

    try {
        // Start Vite server
        viteProcess = await startViteServer();

        // Launch browser
        browser = await launchBrowser();
        const page = await browser.newPage();

        // Set longer timeout for complex models
        page.setDefaultTimeout(CONFIG.timeout);

        // Enable console logging from page
        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                log(`[Browser Error] ${msg.text()}`, 'red');
            }
        });

        // Navigate to benchmark harness
        log('Navigating to benchmark harness...', 'blue');
        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/benchmark-comprehensive.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        // Wait for harness to be ready
        await page.waitForFunction(
            () => window.benchmarkReady === true,
            { timeout: CONFIG.timeout }
        );
        log('Benchmark harness ready\n', 'green');

        // Run benchmarks
        for (let i = 0; i < benchmarks.length; i++) {
            const benchmark = benchmarks[i];
            log(`[${i + 1}/${benchmarks.length}] ${benchmark.name}...`, 'cyan');

            try {
                const result = await runBenchmark(page, benchmark);
                results.push(result);

                if (result.success) {
                    const speedupStr = result.speedup >= 1
                        ? `${colors.green}${result.speedup.toFixed(2)}x faster${colors.reset}`
                        : `${colors.red}${result.speedup.toFixed(2)}x slower${colors.reset}`;
                    log(`  OCC: ${result.occ.avgTime.toFixed(0)}ms, occt-import: ${result.occtImport.avgTime.toFixed(0)}ms (${speedupStr})`);
                } else {
                    log(`  FAILED: ${result.error}`, 'red');
                }
            } catch (e) {
                log(`  ERROR: ${e.message}`, 'red');
                results.push({ name: benchmark.name, success: false, error: e.message });
            }
        }

        // Print results table
        printResultsTable(results);

        // Save to JSON
        saveResultsJson(results);

    } catch (e) {
        log(`\nBenchmark runner error: ${e.message}`, 'red');
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
