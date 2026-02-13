/**
 * Comprehensive Benchmark Runner for OCC Tessellator
 *
 * Compares our pipeline (OpenCascade.js + GPU tessellation) against occt-import-js
 *
 * Usage:
 *   node tests/benchmark-comprehensive.js                 # Run canary suite
 *   node tests/benchmark-comprehensive.js --suite canary  # Run canary suite
 *   node tests/benchmark-comprehensive.js --filter bowl   # Filter by name/path substring
 *   node tests/benchmark-comprehensive.js --max-files 20  # Limit files for quick checks
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { basename, dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

function parseArgs(argv) {
    const parsed = {
        suite: 'canary',
        filter: '',
        maxFiles: null,
        warmupRuns: 0,
        benchmarkRuns: 1,
        timeoutMs: 300000,
    };

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === '--suite' && i + 1 < argv.length) {
            parsed.suite = argv[++i];
        } else if (arg === '--filter' && i + 1 < argv.length) {
            parsed.filter = argv[++i].toLowerCase();
        } else if (arg === '--max-files' && i + 1 < argv.length) {
            parsed.maxFiles = Math.max(1, Number(argv[++i]) || 0) || null;
        } else if (arg === '--warmup' && i + 1 < argv.length) {
            parsed.warmupRuns = Math.max(0, Number(argv[++i]) || 0);
        } else if (arg === '--runs' && i + 1 < argv.length) {
            parsed.benchmarkRuns = Math.max(1, Number(argv[++i]) || 1);
        } else if (arg === '--timeout' && i + 1 < argv.length) {
            parsed.timeoutMs = Math.max(30000, Number(argv[++i]) || 300000);
        } else if (arg === '--help' || arg === '-h') {
            console.log(`Usage:
  node tests/benchmark-comprehensive.js [options]

Options:
  --suite canary          Canary suite (all non-real-world test STEP/STP files)
  --filter PATTERN        Substring filter on path/name
  --max-files N           Limit number of files after filtering
  --warmup N              Warmup runs per file (default: 0)
  --runs N                Benchmark runs per file (default: 1)
  --timeout MS            Per-file timeout in ms (default: 300000)
`);
            process.exit(0);
        }
    }

    return parsed;
}

const args = parseArgs(process.argv.slice(2));

// Configuration
const CONFIG = {
    vitePort: 5175,
    timeout: args.timeoutMs,
    headless: true,
    warmupRuns: args.warmupRuns,
    benchmarkRuns: args.benchmarkRuns,
};

const REAL_WORLD_PATH_PATTERNS = [
    /^step-examples\/Electronic Enclousre\.STEP$/i,
    /^step-examples\/VM-\d+\.STEP$/i,
    /^step-examples\/external\//i,
];
const EXCLUDED_BASENAME_PATTERNS = [
    /rocky_house/i,
    /rotor/i,
];
const BROKEN_CANARY_FILES = new Set([
    'step-examples/c2-holes/2.2-projection/tilted-triangle-no-plane.step',
    'step-examples/complex/air.step',
    'step-examples/complex/nissan.step',
]);

function walkStepFiles(relativeDir) {
    const root = join(PROJECT_ROOT, relativeDir);
    const out = [];

    const recurse = (absDir, relDir) => {
        const entries = fs.readdirSync(absDir, { withFileTypes: true });
        for (const entry of entries) {
            const absPath = join(absDir, entry.name);
            const relPath = `${relDir}/${entry.name}`;
            if (entry.isDirectory()) {
                recurse(absPath, relPath);
                continue;
            }
            if (!entry.isFile()) continue;
            if (!/\.(step|stp)$/i.test(entry.name)) continue;
            out.push(relPath);
        }
    };

    recurse(root, relativeDir);
    return out.sort();
}

function categorizePath(relativePath) {
    const parts = relativePath.split('/');
    return parts.length >= 2 ? parts[1] : 'misc';
}

function shouldExcludeCanary(relativePath) {
    for (const pattern of REAL_WORLD_PATH_PATTERNS) {
        if (pattern.test(relativePath)) return 'real-world';
    }
    if (BROKEN_CANARY_FILES.has(relativePath)) {
        return 'known-broken';
    }
    const fileName = basename(relativePath);
    for (const pattern of EXCLUDED_BASENAME_PATTERNS) {
        if (pattern.test(fileName)) return 'requested-exclusion';
    }
    return null;
}

function percentile(values, pct) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((pct / 100) * sorted.length) - 1));
    return sorted[index];
}

function buildCanaryBenchmarks() {
    const allStepFiles = walkStepFiles('step-examples');
    const excluded = [];
    let selected = [];

    for (const relativePath of allStepFiles) {
        const excludedReason = shouldExcludeCanary(relativePath);
        if (excludedReason) {
            excluded.push({ path: relativePath, reason: excludedReason });
            continue;
        }
        selected.push(relativePath);
    }

    if (args.filter) {
        selected = selected.filter((path) => path.toLowerCase().includes(args.filter));
    }
    if (args.maxFiles != null) {
        selected = selected.slice(0, args.maxFiles);
    }

    const models = selected.map((path) => ({
        name: path.replace('step-examples/', ''),
        path,
        category: categorizePath(path),
    }));

    return { models, excluded };
}

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
        const speedups = successResults.map((r) => r.speedup);
        const oursTimes = successResults.map((r) => r.occ.avgTime);
        const refTimes = successResults.map((r) => r.occtImport.avgTime);
        const avgSpeedup = successResults.reduce((a, r) => a + r.speedup, 0) / successResults.length;
        const medianSpeedup = percentile(speedups, 50);
        const p90Speedup = percentile(speedups, 90);
        const oursMedian = percentile(oursTimes, 50);
        const oursP90 = percentile(oursTimes, 90);
        const refMedian = percentile(refTimes, 50);
        const refP90 = percentile(refTimes, 90);
        const fasterCount = successResults.filter(r => r.speedup >= 1).length;
        const failedCount = results.length - successResults.length;

        log('SUMMARY:', 'bold');
        log(`  Total benchmarks: ${results.length}`, 'cyan');
        log(`  Successful: ${successResults.length}`, 'cyan');
        log(`  Failed: ${failedCount}`, failedCount === 0 ? 'green' : 'red');
        log(`  Speedup median: ${medianSpeedup.toFixed(2)}x`, medianSpeedup >= 1 ? 'green' : 'red');
        log(`  Speedup p90: ${p90Speedup.toFixed(2)}x`, p90Speedup >= 1 ? 'green' : 'red');
        log(`  Average speedup: ${avgSpeedup.toFixed(2)}x`, avgSpeedup >= 1 ? 'green' : 'red');
        log(`  Ours median/p90: ${oursMedian.toFixed(1)}ms / ${oursP90.toFixed(1)}ms`, 'cyan');
        log(`  Ref median/p90: ${refMedian.toFixed(1)}ms / ${refP90.toFixed(1)}ms`, 'cyan');
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
        suite: args.suite,
        config: CONFIG,
        filter: args.filter || null,
        maxFiles: args.maxFiles,
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

    if (args.suite !== 'canary') {
        throw new Error(`Unsupported suite "${args.suite}". This runner currently supports only --suite canary.`);
    }
    const { models: benchmarks, excluded } = buildCanaryBenchmarks();
    log('Running CANARY benchmark suite (all non-real-world STEP test files)', 'yellow');
    if (excluded.length > 0) {
        const reasonCounts = excluded.reduce((acc, e) => {
            acc[e.reason] = (acc[e.reason] || 0) + 1;
            return acc;
        }, {});
        const reasonSummary = Object.entries(reasonCounts)
            .map(([reason, count]) => `${reason}=${count}`)
            .join(', ');
        log(`Excluded files: ${excluded.length} (${reasonSummary})`, 'dim');
    }
    if (args.filter) {
        log(`Filter: "${args.filter}"`, 'dim');
    }
    if (args.maxFiles != null) {
        log(`Max files: ${args.maxFiles}`, 'dim');
    }
    if (benchmarks.length === 0) {
        throw new Error('No benchmark files selected after exclusions/filters');
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
