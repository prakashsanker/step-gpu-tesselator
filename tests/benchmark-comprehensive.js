/**
 * Representative benchmark runner for OCC tessellation vs occt-import-js.
 *
 * Profiles:
 *   canary         Fast iteration set (default)
 *   representative Real-world set including Electronic Enclosure
 *   full           Broader set for milestone/baseline runs
 *
 * Usage:
 *   node tests/benchmark-comprehensive.js
 *   node tests/benchmark-comprehensive.js --suite representative
 *   node tests/benchmark-comprehensive.js --suite full --runs 2 --warmup 1
 *   node tests/benchmark-comprehensive.js --all            # alias for --suite full
 *   node tests/benchmark-comprehensive.js --filter VM      # substring match on name/path
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

const args = process.argv.slice(2);

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

function parseArgs(argv) {
    const parsed = {
        suite: 'canary',
        runs: null,
        warmup: null,
        timeoutMs: null,
        filter: null,
        prewarm: true,
    };

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === '--all') {
            parsed.suite = 'full';
        } else if (arg === '--complex-only') {
            parsed.suite = 'representative';
            parsed.filter = 'VM|Electronic|conical|raw';
        } else if (arg === '--suite') {
            parsed.suite = (argv[i + 1] || '').toLowerCase();
            i += 1;
        } else if (arg === '--runs') {
            parsed.runs = Number(argv[i + 1]);
            i += 1;
        } else if (arg === '--warmup') {
            parsed.warmup = Number(argv[i + 1]);
            i += 1;
        } else if (arg === '--timeout-ms') {
            parsed.timeoutMs = Number(argv[i + 1]);
            i += 1;
        } else if (arg === '--filter') {
            parsed.filter = argv[i + 1] || null;
            i += 1;
        } else if (arg === '--no-prewarm') {
            parsed.prewarm = false;
        } else if (arg === '--help' || arg === '-h') {
            printUsage();
            process.exit(0);
        }
    }

    return parsed;
}

function printUsage() {
    console.log(`Usage:
  node tests/benchmark-comprehensive.js [options]

Options:
  --suite canary|representative|full
  --all                     Alias for --suite full
  --runs N                  Benchmark runs per model
  --warmup N                Warmup runs per model
  --timeout-ms N            Per-run timeout in milliseconds
  --filter PATTERN          Substring filter on model name/path
  --no-prewarm              Disable one-time harness prewarm run
  --help                    Show this help
`);
}

const SUITES = {
    canary: {
        description: 'Fast representative set for tight dev loop',
        timeoutMs: 180000,
        warmupRuns: 0,
        benchmarkRuns: 1,
        models: [
            { name: 'Plate XLarge (holes)', path: 'step-examples/benchmark/plate-xlarge-20x20.step', category: 'holes' },
            { name: 'Cone (primitive)', path: 'step-examples/c4-surfaces/cone.step', category: 'cone' },
            { name: 'Cylinder With Hole (trim)', path: 'step-examples/c6-trimmed/cylinder-with-hole.step', category: 'trimmed' },
            { name: 'BSpline Bowl', path: 'step-examples/c5-bspline/bspline-bowl.step', category: 'bspline' },
            { name: 'Conical Surface (complex)', path: 'step-examples/complex/conical-surface.step', category: 'complex' },
            { name: 'VM-001', path: 'step-examples/VM-001.STEP', category: 'industry' },
        ],
    },
    representative: {
        description: 'Real-world performance gate including electronic enclosure',
        timeoutMs: 360000,
        warmupRuns: 0,
        benchmarkRuns: 1,
        models: [
            { name: 'Electronic Enclosure', path: 'step-examples/Electronic Enclousre.STEP', category: 'enclosure' },
            { name: 'VM-001', path: 'step-examples/VM-001.STEP', category: 'industry' },
            { name: 'VM-002', path: 'step-examples/VM-002.STEP', category: 'industry' },
            { name: 'Raw Material', path: 'step-examples/complex/raw-material.step', category: 'complex' },
            { name: 'Conical Surface (complex)', path: 'step-examples/complex/conical-surface.step', category: 'complex' },
            { name: 'Pipe With Porthole (trim)', path: 'step-examples/c6-trimmed/pipe-with-porthole.step', category: 'trimmed' },
        ],
    },
    full: {
        description: 'Broader milestone suite (still excludes known pathological files)',
        timeoutMs: 420000,
        warmupRuns: 1,
        benchmarkRuns: 2,
        models: [
            { name: 'Simple Square', path: 'step-examples/benchmark/simple-square.step', category: 'simple' },
            { name: 'Plate Medium (holes)', path: 'step-examples/benchmark/plate-medium-5x5.step', category: 'holes' },
            { name: 'Plate XLarge (holes)', path: 'step-examples/benchmark/plate-xlarge-20x20.step', category: 'holes' },
            { name: 'Unit Box', path: 'step-examples/c4-multiface/unit-box.step', category: 'multiface' },
            { name: 'Cone (primitive)', path: 'step-examples/c4-surfaces/cone.step', category: 'cone' },
            { name: 'Cylinder', path: 'step-examples/c4-surfaces/cylinder.step', category: 'curved' },
            { name: 'Sphere', path: 'step-examples/c4-surfaces/sphere.step', category: 'curved' },
            { name: 'Torus', path: 'step-examples/c4-surfaces/torus.step', category: 'curved' },
            { name: 'Cylinder With Hole (trim)', path: 'step-examples/c6-trimmed/cylinder-with-hole.step', category: 'trimmed' },
            { name: 'Pipe With Porthole (trim)', path: 'step-examples/c6-trimmed/pipe-with-porthole.step', category: 'trimmed' },
            { name: 'BSpline Bowl', path: 'step-examples/c5-bspline/bspline-bowl.step', category: 'bspline' },
            { name: 'BSpline Dome', path: 'step-examples/c5-bspline/bspline-dome.step', category: 'bspline' },
            { name: 'Conical Surface (complex)', path: 'step-examples/complex/conical-surface.step', category: 'complex' },
            { name: 'Raw Material', path: 'step-examples/complex/raw-material.step', category: 'complex' },
            { name: 'Electronic Enclosure', path: 'step-examples/Electronic Enclousre.STEP', category: 'enclosure' },
            { name: 'VM-001', path: 'step-examples/VM-001.STEP', category: 'industry' },
            { name: 'VM-002', path: 'step-examples/VM-002.STEP', category: 'industry' },
        ],
    },
};

const EXCLUDED_MODELS = [
    { path: 'step-examples/complex/nissan.step', reason: 'Reference parser failure in baseline runs' },
    { path: 'step-examples/complex/rocky_house.step', reason: 'Timeout (too slow for routine perf loop)' },
    { path: 'step-examples/complex/rotor-201nal.step', reason: 'Timeout (too slow for routine perf loop)' },
];

const LOAD_SUBPHASE_KEYS = [
    'loadStepFile_initOC',
    'loadStepFile_createDoc',
    'loadStepFile_readFile',
    'loadStepFile_transfer',
    'loadStepFile_getTools',
    'loadStepFile_colorParsing',
];

const PHASE_KEYS = [
    'loadStepFile',
    ...LOAD_SUBPHASE_KEYS,
    'extractFacesWithEdges',
    'tessellateOCCShape',
    'tessellatePlanarFace',
    'tessellateCurvedFace',
    'earClipping',
    'computeNormals',
    'meshAssembly',
];

function selectConfig(parsed) {
    const suiteConfig = SUITES[parsed.suite];
    if (!suiteConfig) {
        throw new Error(`Unknown suite: ${parsed.suite}. Use one of: ${Object.keys(SUITES).join(', ')}`);
    }

    const cfg = {
        viteHost: '127.0.0.1',
        vitePort: 5175,
        timeoutMs: Number.isFinite(parsed.timeoutMs) ? parsed.timeoutMs : suiteConfig.timeoutMs,
        headless: true,
        warmupRuns: Number.isFinite(parsed.warmup) ? parsed.warmup : suiteConfig.warmupRuns,
        benchmarkRuns: Number.isFinite(parsed.runs) ? parsed.runs : suiteConfig.benchmarkRuns,
        suite: parsed.suite,
        suiteDescription: suiteConfig.description,
        prewarm: parsed.prewarm !== false,
    };

    let models = [...suiteConfig.models];
    if (parsed.filter) {
        const pattern = parsed.filter.toLowerCase();
        models = models.filter((m) => (`${m.name} ${m.path}`).toLowerCase().includes(pattern));
    }

    return { cfg, models };
}

function percentile(values, p) {
    if (values.length === 0) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    const t = idx - lo;
    return sorted[lo] * (1 - t) + sorted[hi] * t;
}

function mean(values) {
    if (values.length === 0) return null;
    return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function formatMs(value) {
    if (value === null || value === undefined) return 'n/a';
    return `${value.toFixed(1)}ms`;
}

function formatRatio(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
    return `${value.toFixed(2)}x`;
}

function loadStepFile(relativePath) {
    const fullPath = join(PROJECT_ROOT, relativePath);
    const content = fs.readFileSync(fullPath, 'utf8');
    const size = fs.statSync(fullPath).size;
    return { content, size };
}

async function startViteServer(config) {
    return new Promise((resolve, reject) => {
        log('Starting Vite dev server...', 'blue');

        const vite = spawn('npx', ['vite', '--host', config.viteHost, '--port', String(config.vitePort)], {
            cwd: PROJECT_ROOT,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let started = false;
        let stderrBuffer = '';
        const startupTimeout = setTimeout(() => {
            if (!started) {
                vite.kill();
                reject(new Error('Vite server startup timeout'));
            }
        }, 60000);

        vite.stdout.on('data', (chunk) => {
            const text = chunk.toString();
            if (text.includes('Local:') && !started) {
                started = true;
                clearTimeout(startupTimeout);
                const match = text.match(/http:\/\/[^:]+:(\d+)\//);
                const port = match ? Number(match[1]) : config.vitePort;
                log(`Vite server started on port ${port}`, 'green');
                resolve({ process: vite, port });
            }
        });

        vite.stderr.on('data', (chunk) => {
            stderrBuffer += chunk.toString();
        });

        vite.on('close', (code) => {
            if (started) return;
            clearTimeout(startupTimeout);
            const details = stderrBuffer.trim();
            reject(new Error(`Vite exited before ready (code ${code}). ${details}`));
        });

        vite.on('error', (err) => {
            clearTimeout(startupTimeout);
            reject(err);
        });
    });
}

async function launchBrowser() {
    log('Launching browser with WebGPU...', 'blue');
    return puppeteer.launch({
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
}

async function withTimeout(promiseFactory, timeoutMs, label) {
    let timeoutId = null;
    try {
        return await Promise.race([
            promiseFactory(),
            new Promise((_, reject) => {
                timeoutId = setTimeout(() => {
                    reject(new Error(`${label} timed out after ${timeoutMs}ms`));
                }, timeoutMs);
            }),
        ]);
    } finally {
        if (timeoutId !== null) clearTimeout(timeoutId);
    }
}

function summarizePhaseRuns(phaseRuns) {
    const out = {};
    for (const key of PHASE_KEYS) {
        const vals = phaseRuns.map((run) => run[key]).filter((v) => Number.isFinite(v));
        out[key] = mean(vals);
    }
    return out;
}

function summarizeLoadBreakdown(phases, oursMs) {
    if (!phases || !Number.isFinite(phases.loadStepFile) || !Number.isFinite(oursMs) || oursMs <= 0) {
        return null;
    }

    const top = LOAD_SUBPHASE_KEYS
        .map((key) => ({ key, ms: phases[key] || 0 }))
        .filter((entry) => entry.ms > 0)
        .sort((a, b) => b.ms - a.ms)
        .slice(0, 3);

    const topText = top
        .map((entry) => `${entry.key.replace('loadStepFile_', '')}:${entry.ms.toFixed(1)}ms`)
        .join(', ');

    return {
        loadMs: phases.loadStepFile,
        sharePct: (phases.loadStepFile / oursMs) * 100,
        topText,
    };
}

async function runBenchmark(page, model, config) {
    let step;
    try {
        step = loadStepFile(model.path);
    } catch (err) {
        return {
            name: model.name,
            path: model.path,
            category: model.category,
            success: false,
            error: `Failed to load STEP file: ${err.message}`,
        };
    }

    const oursRuns = [];
    const refRuns = [];
    const phaseRuns = [];
    const errors = [];
    let lastSuccess = null;

    for (let i = 0; i < config.warmupRuns; i++) {
        try {
            await withTimeout(
                () => page.evaluate(async (content) => {
                    try {
                        await window.benchmark.runOCC(content);
                    } catch (_) {}
                    try {
                        await window.benchmark.runOcctImport(content);
                    } catch (_) {}
                }, step.content),
                config.timeoutMs,
                `${model.name} warmup`,
            );
        } catch (err) {
            errors.push(`warmup ${i + 1}: ${err.message}`);
        }
    }

    for (let i = 0; i < config.benchmarkRuns; i++) {
        try {
            const run = await withTimeout(
                () => page.evaluate(async (content) => {
                    const ours = await window.benchmark.runOCC(content);
                    const ref = await window.benchmark.runOcctImport(content);
                    return { ours, ref };
                }, step.content),
                config.timeoutMs,
                `${model.name} run ${i + 1}`,
            );

            if (run.ours?.success) {
                oursRuns.push(run.ours.totalTime);
                phaseRuns.push(run.ours.phases || {});
            } else {
                errors.push(`ours run ${i + 1}: ${run.ours?.error || 'unknown error'}`);
            }

            if (run.ref?.success) {
                refRuns.push(run.ref.totalTime);
            } else {
                errors.push(`reference run ${i + 1}: ${run.ref?.error || 'unknown error'}`);
            }

            if (!run.ours?.success || !run.ref?.success) {
                continue;
            }
            lastSuccess = run;
        } catch (err) {
            errors.push(`run ${i + 1}: ${err.message}`);
        }
    }

    if (lastSuccess && oursRuns.length > 0 && refRuns.length > 0) {
        return {
            name: model.name,
            path: model.path,
            category: model.category,
            fileSize: step.size,
            success: true,
            ours: {
                avgMs: mean(oursRuns),
                p50Ms: percentile(oursRuns, 50),
                p90Ms: percentile(oursRuns, 90),
                maxMs: percentile(oursRuns, 100),
                runs: oursRuns,
                phases: summarizePhaseRuns(phaseRuns),
                vertexCount: lastSuccess.ours.vertexCount,
                triangleCount: lastSuccess.ours.triangleCount,
            },
            ref: {
                avgMs: mean(refRuns),
                p50Ms: percentile(refRuns, 50),
                p90Ms: percentile(refRuns, 90),
                maxMs: percentile(refRuns, 100),
                runs: refRuns,
                vertexCount: lastSuccess.ref.vertexCount,
                triangleCount: lastSuccess.ref.triangleCount,
            },
            speedup: mean(refRuns) / mean(oursRuns),
            triangleRatio: lastSuccess.ref.triangleCount > 0
                ? lastSuccess.ours.triangleCount / lastSuccess.ref.triangleCount
                : null,
            warnings: errors,
        };
    }

    return {
        name: model.name,
        path: model.path,
        category: model.category,
        fileSize: step.size,
        success: false,
        error: errors.length > 0 ? errors.join(' | ') : 'No successful run',
    };
}

async function prewarmHarness(page, config) {
    if (!config.prewarm) return;
    const warmFile = loadStepFile('step-examples/benchmark/simple-square.step');
    log('Running one-time benchmark harness prewarm...', 'dim');
    await withTimeout(
        () => page.evaluate(async (content) => {
            try { await window.benchmark.runOCC(content); } catch (_) {}
            try { await window.benchmark.runOcctImport(content); } catch (_) {}
        }, warmFile.content),
        config.timeoutMs,
        'harness prewarm',
    );
}

function printSuiteHeader(config, models) {
    log('\n' + '='.repeat(120), 'blue');
    log(` Representative Benchmark (${config.suite})`, 'bold');
    log(` ${config.suiteDescription}`, 'cyan');
    log(` runs=${config.benchmarkRuns}, warmup=${config.warmupRuns}, timeout=${config.timeoutMs}ms, models=${models.length}`, 'dim');
    log(` prewarm=${config.prewarm ? 'on' : 'off'}`, 'dim');
    log('='.repeat(120), 'blue');

    log('\nExcluded from routine suites:', 'yellow');
    for (const ex of EXCLUDED_MODELS) {
        log(`  - ${ex.path}: ${ex.reason}`, 'dim');
    }
}

function printResultRow(result) {
    if (!result.success) {
        log(`FAIL  ${result.name}: ${result.error}`, 'red');
        return;
    }

    const ours = result.ours.avgMs;
    const ref = result.ref.avgMs;
    const speed = result.speedup;

    const speedLabel = speed >= 1 ? `${speed.toFixed(2)}x faster` : `${(1 / speed).toFixed(2)}x slower`;
    const speedColor = speed >= 1 ? 'green' : 'red';

    log(
        `PASS  ${result.name} | ours=${ours.toFixed(1)}ms | ref=${ref.toFixed(1)}ms | ${speedLabel} | tris=${result.ours.triangleCount}/${result.ref.triangleCount}`,
        speedColor,
    );

    if (speed < 1) {
        const load = summarizeLoadBreakdown(result.ours.phases, ours);
        if (load) {
            const detail = load.topText ? ` | top: ${load.topText}` : '';
            log(
                `      loadStepFile=${load.loadMs.toFixed(1)}ms (${load.sharePct.toFixed(1)}% of ours)${detail}`,
                'dim',
            );
        }
    }
}

function printResultsTable(results) {
    log('\n' + '-'.repeat(154), 'dim');
    const header = [
        'Model'.padEnd(34),
        'SizeKB'.padStart(8),
        'Ours(ms)'.padStart(10),
        'Ref(ms)'.padStart(10),
        'Speedup'.padStart(12),
        'OursTris'.padStart(10),
        'RefTris'.padStart(10),
        'TriRatio'.padStart(10),
    ].join(' | ');
    log(header, 'cyan');
    log('-'.repeat(154), 'dim');

    for (const r of results) {
        if (!r.success) {
            const row = [
                r.name.padEnd(34),
                ((r.fileSize || 0) / 1024).toFixed(1).padStart(8),
                'FAILED'.padStart(10),
                ''.padStart(10),
                ''.padStart(12),
                ''.padStart(10),
                ''.padStart(10),
                ''.padStart(10),
            ].join(' | ');
            log(row, 'red');
            continue;
        }

        const speedStr = r.speedup >= 1
            ? `${r.speedup.toFixed(2)}x`
            : `${(1 / r.speedup).toFixed(2)}x`; // slower shown in color

        const row = [
            r.name.padEnd(34),
            (r.fileSize / 1024).toFixed(1).padStart(8),
            r.ours.avgMs.toFixed(1).padStart(10),
            r.ref.avgMs.toFixed(1).padStart(10),
            speedStr.padStart(12),
            String(r.ours.triangleCount).padStart(10),
            String(r.ref.triangleCount).padStart(10),
            (r.triangleRatio ?? 0).toFixed(2).padStart(10),
        ].join(' | ');

        log(row, r.speedup >= 1 ? 'green' : 'red');
    }

    log('-'.repeat(154), 'dim');
}

function printAggregateSummary(results) {
    const success = results.filter((r) => r.success);
    const failed = results.filter((r) => !r.success);

    log('\nAggregate:', 'bold');
    log(`  successful: ${success.length}/${results.length}`, 'cyan');
    log(`  failed: ${failed.length}/${results.length}`, failed.length > 0 ? 'yellow' : 'green');

    if (success.length === 0) return;

    const speedups = success.map((r) => r.speedup);
    const oursTimes = success.map((r) => r.ours.avgMs);
    const refTimes = success.map((r) => r.ref.avgMs);
    const wins = success.filter((r) => r.speedup >= 1).length;

    const speedMedian = percentile(speedups, 50);
    const speedP90 = percentile(speedups, 90);

    log(`  wins vs occt-import-js: ${wins}/${success.length}`, 'cyan');
    log(`  speedup median: ${formatRatio(speedMedian)}`, speedMedian >= 1 ? 'green' : 'red');
    log(`  speedup p90: ${formatRatio(speedP90)}`, speedP90 >= 1 ? 'green' : 'red');
    log(`  ours avg runtime: ${formatMs(mean(oursTimes))} (p90 ${formatMs(percentile(oursTimes, 90))})`, 'cyan');
    log(`  ref avg runtime: ${formatMs(mean(refTimes))} (p90 ${formatMs(percentile(refTimes, 90))})`, 'cyan');
}

function saveResultsJson(config, models, results) {
    const output = {
        timestamp: new Date().toISOString(),
        suite: config.suite,
        suiteDescription: config.suiteDescription,
        config: {
            benchmarkRuns: config.benchmarkRuns,
            warmupRuns: config.warmupRuns,
            timeoutMs: config.timeoutMs,
            modelCount: models.length,
            prewarm: config.prewarm,
        },
        excluded: EXCLUDED_MODELS,
        results,
    };

    const outPath = join(PROJECT_ROOT, 'tests', 'benchmark-results.json');
    fs.writeFileSync(outPath, JSON.stringify(output, null, 2));
    log(`\nResults saved to ${outPath}`, 'dim');
}

async function main() {
    const parsed = parseArgs(args);
    const { cfg: config, models } = selectConfig(parsed);

    if (models.length === 0) {
        throw new Error('No benchmark models selected. Check --filter or suite.');
    }

    printSuiteHeader(config, models);

    let viteProcess = null;
    let vitePort = config.vitePort;
    let browser = null;
    const results = [];
    let hadFatalError = false;

    try {
        const vite = await startViteServer(config);
        viteProcess = vite.process;
        vitePort = vite.port;
        browser = await launchBrowser();
        const page = await browser.newPage();

        page.setDefaultTimeout(config.timeoutMs);
        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                log(`[Browser Error] ${msg.text()}`, 'red');
            }
        });

        log('Navigating to benchmark harness...', 'blue');
        await page.goto(`http://${config.viteHost}:${vitePort}/tests/benchmark-comprehensive.html`, {
            waitUntil: 'networkidle0',
            timeout: config.timeoutMs,
        });

        await page.waitForFunction(() => window.benchmarkReady === true, {
            timeout: config.timeoutMs,
        });
        log('Benchmark harness ready', 'green');
        await prewarmHarness(page, config);

        for (let i = 0; i < models.length; i++) {
            const model = models[i];
            log(`\n[${i + 1}/${models.length}] ${model.name}`, 'blue');
            const result = await runBenchmark(page, model, config);
            results.push(result);
            printResultRow(result);
        }

        printResultsTable(results);
        printAggregateSummary(results);
        saveResultsJson(config, models, results);
    } catch (err) {
        hadFatalError = true;
        log(`\nBenchmark runner error: ${err.message}`, 'red');
        console.error(err.stack);
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

    const hasFailure = results.some((r) => !r.success);
    process.exit(hadFatalError || hasFailure ? 1 : 0);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
