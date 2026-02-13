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
import { basename, dirname, join } from 'path';
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
        maxFiles: null,
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
        } else if (arg === '--max-files') {
            parsed.maxFiles = Number(argv[i + 1]);
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
  --max-files N             Limit selected model count after filtering
  --no-prewarm              Disable one-time harness prewarm run
  --help                    Show this help
`);
}

const SUITES = {
    canary: {
        description: 'All non-real-world STEP files (larger stability gate)',
        timeoutMs: 300000,
        warmupRuns: 0,
        benchmarkRuns: 1,
        models: [],
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

// Canary policy guardrails:
// - Include VM-001 in canary by default (even though it's real-world).
// - Keep rocky_house/rotor excluded from routine loops.
const CANARY_ALWAYS_INCLUDE = new Set([
    'step-examples/VM-001.STEP',
]);
const CANARY_REQUIRED_EXCLUDE_PATTERNS = [
    /rocky_house/i,
    /rotor-201nal/i,
];

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

const LOAD_SUBPHASE_KEYS = [
    'loadStepFile_initOC',
    'loadStepFile_createDoc',
    'loadStepFile_readFile',
    'loadStepFile_transfer',
    'loadStepFile_getTools',
    'loadStepFile_oneShape',
    'loadStepFile_fsWrite',
    'loadStepFile_fsCleanup',
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

const ELECTRONIC_ENCLOSURE_PATH = 'step-examples/Electronic Enclousre.STEP';
const ENCLOSURE_TREND_PATH = join(PROJECT_ROOT, 'tests', 'benchmark-electronic-trend.jsonl');

const CANARY_QUALITY_GUARDS = [
    {
        path: 'step-examples/c4-surfaces/cone.step',
        minTriangles: 700,
        minTriRatio: 0.35,
        reason: 'Cone seam/twist regressions collapse triangle coverage',
    },
    {
        path: 'step-examples/complex/conical-surface.step',
        minTriangles: 1000,
        minTriRatio: 0.75,
        reason: 'Complex conical trims should not collapse to under-resolved meshes',
    },
];

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
    if (CANARY_ALWAYS_INCLUDE.has(relativePath)) return null;
    for (const pattern of REAL_WORLD_PATH_PATTERNS) {
        if (pattern.test(relativePath)) return 'real-world';
    }
    if (BROKEN_CANARY_FILES.has(relativePath)) return 'known-broken';
    const fileName = basename(relativePath);
    for (const pattern of EXCLUDED_BASENAME_PATTERNS) {
        if (pattern.test(fileName)) return 'requested-exclusion';
    }
    return null;
}

function validateCanaryPolicy(selected, excluded) {
    const missingRequired = [...CANARY_ALWAYS_INCLUDE].filter((path) => !selected.includes(path));
    if (missingRequired.length > 0) {
        throw new Error(`Canary policy violation: required include missing: ${missingRequired.join(', ')}`);
    }

    const excludedPaths = excluded.map((item) => item.path);
    const missingExcluded = CANARY_REQUIRED_EXCLUDE_PATTERNS.filter(
        (pattern) => !excludedPaths.some((path) => pattern.test(path))
    );
    if (missingExcluded.length > 0) {
        throw new Error(`Canary policy violation: expected exclusions missing for patterns: ${missingExcluded.map(String).join(', ')}`);
    }
}

function buildCanaryModels(parsed) {
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

    // Enforce policy before optional filters so default canary cannot silently drift.
    validateCanaryPolicy(selected, excluded);

    if (parsed.filter) {
        const pattern = parsed.filter.toLowerCase();
        selected = selected.filter((path) => path.toLowerCase().includes(pattern));
    }
    if (Number.isFinite(parsed.maxFiles) && parsed.maxFiles > 0) {
        selected = selected.slice(0, parsed.maxFiles);
    }
    if (!parsed.filter) {
        for (const requiredPath of CANARY_ALWAYS_INCLUDE) {
            if (!selected.includes(requiredPath) && allStepFiles.includes(requiredPath)) {
                selected.push(requiredPath);
            }
        }
        selected.sort();
    }

    const models = selected.map((path) => ({
        name: path.replace('step-examples/', ''),
        path,
        category: categorizePath(path),
    }));

    return { models, excluded };
}

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
        filter: parsed.filter,
        maxFiles: parsed.maxFiles,
        prewarm: parsed.prewarm !== false,
    };

    let models = [];
    let excluded = [...EXCLUDED_MODELS];
    if (parsed.suite === 'canary') {
        const canary = buildCanaryModels(parsed);
        models = canary.models;
        excluded = [...excluded, ...canary.excluded];
    } else {
        models = [...suiteConfig.models];
    }

    if (parsed.filter && parsed.suite !== 'canary') {
        const pattern = parsed.filter.toLowerCase();
        models = models.filter((m) => (`${m.name} ${m.path}`).toLowerCase().includes(pattern));
    }

    return { cfg, models, excluded };
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

function speedComparisonFromRatio(ratio) {
    if (!Number.isFinite(ratio) || ratio <= 0) {
        return {
            isFaster: null,
            magnitude: null,
            direction: 'n/a',
            label: 'n/a',
        };
    }
    const isFaster = ratio >= 1;
    const magnitude = isFaster ? ratio : (1 / ratio);
    const direction = isFaster ? 'faster' : 'slower';
    return {
        isFaster,
        magnitude,
        direction,
        label: `${magnitude.toFixed(2)}x ${direction}`,
    };
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

function printSuiteHeader(config, models, excluded) {
    log('\n' + '='.repeat(120), 'blue');
    log(` Representative Benchmark (${config.suite})`, 'bold');
    log(` ${config.suiteDescription}`, 'cyan');
    log(` runs=${config.benchmarkRuns}, warmup=${config.warmupRuns}, timeout=${config.timeoutMs}ms, models=${models.length}`, 'dim');
    log(` prewarm=${config.prewarm ? 'on' : 'off'}`, 'dim');
    log('='.repeat(120), 'blue');

    log('\nExcluded from routine suites:', 'yellow');
    for (const ex of excluded) {
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
    const speed = speedComparisonFromRatio(result.speedup);
    const speedLabel = speed.label;
    const speedColor = speed.isFaster ? 'green' : 'red';

    log(
        `PASS  ${result.name} | ours=${ours.toFixed(1)}ms | ref=${ref.toFixed(1)}ms | ${speedLabel} | tris=${result.ours.triangleCount}/${result.ref.triangleCount}`,
        speedColor,
    );

    if (speed.isFaster === false) {
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
    log('\n' + '-'.repeat(164), 'dim');
    const header = [
        'Model'.padEnd(34),
        'SizeKB'.padStart(8),
        'Ours(ms)'.padStart(10),
        'Ref(ms)'.padStart(10),
        'Speed(vsRef)'.padStart(16),
        'OursTris'.padStart(10),
        'RefTris'.padStart(10),
        'TriRatio'.padStart(10),
    ].join(' | ');
    log(header, 'cyan');
    log('-'.repeat(164), 'dim');

    for (const r of results) {
        if (!r.success) {
            const row = [
                r.name.padEnd(34),
                ((r.fileSize || 0) / 1024).toFixed(1).padStart(8),
                'FAILED'.padStart(10),
                ''.padStart(10),
                ''.padStart(16),
                ''.padStart(10),
                ''.padStart(10),
                ''.padStart(10),
            ].join(' | ');
            log(row, 'red');
            continue;
        }

        const speed = speedComparisonFromRatio(r.speedup);

        const row = [
            r.name.padEnd(34),
            (r.fileSize / 1024).toFixed(1).padStart(8),
            r.ours.avgMs.toFixed(1).padStart(10),
            r.ref.avgMs.toFixed(1).padStart(10),
            speed.label.padStart(16),
            String(r.ours.triangleCount).padStart(10),
            String(r.ref.triangleCount).padStart(10),
            (r.triangleRatio ?? 0).toFixed(2).padStart(10),
        ].join(' | ');

        log(row, speed.isFaster ? 'green' : 'red');
    }

    log('-'.repeat(164), 'dim');
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
    const medianSpeed = speedComparisonFromRatio(speedMedian);
    const p90Speed = speedComparisonFromRatio(speedP90);
    log(`  speedup median: ${medianSpeed.label}`, medianSpeed.isFaster ? 'green' : 'red');
    log(`  speedup p90: ${p90Speed.label}`, p90Speed.isFaster ? 'green' : 'red');
    log(`  ours avg runtime: ${formatMs(mean(oursTimes))} (p90 ${formatMs(percentile(oursTimes, 90))})`, 'cyan');
    log(`  ref avg runtime: ${formatMs(mean(refTimes))} (p90 ${formatMs(percentile(refTimes, 90))})`, 'cyan');
}

function evaluateCanaryQualityGuards(results) {
    const violations = [];
    const byPath = new Map(results.map((r) => [r.path, r]));
    for (const guard of CANARY_QUALITY_GUARDS) {
        const result = byPath.get(guard.path);
        if (!result) {
            violations.push(`${guard.path}: missing from canary selection (${guard.reason})`);
            continue;
        }
        if (!result.success) {
            violations.push(`${guard.path}: benchmark failed (${guard.reason})`);
            continue;
        }
        const oursTris = result.ours?.triangleCount ?? 0;
        const refTris = result.ref?.triangleCount ?? 0;
        const triRatio = refTris > 0 ? oursTris / refTris : null;

        if (Number.isFinite(guard.minTriangles) && oursTris < guard.minTriangles) {
            violations.push(
                `${guard.path}: triangles ${oursTris} < ${guard.minTriangles} (${guard.reason})`
            );
        }
        if (Number.isFinite(guard.minTriRatio) && triRatio !== null && triRatio < guard.minTriRatio) {
            violations.push(
                `${guard.path}: triRatio ${triRatio.toFixed(3)} < ${guard.minTriRatio} (${guard.reason})`
            );
        }
    }
    return violations;
}

function saveResultsJson(config, models, results, excluded) {
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
        excluded,
        results,
    };

    const outPath = join(PROJECT_ROOT, 'tests', 'benchmark-results.json');
    fs.writeFileSync(outPath, JSON.stringify(output, null, 2));
    log(`\nResults saved to ${outPath}`, 'dim');
}

function getElectronicEnclosureResult(results) {
    return results.find((r) => r.success && (r.path === ELECTRONIC_ENCLOSURE_PATH || r.name === 'Electronic Enclosure')) || null;
}

function summarizeLoadShare(phases, oursMs) {
    if (!phases || !Number.isFinite(oursMs) || oursMs <= 0) {
        return { loadMs: null, loadSharePct: null };
    }
    const loadMs = Number(phases.loadStepFile || 0);
    return {
        loadMs,
        loadSharePct: (loadMs / oursMs) * 100,
    };
}

function readLastEnclosureTrendRecord() {
    if (!fs.existsSync(ENCLOSURE_TREND_PATH)) return null;
    const content = fs.readFileSync(ENCLOSURE_TREND_PATH, 'utf8');
    const lines = content
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);
    for (let i = lines.length - 1; i >= 0; i--) {
        try {
            return JSON.parse(lines[i]);
        } catch (_) {
            // Ignore malformed lines and keep scanning backward.
        }
    }
    return null;
}

function saveElectronicEnclosureTrend(config, result) {
    if (!result?.success) return;

    const nowIso = new Date().toISOString();
    const load = summarizeLoadShare(result.ours?.phases, result.ours?.avgMs);
    const record = {
        timestamp: nowIso,
        suite: config.suite,
        oursMs: result.ours.avgMs,
        refMs: result.ref.avgMs,
        speedup: result.speedup,
        oursTris: result.ours.triangleCount,
        refTris: result.ref.triangleCount,
        triRatio: result.triangleRatio,
        loadStepFileMs: load.loadMs,
        loadSharePct: load.loadSharePct,
    };

    const previous = readLastEnclosureTrendRecord();
    fs.appendFileSync(ENCLOSURE_TREND_PATH, `${JSON.stringify(record)}\n`);

    log(`Electronic Enclosure trend saved: ${ENCLOSURE_TREND_PATH}`, 'dim');
    if (!previous) {
        const baselineSpeed = speedComparisonFromRatio(record.speedup);
        log(
            `Electronic Enclosure baseline: ours=${record.oursMs.toFixed(1)}ms, ref=${record.refMs.toFixed(1)}ms, speed=${baselineSpeed.label}`,
            'cyan',
        );
        return;
    }

    const deltaOurs = record.oursMs - previous.oursMs;
    const deltaRef = record.refMs - previous.refMs;
    const oursDir = deltaOurs <= 0 ? 'faster' : 'slower';
    const deltaTri = (record.triRatio ?? 0) - (previous.triRatio ?? 0);
    const currentSpeed = speedComparisonFromRatio(record.speedup);
    const previousSpeed = speedComparisonFromRatio(previous.speedup);
    const speedImprovementRatio = (Number.isFinite(previous.speedup) && previous.speedup > 0)
        ? (record.speedup / previous.speedup)
        : null;
    const speedDeltaText = Number.isFinite(speedImprovementRatio)
        ? (speedImprovementRatio >= 1
            ? `improved by ${speedImprovementRatio.toFixed(3)}x`
            : `regressed by ${(1 / speedImprovementRatio).toFixed(3)}x`)
        : 'change unavailable';
    const trendColor = Number.isFinite(speedImprovementRatio) && speedImprovementRatio >= 1 ? 'green' : 'yellow';
    log(
        `Electronic Enclosure vs previous (${previous.timestamp}): ours ${oursDir} by ${Math.abs(deltaOurs).toFixed(1)}ms, speed ${speedDeltaText} (${currentSpeed.label}, was ${previousSpeed.label}), triRatio Δ${deltaTri.toFixed(3)}, ref Δ${deltaRef.toFixed(1)}ms`,
        trendColor,
    );
}

async function main() {
    const parsed = parseArgs(args);
    const { cfg: config, models, excluded } = selectConfig(parsed);

    if (models.length === 0) {
        throw new Error('No benchmark models selected. Check --filter or suite.');
    }

    printSuiteHeader(config, models, excluded);

    let viteProcess = null;
    let vitePort = config.vitePort;
    let browser = null;
    const results = [];
    let hadFatalError = false;
    let qualityViolations = [];

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
        if (config.suite === 'canary' && !config.filter && !Number.isFinite(config.maxFiles)) {
            qualityViolations = evaluateCanaryQualityGuards(results);
            if (qualityViolations.length > 0) {
                log('\nCanary Quality Gate: FAIL', 'red');
                for (const violation of qualityViolations) {
                    log(`  - ${violation}`, 'red');
                }
            } else {
                log('\nCanary Quality Gate: PASS', 'green');
            }
        }
        saveResultsJson(config, models, results, excluded);
        const enclosureResult = getElectronicEnclosureResult(results);
        if (enclosureResult) {
            saveElectronicEnclosureTrend(config, enclosureResult);
        }
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

    const hasFailure = results.some((r) => !r.success) || qualityViolations.length > 0;
    process.exit(hadFatalError || hasFailure ? 1 : 0);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
