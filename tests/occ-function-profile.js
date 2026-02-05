/**
 * OCC Per-Function Profiler
 *
 * Identifies which functions are the bottlenecks in the OCC tessellation pipeline.
 * Compares earcut vs GPU ear-clipping for different polygon sizes.
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

const CONFIG = {
    vitePort: 5177,
    timeout: 300000,
    headless: true,
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
    magenta: '\x1b[35m',
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

function formatTime(ms) {
    if (ms < 0.01) return `${(ms * 1000).toFixed(2)}µs`;
    if (ms < 1) return `${(ms * 1000).toFixed(1)}µs`;
    if (ms < 1000) return `${ms.toFixed(2)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

function formatPercent(ratio) {
    return `${(ratio * 100).toFixed(1)}%`;
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
                resolve(vite);
            }
        });

        vite.on('error', reject);
    });
}

async function launchBrowser() {
    return puppeteer.launch({
        headless: CONFIG.headless,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=swiftshader',
            '--disable-gpu-sandbox',
            '--no-sandbox',
        ],
    });
}

function loadStepFile(relativePath) {
    return fs.readFileSync(join(PROJECT_ROOT, relativePath), 'utf-8');
}

async function runDetailedProfile(page, stepContent, fileName) {
    log(`\n${'─'.repeat(70)}`, 'dim');
    log(`Profiling: ${fileName}`, 'cyan');
    log(`${'─'.repeat(70)}`, 'dim');

    const result = await page.evaluate(async (text) => {
        return await window.profileOCCDetailed(text);
    }, stepContent);

    if (!result.success) {
        log(`  FAILED: ${result.error}`, 'red');
        return null;
    }

    const t = result.timing;
    const total = result.totalTime;

    log(`\n  Total time: ${formatTime(total)}`, 'bold');
    log(`  Faces: ${t.faceCount}, Vertices: ${t.vertexCount}, Triangles: ${t.triangleCount}`, 'dim');

    log(`\n  Per-Function Breakdown:`, 'bold');

    // Sort by time
    const phases = [
        { name: 'OCC Init (cached)', time: t.occInit },
        { name: 'STEP Parsing', time: t.stepParsing },
        { name: 'Face Extraction', time: t.faceExtraction },
        { name: '  └─ Edge Extraction', time: t.edgeExtraction },
        { name: '  └─ Surface Params', time: t.surfaceParams },
        { name: 'Tessellation', time: t.tessellation.total },
        { name: '  └─ Compute Basis', time: t.tessellation.computeBasis },
        { name: '  └─ Project to 2D', time: t.tessellation.project2D },
        { name: '  └─ Normalize Winding', time: t.tessellation.normalizeWinding },
        { name: '  └─ Bridge Holes', time: t.tessellation.bridgeHoles },
        { name: '  └─ Earcut', time: t.tessellation.earcut },
    ];

    for (const phase of phases) {
        const pct = phase.time / total;
        const bar = '█'.repeat(Math.round(pct * 30));
        const timeStr = formatTime(phase.time).padStart(10);
        const pctStr = formatPercent(pct).padStart(6);

        let color = 'reset';
        if (pct > 0.3) color = 'red';
        else if (pct > 0.1) color = 'yellow';
        else if (pct < 0.01) color = 'dim';

        log(`    ${phase.name.padEnd(25)} ${timeStr} ${pctStr} ${colors[color]}${bar}${colors.reset}`);
    }

    return result;
}

async function runTriangulatorComparison(page, stepContent, fileName) {
    log(`\n${'─'.repeat(70)}`, 'dim');
    log(`Earcut vs GPU: ${fileName}`, 'magenta');
    log(`${'─'.repeat(70)}`, 'dim');

    const result = await page.evaluate(async (text) => {
        return await window.benchmarkTriangulators(text);
    }, stepContent);

    if (!result.success) {
        log(`  FAILED: ${result.error}`, 'red');
        return null;
    }

    if (result.results.length === 0) {
        log(`  No planar faces found`, 'yellow');
        return null;
    }

    // Group by vertex count ranges
    const ranges = [
        { name: '3-10 vertices', min: 3, max: 10, faces: [] },
        { name: '11-25 vertices', min: 11, max: 25, faces: [] },
        { name: '26-50 vertices', min: 26, max: 50, faces: [] },
        { name: '51-100 vertices', min: 51, max: 100, faces: [] },
        { name: '101-256 vertices', min: 101, max: 256, faces: [] },
        { name: '256+ vertices', min: 257, max: Infinity, faces: [] },
    ];

    for (const face of result.results) {
        for (const range of ranges) {
            if (face.vertexCount >= range.min && face.vertexCount <= range.max) {
                range.faces.push(face);
                break;
            }
        }
    }

    log(`\n  Earcut vs GPU Ear-Clipping by Polygon Size:`, 'bold');
    log(`  ${'Range'.padEnd(20)} ${'Count'.padStart(6)} ${'Earcut'.padStart(10)} ${'GPU'.padStart(10)} ${'Winner'.padStart(12)}`);
    log(`  ${'-'.repeat(60)}`);

    for (const range of ranges) {
        if (range.faces.length === 0) continue;

        const avgEarcut = range.faces.reduce((s, f) => s + f.earcut.time, 0) / range.faces.length;
        const avgGPU = range.faces.reduce((s, f) => s + f.gpu.time, 0) / range.faces.length;
        const winner = avgEarcut < avgGPU ? 'earcut' : 'GPU';
        const speedup = avgEarcut < avgGPU
            ? `${(avgGPU / avgEarcut).toFixed(1)}x`
            : `${(avgEarcut / avgGPU).toFixed(1)}x`;

        const winnerStr = winner === 'earcut'
            ? `${colors.green}earcut ${speedup}${colors.reset}`
            : `${colors.cyan}GPU ${speedup}${colors.reset}`;

        log(`  ${range.name.padEnd(20)} ${range.faces.length.toString().padStart(6)} ${formatTime(avgEarcut).padStart(10)} ${formatTime(avgGPU).padStart(10)} ${winnerStr}`);
    }

    // Summary
    const totalEarcutTime = result.results.reduce((s, f) => s + f.earcut.time, 0);
    const totalGPUTime = result.results.reduce((s, f) => s + f.gpu.time, 0);

    log(`\n  Total across ${result.results.length} faces:`);
    log(`    Earcut: ${formatTime(totalEarcutTime)}`, 'green');
    log(`    GPU:    ${formatTime(totalGPUTime)}`, 'cyan');

    const overallWinner = totalEarcutTime < totalGPUTime ? 'Earcut' : 'GPU';
    const overallSpeedup = totalEarcutTime < totalGPUTime
        ? (totalGPUTime / totalEarcutTime).toFixed(2)
        : (totalEarcutTime / totalGPUTime).toFixed(2);

    log(`    Winner: ${overallWinner} (${overallSpeedup}x faster)`, 'bold');

    return result;
}

async function runGPUTessellationProfile(page, stepContent, fileName) {
    log(`\n${'─'.repeat(70)}`, 'dim');
    log(`GPU Tessellation Profile: ${fileName}`, 'cyan');
    log(`${'─'.repeat(70)}`, 'dim');

    const result = await page.evaluate(async (text) => {
        return await window.profileOCCWithGPUTessellation(text);
    }, stepContent);

    if (!result.success) {
        log(`  FAILED: ${result.error}`, 'red');
        return null;
    }

    const p = result.profile;
    const total = result.totalTime;

    log(`\n  Total time: ${formatTime(total)}`, 'bold');
    log(`  Vertices: ${result.vertexCount}, Triangles: ${result.triangleCount}`, 'dim');

    log(`\n  Per-Function Breakdown (GPU Tessellation):`, 'bold');

    // Build phases from profile data
    const phases = [
        { name: 'tessellateOCCShape', time: p.tessellateOCCShape.total, calls: p.tessellateOCCShape.calls },
        { name: '├─ tessellatePlanarFace', time: p.tessellatePlanarFace.total, calls: p.tessellatePlanarFace.calls },
        { name: '│  ├─ occEdgesToPolygon', time: p.occEdgesToPolygon.total, calls: p.occEdgesToPolygon.calls },
        { name: '│  ├─ computeFaceBasis', time: p.computeFaceBasisFromLoop.total, calls: p.computeFaceBasisFromLoop.calls },
        { name: '│  ├─ projectTo2D', time: p.projectFaceLoopsTo2D.total, calls: p.projectFaceLoopsTo2D.calls },
        { name: '│  ├─ normalizeWinding', time: p.normalizeWinding.total, calls: p.normalizeWinding.calls },
        { name: '│  ├─ applyWindingTo3D', time: p.applyWindingTo3D.total, calls: p.applyWindingTo3D.calls },
        { name: '│  ├─ bridgeAllHoles', time: p.bridgeAllHoles.total, calls: p.bridgeAllHoles.calls },
        { name: '│  └─ earClipping (GPU)', time: p.earClipping.total, calls: p.earClipping.calls },
        { name: '├─ tessellateCurvedFace', time: p.tessellateCurvedFace.total, calls: p.tessellateCurvedFace.calls },
        { name: '├─ computeNormals', time: p.computeNormals.total, calls: p.computeNormals.calls },
        { name: '└─ meshAssembly', time: p.meshAssembly.total, calls: p.meshAssembly.calls },
    ];

    for (const phase of phases) {
        if (phase.calls === 0) continue;

        const pct = phase.time / total;
        const bar = '█'.repeat(Math.round(pct * 30));
        const timeStr = formatTime(phase.time).padStart(10);
        const pctStr = formatPercent(pct).padStart(6);
        const callsStr = `(${phase.calls} calls)`.padStart(12);

        let color = 'reset';
        if (pct > 0.3) color = 'red';
        else if (pct > 0.1) color = 'yellow';
        else if (pct < 0.01) color = 'dim';

        log(`    ${phase.name.padEnd(25)} ${timeStr} ${pctStr} ${callsStr} ${colors[color]}${bar}${colors.reset}`);
    }

    // Print report
    log(`\n  Full Report:`);
    log(result.report);

    return result;
}

async function runOCCTImportProfile(page, stepContent, fileName) {
    log(`\n${'─'.repeat(70)}`, 'dim');
    log(`occt-import-js Baseline: ${fileName}`, 'green');
    log(`${'─'.repeat(70)}`, 'dim');

    const result = await page.evaluate(async (text) => {
        return await window.profileOCCTImport(text);
    }, stepContent);

    if (!result.success) {
        log(`  FAILED: ${result.error}`, 'red');
        return null;
    }

    log(`\n  Total time: ${formatTime(result.totalTime)}`, 'bold');
    log(`  Vertices: ${result.vertexCount}, Triangles: ${result.triangleCount}, Meshes: ${result.meshCount}`, 'dim');

    return result;
}

async function runComparison(page, stepContent, fileName) {
    log(`\n${'─'.repeat(70)}`, 'dim');
    log(`Comparison: ${fileName}`, 'cyan');
    log(`${'─'.repeat(70)}`, 'dim');

    // Run occt-import-js
    const occtResult = await page.evaluate(async (text) => {
        return await window.profileOCCTImport(text);
    }, stepContent);

    // Run our GPU tessellation
    const gpuResult = await page.evaluate(async (text) => {
        return await window.profileOCCWithGPUTessellation(text);
    }, stepContent);

    if (!occtResult.success && !gpuResult.success) {
        log(`  Both methods FAILED`, 'red');
        return null;
    }

    log(`\n  ${'Method'.padEnd(25)} ${'Time'.padStart(12)} ${'Vertices'.padStart(10)} ${'Triangles'.padStart(10)}`, 'bold');
    log(`  ${'-'.repeat(60)}`);

    if (occtResult.success) {
        log(`  ${'occt-import-js'.padEnd(25)} ${formatTime(occtResult.totalTime).padStart(12)} ${occtResult.vertexCount.toString().padStart(10)} ${occtResult.triangleCount.toString().padStart(10)}`, 'green');
    } else {
        log(`  ${'occt-import-js'.padEnd(25)} FAILED: ${occtResult.error}`, 'red');
    }

    if (gpuResult.success) {
        log(`  ${'OCC + GPU ear-clipping'.padEnd(25)} ${formatTime(gpuResult.totalTime).padStart(12)} ${gpuResult.vertexCount.toString().padStart(10)} ${gpuResult.triangleCount.toString().padStart(10)}`, 'cyan');
    } else {
        log(`  ${'OCC + GPU ear-clipping'.padEnd(25)} FAILED: ${gpuResult.error}`, 'red');
    }

    if (occtResult.success && gpuResult.success) {
        const ratio = gpuResult.totalTime / occtResult.totalTime;
        const winner = ratio < 1 ? 'OCC + GPU' : 'occt-import-js';
        const speedup = ratio < 1 ? (1 / ratio).toFixed(1) : ratio.toFixed(1);

        if (ratio < 1) {
            log(`\n  Winner: ${colors.cyan}${winner}${colors.reset} (${speedup}x faster)`, 'bold');
        } else {
            log(`\n  Winner: ${colors.green}${winner}${colors.reset} (${speedup}x faster)`, 'bold');
        }
    }

    return { occtResult, gpuResult };
}

async function main() {
    log('\n' + '═'.repeat(80), 'blue');
    log('  OCC Per-Function Profiler', 'bold');
    log('═'.repeat(80) + '\n', 'blue');

    let viteProcess = null;
    let browser = null;

    try {
        viteProcess = await startViteServer();
        log('Vite server started\n', 'green');

        browser = await launchBrowser();
        const page = await browser.newPage();

        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                log(`[Browser Error] ${msg.text()}`, 'red');
            }
        });

        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/occ-function-profile.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        await page.waitForFunction(() => window.profilerReady === true, { timeout: CONFIG.timeout });
        log('Profiler ready\n', 'green');

        // Test files
        const testFiles = [
            { name: 'Simple Square', path: 'step-examples/benchmark/simple-square.step' },
            { name: 'Small (4 holes)', path: 'step-examples/benchmark/plate-small-2x2.step' },
            { name: 'Medium (25 holes)', path: 'step-examples/benchmark/plate-medium-5x5.step' },
            { name: 'Large (100 holes)', path: 'step-examples/benchmark/plate-large-10x10.step' },
            { name: 'Unit Box', path: 'step-examples/c4-multiface/unit-box.step' },
        ];

        // Run detailed profiling
        log('\n' + '═'.repeat(80), 'blue');
        log('  PHASE 1: Per-Function Timing', 'bold');
        log('═'.repeat(80), 'blue');

        for (const file of testFiles) {
            const fullPath = join(PROJECT_ROOT, file.path);
            if (!fs.existsSync(fullPath)) {
                log(`Skipping ${file.name}: file not found`, 'yellow');
                continue;
            }

            const stepContent = loadStepFile(file.path);
            await runDetailedProfile(page, stepContent, file.name);
        }

        // Run GPU tessellation profiling (our actual pipeline)
        log('\n' + '═'.repeat(80), 'magenta');
        log('  PHASE 2: GPU Tessellation Per-Function Profiling', 'bold');
        log('═'.repeat(80), 'magenta');

        for (const file of testFiles) {
            const fullPath = join(PROJECT_ROOT, file.path);
            if (!fs.existsSync(fullPath)) continue;

            const stepContent = loadStepFile(file.path);
            await runGPUTessellationProfile(page, stepContent, file.name);
        }

        // Run comparison: occt-import-js vs OCC + GPU tessellation
        log('\n' + '═'.repeat(80), 'yellow');
        log('  PHASE 3: occt-import-js vs OCC + GPU Tessellation', 'bold');
        log('═'.repeat(80), 'yellow');

        const comparisonResults = [];
        for (const file of testFiles) {
            const fullPath = join(PROJECT_ROOT, file.path);
            if (!fs.existsSync(fullPath)) continue;

            const stepContent = loadStepFile(file.path);
            const result = await runComparison(page, stepContent, file.name);
            if (result) {
                comparisonResults.push({ name: file.name, ...result });
            }
        }

        // Summary table
        log('\n' + '═'.repeat(80), 'yellow');
        log('  SUMMARY: Total Times Comparison', 'bold');
        log('═'.repeat(80), 'yellow');

        log(`\n  ${'Model'.padEnd(25)} ${'occt-import-js'.padStart(15)} ${'OCC+GPU'.padStart(15)} ${'Ratio'.padStart(10)}`);
        log(`  ${'-'.repeat(70)}`);

        for (const r of comparisonResults) {
            if (r.occtResult?.success && r.gpuResult?.success) {
                const ratio = (r.gpuResult.totalTime / r.occtResult.totalTime).toFixed(1);
                const ratioColor = r.gpuResult.totalTime < r.occtResult.totalTime ? colors.cyan : colors.yellow;
                log(`  ${r.name.padEnd(25)} ${formatTime(r.occtResult.totalTime).padStart(15)} ${formatTime(r.gpuResult.totalTime).padStart(15)} ${ratioColor}${ratio}x${colors.reset}`);
            }
        }

        log('\n' + '═'.repeat(80), 'green');
        log('  Profiling Complete', 'bold');
        log('═'.repeat(80) + '\n', 'green');

    } catch (e) {
        log(`\nError: ${e.message}`, 'red');
        console.error(e.stack);
    } finally {
        if (browser) await browser.close();
        if (viteProcess) viteProcess.kill();
    }
}

main();
