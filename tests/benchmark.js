/**
 * Performance Benchmark: GPU Ear Clipping vs occt-import-js
 *
 * Compares:
 * - Your custom GPU-accelerated STEP parser + triangulator
 * - occt-import-js (OpenCASCADE compiled to WebAssembly)
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

// Benchmark configuration
const CONFIG = {
    vitePort: 5174,  // Different port from tests
    timeout: 120000,
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
    return fs.readFileSync(fullPath);
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
        { name: 'Simple Square (no holes)', path: 'step-examples/benchmark/simple-square.step' },
        { name: 'Small (4 holes)', path: 'step-examples/benchmark/plate-small-2x2.step' },
        { name: 'Medium (25 holes)', path: 'step-examples/benchmark/plate-medium-5x5.step' },
        { name: 'Large (100 holes)', path: 'step-examples/benchmark/plate-large-10x10.step' },
        { name: 'XLarge (400 holes)', path: 'step-examples/benchmark/plate-xlarge-20x20.step' },
        // { name: 'XXLarge (900 holes)', path: 'step-examples/benchmark/plate-xxlarge-30x30.step' },
    ];

    const results = [];

    for (const file of benchmarkFiles) {
        log(`\n${'─'.repeat(60)}`, 'dim');
        log(`Benchmarking: ${file.name}`, 'cyan');
        log(`${'─'.repeat(60)}`, 'dim');

        const stepContent = loadStepFile(file.path);
        const stepBase64 = stepContent.toString('base64');

        // Warmup runs
        log(`  Warming up (${CONFIG.warmupRuns} runs)...`, 'dim');
        for (let i = 0; i < CONFIG.warmupRuns; i++) {
            await page.evaluate(async (base64) => {
                await window.benchmark.runGPU(base64);
                await window.benchmark.runOCCT(base64);
            }, stepBase64);
        }

        // Benchmark runs
        const gpuTimes = [];
        const gpuSingleTimes = [];
        const occtTimes = [];
        let gpuResult, gpuSingleResult, occtResult;

        for (let i = 0; i < CONFIG.benchmarkRuns; i++) {
            log(`  Run ${i + 1}/${CONFIG.benchmarkRuns}...`, 'dim');

            // GPU implementation (multi-sync)
            gpuResult = await page.evaluate(async (base64) => {
                return await window.benchmark.runGPU(base64);
            }, stepBase64);

            if (gpuResult.success) {
                gpuTimes.push(gpuResult.totalTime);
            }

            // GPU implementation (single dispatch)
            gpuSingleResult = await page.evaluate(async (base64) => {
                return await window.benchmark.runGPUSingleDispatch(base64);
            }, stepBase64);

            if (gpuSingleResult.success) {
                gpuSingleTimes.push(gpuSingleResult.totalTime);
            }

            // OCCT implementation
            occtResult = await page.evaluate(async (base64) => {
                return await window.benchmark.runOCCT(base64);
            }, stepBase64);

            if (occtResult.success) {
                occtTimes.push(occtResult.totalTime);
            }
        }

        // Calculate averages
        const gpuAvg = gpuTimes.length > 0 ? gpuTimes.reduce((a, b) => a + b) / gpuTimes.length : null;
        const gpuSingleAvg = gpuSingleTimes.length > 0 ? gpuSingleTimes.reduce((a, b) => a + b) / gpuSingleTimes.length : null;
        const occtAvg = occtTimes.length > 0 ? occtTimes.reduce((a, b) => a + b) / occtTimes.length : null;

        // Output results
        log(`\n  Results:`, 'bold');

        if (gpuAvg !== null) {
            log(`    GPU (multi-sync):`, 'green');
            log(`      Total time: ${formatTime(gpuAvg)}`);
            log(`      Triangulation: ${formatTime(gpuResult.triangulationTime)}`);
            log(`      Triangles: ${gpuResult.triangleCount}`);
        } else {
            log(`    GPU (multi-sync): FAILED - ${gpuResult?.error}`, 'red');
        }

        if (gpuSingleAvg !== null) {
            log(`    GPU (single-dispatch):`, 'cyan');
            log(`      Total time: ${formatTime(gpuSingleAvg)}`);
            log(`      Triangulation: ${formatTime(gpuSingleResult.triangulationTime)}`);
            log(`      Triangles: ${gpuSingleResult.triangleCount}`);
        } else {
            log(`    GPU (single-dispatch): FAILED - ${gpuSingleResult?.error}`, 'red');
        }

        if (occtAvg !== null) {
            log(`    OCCT (WebAssembly):`, 'yellow');
            log(`      Total time: ${formatTime(occtAvg)}`);
            log(`      Triangles: ${occtResult.triangleCount}`);
        } else {
            log(`    OCCT (WebAssembly): FAILED - ${occtResult?.error}`, 'red');
        }

        // Comparison
        log(`\n    Comparisons vs OCCT:`);
        if (gpuAvg !== null && occtAvg !== null) {
            const speedup = occtAvg / gpuAvg;
            log(`      Multi-sync: ${formatSpeedup(speedup)}`);
        }
        if (gpuSingleAvg !== null && occtAvg !== null) {
            const speedup = occtAvg / gpuSingleAvg;
            log(`      Single-dispatch: ${formatSpeedup(speedup)}`);
        }
        if (gpuAvg !== null && gpuSingleAvg !== null) {
            const improvement = gpuAvg / gpuSingleAvg;
            log(`    Single vs Multi: ${improvement.toFixed(2)}x ${improvement > 1 ? 'faster' : 'slower'}`);
        }

        results.push({
            name: file.name,
            gpu: gpuAvg !== null ? {
                totalTime: gpuAvg,
                triangulationTime: gpuResult.triangulationTime,
                triangleCount: gpuResult.triangleCount,
            } : null,
            gpuSingle: gpuSingleAvg !== null ? {
                totalTime: gpuSingleAvg,
                triangulationTime: gpuSingleResult.triangulationTime,
                triangleCount: gpuSingleResult.triangleCount,
            } : null,
            occt: occtAvg !== null ? {
                totalTime: occtAvg,
                triangleCount: occtResult.triangleCount,
            } : null,
        });
    }

    return results;
}

function printSummary(results) {
    log(`\n${'═'.repeat(100)}`, 'blue');
    log(`  BENCHMARK SUMMARY`, 'bold');
    log(`${'═'.repeat(100)}`, 'blue');

    console.log('\n');
    console.log('  File                  │ Multi-Sync  │ Single-Disp │ OCCT        │ Single vs OCCT │ Improvement');
    console.log('  ──────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┼────────────');

    for (const r of results) {
        const name = r.name.padEnd(20);
        const multiTime = r.gpu ? formatTime(r.gpu.totalTime).padEnd(11) : 'N/A'.padEnd(11);
        const singleTime = r.gpuSingle ? formatTime(r.gpuSingle.totalTime).padEnd(11) : 'N/A'.padEnd(11);
        const occtTime = r.occt ? formatTime(r.occt.totalTime).padEnd(11) : 'N/A'.padEnd(11);

        let singleVsOcct = 'N/A'.padEnd(14);
        if (r.gpuSingle && r.occt) {
            const ratio = r.occt.totalTime / r.gpuSingle.totalTime;
            singleVsOcct = (ratio >= 1 ? `${ratio.toFixed(2)}x faster` : `${(1/ratio).toFixed(2)}x slower`).padEnd(14);
        }

        let improvement = 'N/A'.padEnd(10);
        if (r.gpu && r.gpuSingle) {
            const ratio = r.gpu.totalTime / r.gpuSingle.totalTime;
            improvement = `${ratio.toFixed(1)}x`.padEnd(10);
        }

        console.log(`  ${name} │ ${multiTime} │ ${singleTime} │ ${occtTime} │ ${singleVsOcct} │ ${improvement}`);
    }

    console.log('\n');

    // Summary stats
    const validSingle = results.filter(r => r.gpuSingle && r.occt);
    if (validSingle.length > 0) {
        const avgVsOcct = validSingle.reduce((sum, r) => sum + r.occt.totalTime / r.gpuSingle.totalTime, 0) / validSingle.length;
        log(`  Single-dispatch vs OCCT average: ${formatSpeedup(avgVsOcct)}`, 'bold');
    }

    const validImprovement = results.filter(r => r.gpu && r.gpuSingle);
    if (validImprovement.length > 0) {
        const avgImprovement = validImprovement.reduce((sum, r) => sum + r.gpu.totalTime / r.gpuSingle.totalTime, 0) / validImprovement.length;
        log(`  Single vs Multi-sync average improvement: ${avgImprovement.toFixed(2)}x`, 'bold');
    }

    log(`${'═'.repeat(100)}\n`, 'blue');
}

async function main() {
    log('\n' + '═'.repeat(80), 'blue');
    log('  GPU Ear Clipping vs occt-import-js Benchmark', 'bold');
    log('═'.repeat(80) + '\n', 'blue');

    let viteProcess = null;
    let browser = null;

    try {
        // Start Vite server
        viteProcess = await startViteServer();

        // Launch browser
        browser = await launchBrowser();
        const page = await browser.newPage();

        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                log(`[Browser Error] ${msg.text()}`, 'red');
            }
        });

        // Navigate to benchmark page
        log('Navigating to benchmark page...', 'blue');
        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/benchmark.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        // Wait for benchmark harness to be ready
        await page.waitForFunction(
            () => window.benchmarkReady === true,
            { timeout: CONFIG.timeout }
        );
        log('Benchmark harness ready\n', 'green');

        // Run benchmarks
        const results = await runBenchmarks(page);

        // Print summary
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
