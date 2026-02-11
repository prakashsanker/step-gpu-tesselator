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
    viteHost: '127.0.0.1',
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

        const vite = spawn('npx', ['vite', '--host', CONFIG.viteHost, '--port', CONFIG.vitePort.toString()], {
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
                await window.benchmark.runGPUBatched(base64);
                await window.benchmark.runGPUHybrid(base64);
                await window.benchmark.runOCCT(base64);
            }, stepBase64);
        }

        // Benchmark runs
        const gpuTimes = [];
        const gpuSingleTimes = [];
        const gpuOptimizedTimes = [];
        const gpuBatchedTimes = [];
        const gpuHybridTimes = [];
        const occtTimes = [];
        let gpuResult, gpuSingleResult, gpuOptimizedResult, gpuBatchedResult, gpuHybridResult, occtResult;

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

            // GPU implementation (optimized - parallel workgroup)
            gpuOptimizedResult = await page.evaluate(async (base64) => {
                return await window.benchmark.runGPUOptimized(base64);
            }, stepBase64);

            if (gpuOptimizedResult.success) {
                gpuOptimizedTimes.push(gpuOptimizedResult.totalTime);
            }

            // GPU implementation (batched - single dispatch for all faces)
            gpuBatchedResult = await page.evaluate(async (base64) => {
                return await window.benchmark.runGPUBatched(base64);
            }, stepBase64);

            if (gpuBatchedResult.success) {
                gpuBatchedTimes.push(gpuBatchedResult.totalTime);
            }

            // GPU implementation (hybrid - GPU batched + CPU fallback for large polygons)
            gpuHybridResult = await page.evaluate(async (base64) => {
                return await window.benchmark.runGPUHybrid(base64);
            }, stepBase64);

            if (gpuHybridResult.success) {
                gpuHybridTimes.push(gpuHybridResult.totalTime);
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
        const gpuOptimizedAvg = gpuOptimizedTimes.length > 0 ? gpuOptimizedTimes.reduce((a, b) => a + b) / gpuOptimizedTimes.length : null;
        const gpuBatchedAvg = gpuBatchedTimes.length > 0 ? gpuBatchedTimes.reduce((a, b) => a + b) / gpuBatchedTimes.length : null;
        const gpuHybridAvg = gpuHybridTimes.length > 0 ? gpuHybridTimes.reduce((a, b) => a + b) / gpuHybridTimes.length : null;
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

        if (gpuOptimizedAvg !== null) {
            log(`    GPU (optimized):`, 'bold');
            log(`      Total time: ${formatTime(gpuOptimizedAvg)}`);
            log(`      Triangulation: ${formatTime(gpuOptimizedResult.triangulationTime)}`);
            log(`      Triangles: ${gpuOptimizedResult.triangleCount}`);
        } else {
            log(`    GPU (optimized): FAILED - ${gpuOptimizedResult?.error}`, 'red');
        }

        if (gpuBatchedAvg !== null) {
            log(`    GPU (batched):`, 'cyan');
            log(`      Total time: ${formatTime(gpuBatchedAvg)}`);
            log(`      Triangulation: ${formatTime(gpuBatchedResult.triangulationTime)}`);
            log(`      Triangles: ${gpuBatchedResult.triangleCount}`);
        } else {
            log(`    GPU (batched): FAILED - ${gpuBatchedResult?.error}`, 'red');
        }

        if (gpuHybridAvg !== null) {
            log(`    GPU (hybrid):`, 'bold');
            log(`      Total time: ${formatTime(gpuHybridAvg)}`);
            log(`      Triangulation: ${formatTime(gpuHybridResult.triangulationTime)}`);
            log(`      Triangles: ${gpuHybridResult.triangleCount}`);
            if (gpuHybridResult.timing) {
                log(`      Breakdown: parse=${formatTime(gpuHybridResult.timing.stepParsing)}, extract=${formatTime(gpuHybridResult.timing.faceExtraction)}, bridge=${formatTime(gpuHybridResult.timing.bridging)}, tri=${formatTime(gpuHybridResult.timing.gpuTriangulation)}`);
            }
        } else {
            log(`    GPU (hybrid): FAILED - ${gpuHybridResult?.error}`, 'red');
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
        if (gpuOptimizedAvg !== null && occtAvg !== null) {
            const speedup = occtAvg / gpuOptimizedAvg;
            log(`      Optimized: ${formatSpeedup(speedup)}`);
        }
        if (gpuBatchedAvg !== null && occtAvg !== null) {
            const speedup = occtAvg / gpuBatchedAvg;
            log(`      Batched: ${formatSpeedup(speedup)}`);
        }
        if (gpuHybridAvg !== null && occtAvg !== null) {
            const speedup = occtAvg / gpuHybridAvg;
            log(`      Hybrid: ${formatSpeedup(speedup)}`);
        }
        if (gpuAvg !== null && gpuBatchedAvg !== null) {
            const improvement = gpuAvg / gpuBatchedAvg;
            log(`    Batched vs Multi: ${improvement.toFixed(2)}x ${improvement > 1 ? 'faster' : 'slower'}`);
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
            gpuOptimized: gpuOptimizedAvg !== null ? {
                totalTime: gpuOptimizedAvg,
                triangulationTime: gpuOptimizedResult.triangulationTime,
                triangleCount: gpuOptimizedResult.triangleCount,
            } : null,
            gpuBatched: gpuBatchedAvg !== null ? {
                totalTime: gpuBatchedAvg,
                triangulationTime: gpuBatchedResult.triangulationTime,
                triangleCount: gpuBatchedResult.triangleCount,
            } : null,
            gpuHybrid: gpuHybridAvg !== null ? {
                totalTime: gpuHybridAvg,
                triangulationTime: gpuHybridResult.triangulationTime,
                triangleCount: gpuHybridResult.triangleCount,
                timing: gpuHybridResult.timing,
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
    log(`\n${'═'.repeat(140)}`, 'blue');
    log(`  BENCHMARK SUMMARY`, 'bold');
    log(`${'═'.repeat(140)}`, 'blue');

    console.log('\n');
    console.log('  File                  │ Batched     │ Hybrid      │ OCCT        │ Hybrid vs OCCT  │ Winner');
    console.log('  ──────────────────────┼─────────────┼─────────────┼─────────────┼─────────────────┼────────────');

    for (const r of results) {
        const name = r.name.padEnd(20);
        const batchedTime = r.gpuBatched ? formatTime(r.gpuBatched.totalTime).padEnd(11) : 'N/A'.padEnd(11);
        const hybridTime = r.gpuHybrid ? formatTime(r.gpuHybrid.totalTime).padEnd(11) : 'N/A'.padEnd(11);
        const occtTime = r.occt ? formatTime(r.occt.totalTime).padEnd(11) : 'N/A'.padEnd(11);

        let hybridVsOcct = 'N/A'.padEnd(15);
        let winner = 'N/A'.padEnd(10);
        if (r.gpuHybrid && r.occt) {
            const ratio = r.occt.totalTime / r.gpuHybrid.totalTime;
            hybridVsOcct = (ratio >= 1 ? `${ratio.toFixed(2)}x faster` : `${(1/ratio).toFixed(2)}x slower`).padEnd(15);
            winner = (ratio >= 1 ? `${colors.green}GPU${colors.reset}` : `${colors.yellow}OCCT${colors.reset}`).padEnd(20);
        }

        console.log(`  ${name} │ ${batchedTime} │ ${hybridTime} │ ${occtTime} │ ${hybridVsOcct} │ ${winner}`);
    }

    console.log('\n');

    // Summary stats
    const validHybrid = results.filter(r => r.gpuHybrid && r.occt);
    if (validHybrid.length > 0) {
        const avgVsOcct = validHybrid.reduce((sum, r) => sum + r.occt.totalTime / r.gpuHybrid.totalTime, 0) / validHybrid.length;
        log(`  Hybrid GPU vs OCCT average: ${formatSpeedup(avgVsOcct)}`, 'bold');

        const gpuWins = validHybrid.filter(r => r.gpuHybrid.totalTime < r.occt.totalTime).length;
        log(`  GPU wins: ${gpuWins}/${validHybrid.length} benchmarks`, 'bold');
    }

    const validBatched = results.filter(r => r.gpuBatched && r.gpuHybrid);
    if (validBatched.length > 0) {
        const avgImprovement = validBatched.reduce((sum, r) => sum + r.gpuBatched.totalTime / r.gpuHybrid.totalTime, 0) / validBatched.length;
        if (avgImprovement > 1.01) {
            log(`  Hybrid vs Batched (large polygon improvement): ${avgImprovement.toFixed(2)}x`, 'bold');
        }
    }

    log(`${'═'.repeat(140)}\n`, 'blue');
}

async function main() {
    log('\n' + '═'.repeat(80), 'blue');
    log('  GPU Ear Clipping vs occt-import-js Benchmark', 'bold');
    log('═'.repeat(80) + '\n', 'blue');

    let viteProcess = null;
    let browser = null;
    let hadFatalError = false;

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
        await page.goto(`http://${CONFIG.viteHost}:${CONFIG.vitePort}/tests/benchmark.html`, {
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
        hadFatalError = true;
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
    process.exit(hadFatalError ? 1 : 0);
}

main();
