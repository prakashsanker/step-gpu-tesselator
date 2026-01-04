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
        { name: 'Small (4 holes)', path: 'step-examples/benchmark/plate-small-2x2.step' },
        { name: 'Medium (25 holes)', path: 'step-examples/benchmark/plate-medium-5x5.step' },
        { name: 'Large (100 holes)', path: 'step-examples/benchmark/plate-large-10x10.step' },
        { name: 'XLarge (400 holes)', path: 'step-examples/benchmark/plate-xlarge-20x20.step' },
        { name: 'XXLarge (900 holes)', path: 'step-examples/benchmark/plate-xxlarge-30x30.step' },
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
        const occtTimes = [];
        let gpuResult, occtResult;

        for (let i = 0; i < CONFIG.benchmarkRuns; i++) {
            log(`  Run ${i + 1}/${CONFIG.benchmarkRuns}...`, 'dim');

            // GPU implementation
            gpuResult = await page.evaluate(async (base64) => {
                return await window.benchmark.runGPU(base64);
            }, stepBase64);

            if (gpuResult.success) {
                gpuTimes.push(gpuResult.totalTime);
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
        const occtAvg = occtTimes.length > 0 ? occtTimes.reduce((a, b) => a + b) / occtTimes.length : null;

        // Output results
        log(`\n  Results:`, 'bold');

        if (gpuAvg !== null) {
            log(`    GPU Implementation:`, 'green');
            log(`      Total time: ${formatTime(gpuAvg)}`);
            log(`      Parse time: ${formatTime(gpuResult.parseTime)}`);
            log(`      Triangulation: ${formatTime(gpuResult.triangulationTime)}`);
            log(`      Triangles: ${gpuResult.triangleCount}`);
            log(`      Vertices: ${gpuResult.vertexCount}`);
        } else {
            log(`    GPU Implementation: FAILED - ${gpuResult?.error}`, 'red');
        }

        if (occtAvg !== null) {
            log(`    OCCT Implementation:`, 'yellow');
            log(`      Total time: ${formatTime(occtAvg)}`);
            log(`      Triangles: ${occtResult.triangleCount}`);
            log(`      Vertices: ${occtResult.vertexCount}`);
        } else {
            log(`    OCCT Implementation: FAILED - ${occtResult?.error}`, 'red');
        }

        if (gpuAvg !== null && occtAvg !== null) {
            const speedup = occtAvg / gpuAvg;
            log(`\n    Comparison: GPU is ${formatSpeedup(speedup)}`);
        }

        results.push({
            name: file.name,
            gpu: gpuAvg !== null ? {
                totalTime: gpuAvg,
                parseTime: gpuResult.parseTime,
                triangulationTime: gpuResult.triangulationTime,
                triangleCount: gpuResult.triangleCount,
                vertexCount: gpuResult.vertexCount,
            } : null,
            occt: occtAvg !== null ? {
                totalTime: occtAvg,
                triangleCount: occtResult.triangleCount,
                vertexCount: occtResult.vertexCount,
            } : null,
        });
    }

    return results;
}

function printSummary(results) {
    log(`\n${'═'.repeat(80)}`, 'blue');
    log(`  BENCHMARK SUMMARY`, 'bold');
    log(`${'═'.repeat(80)}`, 'blue');

    console.log('\n');
    console.log('  File                  │ GPU Time    │ OCCT Time   │ Speedup     │ GPU Tris  │ OCCT Tris');
    console.log('  ──────────────────────┼─────────────┼─────────────┼─────────────┼───────────┼───────────');

    for (const r of results) {
        const name = r.name.padEnd(20);
        const gpuTime = r.gpu ? formatTime(r.gpu.totalTime).padEnd(11) : 'N/A'.padEnd(11);
        const occtTime = r.occt ? formatTime(r.occt.totalTime).padEnd(11) : 'N/A'.padEnd(11);

        let speedup = 'N/A'.padEnd(11);
        if (r.gpu && r.occt) {
            const ratio = r.occt.totalTime / r.gpu.totalTime;
            speedup = (ratio >= 1 ? `${ratio.toFixed(2)}x ▲` : `${(1/ratio).toFixed(2)}x ▼`).padEnd(11);
        }

        const gpuTris = r.gpu ? String(r.gpu.triangleCount).padEnd(9) : 'N/A'.padEnd(9);
        const occtTris = r.occt ? String(r.occt.triangleCount).padEnd(9) : 'N/A'.padEnd(9);

        console.log(`  ${name} │ ${gpuTime} │ ${occtTime} │ ${speedup} │ ${gpuTris} │ ${occtTris}`);
    }

    console.log('\n');

    // Overall speedup
    const validResults = results.filter(r => r.gpu && r.occt);
    if (validResults.length > 0) {
        const avgSpeedup = validResults.reduce((sum, r) => sum + r.occt.totalTime / r.gpu.totalTime, 0) / validResults.length;
        log(`  Average speedup: GPU is ${formatSpeedup(avgSpeedup)}`, 'bold');
    }

    log(`${'═'.repeat(80)}\n`, 'blue');
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
