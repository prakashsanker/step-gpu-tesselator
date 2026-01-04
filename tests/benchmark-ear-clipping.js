/**
 * Benchmark: Ear Clipping Algorithms
 *
 * Compares:
 * 1. Original ear clipping (N-2 GPU syncs)
 * 2. Single-dispatch ear clipping (1 GPU sync, single-threaded)
 * 3. Parallel ear clipping (log N syncs, multi-threaded)
 */

import puppeteer from 'puppeteer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

// Test polygons of increasing size
function generateConvexPolygon(n) {
    const points = [];
    for (let i = 0; i < n; i++) {
        const angle = (i / n) * Math.PI * 2;
        points.push([Math.cos(angle), Math.sin(angle)]);
    }
    return points;
}

function generateStarPolygon(n) {
    // Star with n points, alternating between radius 1 and 0.5
    const points = [];
    for (let i = 0; i < n * 2; i++) {
        const angle = (i / (n * 2)) * Math.PI * 2;
        const radius = i % 2 === 0 ? 1 : 0.5;
        points.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
    }
    return points;
}

async function main() {
    console.log('\n========================================');
    console.log('  Ear Clipping Benchmark');
    console.log('========================================\n');

    const browser = await puppeteer.launch({
        headless: true,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=swiftshader',
            '--disable-gpu-sandbox',
            '--no-sandbox',
        ],
    });

    const page = await browser.newPage();

    // Navigate to test harness
    await page.goto('http://localhost:5173/tests/benchmark-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 30000,
    });

    await page.waitForFunction(() => window.benchmarkReady === true, { timeout: 30000 });
    console.log('Benchmark harness ready\n');

    // Test sizes (optimized version limited to 256 vertices)
    const sizes = [10, 20, 50, 100, 200];

    console.log('Convex Polygon Benchmarks:');
    console.log('─'.repeat(90));
    console.log('Vertices │ Original (ms) │ Single-Dispatch (ms) │ Parallel (ms) │ Optimized (ms) │ Speedup');
    console.log('─'.repeat(90));

    for (const n of sizes) {
        const points = generateConvexPolygon(n);

        const results = await page.evaluate(async (pts) => {
            const iterations = 3;
            const timings = { original: [], singleDispatch: [], parallel: [], optimized: [] };

            for (let i = 0; i < iterations; i++) {
                // Original
                if (window.benchmarkHarness.earClipping) {
                    const start = performance.now();
                    await window.benchmarkHarness.earClipping(pts);
                    timings.original.push(performance.now() - start);
                }

                // Single-dispatch
                if (window.benchmarkHarness.earClippingSingleDispatch) {
                    const start = performance.now();
                    await window.benchmarkHarness.earClippingSingleDispatch(pts);
                    timings.singleDispatch.push(performance.now() - start);
                }

                // Parallel
                if (window.benchmarkHarness.earClippingParallel) {
                    const start = performance.now();
                    await window.benchmarkHarness.earClippingParallel(pts);
                    timings.parallel.push(performance.now() - start);
                }

                // Optimized (single-dispatch + parallel)
                if (window.benchmarkHarness.earClippingOptimized) {
                    const start = performance.now();
                    await window.benchmarkHarness.earClippingOptimized(pts);
                    timings.optimized.push(performance.now() - start);
                }
            }

            // Return median of each
            const median = arr => {
                if (arr.length === 0) return null;
                const sorted = [...arr].sort((a, b) => a - b);
                return sorted[Math.floor(sorted.length / 2)];
            };

            return {
                original: median(timings.original),
                singleDispatch: median(timings.singleDispatch),
                parallel: median(timings.parallel),
                optimized: median(timings.optimized),
            };
        }, points);

        const speedup = results.original && results.optimized
            ? (results.original / results.optimized).toFixed(2)
            : 'N/A';

        console.log(
            `${String(n).padStart(8)} │ ` +
            `${results.original?.toFixed(2).padStart(13) || 'N/A'.padStart(13)} │ ` +
            `${results.singleDispatch?.toFixed(2).padStart(20) || 'N/A'.padStart(20)} │ ` +
            `${results.parallel?.toFixed(2).padStart(13) || 'N/A'.padStart(13)} │ ` +
            `${results.optimized?.toFixed(2).padStart(14) || 'N/A'.padStart(14)} │ ` +
            `${speedup}x`
        );
    }

    console.log('─'.repeat(90));

    console.log('\n\nConcave Star Polygon Benchmarks:');
    console.log('─'.repeat(90));
    console.log('Vertices │ Original (ms) │ Single-Dispatch (ms) │ Parallel (ms) │ Optimized (ms) │ Speedup');
    console.log('─'.repeat(90));

    for (const n of [5, 10, 20, 50, 100]) {
        const points = generateStarPolygon(n);
        const numVertices = points.length;

        const results = await page.evaluate(async (pts) => {
            const iterations = 3;
            const timings = { original: [], singleDispatch: [], parallel: [], optimized: [] };

            for (let i = 0; i < iterations; i++) {
                if (window.benchmarkHarness.earClipping) {
                    const start = performance.now();
                    await window.benchmarkHarness.earClipping(pts);
                    timings.original.push(performance.now() - start);
                }

                if (window.benchmarkHarness.earClippingSingleDispatch) {
                    const start = performance.now();
                    await window.benchmarkHarness.earClippingSingleDispatch(pts);
                    timings.singleDispatch.push(performance.now() - start);
                }

                if (window.benchmarkHarness.earClippingParallel) {
                    const start = performance.now();
                    await window.benchmarkHarness.earClippingParallel(pts);
                    timings.parallel.push(performance.now() - start);
                }

                if (window.benchmarkHarness.earClippingOptimized) {
                    const start = performance.now();
                    await window.benchmarkHarness.earClippingOptimized(pts);
                    timings.optimized.push(performance.now() - start);
                }
            }

            const median = arr => {
                if (arr.length === 0) return null;
                const sorted = [...arr].sort((a, b) => a - b);
                return sorted[Math.floor(sorted.length / 2)];
            };

            return {
                original: median(timings.original),
                singleDispatch: median(timings.singleDispatch),
                parallel: median(timings.parallel),
                optimized: median(timings.optimized),
            };
        }, points);

        const speedup = results.original && results.optimized
            ? (results.original / results.optimized).toFixed(2)
            : 'N/A';

        console.log(
            `${String(numVertices).padStart(8)} │ ` +
            `${results.original?.toFixed(2).padStart(13) || 'N/A'.padStart(13)} │ ` +
            `${results.singleDispatch?.toFixed(2).padStart(20) || 'N/A'.padStart(20)} │ ` +
            `${results.parallel?.toFixed(2).padStart(13) || 'N/A'.padStart(13)} │ ` +
            `${results.optimized?.toFixed(2).padStart(14) || 'N/A'.padStart(14)} │ ` +
            `${speedup}x`
        );
    }

    console.log('─'.repeat(90));

    await browser.close();
    console.log('\nBenchmark complete.\n');
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
