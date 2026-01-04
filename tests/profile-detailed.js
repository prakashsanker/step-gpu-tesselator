/**
 * Detailed Profiler: Understand exactly where time goes in OCCT vs GPU
 *
 * Goal: Identify every bottleneck so we can beat OCCT completely
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

const CONFIG = {
    vitePort: 5175,
    timeout: 120000,
    runs: 5,
};

async function startViteServer() {
    return new Promise((resolve, reject) => {
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

async function main() {
    console.log('\n' + '='.repeat(80));
    console.log('  DETAILED PROFILER: Where Does Time Go?');
    console.log('='.repeat(80) + '\n');

    let viteProcess = null;
    let browser = null;

    try {
        viteProcess = await startViteServer();
        console.log('Vite server started\n');

        browser = await puppeteer.launch({
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

        page.on('console', msg => {
            const text = msg.text();
            if (text.includes('[Profile]') || text.includes('[StepParser]')) {
                console.log(`  ${text}`);
            }
        });

        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/profile-harness.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        await page.waitForFunction(() => window.profileReady === true, { timeout: CONFIG.timeout });
        console.log('Profile harness ready\n');

        // Test files with increasing complexity
        const testFiles = [
            { name: 'Simple (4 vertices)', path: 'step-examples/benchmark/simple-square.step' },
            { name: 'Small (4 holes, ~24 verts)', path: 'step-examples/benchmark/plate-small-2x2.step' },
            { name: 'Medium (25 holes, ~127 verts)', path: 'step-examples/benchmark/plate-medium-5x5.step' },
            { name: 'Large (100 holes, ~502 verts)', path: 'step-examples/benchmark/plate-large-10x10.step' },
        ];

        for (const file of testFiles) {
            console.log('\n' + '-'.repeat(80));
            console.log(`  ${file.name}`);
            console.log('-'.repeat(80));

            const stepContent = fs.readFileSync(join(PROJECT_ROOT, file.path));
            const stepBase64 = stepContent.toString('base64');

            // Run detailed profile
            const result = await page.evaluate(async (base64) => {
                const stepText = atob(base64);

                // Profile our implementation with detailed timing
                const gpuProfile = await window.profileGPU(stepText);

                // Profile OCCT
                const occtProfile = await window.profileOCCT(base64);

                return { gpu: gpuProfile, occt: occtProfile };
            }, stepBase64);

            console.log('\n  GPU Implementation Breakdown:');
            if (result.gpu.success) {
                console.log(`    Total:              ${result.gpu.totalTime.toFixed(2)}ms`);
                console.log(`    ├─ STEP Parsing:    ${result.gpu.parseTime.toFixed(2)}ms (${(result.gpu.parseTime/result.gpu.totalTime*100).toFixed(1)}%)`);
                console.log(`    ├─ Face Extraction: ${result.gpu.extractionTime.toFixed(2)}ms (${(result.gpu.extractionTime/result.gpu.totalTime*100).toFixed(1)}%)`);
                console.log(`    ├─ Projection/Wind: ${result.gpu.projectionTime.toFixed(2)}ms (${(result.gpu.projectionTime/result.gpu.totalTime*100).toFixed(1)}%)`);
                console.log(`    ├─ Hole Bridging:   ${result.gpu.bridgingTime.toFixed(2)}ms (${(result.gpu.bridgingTime/result.gpu.totalTime*100).toFixed(1)}%)`);
                console.log(`    ├─ Triangulation:   ${result.gpu.triangulationTime.toFixed(2)}ms (${(result.gpu.triangulationTime/result.gpu.totalTime*100).toFixed(1)}%)`);
                console.log(`    └─ Mesh Assembly:   ${result.gpu.assemblyTime.toFixed(2)}ms (${(result.gpu.assemblyTime/result.gpu.totalTime*100).toFixed(1)}%)`);
                console.log(`    Faces: ${result.gpu.faceCount}, Triangles: ${result.gpu.triangleCount}`);
            } else {
                console.log(`    FAILED: ${result.gpu.error}`);
            }

            console.log('\n  OCCT Implementation:');
            if (result.occt.success) {
                console.log(`    Total:              ${result.occt.totalTime.toFixed(2)}ms`);
                console.log(`    Meshes: ${result.occt.meshCount}, Triangles: ${result.occt.triangleCount}`);
            } else {
                console.log(`    FAILED: ${result.occt.error}`);
            }

            if (result.gpu.success && result.occt.success) {
                const ratio = result.occt.totalTime / result.gpu.totalTime;
                const status = ratio >= 1 ? `${ratio.toFixed(2)}x FASTER` : `${(1/ratio).toFixed(2)}x slower`;
                console.log(`\n  Comparison: GPU is ${status} than OCCT`);
            }
        }

    } catch (e) {
        console.error(`\nError: ${e.message}`);
        console.error(e.stack);
    } finally {
        if (browser) await browser.close();
        if (viteProcess) viteProcess.kill();
    }
}

main();
