/**
 * Quick test script to verify C4 multi-face STEP files render correctly
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

const C4_FILES = [
    { name: 'two-triangles', path: 'step-examples/c4-multiface/two-triangles.step', expectedFaces: 2 },
    { name: 'tetrahedron', path: 'step-examples/c4-multiface/tetrahedron.step', expectedFaces: 4 },
    { name: 'pyramid', path: 'step-examples/c4-multiface/pyramid.step', expectedFaces: 5 },
    { name: 'triangular-prism', path: 'step-examples/c4-multiface/triangular-prism.step', expectedFaces: 5 },
    { name: 'wedge', path: 'step-examples/c4-multiface/wedge.step', expectedFaces: 5 },
    { name: 'unit-box', path: 'step-examples/c4-multiface/unit-box.step', expectedFaces: 6 },
];

async function main() {
    console.log('\n=== Testing C4 Multi-Face STEP Files ===\n');

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

    // Capture console logs
    page.on('console', msg => {
        if (msg.type() === 'error') {
            console.log(`  [Browser Error] ${msg.text()}`);
        }
    });

    await page.goto('http://localhost:5173/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 30000,
    });

    await page.waitForFunction(() => window.testHarnessReady === true, { timeout: 30000 });
    console.log('Test harness ready\n');

    let passed = 0;
    let failed = 0;

    for (const test of C4_FILES) {
        try {
            const stepContent = fs.readFileSync(join(PROJECT_ROOT, test.path), 'utf8');

            const result = await page.evaluate(async (stepText) => {
                try {
                    const parseResult = await window.testHarness.parseStep(stepText);
                    return parseResult;
                } catch (e) {
                    return { success: false, error: e.message };
                }
            }, stepContent);

            if (!result.success) {
                console.log(`❌ ${test.name}: FAILED - ${result.error}`);
                failed++;
                continue;
            }

            const { vertexCount, triangleCount } = result.mesh;

            // Basic sanity checks
            const issues = [];

            if (vertexCount < 3) {
                issues.push(`too few vertices (${vertexCount})`);
            }

            if (triangleCount < 1) {
                issues.push(`no triangles generated`);
            }

            if (issues.length === 0) {
                console.log(`✅ ${test.name}: ${vertexCount} vertices, ${triangleCount} triangles`);
                passed++;
            } else {
                console.log(`❌ ${test.name}: ${issues.join(', ')}`);
                failed++;
            }

        } catch (e) {
            console.log(`❌ ${test.name}: Exception - ${e.message}`);
            failed++;
        }
    }

    await browser.close();

    console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`);
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
