/**
 * Test if occt-import-js can parse the rocky_house files
 */
import puppeteer from 'puppeteer';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

async function main() {
    const testFiles = [
        'step-examples/complex/rocky_house_table.step',
        'step-examples/complex/air.step',
        'step-examples/c4-surfaces/sphere.step',  // control
    ];

    const browser = await puppeteer.launch({
        headless: true,
        args: ['--no-sandbox'],
    });

    const page = await browser.newPage();

    page.on('console', msg => {
        console.log(`[Browser] ${msg.text()}`);
    });

    // Navigate to benchmark page which has occt-import-js loaded
    await page.goto('http://localhost:5176/tests/benchmark-comprehensive.html', {
        waitUntil: 'networkidle0',
        timeout: 60000
    });

    // Wait for occt-import-js to be ready
    await page.waitForFunction(() => window.benchmarkReady === true, { timeout: 60000 });

    console.log('occt-import-js ready');

    for (const testFile of testFiles) {
        console.log(`\n=== Testing: ${testFile} ===`);

        try {
            const content = fs.readFileSync(join(PROJECT_ROOT, testFile), 'utf8');

            const result = await page.evaluate(async (stepContent) => {
                try {
                    const result = await window.benchmark.runOcctImport(stepContent);
                    return result;
                } catch (e) {
                    return { error: e.message };
                }
            }, content);

            if (result.error) {
                console.log(`  ERROR: ${result.error}`);
            } else {
                console.log(`  Success: ${result.vertexCount} vertices, ${result.triangleCount} triangles, ${result.totalTime?.toFixed(0)}ms`);
            }
        } catch (e) {
            console.log(`  EXCEPTION: ${e.message}`);
        }
    }

    await browser.close();
}

main().catch(console.error);
