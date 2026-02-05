/**
 * Test valid complex STEP files with occt-import-js
 */
import puppeteer from 'puppeteer';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

async function main() {
    const testFiles = [
        'step-examples/complex/nissan.step',
        'step-examples/complex/conical-surface.step',
        'step-examples/complex/cube.step',
        'step-examples/complex/rotor-201nal.step',
        // External files
        'step-examples/VM-002.STEP',
    ];

    const browser = await puppeteer.launch({
        headless: true,
        args: ['--no-sandbox'],
    });

    const page = await browser.newPage();

    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('ERR') || text.includes('Warning')) {
            console.log(`[Browser] ${text}`);
        }
    });

    await page.goto('http://localhost:5176/tests/benchmark-comprehensive.html', {
        waitUntil: 'networkidle0',
        timeout: 60000
    });

    await page.waitForFunction(() => window.benchmarkReady === true, { timeout: 60000 });
    console.log('Ready\n');

    for (const testFile of testFiles) {
        const filePath = join(PROJECT_ROOT, testFile);
        if (!fs.existsSync(filePath)) {
            console.log(`${testFile}: FILE NOT FOUND`);
            continue;
        }

        const content = fs.readFileSync(filePath, 'utf8');
        const sizeMB = (content.length / 1024 / 1024).toFixed(2);

        const result = await page.evaluate(async (stepContent) => {
            try {
                const result = await window.benchmark.runOcctImport(stepContent);
                return result;
            } catch (e) {
                return { error: e.message };
            }
        }, content);

        if (result.error) {
            console.log(`${testFile} (${sizeMB}MB): ERROR - ${result.error}`);
        } else if (result.triangleCount === 0) {
            console.log(`${testFile} (${sizeMB}MB): FAILED - 0 triangles`);
        } else {
            console.log(`${testFile} (${sizeMB}MB): OK - ${result.triangleCount} triangles, ${result.totalTime?.toFixed(0)}ms`);
        }
    }

    await browser.close();
}

main().catch(console.error);
