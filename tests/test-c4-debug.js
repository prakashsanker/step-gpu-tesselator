/**
 * Debug script to check unit-box parsing
 */

import puppeteer from 'puppeteer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

async function main() {
    console.log('\n=== Debug: unit-box.step ===\n');

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

    // Capture ALL console logs
    page.on('console', msg => {
        console.log(`[Browser] ${msg.text()}`);
    });

    await page.goto('http://localhost:5173/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 30000,
    });

    await page.waitForFunction(() => window.testHarnessReady === true, { timeout: 30000 });

    const stepContent = fs.readFileSync(join(PROJECT_ROOT, 'step-examples/c4-multiface/unit-box.step'), 'utf8');

    await page.evaluate(async (stepText) => {
        try {
            await window.testHarness.parseStep(stepText);
        } catch (e) {
            console.error('Parse error:', e.message);
        }
    }, stepContent);

    await browser.close();
}

main().catch(console.error);
