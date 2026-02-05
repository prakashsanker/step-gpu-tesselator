/**
 * Quick diagnostic to understand why rocky_house files fail
 */
import puppeteer from 'puppeteer';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

async function main() {
    const testFile = process.argv[2] || 'step-examples/complex/rocky_house_table.step';

    console.log(`\n=== Diagnosing: ${testFile} ===\n`);

    // Read file
    const filePath = join(PROJECT_ROOT, testFile);
    const content = fs.readFileSync(filePath, 'utf8');
    console.log(`File size: ${content.length} bytes`);

    // Check schema
    const schemaMatch = content.match(/FILE_SCHEMA\(\('([^']+)'\)\)/);
    console.log(`Schema: ${schemaMatch ? schemaMatch[1] : 'Unknown'}`);

    // Count entities
    const entities = {
        ADVANCED_FACE: (content.match(/ADVANCED_FACE/g) || []).length,
        CLOSED_SHELL: (content.match(/CLOSED_SHELL/g) || []).length,
        MANIFOLD_SOLID_BREP: (content.match(/MANIFOLD_SOLID_BREP/g) || []).length,
    };
    console.log('Entities in STEP file:', entities);

    // Launch browser
    console.log('\nLaunching browser...');
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox'],
    });

    const page = await browser.newPage();

    // Capture console output
    page.on('console', msg => {
        const text = msg.text();
        // Filter for interesting messages
        if (text.includes('[') || text.includes('shape') || text.includes('Face') ||
            text.includes('label') || text.includes('Transfer') || text.includes('Error')) {
            console.log(`[Browser] ${text}`);
        }
    });

    page.on('pageerror', err => {
        console.error(`[PageError] ${err.message}`);
    });

    // Navigate to test page
    await page.goto('http://localhost:5176/tests/diagnose-failures.html', {
        waitUntil: 'networkidle0',
        timeout: 60000
    });

    // Wait for OCC to initialize
    await page.waitForFunction(() => {
        return typeof window.testFile === 'function';
    }, { timeout: 30000 });

    console.log('\nRunning diagnostic test...');

    // Run the test
    await page.evaluate((testPath) => {
        window.testFile(testPath);
    }, testFile);

    // Wait for results
    await new Promise(resolve => setTimeout(resolve, 10000));

    // Get output
    const output = await page.evaluate(() => {
        return document.getElementById('output')?.innerText || 'No output';
    });

    console.log('\n=== Results ===');
    console.log(output);

    await browser.close();
}

main().catch(console.error);
