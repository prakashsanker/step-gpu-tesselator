const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    console.log('Starting Vite server...');
    const vite = spawn('npx', ['vite', '--port', '5201'], {
        cwd: projectRoot,
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Vite timeout')), 30000);
        vite.stdout.on('data', (data) => {
            if (data.toString().includes('Local:')) {
                clearTimeout(timeout);
                resolve();
            }
        });
    });

    console.log('Launching browser...');
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--no-sandbox'],
    });

    const page = await browser.newPage();
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('Error') || text.includes('error')) {
            console.log('[Browser ERROR]', text);
        }
    });

    await page.goto('http://localhost:5201/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    await page.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    const testFiles = [
        'cylinder-with-hole.step',
        'cylinder-two-holes.step',
        'full-cylinder-window.step',
        'quarter-cylinder-hole.step',
        'pipe-with-porthole.step',
    ];

    for (const filename of testFiles) {
        console.log(`\n=== Testing ${filename} ===`);
        const stepFile = path.join(projectRoot, 'step-examples/c6-trimmed', filename);

        if (!fs.existsSync(stepFile)) {
            console.log(`  File not found: ${stepFile}`);
            continue;
        }

        const stepContent = fs.readFileSync(stepFile, 'utf-8');

        const result = await page.evaluate(async (stepText) => {
            return await window.testHarness.parseStep(stepText);
        }, stepContent);

        if (result.success) {
            console.log(`  SUCCESS: ${result.mesh.vertexCount} vertices, ${result.mesh.triangleCount} triangles`);
        } else {
            console.log(`  FAILED: ${result.error}`);
        }
    }

    await browser.close();
    vite.kill();
    console.log('\nDone!');
}

main().catch(e => { console.error(e); process.exit(1); });
