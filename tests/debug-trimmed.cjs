const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    console.log('Starting Vite server...');
    const vite = spawn('npx', ['vite', '--port', '5197'], {
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
    page.on('console', msg => console.log('[Browser]', msg.text()));

    await page.goto('http://localhost:5197/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    await page.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    // Test trimmed surfaces
    const trimmedDir = path.join(projectRoot, 'step-examples/c6-trimmed');
    const files = fs.readdirSync(trimmedDir).filter(f => f.endsWith('.step'));

    console.log(`\n=== Testing ${files.length} trimmed surface files ===\n`);

    for (const file of files) {
        const stepFile = path.join(trimmedDir, file);
        const stepContent = fs.readFileSync(stepFile, 'utf-8');

        const result = await page.evaluate(async (stepText) => {
            return await window.testHarness.parseStep(stepText);
        }, stepContent);

        if (result.success) {
            console.log(`✓ ${file}: ${result.mesh.vertexCount} vertices, ${result.mesh.triangleCount} triangles`);
        } else {
            console.log(`✗ ${file}: ${result.error}`);
        }
    }

    await browser.close();
    vite.kill();
}

main().catch(e => { console.error(e); process.exit(1); });
