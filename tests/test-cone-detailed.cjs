const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');

    console.log('Starting Vite server...');
    const vite = spawn('npx', ['vite', '--port', '5199'], {
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
        if (text.includes('[StepParser]') || text.includes('Face')) {
            console.log('[Browser]', text);
        }
    });

    await page.goto('http://localhost:5199/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    await page.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    const stepFile = path.join(projectRoot, 'step-examples/c4-surfaces/cone.step');
    const stepContent = fs.readFileSync(stepFile, 'utf-8');

    const result = await page.evaluate(async (stepText) => {
        return await window.testHarness.parseStep(stepText);
    }, stepContent);

    console.log('\n=== cone.step Result ===');
    console.log('Success:', result.success);
    if (result.success) {
        console.log('Vertices:', result.mesh.vertexCount);
        console.log('Triangles:', result.mesh.triangleCount);
        // For a cone with 16 angular samples:
        // - Conical surface: ~32 triangles
        // - Base cap (full circle): should add more triangles
        console.log('\nExpected: Cone surface (~32 tri) + Base cap (~14 tri) = ~46 triangles');
    } else {
        console.log('Error:', result.error);
    }

    await browser.close();
    vite.kill();
    console.log('Done.');
}

main().catch(e => { console.error(e); process.exit(1); });
