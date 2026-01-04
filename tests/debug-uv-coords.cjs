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
    page.on('console', msg => console.log('[Browser]', msg.text()));

    await page.goto('http://localhost:5199/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    await page.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    // Test half-cylinder first (simpler case)
    console.log('\n=== Testing half-cylinder.step ===');
    const halfCylinderFile = path.join(projectRoot, 'step-examples/c6-trimmed/half-cylinder.step');
    const halfCylinderContent = fs.readFileSync(halfCylinderFile, 'utf-8');

    const halfResult = await page.evaluate(async (stepText) => {
        return await window.testHarness.parseStep(stepText);
    }, halfCylinderContent);

    console.log('Half cylinder:', halfResult.success ?
        `${halfResult.mesh.vertexCount} vertices, ${halfResult.mesh.triangleCount} triangles` :
        halfResult.error);

    // Test cylinder-with-hole
    console.log('\n=== Testing cylinder-with-hole.step ===');
    const stepFile = path.join(projectRoot, 'step-examples/c6-trimmed/cylinder-with-hole.step');
    const stepContent = fs.readFileSync(stepFile, 'utf-8');

    const result = await page.evaluate(async (stepText) => {
        return await window.testHarness.parseStep(stepText);
    }, stepContent);

    console.log('Cylinder with hole:', result.success ?
        `${result.mesh.vertexCount} vertices, ${result.mesh.triangleCount} triangles` :
        result.error);

    if (result.success) {
        // Get first few vertex positions to verify they're on the cylinder
        const positions = result.mesh.positions;
        console.log('\nFirst 5 vertex positions:');
        for (let i = 0; i < Math.min(5, positions.length / 3); i++) {
            const x = positions[i * 3];
            const y = positions[i * 3 + 1];
            const z = positions[i * 3 + 2];
            const radius = Math.sqrt(x*x + y*y);
            console.log(`  v${i}: (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}) radius=${radius.toFixed(2)}`);
        }
    }

    await browser.close();
    vite.kill();
}

main().catch(e => { console.error(e); process.exit(1); });
