const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');

(async () => {
    const vite = spawn('npx', ['vite', '--port', '5199'], { stdio: 'pipe' });
    await new Promise(r => setTimeout(r, 3000));

    const browser = await puppeteer.launch({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--use-gl=swiftshader']
    });

    const page = await browser.newPage();
    page.on('console', msg => console.log('[BROWSER]', msg.text()));

    await page.goto('http://localhost:5199/tests/test-harness.html', { waitUntil: 'networkidle0' });
    await page.waitForFunction(() => window.testHarnessReady === true, { timeout: 10000 });

    const stepContent = fs.readFileSync('step-examples/basics/square-with-triangle-hole.step', 'utf-8');

    const result = await page.evaluate(async (stepText) => {
        return await window.testHarness.parseStep(stepText);
    }, stepContent);

    console.log('Result:', result.success ? 'SUCCESS' : 'FAILED');
    if (result.error) console.log('Error:', result.error);
    if (result.mesh) console.log('Mesh:', result.mesh.vertexCount, 'vertices,', result.mesh.triangleCount, 'triangles');

    await browser.close();
    vite.kill();
    process.exit(0);
})();
