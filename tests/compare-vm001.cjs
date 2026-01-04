const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    console.log('Starting Vite server...');
    const vite = spawn('npx', ['vite', '--port', '5220'], {
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

    const stepFile = path.join(projectRoot, 'step-examples/VM-001.STEP');
    const stepContent = fs.readFileSync(stepFile, 'utf-8');
    const stepBase64 = Buffer.from(fs.readFileSync(stepFile)).toString('base64');
    const fileSize = fs.statSync(stepFile).size;

    console.log(`\nFile: VM-001.STEP (${(fileSize / 1024).toFixed(1)} KB)\n`);
    console.log('='.repeat(60));

    // Test Custom Tessellator
    console.log('\n[1] Custom WebGPU Tessellator');
    console.log('-'.repeat(60));

    const customPage = await browser.newPage();
    await customPage.goto('http://localhost:5220/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });
    await customPage.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    const customResult = await customPage.evaluate(async (stepText) => {
        const start = performance.now();
        const result = await window.testHarness.parseStep(stepText);
        const totalTime = performance.now() - start;

        if (result.success) {
            return {
                success: true,
                vertices: result.mesh.vertexCount,
                triangles: result.mesh.triangleCount,
                totalTime,
                parseTime: result.mesh.parseTime || 0,
                triangulationTime: result.mesh.triangulationTime || 0,
            };
        }
        return { success: false, error: result.error };
    }, stepContent);

    if (customResult.success) {
        console.log(`  Vertices:     ${customResult.vertices.toLocaleString()}`);
        console.log(`  Triangles:    ${customResult.triangles.toLocaleString()}`);
        console.log(`  Total Time:   ${customResult.totalTime.toFixed(0)} ms`);
    } else {
        console.log(`  FAILED: ${customResult.error}`);
    }
    await customPage.close();

    // Test OCCT Import
    console.log('\n[2] OpenCASCADE (occt-import-js)');
    console.log('-'.repeat(60));

    const occtPage = await browser.newPage();
    occtPage.on('console', msg => console.log('  [Browser]', msg.text()));
    occtPage.on('pageerror', err => console.log('  [PageError]', err.message));

    await occtPage.goto('http://localhost:5220/tests/occt-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    // Wait a bit for module to load
    await new Promise(r => setTimeout(r, 2000));

    // Check if harness is ready
    const isReady = await occtPage.evaluate(() => {
        return typeof window.occtHarness !== 'undefined' && typeof window.occtHarness.parseStep === 'function';
    });

    if (!isReady) {
        console.log('  Harness not ready, checking window...');
        const windowKeys = await occtPage.evaluate(() => Object.keys(window).filter(k => k.includes('occt') || k.includes('harness')));
        console.log('  Window keys with occt/harness:', windowKeys);
    }

    let occtResult = { success: false, error: 'Harness not ready' };
    if (isReady) {
        occtResult = await occtPage.evaluate(async (base64Data) => {
            return await window.occtHarness.parseStep(base64Data);
        }, stepBase64);
    }

    if (occtResult.success) {
        console.log(`  Vertices:     ${occtResult.vertices.toLocaleString()}`);
        console.log(`  Triangles:    ${occtResult.triangles.toLocaleString()}`);
        console.log(`  Mesh Count:   ${occtResult.meshCount}`);
        console.log(`  WASM Init:    ${occtResult.initTime.toFixed(0)} ms`);
        console.log(`  Parse Time:   ${occtResult.parseTime.toFixed(0)} ms`);
        console.log(`  Total Time:   ${occtResult.totalTime.toFixed(0)} ms`);
    } else {
        console.log(`  FAILED: ${occtResult.error}`);
    }
    await occtPage.close();

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('COMPARISON SUMMARY');
    console.log('='.repeat(60));

    if (customResult.success && occtResult.success) {
        const speedup = occtResult.totalTime / customResult.totalTime;
        const vertDiff = ((customResult.vertices - occtResult.vertices) / occtResult.vertices * 100).toFixed(1);
        const triDiff = ((customResult.triangles - occtResult.triangles) / occtResult.triangles * 100).toFixed(1);

        console.log(`\n                    Custom      OCCT`);
        console.log(`  Vertices:      ${String(customResult.vertices).padStart(10)}  ${String(occtResult.vertices).padStart(10)}  (${vertDiff}%)`);
        console.log(`  Triangles:     ${String(customResult.triangles).padStart(10)}  ${String(occtResult.triangles).padStart(10)}  (${triDiff}%)`);
        console.log(`  Time (ms):     ${String(customResult.totalTime.toFixed(0)).padStart(10)}  ${String(occtResult.totalTime.toFixed(0)).padStart(10)}  (${speedup.toFixed(1)}x)`);
    }

    await browser.close();
    vite.kill();
    console.log('\nDone!');
}

main().catch(e => { console.error(e); process.exit(1); });
