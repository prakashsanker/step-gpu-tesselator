const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');

(async () => {
    // Start Vite server
    const vite = spawn('npx', ['vite', '--port', '5198'], { stdio: 'pipe' });
    await new Promise(r => setTimeout(r, 3000));

    const browser = await puppeteer.launch({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--use-gl=swiftshader']
    });

    const page = await browser.newPage();
    page.on('console', msg => console.log('[BROWSER]', msg.text()));

    await page.goto('http://localhost:5198/tests/test-harness.html', { waitUntil: 'networkidle0' });
    await page.waitForFunction(() => window.testHarnessReady === true, { timeout: 10000 });

    // Get the STEP content
    const stepContent = fs.readFileSync('step-examples/basics/square-with-triangle-hole.step', 'utf-8');

    // Parse face bounds to see raw data
    const bounds = await page.evaluate((stepText) => {
        return window.testHarness.parseFaceBounds(stepText);
    }, stepContent);

    console.log('\n=== FACE BOUNDS ===');
    console.log('Outer vertices:', bounds.outerVertexCount);
    console.log('Hole count:', bounds.holeCount);
    if (bounds.outer) {
        console.log('Outer loop:');
        bounds.outer.forEach((p, i) => console.log(`  [${i}] (${p[0]}, ${p[1]}, ${p[2]})`));
    }
    if (bounds.holes && bounds.holes.length > 0) {
        bounds.holes.forEach((hole, h) => {
            console.log(`Hole ${h}:`);
            hole.forEach((p, i) => console.log(`  [${i}] (${p[0]}, ${p[1]}, ${p[2]})`));
        });
    }

    // Test winding
    const winding = await page.evaluate((stepText) => {
        return window.testHarness.testWinding(stepText);
    }, stepContent);

    console.log('\n=== WINDING ===');
    console.log('Raw outer area:', winding.rawOuterArea);
    console.log('Raw hole areas:', winding.rawHoleAreas);
    console.log('Normalized outer area:', winding.normalizedOuterArea);
    console.log('Normalized hole areas:', winding.normalizedHoleAreas);
    console.log('Outer reversed:', winding.outerReversed);
    console.log('Holes reversed:', winding.holesReversed);

    // Test topology
    const topology = await page.evaluate((stepText) => {
        return window.testHarness.testTopology(stepText);
    }, stepContent);

    console.log('\n=== TOPOLOGY ===');
    console.log('Valid:', topology.valid);
    if (topology.errors && topology.errors.length > 0) {
        console.log('Errors:', topology.errors);
    }

    // Try to parse full mesh - this will fail due to WebGPU, but we can see the bridging logs
    console.log('\n=== ATTEMPTING FULL PARSE (may fail on WebGPU) ===');
    const result = await page.evaluate(async (stepText) => {
        try {
            const mesh = await window.testHarness.parseStepToMesh(stepText);
            return {
                success: true,
                mesh: {
                    positions: Array.from(mesh.positions),
                    indices: Array.from(mesh.indices),
                    vertexCount: mesh.positions.length / 3,
                    triangleCount: mesh.indices.length / 3
                }
            };
        } catch (e) {
            return { success: false, error: e.message };
        }
    }, stepContent);

    console.log('\n=== RESULT ===');
    if (result.success) {
        console.log('Success!');
        console.log('Vertex count:', result.mesh.vertexCount);
        console.log('Triangle count:', result.mesh.triangleCount);

        console.log('\nVertices:');
        for (let i = 0; i < result.mesh.vertexCount; i++) {
            const x = result.mesh.positions[i * 3];
            const y = result.mesh.positions[i * 3 + 1];
            const z = result.mesh.positions[i * 3 + 2];
            console.log(`  [${i}] (${x}, ${y}, ${z})`);
        }

        console.log('\nTriangles:');
        for (let i = 0; i < result.mesh.triangleCount; i++) {
            const a = result.mesh.indices[i * 3];
            const b = result.mesh.indices[i * 3 + 1];
            const c = result.mesh.indices[i * 3 + 2];
            console.log(`  [${i}] ${a} → ${b} → ${c}`);
        }
    } else {
        console.log('Failed:', result.error);
    }

    await browser.close();
    vite.kill();
    process.exit(0);
})();
