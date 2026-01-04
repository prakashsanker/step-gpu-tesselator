const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    console.log('Starting Vite server...');
    const vite = spawn('npx', ['vite', '--port', '5202'], {
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

    // Collect errors
    const errors = [];
    page.on('console', msg => {
        if (msg.type() === 'error') {
            errors.push(msg.text());
        }
    });
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('http://localhost:5202/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    await page.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    // Categories of test files from simple to complex
    const testCategories = [
        {
            name: 'Basic Surfaces (C4)',
            files: [
                'c4-surfaces/cylinder.step',
                'c4-surfaces/sphere.step',
                'c4-surfaces/cone.step',
                'c4-surfaces/torus.step',
            ]
        },
        {
            name: 'Multi-Face Models (C4)',
            files: [
                'c4-multiface/tetrahedron.step',
                'c4-multiface/pyramid.step',
                'c4-multiface/triangular-prism.step',
                'c4-multiface/wedge.step',
                'c4-multiface/unit-box.step',
            ]
        },
        {
            name: 'B-Spline Surfaces (C5)',
            files: [
                'c5-bspline/simple-bspline-surface.step',
                'c5-bspline/bspline-dome.step',
                'c5-bspline/bspline-bowl.step',
                'c5-bspline/bspline-saddle.step',
                'c5-bspline/bspline-wave.step',
            ]
        },
        {
            name: 'Trimmed Curved Surfaces (C6)',
            files: [
                'c6-trimmed/half-cylinder.step',
                'c6-trimmed/cylinder-with-hole.step',
                'c6-trimmed/cylinder-two-holes.step',
                'c6-trimmed/quarter-cylinder-hole.step',
                'c6-trimmed/full-cylinder-window.step',
                'c6-trimmed/pipe-with-porthole.step',
            ]
        },
        {
            name: 'Complex Planar Holes',
            files: [
                'c2-holes/2.5-triangulation/square-with-4-holes.step',
                'c2-holes/2.5-triangulation/rectangle-with-6-holes.step',
                'c2-holes/2.5-triangulation/hexagon-with-3-holes.step',
            ]
        },
        {
            name: 'Benchmark Plates',
            files: [
                'benchmark/plate-medium-5x5.step',
                'benchmark/plate-large-10x10.step',
                'benchmark/plate-xlarge-20x20.step',
            ]
        },
        {
            name: 'Real-World CAD Files',
            files: [
                'complex/cube.step',
                'complex/conical-surface.step',
                'complex/raw-material.step',
                'complex/nissan.step',
                'complex/air.step',
                'VM-002.STEP',
                'VM-001.STEP',
                'VM-004.STEP',
            ]
        },
    ];

    console.log('\n' + '='.repeat(70));
    console.log('  TESSELLATOR STRESS TEST');
    console.log('='.repeat(70));

    const results = [];

    for (const category of testCategories) {
        console.log(`\n${'─'.repeat(70)}`);
        console.log(`  ${category.name}`);
        console.log(`${'─'.repeat(70)}`);

        for (const relPath of category.files) {
            const stepFile = path.join(projectRoot, 'step-examples', relPath);
            const filename = path.basename(relPath);

            if (!fs.existsSync(stepFile)) {
                console.log(`  ⚠ ${filename.padEnd(35)} FILE NOT FOUND`);
                continue;
            }

            const fileSize = fs.statSync(stepFile).size;
            const fileSizeStr = fileSize > 1024 * 1024
                ? `${(fileSize / 1024 / 1024).toFixed(1)}MB`
                : fileSize > 1024
                    ? `${(fileSize / 1024).toFixed(1)}KB`
                    : `${fileSize}B`;

            const stepContent = fs.readFileSync(stepFile, 'utf-8');
            errors.length = 0; // Clear errors

            const startTime = Date.now();

            try {
                const result = await page.evaluate(async (stepText) => {
                    try {
                        return await window.testHarness.parseStep(stepText);
                    } catch (e) {
                        return { success: false, error: e.message || String(e) };
                    }
                }, stepContent);

                const elapsed = Date.now() - startTime;

                if (result.success) {
                    const verts = result.mesh.vertexCount;
                    const tris = result.mesh.triangleCount;
                    console.log(`  ✓ ${filename.padEnd(35)} ${fileSizeStr.padStart(8)} → ${String(verts).padStart(6)} verts, ${String(tris).padStart(6)} tris  (${elapsed}ms)`);
                    results.push({ file: relPath, success: true, vertices: verts, triangles: tris, time: elapsed, size: fileSize });
                } else {
                    console.log(`  ✗ ${filename.padEnd(35)} ${fileSizeStr.padStart(8)} → FAILED: ${result.error?.slice(0, 50)}`);
                    results.push({ file: relPath, success: false, error: result.error, time: elapsed, size: fileSize });
                }
            } catch (e) {
                const elapsed = Date.now() - startTime;
                console.log(`  ✗ ${filename.padEnd(35)} ${fileSizeStr.padStart(8)} → ERROR: ${e.message?.slice(0, 50)}`);
                results.push({ file: relPath, success: false, error: e.message, time: elapsed, size: fileSize });
            }
        }
    }

    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('  SUMMARY');
    console.log('='.repeat(70));

    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);

    console.log(`  Total files tested: ${results.length}`);
    console.log(`  Successful: ${successful.length}`);
    console.log(`  Failed: ${failed.length}`);

    if (successful.length > 0) {
        const totalVerts = successful.reduce((s, r) => s + r.vertices, 0);
        const totalTris = successful.reduce((s, r) => s + r.triangles, 0);
        const maxVerts = Math.max(...successful.map(r => r.vertices));
        const maxTris = Math.max(...successful.map(r => r.triangles));
        const maxFile = successful.find(r => r.triangles === maxTris);

        console.log(`\n  Total vertices generated: ${totalVerts.toLocaleString()}`);
        console.log(`  Total triangles generated: ${totalTris.toLocaleString()}`);
        console.log(`  Largest mesh: ${maxVerts.toLocaleString()} verts, ${maxTris.toLocaleString()} tris (${path.basename(maxFile.file)})`);
    }

    if (failed.length > 0) {
        console.log('\n  Failed files:');
        for (const f of failed) {
            console.log(`    - ${f.file}: ${f.error?.slice(0, 60)}`);
        }
    }

    await browser.close();
    vite.kill();
    console.log('\n' + '='.repeat(70));
}

main().catch(e => { console.error(e); process.exit(1); });
