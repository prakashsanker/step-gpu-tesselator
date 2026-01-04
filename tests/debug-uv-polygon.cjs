const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    console.log('Starting Vite server...');
    const vite = spawn('npx', ['vite', '--port', '5200'], {
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

    await page.goto('http://localhost:5200/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    await page.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    // Test half-cylinder first (simpler - no hole)
    console.log('\n=== Testing half-cylinder.step (no hole) ===');
    const halfCylinderFile = path.join(projectRoot, 'step-examples/c6-trimmed/half-cylinder.step');
    const halfCylinderContent = fs.readFileSync(halfCylinderFile, 'utf-8');

    const halfResult = await page.evaluate(async (stepText) => {
        const result = await window.testHarness.parseStep(stepText);
        if (result.success) {
            // Get first 10 vertex positions
            const positions = result.mesh.positions;
            const verts = [];
            for (let i = 0; i < Math.min(10, positions.length / 3); i++) {
                verts.push({
                    x: positions[i * 3],
                    y: positions[i * 3 + 1],
                    z: positions[i * 3 + 2]
                });
            }
            // Get first 10 triangles
            const indices = result.mesh.indices;
            const tris = [];
            for (let i = 0; i < Math.min(10, indices.length / 3); i++) {
                tris.push([indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]]);
            }
            return {
                success: true,
                vertexCount: result.mesh.vertexCount,
                triangleCount: result.mesh.triangleCount,
                vertices: verts,
                triangles: tris
            };
        }
        return result;
    }, halfCylinderContent);

    if (halfResult.success) {
        console.log(`Vertices: ${halfResult.vertexCount}, Triangles: ${halfResult.triangleCount}`);
        console.log('\nFirst 10 3D vertex positions:');
        halfResult.vertices.forEach((v, i) => {
            const r = Math.sqrt(v.x * v.x + v.y * v.y);
            console.log(`  v${i}: (${v.x.toFixed(2)}, ${v.y.toFixed(2)}, ${v.z.toFixed(2)}) r=${r.toFixed(2)}`);
        });
        console.log('\nFirst 10 triangles:');
        halfResult.triangles.forEach((t, i) => {
            console.log(`  t${i}: [${t[0]}, ${t[1]}, ${t[2]}]`);
        });

        // Check if triangles form a fan pattern
        const firstVertexCounts = {};
        halfResult.triangles.forEach(t => {
            t.forEach(v => {
                firstVertexCounts[v] = (firstVertexCounts[v] || 0) + 1;
            });
        });
        const mostCommonVertex = Object.entries(firstVertexCounts).sort((a, b) => b[1] - a[1])[0];
        console.log(`\nMost common vertex in triangles: v${mostCommonVertex[0]} appears ${mostCommonVertex[1]} times`);
    } else {
        console.log('Error:', halfResult.error);
    }

    await browser.close();
    vite.kill();
}

main().catch(e => { console.error(e); process.exit(1); });
