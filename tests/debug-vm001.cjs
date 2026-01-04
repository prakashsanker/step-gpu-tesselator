const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    console.log('Starting Vite server...');
    const vite = spawn('npx', ['vite', '--port', '5203'], {
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

    await page.goto('http://localhost:5203/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });

    await page.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    const stepFile = path.join(projectRoot, 'step-examples/VM-001.STEP');
    const stepContent = fs.readFileSync(stepFile, 'utf-8');

    console.log('\n=== Parsing VM-001.STEP ===');
    const result = await page.evaluate(async (stepText) => {
        const result = await window.testHarness.parseStep(stepText);
        if (result.success) {
            const positions = result.mesh.positions;
            const indices = result.mesh.indices;
            
            // Check for NaN/Infinity
            let nanCount = 0;
            let infCount = 0;
            let minPos = [Infinity, Infinity, Infinity];
            let maxPos = [-Infinity, -Infinity, -Infinity];
            
            for (let i = 0; i < positions.length; i += 3) {
                for (let j = 0; j < 3; j++) {
                    const v = positions[i + j];
                    if (isNaN(v)) nanCount++;
                    if (!isFinite(v)) infCount++;
                    if (isFinite(v)) {
                        minPos[j] = Math.min(minPos[j], v);
                        maxPos[j] = Math.max(maxPos[j], v);
                    }
                }
            }
            
            return {
                success: true,
                vertexCount: result.mesh.vertexCount,
                triangleCount: result.mesh.triangleCount,
                nanCount,
                infCount,
                boundingBox: { min: minPos, max: maxPos },
                samplePositions: Array.from(positions.slice(0, 15)),
                sampleIndices: Array.from(indices.slice(0, 30)),
                maxIndex: Math.max(...indices),
            };
        }
        return result;
    }, stepContent);

    console.log('Result:', JSON.stringify(result, null, 2));

    await browser.close();
    vite.kill();
}

main().catch(e => { console.error(e); process.exit(1); });
