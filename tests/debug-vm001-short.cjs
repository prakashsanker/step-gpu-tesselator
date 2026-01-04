const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    const vite = spawn('npx', ['vite', '--port', '5204'], {
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

    const browser = await puppeteer.launch({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--no-sandbox'],
    });

    const page = await browser.newPage();
    await page.goto('http://localhost:5204/tests/test-harness.html', {
        waitUntil: 'networkidle0',
        timeout: 60000,
    });
    await page.waitForFunction(() => window.testHarness && window.testHarness.parseStep);

    const stepFile = path.join(projectRoot, 'step-examples/VM-001.STEP');
    const stepContent = fs.readFileSync(stepFile, 'utf-8');

    const result = await page.evaluate(async (stepText) => {
        const result = await window.testHarness.parseStep(stepText);
        if (result.success) {
            const positions = result.mesh.positions;
            const indices = result.mesh.indices;
            
            let nanCount = 0, infCount = 0;
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
                maxIndex: Math.max(...indices),
                positionsLength: positions.length,
                indicesLength: indices.length,
            };
        }
        return result;
    }, stepContent);

    console.log('Result:', JSON.stringify(result, null, 2));

    await browser.close();
    vite.kill();
}

main().catch(e => { console.error(e); process.exit(1); });
