const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    const vite = spawn('npx', ['vite', '--port', '5205'], {
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
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('warn') || text.includes('Missing') || text.includes('Invalid') || text.includes('Unknown')) {
            console.log('[WARN]', text);
        }
    });
    
    await page.goto('http://localhost:5205/tests/test-harness.html', {
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
            
            // Find first NaN position
            for (let i = 0; i < positions.length; i += 3) {
                if (isNaN(positions[i]) || isNaN(positions[i+1]) || isNaN(positions[i+2])) {
                    return {
                        success: true,
                        firstNaNIndex: i / 3,
                        nanCount: [...positions].filter(v => isNaN(v)).length,
                        totalVertices: positions.length / 3,
                    };
                }
            }
            return { success: true, nanCount: 0, totalVertices: positions.length / 3 };
        }
        return result;
    }, stepContent);

    console.log('Result:', JSON.stringify(result, null, 2));

    await browser.close();
    vite.kill();
}

main().catch(e => { console.error(e); process.exit(1); });
