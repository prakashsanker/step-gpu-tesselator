const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

async function main() {
    const projectRoot = path.join(__dirname, '..');
    const vite = spawn('npx', ['vite', '--port', '5206'], {
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
    let faceCount = 0;
    page.on('console', msg => {
        const text = msg.text();
        // Track faces being processed
        if (text.includes('Processing face')) {
            faceCount++;
            if (faceCount <= 5) console.log('[Face]', text);
        }
        if (text.includes('Planar face') || text.includes('Curved surface')) {
            if (faceCount <= 5) console.log('[Type]', text);
        }
        if (text.includes('NaN') || text.includes('undefined')) {
            console.log('[Issue]', text);
        }
    });
    
    await page.goto('http://localhost:5206/tests/test-harness.html', {
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
            
            // Find first few NaN positions and their triangle indices
            const nanVertices = [];
            for (let i = 0; i < positions.length && nanVertices.length < 5; i += 3) {
                if (isNaN(positions[i]) || isNaN(positions[i+1]) || isNaN(positions[i+2])) {
                    nanVertices.push({
                        index: i / 3,
                        values: [positions[i], positions[i+1], positions[i+2]]
                    });
                }
            }
            
            // Get first valid vertex for comparison
            let firstValid = null;
            for (let i = 0; i < positions.length; i += 3) {
                if (!isNaN(positions[i]) && !isNaN(positions[i+1]) && !isNaN(positions[i+2])) {
                    firstValid = { index: i / 3, values: [positions[i], positions[i+1], positions[i+2]] };
                    break;
                }
            }
            
            return {
                success: true,
                nanVertices,
                firstValid,
                nanCount: [...positions].filter(v => isNaN(v)).length,
                totalVertices: positions.length / 3,
            };
        }
        return result;
    }, stepContent);

    console.log('\nResult:', JSON.stringify(result, null, 2));
    console.log(`\nTotal faces processed: ${faceCount}`);

    await browser.close();
    vite.kill();
}

main().catch(e => { console.error(e); process.exit(1); });
