/**
 * AI-Powered Visual Comparison Test Runner
 *
 * Uses Claude's vision capabilities to compare renders instead of pixel matching.
 * This catches semantic differences that pixel comparison misses.
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join, relative } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

// Configuration
const CONFIG = {
    viteHost: '127.0.0.1',
    vitePort: 5176,
    timeout: 180000,
    headless: true,
    screenshotDir: join(__dirname, 'visual-results-ai'),
    // Use OpenRouter for Claude access
    openRouterModel: 'anthropic/claude-sonnet-4',
    openRouterApiKey: process.env.OPENROUTER_API_KEY,
};

// Colors for console output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    dim: '\x1b[2m',
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

/**
 * Find all STEP files
 */
function findStepFiles() {
    const stepDir = join(PROJECT_ROOT, 'step-examples');
    const files = [];

    function walk(dir) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = join(dir, entry.name);
            if (entry.isDirectory()) {
                walk(fullPath);
            } else if (entry.name.endsWith('.step') || entry.name.endsWith('.stp')) {
                files.push(fullPath);
            }
        }
    }

    walk(stepDir);
    return files.sort();
}

/**
 * Start Vite dev server
 */
async function startViteServer() {
    return new Promise((resolve, reject) => {
        log('Starting Vite dev server...', 'blue');

        const vite = spawn('npx', ['vite', '--host', CONFIG.viteHost, '--port', CONFIG.vitePort.toString()], {
            cwd: PROJECT_ROOT,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let started = false;
        const timeout = setTimeout(() => {
            if (!started) {
                vite.kill();
                reject(new Error('Vite server startup timeout'));
            }
        }, 30000);

        vite.stdout.on('data', (data) => {
            const output = data.toString();
            if (output.includes('Local:') && !started) {
                started = true;
                clearTimeout(timeout);
                log(`Vite server started on port ${CONFIG.vitePort}`, 'green');
                resolve(vite);
            }
        });

        vite.stderr.on('data', (data) => {
            const output = data.toString();
            if (output.includes('error')) {
                console.error('Vite error:', output);
            }
        });

        vite.on('error', (err) => {
            clearTimeout(timeout);
            reject(err);
        });
    });
}

/**
 * Launch browser with WebGPU support
 */
async function launchBrowser() {
    log('Launching browser with WebGPU...', 'blue');

    const browser = await puppeteer.launch({
        headless: CONFIG.headless,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=swiftshader',
            '--enable-gpu-rasterization',
            '--disable-gpu-sandbox',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--window-size=1600,900',
        ],
    });

    return browser;
}

/**
 * Convert data URL to base64
 */
function dataUrlToBase64(dataUrl) {
    return dataUrl.replace(/^data:image\/png;base64,/, '');
}

/**
 * Use Claude via OpenRouter to compare two renders
 */
async function compareWithClaude(oursBase64, refBase64, fileName) {
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${CONFIG.openRouterApiKey}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/anthropics/claude-code',
            'X-Title': 'STEP Tessellator Visual Tests',
        },
        body: JSON.stringify({
            model: CONFIG.openRouterModel,
            max_tokens: 1024,
            messages: [
                {
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: `You are comparing two 3D renders of a CAD model called "${fileName}".

Image 1 is "Our Pipeline" render.
Image 2 is the "Reference" render (known to be correct).

Compare these two renders and determine if they show the SAME 3D shape. Look for:
1. Missing geometry (holes, surfaces, features not present)
2. Wrong shape (different curvature, angles, proportions)
3. Extra geometry that shouldn't be there
4. Orientation differences

Respond with EXACTLY this format:
RESULT: PASS or FAIL
REASON: One sentence explaining why

Be strict - if there's ANY visible difference in the 3D geometry, it should FAIL.`
                        },
                        {
                            type: 'image_url',
                            image_url: {
                                url: `data:image/png;base64,${oursBase64}`,
                            },
                        },
                        {
                            type: 'image_url',
                            image_url: {
                                url: `data:image/png;base64,${refBase64}`,
                            },
                        },
                    ],
                },
            ],
        }),
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`OpenRouter API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    const text = data.choices[0].message.content;

    const resultMatch = text.match(/RESULT:\s*(PASS|FAIL)/i);
    const reasonMatch = text.match(/REASON:\s*(.+)/i);

    return {
        pass: resultMatch ? resultMatch[1].toUpperCase() === 'PASS' : false,
        reason: reasonMatch ? reasonMatch[1].trim() : text,
        rawResponse: text,
    };
}

/**
 * Test a single STEP file
 */
async function testFile(page, filePath, saveScreenshots = false) {
    const relativePath = relative(PROJECT_ROOT, filePath);
    const fileName = relativePath.replace(/\//g, '_').replace(/\.(step|stp)$/i, '');

    try {
        // Run the test using the exposed API
        await page.evaluate(async (testPath) => {
            await window.visualValidation.runTest(testPath);
        }, relativePath);

        // Wait for rendering to complete
        await new Promise(r => setTimeout(r, 500));

        // Get stats
        const stats = await page.evaluate(() => {
            return window.visualValidation.getStats();
        });

        // Parse triangle counts
        const parseTriCount = (str) => {
            if (!str || str === '-' || str === 'Failed') return null;
            const match = str.match(/(\d+)\s*tris/);
            return match ? parseInt(match[1], 10) : null;
        };

        const oursTriangles = parseTriCount(stats.occ);
        const refTriangles = parseTriCount(stats.occt);

        if (oursTriangles === null || refTriangles === null) {
            return {
                file: relativePath,
                status: 'error',
                error: oursTriangles === null ? 'Our pipeline failed' : 'Reference failed',
                ours: { triangles: oursTriangles },
                reference: { triangles: refTriangles },
            };
        }

        // Get screenshots
        const [oursDataUrl, refDataUrl] = await page.evaluate(() => {
            return [
                window.visualValidation.getCanvasDataUrl('canvas-occ'),
                window.visualValidation.getCanvasDataUrl('canvas-occt')
            ];
        });

        const oursBase64 = dataUrlToBase64(oursDataUrl);
        const refBase64 = dataUrlToBase64(refDataUrl);

        // Save screenshots if requested
        if (saveScreenshots) {
            const dir = join(CONFIG.screenshotDir, fileName);
            fs.mkdirSync(dir, { recursive: true });
            fs.writeFileSync(join(dir, 'ours.png'), Buffer.from(oursBase64, 'base64'));
            fs.writeFileSync(join(dir, 'reference.png'), Buffer.from(refBase64, 'base64'));
        }

        // Use Claude to compare
        const comparison = await compareWithClaude(oursBase64, refBase64, relativePath);

        return {
            file: relativePath,
            status: comparison.pass ? 'pass' : 'fail',
            reason: comparison.reason,
            ours: { triangles: oursTriangles },
            reference: { triangles: refTriangles },
        };

    } catch (e) {
        return {
            file: relativePath,
            status: 'error',
            error: e.message,
            ours: null,
            reference: null,
        };
    }
}

/**
 * Main test runner
 */
async function main() {
    const args = process.argv.slice(2);
    const saveScreenshots = args.includes('--save-screenshots') || args.includes('--save');
    const filterPattern = args.find(a => !a.startsWith('--'));

    log('\n========================================', 'blue');
    log('  AI Visual Comparison Test Suite', 'blue');
    log('========================================\n', 'blue');

    log(`Using Claude via OpenRouter (${CONFIG.openRouterModel}) for visual comparison`, 'dim');

    if (!CONFIG.openRouterApiKey) {
        log('ERROR: OPENROUTER_API_KEY environment variable not set', 'red');
        process.exit(1);
    }

    // Create results directory
    if (saveScreenshots) {
        fs.mkdirSync(CONFIG.screenshotDir, { recursive: true });
        log(`Screenshots will be saved to: ${CONFIG.screenshotDir}`, 'dim');
    }

    let viteProcess = null;
    let browser = null;
    const results = [];

    try {
        // Start Vite
        viteProcess = await startViteServer();

        // Launch browser
        browser = await launchBrowser();
        const page = await browser.newPage();

        await page.setViewport({ width: 1600, height: 900 });

        // Navigate to visual-validation.html
        log('Loading visual validation page...', 'blue');
        await page.goto(`http://${CONFIG.viteHost}:${CONFIG.vitePort}/tests/visual-validation.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        // Wait for initialization
        await page.waitForFunction(
            () => window.visualValidationReady === true,
            { timeout: CONFIG.timeout }
        );
        log('Visual validation harness ready\n', 'green');

        // Find STEP files
        let stepFiles = findStepFiles();
        log(`Found ${stepFiles.length} STEP files\n`, 'blue');

        // Filter if pattern provided
        if (filterPattern) {
            stepFiles = stepFiles.filter(f => f.toLowerCase().includes(filterPattern.toLowerCase()));
            log(`Filtered to ${stepFiles.length} files matching "${filterPattern}"\n`, 'yellow');
        }

        // Test each file
        let passed = 0;
        let failed = 0;
        let errors = 0;

        for (let i = 0; i < stepFiles.length; i++) {
            const file = stepFiles[i];
            const relativePath = relative(PROJECT_ROOT, file);
            const progress = `[${i + 1}/${stepFiles.length}]`;

            const result = await testFile(page, file, saveScreenshots);
            results.push(result);

            if (result.status === 'pass') {
                passed++;
                console.log(`${colors.dim}${progress}${colors.reset} ${colors.green}PASS${colors.reset} ${relativePath}`);
                console.log(`${colors.dim}       ${result.reason}${colors.reset}`);
                console.log(`${colors.dim}       Triangles: ours=${result.ours.triangles}, ref=${result.reference.triangles}${colors.reset}`);
            } else if (result.status === 'fail') {
                failed++;
                console.log(`${colors.dim}${progress}${colors.reset} ${colors.red}FAIL${colors.reset} ${relativePath}`);
                console.log(`${colors.red}       ${result.reason}${colors.reset}`);
                console.log(`${colors.dim}       Triangles: ours=${result.ours.triangles}, ref=${result.reference.triangles}${colors.reset}`);
            } else {
                errors++;
                console.log(`${colors.dim}${progress}${colors.reset} ${colors.yellow}ERROR${colors.reset} ${relativePath}`);
                console.log(`${colors.yellow}       ${result.error}${colors.reset}`);
            }
        }

        // Summary
        log('\n========================================', 'blue');
        log('  SUMMARY', 'blue');
        log('========================================', 'blue');
        log(`  Passed: ${passed}`, 'green');
        log(`  Failed: ${failed}`, failed > 0 ? 'red' : 'green');
        log(`  Errors: ${errors}`, errors > 0 ? 'yellow' : 'green');
        log(`  Total:  ${stepFiles.length}`, 'blue');
        log(`  Pass Rate: ${((passed / stepFiles.length) * 100).toFixed(1)}%`, passed === stepFiles.length ? 'green' : 'yellow');
        log('========================================\n', 'blue');

        // Show failed files
        if (failed > 0) {
            log('Failed files:', 'red');
            for (const result of results.filter(r => r.status === 'fail')) {
                log(`  ${result.file}`, 'red');
                log(`    ${result.reason}`, 'dim');
            }
            log('');
        }

        // Save JSON report
        const reportPath = join(CONFIG.screenshotDir, 'ai-test-report.json');
        fs.mkdirSync(CONFIG.screenshotDir, { recursive: true });
        fs.writeFileSync(reportPath, JSON.stringify({
            timestamp: new Date().toISOString(),
            config: { model: CONFIG.openRouterModel },
            summary: { passed, failed, errors, total: stepFiles.length },
            results
        }, null, 2));

        log(`Report saved to: ${reportPath}`, 'dim');

    } finally {
        if (browser) await browser.close();
        if (viteProcess) viteProcess.kill();
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
