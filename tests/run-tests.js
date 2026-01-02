/**
 * Puppeteer Test Runner for WebGPU Ear Clipping
 *
 * This script:
 * 1. Starts a Vite dev server
 * 2. Launches Chrome with WebGPU enabled
 * 3. Runs all test suites
 * 4. Reports results and exits with appropriate code
 */

import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

// Test configuration
const CONFIG = {
    vitePort: 5173,
    timeout: 60000,
    headless: true,
    slowMo: 0,
};

// Colors for terminal output
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

function logTest(name, passed, message = '') {
    const icon = passed ? '✓' : '✗';
    const color = passed ? 'green' : 'red';
    log(`  ${icon} ${name}${message ? ': ' + message : ''}`, color);
}

/**
 * Start Vite dev server
 */
async function startViteServer() {
    return new Promise((resolve, reject) => {
        log('Starting Vite dev server...', 'blue');

        const vite = spawn('npx', ['vite', '--port', CONFIG.vitePort.toString()], {
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
            // Vite sometimes logs to stderr
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
        slowMo: CONFIG.slowMo,
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan=swiftshader',
            '--enable-gpu-rasterization',
            '--disable-gpu-sandbox',
            '--no-sandbox',
            '--disable-setuid-sandbox',
        ],
    });

    return browser;
}

/**
 * Wait for test harness to be ready
 */
async function waitForHarness(page) {
    await page.waitForFunction(
        () => window.testHarnessReady === true,
        { timeout: CONFIG.timeout }
    );
}

/**
 * Load a STEP file from disk
 */
function loadStepFile(relativePath) {
    const fullPath = join(PROJECT_ROOT, relativePath);
    return fs.readFileSync(fullPath, 'utf8');
}

// ============================================================================
// TEST SUITES
// ============================================================================

/**
 * Test Suite: Basic Convex Polygons
 */
async function testConvexPolygons(page) {
    log('\n[Suite] Convex Polygons', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Triangle (3 vertices)',
            points: [[0, 0], [1, 0], [0.5, 1]],
            expectedTriangles: 1,
        },
        {
            name: 'Square (4 vertices)',
            points: [[0, 0], [1, 0], [1, 1], [0, 1]],
            expectedTriangles: 2,
        },
        {
            name: 'Pentagon (5 vertices)',
            points: [[0, 0], [2, 0], [2.5, 1.5], [1, 2.5], [-0.5, 1.5]],
            expectedTriangles: 3,
        },
        {
            name: 'Hexagon (6 vertices)',
            points: [[1, 0], [2, 0], [2.5, 1], [2, 2], [1, 2], [0.5, 1]],
            expectedTriangles: 4,
        },
        {
            name: 'Unit square CCW',
            points: [[0, 0], [1, 0], [1, 1], [0, 1]],
            expectedTriangles: 2,
        },
    ];

    for (const test of tests) {
        try {
            const result = await page.evaluate(async (points) => {
                const res = await window.testHarness.runEarClipping(points);
                if (!res.success) return res;

                const validation = window.testHarness.validateTriangulation(points, res.triangles);
                return { ...res, validation };
            }, test.points);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            const triangleCount = result.triangles.length;
            const validationPassed = result.validation.valid;
            const countCorrect = triangleCount === test.expectedTriangles;

            if (countCorrect && validationPassed) {
                logTest(test.name, true, `${triangleCount} triangles`);
                passed++;
            } else {
                const issues = [];
                if (!countCorrect) issues.push(`expected ${test.expectedTriangles} triangles, got ${triangleCount}`);
                if (!validationPassed) issues.push(...result.validation.errors);
                logTest(test.name, false, issues.join('; '));
                failed++;
            }
        } catch (e) {
            logTest(test.name, false, e.message);
            failed++;
        }
    }

    return { passed, failed };
}

/**
 * Test Suite: Concave Polygons
 */
async function testConcavePolygons(page) {
    log('\n[Suite] Concave Polygons', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Arrow shape (5 vertices, 1 reflex)',
            // An arrow pointing right with a notch on top
            points: [[0, 0], [2, 0], [2, 2], [1, 1], [0, 2]],
            expectedTriangles: 3,
        },
        {
            name: 'L-shape (6 vertices, 1 reflex)',
            points: [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]],
            expectedTriangles: 4,
        },
        {
            name: 'Star-like indent (5 vertices)',
            points: [[0, 0], [3, 0], [3, 3], [1.5, 2], [0, 3]],
            expectedTriangles: 3,
        },
        {
            name: 'Deep concave (6 vertices)',
            // A rectangle with a deep V cut on the right
            points: [[0, 0], [3, 0], [3, 1], [1.5, 1.5], [3, 2], [0, 2]],
            expectedTriangles: 4,
        },
    ];

    for (const test of tests) {
        try {
            const result = await page.evaluate(async (points) => {
                const res = await window.testHarness.runEarClipping(points);
                if (!res.success) return res;

                const validation = window.testHarness.validateTriangulation(points, res.triangles);
                return { ...res, validation };
            }, test.points);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            const triangleCount = result.triangles.length;
            const validationPassed = result.validation.valid;
            const countCorrect = triangleCount === test.expectedTriangles;

            if (countCorrect && validationPassed) {
                logTest(test.name, true, `${triangleCount} triangles`);
                passed++;
            } else {
                const issues = [];
                if (!countCorrect) issues.push(`expected ${test.expectedTriangles} triangles, got ${triangleCount}`);
                if (!validationPassed) issues.push(...result.validation.errors);
                logTest(test.name, false, issues.join('; '));
                failed++;
            }
        } catch (e) {
            logTest(test.name, false, e.message);
            failed++;
        }
    }

    return { passed, failed };
}

/**
 * Test Suite: Edge Cases
 */
async function testEdgeCases(page) {
    log('\n[Suite] Edge Cases', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Minimum polygon (3 vertices)',
            points: [[0, 0], [1, 0], [0.5, 0.866]],
            expectedTriangles: 1,
        },
        {
            name: 'Very thin triangle',
            points: [[0, 0], [100, 0], [50, 0.001]],
            expectedTriangles: 1,
        },
        {
            name: 'Large polygon (10 vertices)',
            points: [
                [0, 0], [2, 0], [3, 1], [3, 2], [2, 3],
                [1, 3], [0, 2], [-0.5, 1.5], [-0.2, 0.8], [0.2, 0.3]
            ],
            expectedTriangles: 8,
        },
        {
            name: 'Regular octagon',
            points: (() => {
                const pts = [];
                for (let i = 0; i < 8; i++) {
                    const angle = (i * Math.PI * 2) / 8;
                    pts.push([Math.cos(angle), Math.sin(angle)]);
                }
                return pts;
            })(),
            expectedTriangles: 6,
        },
    ];

    for (const test of tests) {
        try {
            const result = await page.evaluate(async (points) => {
                const res = await window.testHarness.runEarClipping(points);
                if (!res.success) return res;

                const validation = window.testHarness.validateTriangulation(points, res.triangles);
                return { ...res, validation };
            }, test.points);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            const triangleCount = result.triangles.length;
            const validationPassed = result.validation.valid;
            const countCorrect = triangleCount === test.expectedTriangles;

            if (countCorrect && validationPassed) {
                logTest(test.name, true, `${triangleCount} triangles`);
                passed++;
            } else {
                const issues = [];
                if (!countCorrect) issues.push(`expected ${test.expectedTriangles} triangles, got ${triangleCount}`);
                if (!validationPassed) issues.push(...result.validation.errors);
                logTest(test.name, false, issues.join('; '));
                failed++;
            }
        } catch (e) {
            logTest(test.name, false, e.message);
            failed++;
        }
    }

    return { passed, failed };
}

/**
 * Test Suite: STEP File Integration
 */
async function testStepFiles(page) {
    log('\n[Suite] STEP File Integration', 'blue');
    let passed = 0;
    let failed = 0;

    const stepFiles = [
        {
            name: 'Simple Triangle',
            path: 'step-examples/basics/triangle.step',
            expectedVertices: 3,
            expectedTriangles: 1,
        },
        {
            name: 'CCW Square',
            path: 'step-examples/basics/ccw-square.step',
            expectedVertices: 4,
            expectedTriangles: 2,
        },
        {
            name: 'CW Square (should auto-reverse)',
            path: 'step-examples/basics/cw-square.step',
            expectedVertices: 4,
            expectedTriangles: 2,
        },
        {
            name: 'Convex Hexagon',
            path: 'step-examples/basics/hexagon.step',
            expectedVertices: 6,
            expectedTriangles: 4,
        },
        {
            name: 'Convex Heptagon',
            path: 'step-examples/basics/convex-heptagon.step',
            expectedVertices: 7,
            expectedTriangles: 5,
        },
        {
            name: 'Regular Octagon',
            path: 'step-examples/basics/octagon.step',
            expectedVertices: 8,
            expectedTriangles: 6,
        },
        {
            name: 'CCW Concave Pentagon',
            path: 'step-examples/basics/ccw-pentagon-concave.step',
            expectedVertices: 5,
            expectedTriangles: 3,
        },
        {
            name: 'CW Concave Pentagon',
            path: 'step-examples/basics/cw-pentagon-concave.step',
            expectedVertices: 5,
            expectedTriangles: 3,
        },
        {
            name: 'L-Shape (6 vertices)',
            path: 'step-examples/basics/l-shape.step',
            expectedVertices: 6,
            expectedTriangles: 4,
        },
        {
            name: 'Arrow Shape (7 vertices)',
            path: 'step-examples/basics/arrow.step',
            expectedVertices: 7,
            expectedTriangles: 5,
        },
        {
            name: 'Tilted Rectangle',
            path: 'step-examples/basics/tilted-rectangle.step',
            expectedVertices: 4,
            expectedTriangles: 2,
        },
        {
            name: 'Convex Pentagon (convexity test)',
            path: 'step-examples/convexity-step-files/convex_pentagon_simple.step',
            expectedVertices: 5,
            expectedTriangles: 3,
        },
        {
            name: 'Concave Pentagon Single Reflex',
            path: 'step-examples/convexity-step-files/concave_pentagon_single_reflex.step',
            expectedVertices: 5,
            expectedTriangles: 3,
        },
    ];

    for (const test of stepFiles) {
        try {
            // Load STEP file
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

            // Parse in browser
            const result = await page.evaluate(async (stepText) => {
                return await window.testHarness.parseStep(stepText);
            }, stepContent);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            const vertexCount = result.mesh.vertexCount;
            const triangleCount = result.mesh.triangleCount;

            const verticesOk = vertexCount === test.expectedVertices;
            const trianglesOk = triangleCount === test.expectedTriangles;

            if (verticesOk && trianglesOk) {
                logTest(test.name, true, `${vertexCount} vertices, ${triangleCount} triangles`);
                passed++;
            } else {
                const issues = [];
                if (!verticesOk) issues.push(`expected ${test.expectedVertices} vertices, got ${vertexCount}`);
                if (!trianglesOk) issues.push(`expected ${test.expectedTriangles} triangles, got ${triangleCount}`);
                logTest(test.name, false, issues.join('; '));
                failed++;
            }
        } catch (e) {
            logTest(test.name, false, e.message);
            failed++;
        }
    }

    return { passed, failed };
}

/**
 * Test Suite: Winding Order Detection
 */
async function testWindingOrder(page) {
    log('\n[Suite] Winding Order Detection', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'CCW Square',
            points: [[0, 0], [1, 0], [1, 1], [0, 1]],
            expectedCCW: true,
        },
        {
            name: 'CW Square',
            points: [[0, 0], [0, 1], [1, 1], [1, 0]],
            expectedCCW: false,
        },
        {
            name: 'CCW Triangle',
            points: [[0, 0], [1, 0], [0.5, 1]],
            expectedCCW: true,
        },
        {
            name: 'CW Triangle',
            points: [[0, 0], [0.5, 1], [1, 0]],
            expectedCCW: false,
        },
        {
            name: 'CCW Pentagon',
            points: [[0, 0], [2, 0], [2.5, 1.5], [1, 2.5], [-0.5, 1.5]],
            expectedCCW: true,
        },
    ];

    for (const test of tests) {
        try {
            const result = await page.evaluate(async (points) => {
                return await window.testHarness.checkWinding(points);
            }, test.points);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            if (result.isCCW === test.expectedCCW) {
                logTest(test.name, true, `CCW: ${result.isCCW}`);
                passed++;
            } else {
                logTest(test.name, false, `expected CCW=${test.expectedCCW}, got ${result.isCCW}`);
                failed++;
            }
        } catch (e) {
            logTest(test.name, false, e.message);
            failed++;
        }
    }

    return { passed, failed };
}

/**
 * Test Suite: Triangle Output Validity
 */
async function testTriangleValidity(page) {
    log('\n[Suite] Triangle Output Validity', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'All indices reference valid vertices',
            points: [[0, 0], [1, 0], [1, 1], [0, 1]],
        },
        {
            name: 'No degenerate triangles',
            points: [[0, 0], [2, 0], [2, 2], [1, 1], [0, 2]],
        },
        {
            name: 'Total area preserved (hexagon)',
            points: [[1, 0], [2, 0], [2.5, 1], [2, 2], [1, 2], [0.5, 1]],
        },
    ];

    for (const test of tests) {
        try {
            const result = await page.evaluate(async (points) => {
                const res = await window.testHarness.runEarClipping(points);
                if (!res.success) return res;

                const validation = window.testHarness.validateTriangulation(points, res.triangles);
                return { ...res, validation };
            }, test.points);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            if (result.validation.valid) {
                logTest(test.name, true, `area diff: ${Math.abs(result.validation.polygonArea - result.validation.triangleAreaSum).toExponential(2)}`);
                passed++;
            } else {
                logTest(test.name, false, result.validation.errors.join('; '));
                failed++;
            }
        } catch (e) {
            logTest(test.name, false, e.message);
            failed++;
        }
    }

    return { passed, failed };
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
    log('\n========================================', 'blue');
    log('  WebGPU Ear Clipping Test Suite', 'blue');
    log('========================================\n', 'blue');

    let viteProcess = null;
    let browser = null;
    let totalPassed = 0;
    let totalFailed = 0;

    try {
        // Start Vite server
        viteProcess = await startViteServer();

        // Launch browser
        browser = await launchBrowser();
        const page = await browser.newPage();

        // Enable console logging from page
        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                log(`[Browser Error] ${msg.text()}`, 'red');
            }
        });

        // Navigate to test harness
        log(`Navigating to test harness...`, 'blue');
        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/test-harness.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        // Wait for harness to be ready
        await waitForHarness(page);
        log('Test harness ready\n', 'green');

        // Check WebGPU availability
        const gpuCheck = await page.evaluate(async () => {
            return await window.testHarness.checkWebGPU();
        });

        if (!gpuCheck.available) {
            log(`WebGPU not available: ${gpuCheck.error}`, 'red');
            log('Tests cannot run without WebGPU support.', 'red');
            process.exit(1);
        }

        log('WebGPU is available\n', 'green');

        // Run test suites
        const suites = [
            testConvexPolygons,
            testConcavePolygons,
            testEdgeCases,
            testWindingOrder,
            testTriangleValidity,
            testStepFiles,
        ];

        for (const suite of suites) {
            const result = await suite(page);
            totalPassed += result.passed;
            totalFailed += result.failed;
        }

        // Summary
        log('\n========================================', 'blue');
        log('  SUMMARY', 'blue');
        log('========================================', 'blue');
        log(`  Passed: ${totalPassed}`, 'green');
        log(`  Failed: ${totalFailed}`, totalFailed > 0 ? 'red' : 'green');
        log(`  Total:  ${totalPassed + totalFailed}`, 'blue');
        log('========================================\n', 'blue');

    } catch (e) {
        log(`\nTest runner error: ${e.message}`, 'red');
        console.error(e.stack);
        totalFailed = 1;
    } finally {
        // Cleanup
        if (browser) {
            await browser.close();
            log('Browser closed', 'dim');
        }
        if (viteProcess) {
            viteProcess.kill();
            log('Vite server stopped', 'dim');
        }
    }

    process.exit(totalFailed > 0 ? 1 : 0);
}

main();
