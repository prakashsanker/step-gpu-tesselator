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

/**
 * Test Suite: Face Bounds Parsing (outer boundary + holes)
 */
async function testFaceBoundsParsing(page) {
    log('\n[Suite] Face Bounds Parsing', 'blue');
    let passed = 0;
    let failed = 0;

    const stepFiles = [
        {
            name: 'Simple square (no holes)',
            path: 'step-examples/basics/ccw-square.step',
            expectedOuterVertices: 4,
            expectedHoleCount: 0,
            expectedHoleVertices: [],
        },
        {
            name: 'Triangle (no holes)',
            path: 'step-examples/basics/triangle.step',
            expectedOuterVertices: 3,
            expectedHoleCount: 0,
            expectedHoleVertices: [],
        },
        {
            name: 'Square with triangular hole',
            path: 'step-examples/basics/square-with-triangle-hole.step',
            expectedOuterVertices: 4,
            expectedHoleCount: 1,
            expectedHoleVertices: [3],
        },
        {
            name: 'Square with two triangular holes',
            path: 'step-examples/basics/square-with-two-holes.step',
            expectedOuterVertices: 4,
            expectedHoleCount: 2,
            expectedHoleVertices: [3, 3],
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

            // Parse face bounds in browser
            const result = await page.evaluate((stepText) => {
                return window.testHarness.parseFaceBounds(stepText);
            }, stepContent);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            const outerOk = result.outerVertexCount === test.expectedOuterVertices;
            const holeCountOk = result.holeCount === test.expectedHoleCount;
            const holeVerticesOk = test.expectedHoleVertices.every(
                (expected, i) => result.holeVertexCounts[i] === expected
            );

            if (outerOk && holeCountOk && holeVerticesOk) {
                const holeInfo = result.holeCount > 0
                    ? `, ${result.holeCount} hole(s) with [${result.holeVertexCounts.join(', ')}] vertices`
                    : ', no holes';
                logTest(test.name, true, `outer: ${result.outerVertexCount} vertices${holeInfo}`);
                passed++;
            } else {
                const issues = [];
                if (!outerOk) issues.push(`expected ${test.expectedOuterVertices} outer vertices, got ${result.outerVertexCount}`);
                if (!holeCountOk) issues.push(`expected ${test.expectedHoleCount} holes, got ${result.holeCount}`);
                if (!holeVerticesOk) issues.push(`hole vertex counts mismatch: expected [${test.expectedHoleVertices}], got [${result.holeVertexCounts}]`);
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
 * Test Suite: 3D to 2D Projection
 */
async function testProjection(page) {
    log('\n[Suite] 3D to 2D Projection', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Flat square on XY plane (z=0)',
            path: 'step-examples/basics/ccw-square.step',
            expectedVertices: 4,
            expectUsedStepPlane: true,
            expectPlanar: true,
            expectNonZeroArea: true,
        },
        {
            name: 'Tilted square (45° around Y axis)',
            path: 'step-examples/projection/tilted-square-45deg.step',
            expectedVertices: 4,
            expectUsedStepPlane: true,
            expectPlanar: true,
            expectNonZeroArea: true,
        },
        {
            name: 'Vertical wall on XZ plane',
            path: 'step-examples/projection/vertical-wall-xz.step',
            expectedVertices: 4,
            expectUsedStepPlane: true,
            expectPlanar: true,
            expectNonZeroArea: true,
        },
        {
            name: 'Tilted hexagon (45° around X axis)',
            path: 'step-examples/projection/tilted-hexagon.step',
            expectedVertices: 6,
            expectUsedStepPlane: true,
            expectPlanar: true,
            expectNonZeroArea: true,
        },
        {
            name: 'Tilted triangle (no PLANE - geometric fallback)',
            path: 'step-examples/projection/tilted-triangle-no-plane.step',
            expectedVertices: 3,
            expectUsedStepPlane: false,  // Should use geometric fallback
            expectPlanar: true,
            expectNonZeroArea: true,
        },
        {
            name: 'Square with hole (projection preserves hole)',
            path: 'step-examples/basics/square-with-triangle-hole.step',
            expectedVertices: 4,
            expectedHoles: 1,
            expectUsedStepPlane: true,
            expectPlanar: true,
            expectNonZeroArea: true,
        },
    ];

    for (const test of tests) {
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

            // Run projection test in browser
            const result = await page.evaluate((stepText) => {
                return window.testHarness.testProjection(stepText);
            }, stepContent);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            // Validate results
            const issues = [];

            // Check vertex count
            if (result.outerVertexCount !== test.expectedVertices) {
                issues.push(`expected ${test.expectedVertices} vertices, got ${result.outerVertexCount}`);
            }

            // Check hole count if specified
            if (test.expectedHoles !== undefined && result.holeCount !== test.expectedHoles) {
                issues.push(`expected ${test.expectedHoles} holes, got ${result.holeCount}`);
            }

            // Check if used STEP plane data
            if (result.usedStepPlane !== test.expectUsedStepPlane) {
                issues.push(`expected usedStepPlane=${test.expectUsedStepPlane}, got ${result.usedStepPlane}`);
            }

            // Check planarity (maxZDeviation should be near 0)
            const planarityThreshold = 1e-6;
            const isPlanar = result.maxZDeviation < planarityThreshold;
            if (test.expectPlanar && !isPlanar) {
                issues.push(`not planar: maxZ=${result.maxZDeviation.toExponential(2)}`);
            }

            // Check non-zero signed area
            const areaThreshold = 1e-10;
            const hasArea = Math.abs(result.signedArea2d) > areaThreshold;
            if (test.expectNonZeroArea && !hasArea) {
                issues.push(`zero area: ${result.signedArea2d.toExponential(2)}`);
            }

            // Check that 2D points match 3D point count
            if (result.outer2d.length !== result.outer3d.length) {
                issues.push(`2D/3D count mismatch: ${result.outer2d.length} vs ${result.outer3d.length}`);
            }

            if (issues.length === 0) {
                const planeInfo = result.usedStepPlane ? 'STEP plane' : 'geometric';
                logTest(test.name, true, `${result.outerVertexCount}v, area=${result.signedArea2d.toFixed(2)}, ${planeInfo}`);
                passed++;
            } else {
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
 * Test Suite: Winding Normalization (C2.3)
 */
async function testWindingNormalization(page) {
    log('\n[Suite] Winding Normalization', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Square with CW winding (should reverse outer)',
            path: 'step-examples/winding/square-cw.step',
            expectOuterReversed: true,
            expectHolesReversed: [],
            expectNormalizedOuterPositive: true,  // After normalization, outer should be CCW (positive area)
        },
        {
            name: 'Square with CCW hole (should reverse hole)',
            path: 'step-examples/winding/square-with-ccw-hole.step',
            expectOuterReversed: false,
            expectHolesReversed: [true],  // Hole should be reversed to CW
            expectNormalizedOuterPositive: true,
            expectNormalizedHolesNegative: [true],  // Hole should have negative area after normalization
        },
        {
            name: 'Both wrong (CW outer + CCW hole)',
            path: 'step-examples/winding/both-wrong.step',
            expectOuterReversed: true,
            expectHolesReversed: [true],
            expectNormalizedOuterPositive: true,
            expectNormalizedHolesNegative: [true],
        },
        {
            name: 'Correct winding (CCW outer + CW hole)',
            path: 'step-examples/winding/correct-winding.step',
            expectOuterReversed: false,
            expectHolesReversed: [false],  // Already correct, no reversal needed
            expectNormalizedOuterPositive: true,
            expectNormalizedHolesNegative: [true],
        },
    ];

    for (const test of tests) {
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

            // Run winding test in browser
            const result = await page.evaluate((stepText) => {
                return window.testHarness.testWinding(stepText);
            }, stepContent);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            // Validate results
            const issues = [];

            // Check if outer was reversed correctly
            if (result.outerReversed !== test.expectOuterReversed) {
                issues.push(`outer reversed: expected ${test.expectOuterReversed}, got ${result.outerReversed}`);
            }

            // Check if holes were reversed correctly
            for (let i = 0; i < test.expectHolesReversed.length; i++) {
                if (result.holesReversed[i] !== test.expectHolesReversed[i]) {
                    issues.push(`hole[${i}] reversed: expected ${test.expectHolesReversed[i]}, got ${result.holesReversed[i]}`);
                }
            }

            // Check normalized outer area is positive (CCW)
            if (test.expectNormalizedOuterPositive && result.normalizedOuterArea <= 0) {
                issues.push(`normalized outer area should be positive (CCW), got ${result.normalizedOuterArea.toFixed(4)}`);
            }

            // Check normalized hole areas are negative (CW)
            if (test.expectNormalizedHolesNegative) {
                for (let i = 0; i < test.expectNormalizedHolesNegative.length; i++) {
                    if (test.expectNormalizedHolesNegative[i] && result.normalizedHoleAreas[i] >= 0) {
                        issues.push(`normalized hole[${i}] area should be negative (CW), got ${result.normalizedHoleAreas[i].toFixed(4)}`);
                    }
                }
            }

            if (issues.length === 0) {
                const reversals = [];
                if (result.outerReversed) reversals.push('outer');
                for (let i = 0; i < result.holesReversed.length; i++) {
                    if (result.holesReversed[i]) reversals.push(`hole${i}`);
                }
                const reversalInfo = reversals.length > 0 ? `reversed: [${reversals.join(', ')}]` : 'no reversals needed';
                logTest(test.name, true, reversalInfo);
                passed++;
            } else {
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
 * Test Suite: Topology Validation (C2.4)
 */
async function testTopologyValidation(page) {
    log('\n[Suite] Topology Validation', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Valid: square with centered hole',
            path: 'step-examples/topology/valid-square-with-hole.step',
            expectValid: true,
        },
        {
            name: 'Valid: simple square (no holes)',
            path: 'step-examples/basics/ccw-square.step',
            expectValid: true,
        },
        {
            name: 'Valid: existing square with hole',
            path: 'step-examples/basics/square-with-triangle-hole.step',
            expectValid: true,
        },
        {
            name: 'Invalid: hole outside outer boundary',
            path: 'step-examples/topology/hole-outside-outer.step',
            expectValid: false,
            expectErrorContains: 'outside outer boundary',
        },
        {
            name: 'Invalid: holes intersect each other',
            path: 'step-examples/topology/holes-intersect.step',
            expectValid: false,
            expectErrorContains: 'intersects',
        },
        {
            name: 'Invalid: self-intersecting outer loop',
            path: 'step-examples/topology/self-intersecting-outer.step',
            expectValid: false,
            expectErrorContains: 'Self-intersection',
        },
    ];

    for (const test of tests) {
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

            // Run topology test in browser
            const result = await page.evaluate((stepText) => {
                return window.testHarness.testTopology(stepText);
            }, stepContent);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            // Validate results
            const issues = [];

            // Check validity matches expectation
            if (result.valid !== test.expectValid) {
                issues.push(`expected valid=${test.expectValid}, got ${result.valid}`);
                if (result.errors.length > 0) {
                    issues.push(`errors: ${result.errors.join('; ')}`);
                }
            }

            // For invalid cases, check error message contains expected text
            if (!test.expectValid && test.expectErrorContains) {
                const hasExpectedError = result.errors.some(e =>
                    e.toLowerCase().includes(test.expectErrorContains.toLowerCase())
                );
                if (!hasExpectedError) {
                    issues.push(`expected error containing "${test.expectErrorContains}", got: ${result.errors.join('; ')}`);
                }
            }

            if (issues.length === 0) {
                const info = test.expectValid
                    ? 'topology OK'
                    : `detected: ${result.errors[0]}`;
                logTest(test.name, true, info);
                passed++;
            } else {
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
 * Test Suite: Hole Triangulation (C2.5)
 * Tests the hole bridging algorithm with visual verification
 */
async function testHoleTriangulation(page) {
    log('\n[Suite] Hole Triangulation (C2.5)', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Square with triangle hole (existing)',
            path: 'step-examples/basics/square-with-triangle-hole.step',
            expectedOuterVertices: 4,
            expectedHoleVertices: 3,
            // After bridging: 4 outer + 3 hole + 2 bridge duplicates = 9
            expectedMergedVertices: 9,
            // Triangles = mergedVertices - 2 = 7
            expectedTriangles: 7,
        },
        {
            name: 'Square with two holes',
            path: 'step-examples/basics/square-with-two-holes.step',
            expectedOuterVertices: 4,
            expectedHoles: 2,
            // 4 + 3 + 2 (first hole bridged) + 3 + 2 (second hole bridged) = 14
            expectedMergedVertices: 14,
            expectedTriangles: 12,
        },
        {
            name: 'Square with square hole',
            path: 'step-examples/holes/square-with-square-hole.step',
            expectedOuterVertices: 4,
            expectedHoleVertices: 4,
            // 4 outer + 4 hole + 2 bridge = 10
            expectedMergedVertices: 10,
            expectedTriangles: 8,
        },
        {
            name: 'Square with right-side hole',
            path: 'step-examples/holes/square-with-right-hole.step',
            expectedOuterVertices: 4,
            expectedHoleVertices: 3,
            expectedMergedVertices: 9,
            expectedTriangles: 7,
        },
        {
            name: 'Square with three holes',
            path: 'step-examples/holes/square-with-three-holes.step',
            expectedOuterVertices: 4,
            expectedHoles: 3,
            // 4 + (3+2)*3 = 4 + 15 = 19
            expectedMergedVertices: 19,
            expectedTriangles: 17,
        },
        {
            name: 'Hexagon with triangle hole',
            path: 'step-examples/holes/hexagon-with-triangle-hole.step',
            expectedOuterVertices: 6,
            expectedHoleVertices: 3,
            // 6 outer + 3 hole + 2 bridge = 11
            expectedMergedVertices: 11,
            expectedTriangles: 9,
        },
        {
            name: 'Triangle with triangle hole',
            path: 'step-examples/holes/triangle-with-triangle-hole.step',
            expectedOuterVertices: 3,
            expectedHoleVertices: 3,
            // 3 outer + 3 hole + 2 bridge = 8
            expectedMergedVertices: 8,
            expectedTriangles: 6,
        },
        {
            name: 'Pentagon with hole',
            path: 'step-examples/holes/pentagon-with-hole.step',
            expectedOuterVertices: 5,
            expectedHoleVertices: 3,
            // 5 outer + 3 hole + 2 bridge = 10
            expectedMergedVertices: 10,
            expectedTriangles: 8,
        },
        {
            name: 'Concentric squares (washer)',
            path: 'step-examples/holes/concentric-squares.step',
            expectedOuterVertices: 4,
            expectedHoleVertices: 4,
            // 4 outer + 4 hole + 2 bridge = 10
            expectedMergedVertices: 10,
            expectedTriangles: 8,
        },
        {
            name: 'Octagon with square hole',
            path: 'step-examples/holes/octagon-with-square-hole.step',
            expectedOuterVertices: 8,
            expectedHoleVertices: 4,
            // 8 outer + 4 hole + 2 bridge = 14
            expectedMergedVertices: 14,
            expectedTriangles: 12,
        },
        {
            name: 'Thin rectangle with slot',
            path: 'step-examples/holes/thin-rectangle-with-slot.step',
            expectedOuterVertices: 4,
            expectedHoleVertices: 4,
            // 4 outer + 4 hole + 2 bridge = 10
            expectedMergedVertices: 10,
            expectedTriangles: 8,
        },
    ];

    for (const test of tests) {
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

            // Parse and triangulate
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

            // Validate
            const issues = [];

            if (test.expectedMergedVertices && vertexCount !== test.expectedMergedVertices) {
                issues.push(`expected ${test.expectedMergedVertices} merged vertices, got ${vertexCount}`);
            }

            if (test.expectedTriangles && triangleCount !== test.expectedTriangles) {
                issues.push(`expected ${test.expectedTriangles} triangles, got ${triangleCount}`);
            }

            // Check triangle count matches n-2 formula for merged polygon
            const expectedFromFormula = vertexCount - 2;
            if (triangleCount !== expectedFromFormula) {
                issues.push(`triangle count ${triangleCount} != vertices-2 (${expectedFromFormula})`);
            }

            if (issues.length === 0) {
                logTest(test.name, true, `${vertexCount} vertices, ${triangleCount} triangles`);
                passed++;
            } else {
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
 * Test Suite: Visual Hole Rendering with Screenshots
 */
async function testVisualHoleRendering(page, browser) {
    log('\n[Suite] Visual Hole Rendering (Screenshots)', 'blue');
    let passed = 0;
    let failed = 0;

    // Create screenshots directory
    const screenshotDir = join(PROJECT_ROOT, 'tests', 'screenshots');
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }

    // Open visual test page
    const visualPage = await browser.newPage();
    await visualPage.goto(`http://localhost:${CONFIG.vitePort}/tests/visual-test.html`, {
        waitUntil: 'networkidle0',
        timeout: CONFIG.timeout,
    });

    // Wait for visual test to be ready
    await visualPage.waitForFunction(
        () => window.visualTestReady === true,
        { timeout: CONFIG.timeout }
    );

    const tests = [
        {
            name: 'square-with-triangle-hole',
            path: 'step-examples/basics/square-with-triangle-hole.step',
        },
        {
            name: 'square-with-two-holes',
            path: 'step-examples/basics/square-with-two-holes.step',
        },
        {
            name: 'square-with-square-hole',
            path: 'step-examples/holes/square-with-square-hole.step',
        },
        {
            name: 'square-with-three-holes',
            path: 'step-examples/holes/square-with-three-holes.step',
        },
        {
            name: 'hexagon-with-triangle-hole',
            path: 'step-examples/holes/hexagon-with-triangle-hole.step',
        },
    ];

    for (const test of tests) {
        try {
            // Load STEP file
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(`Visual: ${test.name}`, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

            // Render in visual page
            const result = await visualPage.evaluate(async (stepText, testName) => {
                return await window.visualTest.loadAndRender(stepText, testName);
            }, stepContent, test.name);

            if (!result.success) {
                logTest(`Visual: ${test.name}`, false, result.error);
                failed++;
                continue;
            }

            // Take screenshot
            const screenshotPath = join(screenshotDir, `${test.name}.png`);
            await visualPage.screenshot({
                path: screenshotPath,
                clip: { x: 0, y: 0, width: 800, height: 600 }
            });

            logTest(`Visual: ${test.name}`, true, `${result.triangleCount} triangles, screenshot saved`);
            passed++;

        } catch (e) {
            logTest(`Visual: ${test.name}`, false, e.message);
            failed++;
        }
    }

    await visualPage.close();
    log(`  Screenshots saved to: ${screenshotDir}`, 'dim');

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
            testFaceBoundsParsing,
            testProjection,
            testWindingNormalization,
            testTopologyValidation,
            testHoleTriangulation,
        ];

        for (const suite of suites) {
            const result = await suite(page);
            totalPassed += result.passed;
            totalFailed += result.failed;
        }

        // Run visual tests (needs browser reference for new page)
        const visualResult = await testVisualHoleRendering(page, browser);
        totalPassed += visualResult.passed;
        totalFailed += visualResult.failed;

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
