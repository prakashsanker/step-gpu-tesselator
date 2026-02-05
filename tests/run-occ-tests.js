/**
 * Puppeteer Test Runner for OCC-based STEP Tessellator
 *
 * This script:
 * 1. Starts a Vite dev server
 * 2. Launches Chrome with WebGPU enabled
 * 3. Runs correctness tests for the OCC tessellator
 * 4. Runs comparison tests against occt-import-js
 * 5. Reports results and exits with appropriate code
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
    vitePort: 5174,  // Different port to avoid conflicts
    timeout: 120000, // Longer timeout for OCC initialization
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
    cyan: '\x1b[36m',
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
 * Test Suite: Basic Convex Polygons (GPU Ear Clipping)
 */
async function testConvexPolygons(page) {
    log('\n[Suite] Convex Polygons (GPU Ear Clipping)', 'blue');
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
 * Test Suite: Concave Polygons (GPU Ear Clipping)
 */
async function testConcavePolygons(page) {
    log('\n[Suite] Concave Polygons (GPU Ear Clipping)', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Arrow shape (5 vertices, 1 reflex)',
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
 * Test Suite: OCC STEP File Parsing
 */
async function testOCCStepParsing(page) {
    log('\n[Suite] OCC STEP File Parsing', 'blue');
    let passed = 0;
    let failed = 0;

    const stepFiles = [
        {
            name: 'Simple Triangle',
            path: 'step-examples/c1-triangulation/convex/triangle.step',
            minVertices: 3,
            minTriangles: 1,
        },
        {
            name: 'CCW Square',
            path: 'step-examples/c1-triangulation/convex/ccw-square.step',
            minVertices: 4,
            minTriangles: 2,
        },
        {
            name: 'Convex Hexagon',
            path: 'step-examples/c1-triangulation/convex/hexagon.step',
            minVertices: 6,
            minTriangles: 4,
        },
        {
            name: 'L-Shape (concave)',
            path: 'step-examples/c1-triangulation/concave/l-shape.step',
            minVertices: 6,
            minTriangles: 4,
        },
        {
            name: 'Arrow Shape (concave)',
            path: 'step-examples/c1-triangulation/concave/arrow.step',
            minVertices: 7,
            minTriangles: 5,
        },
    ];

    for (const test of stepFiles) {
        try {
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

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

            const verticesOk = vertexCount >= test.minVertices;
            const trianglesOk = triangleCount >= test.minTriangles;

            if (verticesOk && trianglesOk) {
                logTest(test.name, true, `${vertexCount} vertices, ${triangleCount} triangles, ${result.timing.total.toFixed(0)}ms`);
                passed++;
            } else {
                const issues = [];
                if (!verticesOk) issues.push(`expected >= ${test.minVertices} vertices, got ${vertexCount}`);
                if (!trianglesOk) issues.push(`expected >= ${test.minTriangles} triangles, got ${triangleCount}`);
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
 * Test Suite: Multi-Face Models
 */
async function testMultiFaceModels(page) {
    log('\n[Suite] Multi-Face Models', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Tetrahedron (4 faces)',
            path: 'step-examples/c4-multiface/tetrahedron.step',
            minTriangles: 4,
        },
        {
            name: 'Pyramid (5 faces)',
            path: 'step-examples/c4-multiface/pyramid.step',
            minTriangles: 6,
        },
        {
            name: 'Unit Box (6 faces)',
            path: 'step-examples/c4-multiface/unit-box.step',
            minTriangles: 12,
        },
        {
            name: 'Triangular Prism (5 faces)',
            path: 'step-examples/c4-multiface/triangular-prism.step',
            minTriangles: 8,
        },
    ];

    for (const test of tests) {
        try {
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

            const result = await page.evaluate(async (stepText) => {
                return await window.testHarness.parseStep(stepText);
            }, stepContent);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            const triangleCount = result.mesh.triangleCount;

            if (triangleCount >= test.minTriangles) {
                logTest(test.name, true, `${result.mesh.vertexCount} vertices, ${triangleCount} triangles`);
                passed++;
            } else {
                logTest(test.name, false, `expected >= ${test.minTriangles} triangles, got ${triangleCount}`);
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
 * Test Suite: Curved Surfaces
 */
async function testCurvedSurfaces(page) {
    log('\n[Suite] Curved Surfaces', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Cylinder',
            path: 'step-examples/c4-surfaces/cylinder.step',
            minTriangles: 10,
        },
        {
            name: 'Sphere',
            path: 'step-examples/c4-surfaces/sphere.step',
            minTriangles: 10,
        },
        {
            name: 'Cone',
            path: 'step-examples/c4-surfaces/cone.step',
            minTriangles: 10,
        },
        {
            name: 'Torus',
            path: 'step-examples/c4-surfaces/torus.step',
            minTriangles: 20,
        },
    ];

    for (const test of tests) {
        try {
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

            const result = await page.evaluate(async (stepText) => {
                return await window.testHarness.parseStep(stepText);
            }, stepContent);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            const triangleCount = result.mesh.triangleCount;

            if (triangleCount >= test.minTriangles) {
                logTest(test.name, true, `${result.mesh.vertexCount} vertices, ${triangleCount} triangles`);
                passed++;
            } else {
                logTest(test.name, false, `expected >= ${test.minTriangles} triangles, got ${triangleCount}`);
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
 * Test Suite: Holes and Complex Faces
 */
async function testHolesAndComplexFaces(page) {
    log('\n[Suite] Holes and Complex Faces', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Square with triangle hole',
            path: 'step-examples/c2-holes/2.5-triangulation/square-with-triangle-hole.step',
            minTriangles: 7,
        },
        {
            name: 'Square with two holes',
            path: 'step-examples/c2-holes/2.5-triangulation/square-with-two-holes.step',
            minTriangles: 10,
        },
        {
            name: 'Hexagon with triangle hole',
            path: 'step-examples/c2-holes/2.5-triangulation/hexagon-with-triangle-hole.step',
            minTriangles: 9,
        },
    ];

    for (const test of tests) {
        try {
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

            const result = await page.evaluate(async (stepText) => {
                return await window.testHarness.parseStep(stepText);
            }, stepContent);

            if (!result.success) {
                logTest(test.name, false, result.error);
                failed++;
                continue;
            }

            const triangleCount = result.mesh.triangleCount;

            if (triangleCount >= test.minTriangles) {
                logTest(test.name, true, `${result.mesh.vertexCount} vertices, ${triangleCount} triangles`);
                passed++;
            } else {
                logTest(test.name, false, `expected >= ${test.minTriangles} triangles, got ${triangleCount}`);
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
 * Test Suite: External Real-World STEP Files
 * Files from steptools.com AP203e2 sample collection (Catia V5 exports)
 */
async function testExternalRealWorldFiles(page) {
    log('\n[Suite] External Real-World Files (steptools.com)', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        // 123Block variants (simple block with different annotations)
        {
            name: '123Block_Color (block with colors)',
            path: 'step-examples/external/steptools/123Block_Color.stp',
            minVertices: 8,
            minTriangles: 12,
        },
        {
            name: '123Block_Dimension (block with dimensions)',
            path: 'step-examples/external/steptools/123Block_Dimension.stp',
            minVertices: 8,
            minTriangles: 12,
        },
        {
            name: '123Block_Short_Note (block with notes)',
            path: 'step-examples/external/steptools/123Block_Short_Note.stp',
            minVertices: 8,
            minTriangles: 12,
        },
        // Boxy variants (complex Catia models with various GD&T annotations)
        {
            name: 'boxy_with_cylindricity',
            path: 'step-examples/external/steptools/boxy_with_cylindricity.stp',
            minVertices: 1000,
            minTriangles: 1000,
        },
        {
            name: 'boxy_with_diamsize',
            path: 'step-examples/external/steptools/boxy_with_diamsize.stp',
            minVertices: 1000,
            minTriangles: 1000,
        },
        {
            name: 'boxy_with_flatness',
            path: 'step-examples/external/steptools/boxy_with_flatness.stp',
            minVertices: 1000,
            minTriangles: 1000,
        },
        {
            name: 'boxy_with_limitsandfits',
            path: 'step-examples/external/steptools/boxy_with_limitsandfits.stp',
            minVertices: 1000,
            minTriangles: 1000,
        },
        {
            name: 'boxy_with_linearsize',
            path: 'step-examples/external/steptools/boxy_with_linearsize.stp',
            minVertices: 1000,
            minTriangles: 1000,
        },
        {
            name: 'boxy_with_perp',
            path: 'step-examples/external/steptools/boxy_with_perp.stp',
            minVertices: 1000,
            minTriangles: 1000,
        },
        {
            name: 'boxy_with_surfacetex',
            path: 'step-examples/external/steptools/boxy_with_surfacetex.stp',
            minVertices: 1000,
            minTriangles: 1000,
        },
    ];

    for (const test of tests) {
        try {
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

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

            const verticesOk = vertexCount >= test.minVertices;
            const trianglesOk = triangleCount >= test.minTriangles;

            if (verticesOk && trianglesOk) {
                logTest(test.name, true, `${vertexCount} vertices, ${triangleCount} triangles, ${result.timing.total.toFixed(0)}ms`);
                passed++;
            } else {
                const issues = [];
                if (!verticesOk) issues.push(`expected >= ${test.minVertices} vertices, got ${vertexCount}`);
                if (!trianglesOk) issues.push(`expected >= ${test.minTriangles} triangles, got ${triangleCount}`);
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
 * Test Suite: AP214 STEPnet Files
 * Files from steptools.com AP214 collection (various CAD sources)
 */
async function testAP214StepnetFiles(page) {
    log('\n[Suite] AP214 STEPnet Files (multi-CAD sources)', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        // AS1 assembly model from various CAD systems
        {
            name: 'as1-ac-214 (AutoCAD)',
            path: 'step-examples/external/steptools-ap214/as1-ac-214.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        {
            name: 'as1-ec-214 (Euclid)',
            path: 'step-examples/external/steptools-ap214/as1-ec-214.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        {
            name: 'as1-md-214 (MicroStation)',
            path: 'step-examples/external/steptools-ap214/as1-md-214.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        // NOTE: as1-tc-214 (Theorem Solutions) skipped - causes stack overflow, needs investigation
        {
            name: 'as1-ug-214 (Unigraphics)',
            path: 'step-examples/external/steptools-ap214/as1-ug-214.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        // D2 and F1 models from debis
        {
            name: 'd2-db-214 (debis)',
            path: 'step-examples/external/steptools-ap214/d2-db-214.stp',
            minVertices: 20,
            minTriangles: 20,
        },
        {
            name: 'f1-db-214 (debis)',
            path: 'step-examples/external/steptools-ap214/f1-db-214.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        // IO1 model from various CAD systems
        {
            name: 'io1-ac-214 (AutoCAD)',
            path: 'step-examples/external/steptools-ap214/io1-ac-214.stp',
            minVertices: 50,
            minTriangles: 50,
        },
        {
            name: 'io1-ca-214 (CADDS)',
            path: 'step-examples/external/steptools-ap214/io1-ca-214.stp',
            minVertices: 50,
            minTriangles: 50,
        },
        {
            name: 'io1-ec-214 (Euclid)',
            path: 'step-examples/external/steptools-ap214/io1-ec-214.stp',
            minVertices: 50,
            minTriangles: 50,
        },
        {
            name: 'io1-md-214 (MicroStation)',
            path: 'step-examples/external/steptools-ap214/io1-md-214.stp',
            minVertices: 50,
            minTriangles: 50,
        },
        // NOTE: io1-tc-214 (Theorem Solutions) skipped - causes stack overflow, needs investigation
        {
            name: 'io1-ug-214 (Unigraphics)',
            path: 'step-examples/external/steptools-ap214/io1-ug-214.stp',
            minVertices: 50,
            minTriangles: 50,
        },
    ];

    for (const test of tests) {
        try {
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

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

            const verticesOk = vertexCount >= test.minVertices;
            const trianglesOk = triangleCount >= test.minTriangles;

            if (verticesOk && trianglesOk) {
                logTest(test.name, true, `${vertexCount} vertices, ${triangleCount} triangles, ${result.timing.total.toFixed(0)}ms`);
                passed++;
            } else {
                const issues = [];
                if (!verticesOk) issues.push(`expected >= ${test.minVertices} vertices, got ${vertexCount}`);
                if (!trianglesOk) issues.push(`expected >= ${test.minTriangles} triangles, got ${triangleCount}`);
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
 * Test Suite: AP224 Manufacturing Features Files
 * Files from steptools.com AP224 collection (RPTS MP 6.0, RAMP)
 */
async function testAP224ManufacturingFiles(page) {
    log('\n[Suite] AP224 Manufacturing Features Files', 'blue');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'ap224_995277945 (RAMP)',
            path: 'step-examples/external/steptools-ap224/ap224_995277945.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        {
            name: 'ap224_995288709 (RAMP)',
            path: 'step-examples/external/steptools-ap224/ap224_995288709.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        {
            name: 'ap224_995315479 (RAMP)',
            path: 'step-examples/external/steptools-ap224/ap224_995315479.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        {
            name: 'ap224_995602415 (RAMP)',
            path: 'step-examples/external/steptools-ap224/ap224_995602415.stp',
            minVertices: 50,
            minTriangles: 50,
        },
        {
            name: 'ap224_997423743 (RAMP)',
            path: 'step-examples/external/steptools-ap224/ap224_997423743.stp',
            minVertices: 100,
            minTriangles: 100,
        },
        {
            name: 'ap224_997865309 (RAMP)',
            path: 'step-examples/external/steptools-ap224/ap224_997865309.stp',
            minVertices: 100,
            minTriangles: 100,
        },
    ];

    for (const test of tests) {
        try {
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

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

            const verticesOk = vertexCount >= test.minVertices;
            const trianglesOk = triangleCount >= test.minTriangles;

            if (verticesOk && trianglesOk) {
                logTest(test.name, true, `${vertexCount} vertices, ${triangleCount} triangles, ${result.timing.total.toFixed(0)}ms`);
                passed++;
            } else {
                const issues = [];
                if (!verticesOk) issues.push(`expected >= ${test.minVertices} vertices, got ${vertexCount}`);
                if (!trianglesOk) issues.push(`expected >= ${test.minTriangles} triangles, got ${triangleCount}`);
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
 * Test Suite: Comparison with occt-import-js
 */
async function testComparisonWithOcctImport(page) {
    log('\n[Suite] Comparison with occt-import-js', 'cyan');
    let passed = 0;
    let failed = 0;

    const tests = [
        {
            name: 'Simple Square',
            path: 'step-examples/benchmark/simple-square.step',
            triangleTolerance: 0.5, // 50% tolerance for simple shapes
        },
        {
            name: 'Unit Box',
            path: 'step-examples/c4-multiface/unit-box.step',
            triangleTolerance: 0.3,
        },
        {
            name: 'Cylinder',
            path: 'step-examples/c4-surfaces/cylinder.step',
            triangleTolerance: 0.5,
        },
        {
            name: 'Sphere',
            path: 'step-examples/c4-surfaces/sphere.step',
            triangleTolerance: 0.5,
        },
    ];

    for (const test of tests) {
        try {
            let stepContent;
            try {
                stepContent = loadStepFile(test.path);
            } catch (e) {
                logTest(test.name, false, `Failed to load file: ${e.message}`);
                failed++;
                continue;
            }

            // Parse with both parsers
            const result = await page.evaluate(async (stepText, tolerance) => {
                const occResult = await window.testHarness.parseStep(stepText);
                const occtResult = await window.testHarness.parseStepOcctImport(stepText);

                const comparison = window.testHarness.compareMeshes(occResult, occtResult, tolerance);

                return {
                    occResult,
                    occtResult,
                    comparison
                };
            }, stepContent, test.triangleTolerance);

            if (!result.comparison.valid) {
                logTest(test.name, false, result.comparison.issues.join('; '));
                failed++;
                continue;
            }

            const occTri = result.comparison.occTriangles;
            const occtTri = result.comparison.occtTriangles;
            const diff = (result.comparison.triangleDiff * 100).toFixed(1);

            logTest(test.name, true, `OCC: ${occTri} tris, occt-import: ${occtTri} tris (${diff}% diff)`);
            passed++;
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
    log('  OCC Tessellator Test Suite', 'blue');
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
        log(`Navigating to OCC test harness...`, 'blue');
        await page.goto(`http://localhost:${CONFIG.vitePort}/tests/occ-test-harness.html`, {
            waitUntil: 'networkidle0',
            timeout: CONFIG.timeout,
        });

        // Wait for harness to be ready
        log('Waiting for test harness initialization...', 'blue');
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
            testWindingOrder,
            testOCCStepParsing,
            testMultiFaceModels,
            testCurvedSurfaces,
            testHolesAndComplexFaces,
            testExternalRealWorldFiles,
            testAP214StepnetFiles,
            testAP224ManufacturingFiles,
            testComparisonWithOcctImport,
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
