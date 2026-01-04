/**
 * Test monotone decomposition and triangulation
 */

import {
    computeMonotoneDecomposition,
    extractMonotonePieces,
    triangulateMonotone,
    triangulateWithMonotoneDecomposition,
} from '../src/monotone-decomposition.ts';

// Test cases
const tests = [
    {
        name: 'Simple triangle',
        polygon: [[0, 0], [1, 0], [0.5, 1]],
        expectedTriangles: 1,
    },
    {
        name: 'Square',
        polygon: [[0, 0], [1, 0], [1, 1], [0, 1]],
        expectedTriangles: 2,
    },
    {
        name: 'Pentagon (convex)',
        polygon: [
            [0.5, 0],
            [1, 0.4],
            [0.8, 1],
            [0.2, 1],
            [0, 0.4],
        ],
        expectedTriangles: 3,
    },
    {
        name: 'L-shape (concave)',
        polygon: [
            [0, 0],
            [2, 0],
            [2, 1],
            [1, 1],
            [1, 2],
            [0, 2],
        ],
        expectedTriangles: 4,
    },
    {
        name: 'Star shape (multiple reflex vertices)',
        polygon: [
            [0.5, 0],
            [0.6, 0.4],
            [1, 0.5],
            [0.6, 0.6],
            [0.5, 1],
            [0.4, 0.6],
            [0, 0.5],
            [0.4, 0.4],
        ],
        expectedTriangles: 6,
    },
    {
        name: 'Large convex polygon (20-gon)',
        polygon: Array.from({ length: 20 }, (_, i) => {
            const angle = (i / 20) * Math.PI * 2;
            return [Math.cos(angle), Math.sin(angle)];
        }),
        expectedTriangles: 18,
    },
    {
        name: 'Large polygon (50-gon)',
        polygon: Array.from({ length: 50 }, (_, i) => {
            const angle = (i / 50) * Math.PI * 2;
            return [Math.cos(angle), Math.sin(angle)];
        }),
        expectedTriangles: 48,
    },
    {
        name: 'Large polygon (100-gon)',
        polygon: Array.from({ length: 100 }, (_, i) => {
            const angle = (i / 100) * Math.PI * 2;
            return [Math.cos(angle), Math.sin(angle)];
        }),
        expectedTriangles: 98,
    },
    {
        name: 'Large polygon (500-gon)',
        polygon: Array.from({ length: 500 }, (_, i) => {
            const angle = (i / 500) * Math.PI * 2;
            return [Math.cos(angle), Math.sin(angle)];
        }),
        expectedTriangles: 498,
    },
];

function verifyTriangulation(polygon, triangles) {
    // Check expected number of triangles
    const n = polygon.length;
    const expectedCount = n - 2;

    if (triangles.length !== expectedCount) {
        return { valid: false, error: `Expected ${expectedCount} triangles, got ${triangles.length}` };
    }

    // Check all indices are valid
    for (const tri of triangles) {
        for (const idx of tri) {
            if (idx < 0 || idx >= n) {
                return { valid: false, error: `Invalid index ${idx} (n=${n})` };
            }
        }
    }

    // Check total area matches polygon area
    function signedArea(pts) {
        let area = 0;
        for (let i = 0; i < pts.length; i++) {
            const j = (i + 1) % pts.length;
            area += pts[i][0] * pts[j][1];
            area -= pts[j][0] * pts[i][1];
        }
        return area / 2;
    }

    function triangleArea(p1, p2, p3) {
        return Math.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2;
    }

    const polygonArea = Math.abs(signedArea(polygon));
    let trianglesArea = 0;
    for (const tri of triangles) {
        trianglesArea += triangleArea(
            polygon[tri[0]],
            polygon[tri[1]],
            polygon[tri[2]]
        );
    }

    const areaDiff = Math.abs(polygonArea - trianglesArea) / polygonArea;
    if (areaDiff > 0.01) {
        return { valid: false, error: `Area mismatch: polygon=${polygonArea.toFixed(4)}, triangles=${trianglesArea.toFixed(4)} (diff=${(areaDiff * 100).toFixed(2)}%)` };
    }

    return { valid: true };
}

console.log('\n' + '='.repeat(60));
console.log('  Monotone Decomposition Tests');
console.log('='.repeat(60) + '\n');

let passed = 0;
let failed = 0;

for (const test of tests) {
    const start = performance.now();
    const triangles = triangulateWithMonotoneDecomposition(test.polygon);
    const time = performance.now() - start;

    const verification = verifyTriangulation(test.polygon, triangles);

    if (verification.valid && triangles.length === test.expectedTriangles) {
        console.log(`✓ ${test.name}: ${triangles.length} triangles (${time.toFixed(2)}ms)`);
        passed++;
    } else {
        console.log(`✗ ${test.name}: ${verification.error || `expected ${test.expectedTriangles}, got ${triangles.length}`}`);
        failed++;
    }
}

console.log('\n' + '-'.repeat(60));
console.log(`Results: ${passed} passed, ${failed} failed`);
console.log('-'.repeat(60) + '\n');

// Performance benchmark
console.log('Performance Benchmark:');
console.log('-'.repeat(40));

const sizes = [100, 200, 500, 1000, 2000];
for (const size of sizes) {
    const polygon = Array.from({ length: size }, (_, i) => {
        const angle = (i / size) * Math.PI * 2;
        return [Math.cos(angle), Math.sin(angle)];
    });

    const iterations = 5;
    const times = [];

    for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        triangulateWithMonotoneDecomposition(polygon);
        times.push(performance.now() - start);
    }

    const avg = times.reduce((a, b) => a + b) / times.length;
    console.log(`  ${size.toString().padStart(5)} vertices: ${avg.toFixed(2)}ms`);
}

console.log('\n');
