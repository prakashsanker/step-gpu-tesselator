/**
 * Debug monotone triangulation
 */

import { triangulateMonotone } from '../src/monotone-decomposition.ts';

// Test square - should produce 2 triangles
const square = [[0, 0], [1, 0], [1, 1], [0, 1]];
console.log('Square vertices:', square);

const triangles = triangulateMonotone(square);
console.log('Triangles:', triangles);
console.log('Expected: 2, Got:', triangles.length);

// Test with different vertex orderings
console.log('\n--- Testing with different starting points ---');

for (let start = 0; start < 4; start++) {
    const rotated = [];
    for (let i = 0; i < 4; i++) {
        rotated.push(square[(i + start) % 4]);
    }
    const tris = triangulateMonotone(rotated);
    console.log(`Start at ${start}:`, tris.length, 'triangles');
}
