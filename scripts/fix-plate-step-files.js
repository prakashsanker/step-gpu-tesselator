#!/usr/bin/env node
/**
 * Fix plate benchmark STEP files by adding correct direction vectors for edges.
 *
 * The original files had all LINE entities using a shared VECTOR pointing in +X direction,
 * which caused OCC to fail to extract edges correctly for triangular holes.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BENCHMARK_DIR = path.join(__dirname, '..', 'step-examples', 'benchmark');

// Direction vectors needed for the plate files:
// - Outer boundary: +X, +Y, -X, -Y
// - Triangular holes: +Y (vertical edge), diagonal-down-right, diagonal-down-left
const DIRECTION_BLOCK = `
/* Direction vectors for edges */
#600 = DIRECTION('', (1.0, 0.0, 0.0));    /* +X direction */
#601 = DIRECTION('', (0.0, 1.0, 0.0));    /* +Y direction */
#602 = DIRECTION('', (-1.0, 0.0, 0.0));   /* -X direction */
#603 = DIRECTION('', (0.0, -1.0, 0.0));   /* -Y direction */
#604 = DIRECTION('', (0.8944271909999159, -0.4472135954999579, 0.0));   /* diagonal down-right */
#605 = DIRECTION('', (-0.8944271909999159, -0.4472135954999579, 0.0));  /* diagonal down-left */

#610 = VECTOR('', #600, 1.0);  /* +X */
#611 = VECTOR('', #601, 1.0);  /* +Y */
#612 = VECTOR('', #602, 1.0);  /* -X */
#613 = VECTOR('', #603, 1.0);  /* -Y */
#614 = VECTOR('', #604, 1.0);  /* diagonal down-right */
#615 = VECTOR('', #605, 1.0);  /* diagonal down-left */
`;

function fixPlateFile(filePath) {
    console.log(`Processing: ${path.basename(filePath)}`);

    let content = fs.readFileSync(filePath, 'utf8');

    // Check if already fixed (has #600 direction)
    if (content.includes('#600 = DIRECTION')) {
        console.log('  Already fixed, skipping.');
        return;
    }

    // Remove the old shared vector line: #6 = VECTOR('', #3, 1.0);
    content = content.replace(/#6 = VECTOR\('', #3, 1\.0\);\n?/g, '');

    // Insert direction block after the PLANE definition
    content = content.replace(
        /(#5 = PLANE\('', #4\);)\n/,
        `$1\n${DIRECTION_BLOCK}\n`
    );

    // Fix outer boundary LINE references (4 edges)
    // Edge 1: (0,0) -> (100,0) is +X, use #610
    // Edge 2: (100,0) -> (100,100) is +Y, use #611
    // Edge 3: (100,100) -> (0,100) is -X, use #612
    // Edge 4: (0,100) -> (0,0) is -Y, use #613

    // The outer boundary lines are #15, #16, #17, #18
    content = content.replace(/#15 = LINE\('', #7, #6\);/, "#15 = LINE('', #7, #610);   /* (0,0) -> (100,0): +X */");
    content = content.replace(/#16 = LINE\('', #8, #6\);/, "#16 = LINE('', #8, #611);   /* (100,0) -> (100,100): +Y */");
    content = content.replace(/#17 = LINE\('', #9, #6\);/, "#17 = LINE('', #9, #612);   /* (100,100) -> (0,100): -X */");
    content = content.replace(/#18 = LINE\('', #10, #6\);/, "#18 = LINE('', #10, #613);  /* (0,100) -> (0,0): -Y */");

    // Fix all hole LINE references
    // Each triangular hole has 3 edges:
    // - First edge: vertical +Y (point to point+3 in Y), use #611
    // - Second edge: diagonal down-right, use #614
    // - Third edge: diagonal down-left, use #615

    // Pattern for hole LINEs: they come in groups of 3
    // First LINE in group (vertical +Y): LINE('', #XX, #6) where XX is first point of hole
    // We need to identify which LINE is which based on the pattern

    // The holes follow a pattern where each hole has:
    // - 3 CARTESIAN_POINTs
    // - 3 VERTEX_POINTs
    // - 3 LINEs (referencing #6)
    // - 3 EDGE_CURVEs
    // - 3 ORIENTED_EDGEs
    // - 1 EDGE_LOOP
    // - 1 FACE_BOUND

    // Find all LINE definitions that reference #6 (after outer boundary)
    // and replace them with the correct vector based on their position in the hole

    // Match pattern: /* ==== Hole N ... */ followed by the hole definition
    const holePattern = /\/\* ==== Hole \d+.*?==== \*\/\n([\s\S]*?)(?=\/\* ==== Hole|\/* ==== Advanced face|$)/g;

    let holeIndex = 0;
    content = content.replace(holePattern, (match, holeContent) => {
        holeIndex++;

        // Find the 3 LINE definitions in this hole and replace #6 with correct vectors
        let lineCount = 0;
        const fixedHoleContent = holeContent.replace(
            /LINE\('', (#\d+), #6\)/g,
            (lineMatch, pointRef) => {
                lineCount++;
                if (lineCount === 1) {
                    return `LINE('', ${pointRef}, #611)`;  // +Y (vertical)
                } else if (lineCount === 2) {
                    return `LINE('', ${pointRef}, #614)`;  // diagonal down-right
                } else {
                    return `LINE('', ${pointRef}, #615)`;  // diagonal down-left
                }
            }
        );

        return match.replace(holeContent, fixedHoleContent);
    });

    // Write the fixed file
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`  Fixed ${holeIndex} holes.`);
}

// Process all plate files
const plateFiles = [
    'plate-medium-5x5.step',
    'plate-large-10x10.step',
    'plate-xlarge-20x20.step',
    'plate-xxlarge-30x30.step'
];

console.log('Fixing plate benchmark STEP files...\n');

for (const file of plateFiles) {
    const filePath = path.join(BENCHMARK_DIR, file);
    if (fs.existsSync(filePath)) {
        fixPlateFile(filePath);
    } else {
        console.log(`File not found: ${file}`);
    }
}

console.log('\nDone!');
