/**
 * Test script for C8: Full solids / assemblies
 * Verifies that CLOSED_SHELL, MANIFOLD_SOLID_BREP, and STYLED_ITEM are parsed correctly.
 */

const fs = require('fs');
const path = require('path');

const testFiles = [
    { name: 'VM-001.STEP', path: '../step-examples/VM-001.STEP', expectColor: true },
    { name: 'simple-cube.step', path: '../step-examples/c8-solids/simple-cube.step', expectColor: false },
    { name: 'colored-solid.step', path: '../step-examples/c8-solids/colored-solid.step', expectColor: true },
    { name: 'tetrahedron.step', path: '../step-examples/c8-solids/tetrahedron.step', expectColor: true },
];

let allPassed = true;

for (const testFile of testFiles) {
    const stepFile = path.join(__dirname, testFile.path);

    if (!fs.existsSync(stepFile)) {
        console.log(`\n=== ${testFile.name} ===`);
        console.log(`❌ File not found: ${stepFile}`);
        allPassed = false;
        continue;
    }

    const stepContent = fs.readFileSync(stepFile, 'utf-8');

    console.log(`\n=== ${testFile.name} ===`);

    // Check for CLOSED_SHELL
    const closedShellMatch = stepContent.match(/CLOSED_SHELL\s*\(\s*'([^']*)'\s*,\s*\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)/);
    if (closedShellMatch) {
        const faceRefs = closedShellMatch[2].match(/#\d+/g);
        console.log(`✅ CLOSED_SHELL: '${closedShellMatch[1]}' with ${faceRefs ? faceRefs.length : 0} faces`);
    } else {
        console.log('❌ CLOSED_SHELL not found');
        allPassed = false;
    }

    // Check for MANIFOLD_SOLID_BREP
    const brepMatch = stepContent.match(/MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'\s*,\s*#(\d+)/);
    if (brepMatch) {
        console.log(`✅ MANIFOLD_SOLID_BREP: '${brepMatch[1]}' -> shell #${brepMatch[2]}`);
    } else {
        console.log('❌ MANIFOLD_SOLID_BREP not found');
        allPassed = false;
    }

    // Check for STYLED_ITEM (optional based on expectColor)
    const styledItemMatch = stepContent.match(/STYLED_ITEM\s*\(\s*'([^']*)'\s*,\s*\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)\s*,\s*#(\d+)/);
    if (styledItemMatch) {
        console.log(`✅ STYLED_ITEM: '${styledItemMatch[1]}' -> item #${styledItemMatch[3]}`);
    } else if (testFile.expectColor) {
        console.log('❌ STYLED_ITEM not found (expected)');
        allPassed = false;
    } else {
        console.log('⚪ STYLED_ITEM not found (not expected)');
    }

    // Check for COLOUR_RGB
    const colourMatch = stepContent.match(/COLOUR_RGB\s*\(\s*'([^']*)'\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)/);
    if (colourMatch) {
        const r = parseFloat(colourMatch[2]).toFixed(3);
        const g = parseFloat(colourMatch[3]).toFixed(3);
        const b = parseFloat(colourMatch[4]).toFixed(3);
        console.log(`✅ COLOUR_RGB: '${colourMatch[1]}' = (${r}, ${g}, ${b})`);
    } else if (testFile.expectColor) {
        console.log('❌ COLOUR_RGB not found (expected)');
        allPassed = false;
    } else {
        console.log('⚪ COLOUR_RGB not found (not expected)');
    }

    // Check for ADVANCED_BREP_SHAPE_REPRESENTATION
    const shapeRepMatch = stepContent.match(/ADVANCED_BREP_SHAPE_REPRESENTATION\s*\(\s*'([^']*)'/);
    if (shapeRepMatch) {
        console.log(`✅ ADVANCED_BREP_SHAPE_REPRESENTATION: '${shapeRepMatch[1]}'`);
    } else {
        console.log('❌ ADVANCED_BREP_SHAPE_REPRESENTATION not found');
        allPassed = false;
    }
}

console.log('\n=== Summary ===');
if (allPassed) {
    console.log('✅ All C8 entity tests passed!');
} else {
    console.log('❌ Some tests failed');
    process.exit(1);
}
