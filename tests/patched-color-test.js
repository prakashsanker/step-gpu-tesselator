// Test the patched occt-import-js library for color extraction
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const require = createRequire(import.meta.url);

// Load the patched library (use .cjs for CommonJS in "type": "module" project)
const patchedLibPath = path.resolve(__dirname, '../lib/occt-import-js-patched/occt-import-js.cjs');

async function testColorExtraction() {
  console.log('Loading patched occt-import-js library...');

  // Use require to load the CommonJS-style library
  const occtimportjs = require(patchedLibPath);
  const occ = await occtimportjs();

  const testFiles = [
    '../step-examples/c8-solids/colored-solid.step',
    '../step-examples/complex/rocky_house.step',
  ];

  for (const relPath of testFiles) {
    const filePath = path.resolve(__dirname, relPath);
    if (!fs.existsSync(filePath)) {
      console.log(`File not found: ${filePath}`);
      continue;
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log(`Testing: ${path.basename(filePath)}`);
    console.log('='.repeat(60));

    const stepData = fs.readFileSync(filePath);
    const result = occ.ReadStepFile(new Uint8Array(stepData), {
      linearUnit: 'millimeter'
    });

    if (!result.success) {
      console.log('Failed to parse STEP file');
      continue;
    }

    console.log(`Meshes: ${result.meshes.length}`);

    let meshColorCount = 0;
    let totalFaces = 0;
    let faceColorCount = 0;
    const uniqueColors = new Set();

    for (const mesh of result.meshes) {
      // Check mesh-level color
      if (mesh.color) {
        meshColorCount++;
      }

      // Check face-level colors (color is an array [r, g, b])
      if (mesh.brep_faces) {
        for (const face of mesh.brep_faces) {
          totalFaces++;
          if (face.color) {
            faceColorCount++;
            const colorKey = `rgb(${Math.round(face.color[0] * 255)}, ${Math.round(face.color[1] * 255)}, ${Math.round(face.color[2] * 255)})`;
            uniqueColors.add(colorKey);
          }
        }
      }
    }

    console.log(`\nMesh color: ${meshColorCount}/${result.meshes.length}`);
    console.log(`Face colors: ${faceColorCount}/${totalFaces}`);
    console.log(`Unique colors: ${uniqueColors.size}`);

    if (uniqueColors.size > 0) {
      console.log('\nColors found:');
      let i = 0;
      for (const color of uniqueColors) {
        if (i++ >= 10) {
          console.log(`  ... and ${uniqueColors.size - 10} more`);
          break;
        }
        console.log(`  ${color}`);
      }
    }

    // Success criteria
    const success = faceColorCount > 0;
    console.log(`\n${success ? '✅ SUCCESS' : '❌ FAILED'}: ${success ? 'Face colors are now extracted!' : 'Face colors still not working'}`);
  }
}

testColorExtraction().catch(console.error);
