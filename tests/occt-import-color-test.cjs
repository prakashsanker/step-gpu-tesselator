/**
 * Test occt-import-js native color extraction
 * Run: node tests/occt-import-color-test.cjs [step-file]
 */

const fs = require('fs');
const path = require('path');

async function testOcctImportColors(stepFilePath) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing: ${path.basename(stepFilePath)}`);
  console.log('='.repeat(60));

  // Dynamic import for ESM module
  const occtimportjs = (await import('occt-import-js')).default;
  const occt = await occtimportjs();

  const stepContent = fs.readFileSync(stepFilePath, 'utf8');
  console.log(`File size: ${(stepContent.length / 1024).toFixed(1)} KB`);

  const encoder = new TextEncoder();
  const fileBuffer = encoder.encode(stepContent);

  console.log('Parsing with occt-import-js...');
  const startTime = Date.now();
  const result = occt.ReadStepFile(fileBuffer, null);
  console.log(`Parse time: ${Date.now() - startTime}ms`);

  if (!result || !result.success) {
    console.log('ERROR: Failed to parse STEP file');
    return;
  }

  console.log(`\nResult: ${result.meshes.length} meshes`);

  let totalFaces = 0;
  let facesWithColor = 0;
  let meshesWithColor = 0;
  const uniqueColors = new Set();

  for (let meshIdx = 0; meshIdx < result.meshes.length; meshIdx++) {
    const mesh = result.meshes[meshIdx];
    const vertexCount = mesh.attributes.position.array.length / 3;
    const triangleCount = mesh.index.array.length / 3;

    // Check mesh-level color
    if (mesh.color && mesh.color.length >= 3) {
      meshesWithColor++;
      const colorKey = `${mesh.color[0].toFixed(3)},${mesh.color[1].toFixed(3)},${mesh.color[2].toFixed(3)}`;
      uniqueColors.add(colorKey);
      if (meshIdx < 5) {
        console.log(`  Mesh ${meshIdx}: ${vertexCount} verts, ${triangleCount} tris, color: RGB(${mesh.color.map(c => c.toFixed(2)).join(', ')})`);
      }
    } else if (meshIdx < 5) {
      console.log(`  Mesh ${meshIdx}: ${vertexCount} verts, ${triangleCount} tris, no mesh color`);
    }

    // Check brep_faces colors
    if (mesh.brep_faces) {
      totalFaces += mesh.brep_faces.length;
      for (const face of mesh.brep_faces) {
        if (face.color && face.color.length >= 3) {
          facesWithColor++;
          const colorKey = `${face.color[0].toFixed(3)},${face.color[1].toFixed(3)},${face.color[2].toFixed(3)}`;
          uniqueColors.add(colorKey);
        }
      }
    }
  }

  console.log(`\n[SUMMARY]`);
  console.log(`  Total meshes: ${result.meshes.length}`);
  console.log(`  Meshes with color: ${meshesWithColor}`);
  console.log(`  Total brep_faces: ${totalFaces}`);
  console.log(`  Faces with native color: ${facesWithColor}`);
  console.log(`  Unique colors found: ${uniqueColors.size}`);

  if (uniqueColors.size > 0 && uniqueColors.size <= 30) {
    console.log(`\n[COLORS]`);
    for (const color of uniqueColors) {
      const [r, g, b] = color.split(',').map(parseFloat);
      console.log(`  RGB(${r.toFixed(2)}, ${g.toFixed(2)}, ${b.toFixed(2)})`);
    }
  }

  if (facesWithColor > 0 || meshesWithColor > 0) {
    console.log(`\n[SUCCESS] occt-import-js extracted native colors!`);
  } else {
    console.log(`\n[WARNING] No native colors found - may need STEP text parser fallback`);
  }
}

// Main
const args = process.argv.slice(2);
const testFiles = args.length > 0
  ? args
  : [
      './step-examples/c8-solids/colored-solid.step',
      './step-examples/complex/rocky_house.step',
    ];

(async () => {
  for (const file of testFiles) {
    const fullPath = path.resolve(file);
    if (fs.existsSync(fullPath)) {
      await testOcctImportColors(fullPath);
    } else {
      console.log(`Skipping ${file} (not found)`);
    }
  }
})();
