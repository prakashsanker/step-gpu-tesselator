// Compare color extraction between original and patched occt-import-js
const path = require('path');
const fs = require('fs');

const originalOcc = require('occt-import-js');
const patchedOcc = require('../lib/occt-import-js-patched/occt-import-js.cjs');

async function testFile(filePath, occ, label) {
  const m = await occ();
  const stepData = fs.readFileSync(filePath);
  const result = m.ReadStepFile(new Uint8Array(stepData), null);

  if (!result.success) return { meshColorCount: 0, faceColorCount: 0, totalFaces: 0 };

  let meshColorCount = 0;
  let faceColorCount = 0;
  let totalFaces = 0;

  for (const mesh of result.meshes) {
    if (mesh.color) meshColorCount++;
    if (mesh.brep_faces) {
      for (const face of mesh.brep_faces) {
        totalFaces++;
        if (face.color) faceColorCount++;
      }
    }
  }

  return { meshColorCount, faceColorCount, totalFaces, meshCount: result.meshes.length };
}

async function main() {
  const testFiles = [
    path.resolve(__dirname, '../step-examples/c8-solids/colored-solid.step'),
    path.resolve(__dirname, '../step-examples/complex/rocky_house.step'),
  ];

  for (const filePath of testFiles) {
    if (!fs.existsSync(filePath)) continue;

    console.log(`\n${'='.repeat(60)}`);
    console.log(`File: ${path.basename(filePath)}`);
    console.log('='.repeat(60));

    const orig = await testFile(filePath, originalOcc, 'original');
    const patched = await testFile(filePath, patchedOcc, 'patched');

    console.log('\nOriginal occt-import-js:');
    console.log(`  Mesh colors: ${orig.meshColorCount}/${orig.meshCount}`);
    console.log(`  Face colors: ${orig.faceColorCount}/${orig.totalFaces}`);

    console.log('\nPatched occt-import-js:');
    console.log(`  Mesh colors: ${patched.meshColorCount}/${patched.meshCount}`);
    console.log(`  Face colors: ${patched.faceColorCount}/${patched.totalFaces}`);

    const improvement = patched.faceColorCount - orig.faceColorCount;
    if (improvement > 0) {
      console.log(`\n✅ IMPROVEMENT: +${improvement} face colors extracted!`);
    } else if (improvement === 0) {
      console.log(`\n= No change in face colors`);
    } else {
      console.log(`\n❌ REGRESSION: ${improvement} face colors`);
    }
  }
}

main().catch(console.error);
