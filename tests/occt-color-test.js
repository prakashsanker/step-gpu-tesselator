// Test script to diagnose occt-import-js color extraction
import occtimportjs from 'occt-import-js';
import fs from 'fs';
import path from 'path';

async function testColorExtraction(stepFilePath) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing: ${path.basename(stepFilePath)}`);
  console.log('='.repeat(60));

  const fileBuffer = fs.readFileSync(stepFilePath);

  const occt = await occtimportjs();
  const result = occt.ReadStepFile(new Uint8Array(fileBuffer), null);

  if (!result || !result.success) {
    console.log('FAILED to parse STEP file');
    return;
  }

  console.log(`\nTotal meshes: ${result.meshes.length}`);

  // Count mesh-level colors
  let meshesWithColor = 0;
  const meshColors = new Set();

  // Count face-level colors
  let totalFaces = 0;
  let facesWithColor = 0;
  const faceColors = new Set();

  for (let i = 0; i < result.meshes.length; i++) {
    const mesh = result.meshes[i];

    if (mesh.color) {
      meshesWithColor++;
      meshColors.add(JSON.stringify(mesh.color));
    }

    if (mesh.brep_faces) {
      totalFaces += mesh.brep_faces.length;
      for (const face of mesh.brep_faces) {
        if (face.color) {
          facesWithColor++;
          faceColors.add(JSON.stringify(face.color));
        }
      }
    }

    // Log first few meshes in detail
    if (i < 3) {
      console.log(`\nMesh ${i}:`);
      console.log(`  name: ${mesh.name || '(unnamed)'}`);
      console.log(`  vertices: ${mesh.attributes?.position?.array?.length / 3}`);
      console.log(`  color: ${JSON.stringify(mesh.color)}`);
      console.log(`  brep_faces: ${mesh.brep_faces?.length || 0}`);
      if (mesh.brep_faces && mesh.brep_faces.length > 0) {
        console.log(`  brep_faces[0]: ${JSON.stringify(mesh.brep_faces[0])}`);
        const facesWithColorInMesh = mesh.brep_faces.filter(f => f.color).length;
        console.log(`  faces with color: ${facesWithColorInMesh}/${mesh.brep_faces.length}`);
      }
    }
  }

  console.log(`\n--- Summary ---`);
  console.log(`Meshes with mesh.color: ${meshesWithColor}/${result.meshes.length}`);
  console.log(`Unique mesh colors: ${meshColors.size}`);
  console.log(`Total brep_faces: ${totalFaces}`);
  console.log(`Faces with face.color: ${facesWithColor}/${totalFaces}`);
  console.log(`Unique face colors: ${faceColors.size}`);

  if (meshColors.size > 0) {
    console.log(`\nMesh colors found:`);
    for (const c of meshColors) {
      console.log(`  ${c}`);
    }
  }

  if (faceColors.size > 0) {
    console.log(`\nFace colors found (first 10):`);
    let count = 0;
    for (const c of faceColors) {
      if (count++ >= 10) break;
      console.log(`  ${c}`);
    }
  }
}

// Test files
const testFiles = [
  './step-examples/c8-solids/colored-solid.step',
  './step-examples/complex/rocky_house.step',
];

async function main() {
  for (const file of testFiles) {
    const fullPath = path.resolve(process.cwd(), file);
    if (fs.existsSync(fullPath)) {
      await testColorExtraction(fullPath);
    } else {
      console.log(`File not found: ${file}`);
    }
  }
}

main().catch(console.error);
