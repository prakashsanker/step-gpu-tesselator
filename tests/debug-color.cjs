// Debug script to check color object structure
const path = require('path');
const fs = require('fs');

const occ = require('../lib/occt-import-js-patched/occt-import-js.cjs');

(async () => {
  const m = await occ();
  const stepData = fs.readFileSync(path.resolve(__dirname, '../step-examples/c8-solids/colored-solid.step'));
  const result = m.ReadStepFile(new Uint8Array(stepData), null);

  console.log('Result success:', result.success);
  if (result.meshes.length > 0) {
    const mesh = result.meshes[0];
    console.log('Mesh color:', mesh.color);
    if (mesh.brep_faces && mesh.brep_faces.length > 0) {
      const face = mesh.brep_faces[0];
      console.log('Face color object:', face.color);
      console.log('Face color keys:', face.color ? Object.keys(face.color) : 'null');
      if (face.color) {
        console.log('r:', face.color.r, 'type:', typeof face.color.r);
        console.log('g:', face.color.g, 'type:', typeof face.color.g);
        console.log('b:', face.color.b, 'type:', typeof face.color.b);
        // Try to iterate properties
        for (const key in face.color) {
          console.log(`  ${key}:`, face.color[key]);
        }
      }
    }
  }
})();
