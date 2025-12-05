// src/main.ts

// --- Imports from Three.js ---
// THREE: core library (Scene, Camera, Mesh, Materials, etc)
import * as THREE from "three";
// OrbitControls: helper for mouse-based orbit / zoom / pan on the camera
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// --- occt-import-js global declaration ---
// In index.html, we included:
//   <script src="/occt-import-js.js"></script>
// That script defines a global function `occtimportjs()` on window.
// TypeScript doesn't know about it by default, so we declare it here.
//
// It returns a Promise that resolves to an object exposing OCCT functions,
// including `ReadStepFile`.
declare function occtimportjs(): Promise<any>;

//
// ---------------------------------------------------------
// 1. THREE.JS SCENE + CAMERA + RENDERER SETUP
// ---------------------------------------------------------
//

// Create a Scene = the root container for all 3D objects
const scene = new THREE.Scene();
// Set a dark-ish background color (you can change this)
scene.background = new THREE.Color(0x202020);

// Create a perspective camera
//  - fov: 60 degrees
//  - aspect: window width / height
//  - near clip: 0.1, far clip: 10000 (in scene units)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  10000
);
// Put the camera somewhere in space, looking at the origin.
// (We’ll later re-position it after loading a STEP model).
camera.position.set(200, 200, 200);
camera.lookAt(0, 0, 0);

// Create a WebGL renderer and attach it to the page
const renderer = new THREE.WebGLRenderer({ antialias: true });
// Match renderer size to current window size
renderer.setSize(window.innerWidth, window.innerHeight);
// Append the renderer's <canvas> element to the document body
document.body.appendChild(renderer.domElement);

//
// ---------------------------------------------------------
// 2. LIGHTS
// ---------------------------------------------------------
//

// Directional light = like a sun shining on the model
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(200, 300, 200);
scene.add(dirLight);

// Ambient light = base level of light everywhere
scene.add(new THREE.AmbientLight(0xffffff, 0.3));

//
// ---------------------------------------------------------
// 3. ORBIT CONTROLS (MOUSE INTERACTION)
// ---------------------------------------------------------
//

// OrbitControls lets you:
//  - left drag to orbit
//  - scroll to zoom
//  - right drag / middle drag to pan (depending on config)
const controls = new OrbitControls(camera, renderer.domElement);
// The point the camera orbits around (we’ll center models here)
controls.target.set(0, 0, 0);
controls.update();

//
// ---------------------------------------------------------
// 4. HANDLE WINDOW RESIZE
// ---------------------------------------------------------
//

// On resize, update:
//  - camera aspect ratio
//  - camera projection matrix
//  - renderer size
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

//
// ---------------------------------------------------------
// 5. RENDER LOOP
// ---------------------------------------------------------
//

// The main animation loop: called every frame
function animate() {
  requestAnimationFrame(animate);

  // Render the entire scene from the camera’s point of view
  renderer.render(scene, camera);
}
// Start the loop
animate();

//
// ---------------------------------------------------------
// 6. LOAD & CONVERT STEP FILE → THREE.JS MESHES
// ---------------------------------------------------------
//

// This function is called when the user selects a STEP file.
// It:
//
// 1. Initializes the OCCT WebAssembly module (occt-import-js).
// 2. Reads the file into a Uint8Array.
// 3. Calls ReadStepFile(...) to let OCCT parse & tessellate it.
// 4. Converts the resulting meshes into Three.js BufferGeometries.
// 5. Adds them to the scene.
async function loadStepFile(file: File) {
  // 1) Initialize OCCT WASM module (loads the .wasm behind the scenes)
  console.log("Initializing OCCT WASM module");
  // set a timer
  const startTime = Date.now();
  const occt = await occtimportjs();

  // 2) Read the STEP file into a Uint8Array
  const buffer = await file.arrayBuffer();
  const fileContent = new Uint8Array(buffer);

  // 3) Ask OCCT to parse & tessellate the STEP file
  //
  // ReadStepFile signature (simplified):
  //   ReadStepFile(fileContent: Uint8Array, options: any | null)
  //
  // Passing `null` for options uses defaults.
  const result = occt.ReadStepFile(fileContent, null);
  const endTime = Date.now();

  const difference = endTime - startTime;
  console.log(`OCCT WASM module initialized in ${difference} milliseconds`);


  // Check if the import succeeded
  if (!result.success) {
    console.error("STEP import failed:", result);
    alert("Failed to import STEP file");
    return;
  }

  console.log("STEP import result:", result);

  // Optionally, clear any previous model loaded in the scene
  clearSceneMeshes();

  // Create a group to hold all meshes from this STEP file
  const group = new THREE.Group();

  // `result.meshes` is an array of mesh objects.
  // Each mesh looks roughly like:
  // {
  //   name: string,
  //   color: [r, g, b] (0–255),
  //   attributes: {
  //     position: { array: number[] },
  //     normal?: { array: number[] }
  //   },
  //   index: { array: number[] }
  // }
  //
  // We’ll loop over them and build a Three.js Mesh for each.
  for (const mesh of result.meshes as any[]) {
    // Positions = flat number array [x0, y0, z0, x1, y1, z1, ...]
    const posArray = mesh.attributes.position.array as number[];
    // Indices = which vertices form each triangle
    const idxArray = mesh.index.array as number[];
    // Normals (may or may not exist, depending on what the importer gave us)
    const normalArray = mesh.attributes.normal
      ? (mesh.attributes.normal.array as number[])
      : null;

    // Create a BufferGeometry to hold our mesh data on the GPU
    const geometry = new THREE.BufferGeometry();

    // Create a position attribute: each vertex has 3 components (x,y,z)
    const positionAttr = new THREE.Float32BufferAttribute(posArray, 3);
    geometry.setAttribute("position", positionAttr);

    // If we have normals and they match positions in length, use them.
    // Otherwise, let Three.js approximate normals for lighting.
    if (normalArray && normalArray.length === posArray.length) {
      const normalAttr = new THREE.Float32BufferAttribute(normalArray, 3);
      geometry.setAttribute("normal", normalAttr);
    } else {
      geometry.computeVertexNormals();
    }

    if (idxArray && idxArray.length > 0) {
      geometry.setIndex(idxArray); // Three.js accepts a number[] here
    }

    // Create a PBR-ish material (MeshStandardMaterial) for nice lighting
    const material = new THREE.MeshStandardMaterial({
      // If mesh.color is present, it's [R,G,B] in 0–255 range → convert to 0–1
      color: mesh.color
        ? new THREE.Color(mesh.color[0] / 255, mesh.color[1] / 255, mesh.color[2] / 255)
        : new THREE.Color(0xdddddd), // default light gray
      metalness: 0.1,
      roughness: 0.7,
      side: THREE.DoubleSide, // show both sides of faces
    });

    // Build the actual Three.js Mesh and give it a name for debugging
    const threeMesh = new THREE.Mesh(geometry, material);
    threeMesh.name = mesh.name || "STEP_Mesh";

    // Add this mesh to our group
    group.add(threeMesh);
  }

  // Center the group at the origin and reposition camera to frame it
  centerGroup(group);

  // Finally, add the group to the scene so it will be rendered
  scene.add(group);
}

//
// ---------------------------------------------------------
// 7. HELPERS: CLEAR MESHES & CENTER MODEL
// ---------------------------------------------------------
//

// Remove previous meshes/groups from the scene, but keep camera / lights.
//
// This is a simple version; you could refine it by tagging the group
// you add and only removing that, etc.
function clearSceneMeshes() {
  const toRemove: THREE.Object3D[] = [];

  scene.traverse((obj) => {
    // We consider both Mesh and Group as "model content"
    if (obj instanceof THREE.Mesh || obj instanceof THREE.Group) {
      // Be careful not to remove scene or camera itself
      if (obj !== scene && obj !== camera) {
        toRemove.push(obj);
      }
    }
  });

  // Actually remove the gathered objects from the scene graph
  toRemove.forEach((obj) => {
    if (obj.parent) obj.parent.remove(obj);
  });
}

// Center a group at the origin and position the camera so the whole model is visible.
function centerGroup(group: THREE.Group) {
  // Compute a bounding box for the entire group
  const box = new THREE.Box3().setFromObject(group);
  // Get the center of that box
  const center = box.getCenter(new THREE.Vector3());

  // Move the group so its center goes to (0,0,0)
  group.position.sub(center);

  // Compute an approximate "size" (diagonal length) of the bounding box
  const size = box.getSize(new THREE.Vector3()).length();

  // Place camera at some distance proportional to the model size
  const distance = size * 1.5 || 200;
  camera.position.set(distance, distance, distance);

  // Make OrbitControls orbit around the origin
  controls.target.set(0, 0, 0);
  controls.update();
}

//
// ---------------------------------------------------------
// 8. WIRE UP THE FILE INPUT TO LOAD STEP
// ---------------------------------------------------------
//

// Get the <input type="file" id="file-input"> from index.html
const fileInput = document.getElementById("file-input") as HTMLInputElement | null;

if (fileInput) {
  // When the user selects a file...
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) {
      // ...call our loadStepFile() function
      loadStepFile(file).catch((err) => {
        console.error(err);
        alert("Error loading STEP file (see console).");
      });
    }
  });
}
