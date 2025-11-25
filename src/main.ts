import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";


declare function occtimportjs(): Promise<any>;


const scene = new THREE.Scene();

scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(200,200,200);

camera.lookAt(0,0,0);

const renderer = new THREE.WebGLRenderer( { antialias: true } );
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);


const dirLight = new THREE.DirectionalLight(0xffffff, 1);
dirLight.position.set(200,300,200);
scene.add(dirLight);

scene.add(new THREE.AmbientLight(0xffffff, 0.3));

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0,0,0);
controls.update();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

async function loadStepFile(file: File) {
  const occt = await occtimportjs();
  const buffer = await file.arrayBuffer();
  const fileContent = new Uint8Array(buffer);

  console.log(file);
  const result = occt.ReadStepFile(fileContent, null);

  console.log("RESULT");
  console.log(result.meshes[0]);
  console.log(result.logs);

  if (!result.success) {
    console.error("STEP file loading failed:", result);
    alert("failed to load STEP file");
    return;
  }

  clearSceneMeshes();

  const group = new THREE.Group();
  for (const mesh of result.meshes as any[]){
    const posArray = mesh.attributes.position.array as number[];
    const idxArray = mesh.index.array as number[];
    const normalArray = mesh.attributes.normal ? (mesh.attributes.normal.array as number[]) : null;

    console.log({
      positions: posArray.length,
      indices: idxArray.length,
      normals: normalArray?.length
    });

    const geometry = new THREE.BufferGeometry();
    const positionAttr = new THREE.Float32BufferAttribute(posArray, 3);


    geometry.setAttribute("position", positionAttr);

    if (normalArray && normalArray.length === posArray.length) {
      const normalAttr = new THREE.Float32BufferAttribute(normalArray, 3);
      geometry.setAttribute("normal", normalAttr);
    } else {
      geometry.computeVertexNormals();
    }

    if (idxArray && idxArray.length > 0) {
      geometry.setIndex(idxArray); // Three.js accepts a number[] here
    }

    console.log("MESH COLOR");
    console.log(mesh.color);

    const material = new THREE.MeshStandardMaterial({
      // If mesh.color is present, it's [R,G,B] in 0–255 range → convert to 0–1
      color: new THREE.Color(0xdddddd), // default light gray
      metalness: 0.1,
      roughness: 0.7,
      side: THREE.DoubleSide, // show both sides of faces
    });

    const threeMesh = new THREE.Mesh(geometry, material);
    threeMesh.name = mesh.name || "STEP_Mesh";
    group.add(threeMesh);
  }

  centerGroup(group);
  scene.add(group);
}

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

function centerGroup(group: THREE.Group) {
  const box = new THREE.Box3().setFromObject(group);

  if (box.isEmpty()) {
    console.warn("centerGroup: empty bounding box");
    return;
  }

  const center = box.getCenter(new THREE.Vector3());
  const sizeVec = box.getSize(new THREE.Vector3());
  const size = sizeVec.length() || 10; // diagonal length

  // Move group so its center is at the origin
  group.position.sub(center);

  // Choose a camera distance proportional to the model size
  const distance = size * 1.2;

  // Make sure the camera frustum actually covers the model
  camera.near = Math.max(size / 1000, 0.1);
  camera.far = size * 10;
  camera.updateProjectionMatrix();

  camera.position.set(distance, distance, distance);

  controls.target.set(0, 0, 0);
  controls.update();

  console.log("BBox size:", sizeVec, "diag:", size, "cam distance:", distance, "near/far:", camera.near, camera.far);
}

// function centerGroup(group: THREE.Group) {
//   const box = new THREE.Box3().setFromObject(group);
//   const center = box.getCenter(new THREE.Vector3());

//   group.position.sub(center);

//   const size = box.getSize(new THREE.Vector3()).length();

//   const distance = size * 1.5 || 200;
//   camera.position.set(distance, distance, distance);

//   controls.target.set(0,0,0);
//   controls.update();
// }

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

animate();