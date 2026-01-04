import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import occtimportjs from "occt-import-js";

let renderer: THREE.WebGLRenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let currentModel: THREE.Group | null = null;

function init() {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x404040);

  camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    100000
  );
  camera.position.set(100, 100, 100);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // Lights
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x888888, 1.5);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
  dirLight.position.set(100, 200, 100);
  scene.add(dirLight);

  const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
  fillLight.position.set(-100, 50, -100);
  scene.add(fillLight);

  window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

function buildThreeMesh(geometryMesh: {
  attributes: {
    position: { array: number[] };
    normal?: { array: number[] };
  };
  index: { array: number[] };
  color?: number[];
  name?: string;
}): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();

  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(geometryMesh.attributes.position.array, 3)
  );

  if (geometryMesh.attributes.normal) {
    geometry.setAttribute(
      "normal",
      new THREE.Float32BufferAttribute(geometryMesh.attributes.normal.array, 3)
    );
  }

  const index = Uint32Array.from(geometryMesh.index.array);
  geometry.setIndex(new THREE.BufferAttribute(index, 1));

  const material = new THREE.MeshStandardMaterial({
    color: geometryMesh.color
      ? new THREE.Color(
          geometryMesh.color[0],
          geometryMesh.color[1],
          geometryMesh.color[2]
        )
      : 0x6699ff,
    metalness: 0.2,
    roughness: 0.5,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = geometryMesh.name || "mesh";
  return mesh;
}

function fitCameraToModel(group: THREE.Group) {
  const box = new THREE.Box3().setFromObject(group);
  const center = new THREE.Vector3();
  box.getCenter(center);
  const size = new THREE.Vector3();
  box.getSize(size);

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  const cameraDistance = maxDim / (2 * Math.tan(fov / 2)) * 1.5;

  camera.position.set(
    center.x + cameraDistance * 0.7,
    center.y + cameraDistance * 0.7,
    center.z + cameraDistance * 0.7
  );
  camera.lookAt(center);
  camera.near = 0.01;
  camera.far = Math.max(1000, cameraDistance * 10);
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();

  // Add grid
  const gridSize = Math.ceil(maxDim * 2);
  const existingGrid = scene.getObjectByName("grid");
  if (existingGrid) scene.remove(existingGrid);
  const grid = new THREE.GridHelper(gridSize, Math.min(gridSize, 20));
  grid.name = "grid";
  grid.position.y = box.min.y;
  scene.add(grid);
}

async function loadStepFile(file: File) {
  const statsEl = document.getElementById("stats")!;
  statsEl.textContent = "Loading OCCT...";

  const startInit = performance.now();
  const occt = await occtimportjs();
  const initTime = performance.now() - startInit;

  statsEl.textContent = "Parsing STEP file...";

  const arrayBuffer = await file.arrayBuffer();
  const fileBuffer = new Uint8Array(arrayBuffer);

  const startParse = performance.now();
  const result = occt.ReadStepFile(fileBuffer, null);
  const parseTime = performance.now() - startParse;

  // Remove old model
  if (currentModel) {
    scene.remove(currentModel);
    currentModel = null;
  }

  // Build meshes
  const group = new THREE.Group();
  let totalVertices = 0;
  let totalTriangles = 0;

  for (const resultMesh of result.meshes) {
    const mesh = buildThreeMesh(resultMesh);
    group.add(mesh);
    totalVertices += resultMesh.attributes.position.array.length / 3;
    totalTriangles += resultMesh.index.array.length / 3;
  }

  scene.add(group);
  currentModel = group;

  fitCameraToModel(group);

  statsEl.innerHTML = `
    WASM Init: ${initTime.toFixed(0)}ms<br>
    Parse+Tessellate: ${parseTime.toFixed(0)}ms<br>
    Total: ${(initTime + parseTime).toFixed(0)}ms<br>
    Vertices: ${totalVertices.toLocaleString()}<br>
    Triangles: ${totalTriangles.toLocaleString()}
  `;
}

// Initialize
init();

// File input handler
const fileInput = document.getElementById("file-input") as HTMLInputElement;
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) {
    loadStepFile(file).catch((err) => {
      console.error("Error loading STEP file:", err);
      const statsEl = document.getElementById("stats")!;
      statsEl.innerHTML = `<span style="color: #f44">Error: ${err.message}</span>`;
    });
  }
});
