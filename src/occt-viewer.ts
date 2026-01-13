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

  // Use logarithmic depth buffer to reduce z-fighting
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    logarithmicDepthBuffer: true
  });
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

interface BrepFace {
  first: number;
  last: number;
  color?: number[];
}

interface OcctMesh {
  attributes: {
    position: { array: number[] };
    normal?: { array: number[] };
  };
  index: { array: number[] };
  color?: number[];
  name?: string;
  brep_faces?: BrepFace[];
}

function buildThreeMesh(geometryMesh: OcctMesh): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();

  // Convert from Z-up (CAD convention) to Y-up (Three.js convention)
  // Rotate -90 degrees around X axis: (x, y, z) -> (x, z, -y)
  const positions = geometryMesh.attributes.position.array;
  const convertedPositions = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    convertedPositions[i] = positions[i];          // x stays the same
    convertedPositions[i + 1] = positions[i + 2];  // y = old z
    convertedPositions[i + 2] = -positions[i + 1]; // z = -old y
  }

  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(convertedPositions, 3)
  );

  if (geometryMesh.attributes.normal) {
    // Also convert normals
    const normals = geometryMesh.attributes.normal.array;
    const convertedNormals = new Float32Array(normals.length);
    for (let i = 0; i < normals.length; i += 3) {
      convertedNormals[i] = normals[i];          // x stays the same
      convertedNormals[i + 1] = normals[i + 2];  // y = old z
      convertedNormals[i + 2] = -normals[i + 1]; // z = -old y
    }
    geometry.setAttribute(
      "normal",
      new THREE.Float32BufferAttribute(convertedNormals, 3)
    );
  }

  const index = Uint32Array.from(geometryMesh.index.array);
  geometry.setIndex(new THREE.BufferAttribute(index, 1));

  // Check if we have per-face colors from brep_faces
  // Note: face.color can be null or undefined, so check for both
  const hasPerFaceColors = geometryMesh.brep_faces &&
    geometryMesh.brep_faces.some(face => face.color != null);

  // Debug: log color detection
  console.log(`[buildThreeMesh] ${geometryMesh.name}: hasPerFaceColors=${hasPerFaceColors}, brep_faces=${geometryMesh.brep_faces?.length}, mesh.color=${JSON.stringify(geometryMesh.color)}`);

  if (hasPerFaceColors && geometryMesh.brep_faces) {
    // Create vertex colors array - need to color each vertex based on the face it belongs to
    const numVertices = positions.length / 3;
    const vertexColors = new Float32Array(numVertices * 3);

    // Default color (light blue)
    const defaultColor = [0.4, 0.6, 1.0];
    for (let i = 0; i < numVertices; i++) {
      vertexColors[i * 3] = defaultColor[0];
      vertexColors[i * 3 + 1] = defaultColor[1];
      vertexColors[i * 3 + 2] = defaultColor[2];
    }

    // Apply per-face colors by iterating through triangles
    for (const face of geometryMesh.brep_faces) {
      const color = face.color || defaultColor;
      // first and last are triangle indices (0-based)
      // Each triangle has 3 indices in the index array
      for (let triIdx = face.first; triIdx <= face.last; triIdx++) {
        // Get the 3 vertex indices for this triangle
        const baseIdx = triIdx * 3;
        for (let j = 0; j < 3; j++) {
          const vertexIdx = index[baseIdx + j];
          if (vertexIdx !== undefined) {
            vertexColors[vertexIdx * 3] = color[0];
            vertexColors[vertexIdx * 3 + 1] = color[1];
            vertexColors[vertexIdx * 3 + 2] = color[2];
          }
        }
      }
    }

    geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(vertexColors, 3)
    );

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.2,
      roughness: 0.5,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = geometryMesh.name || "mesh";
    return mesh;
  }

  // Fallback to single color for mesh
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

  // Better near/far ratio
  const nearPlane = Math.max(0.1, maxDim * 0.001);
  const farPlane = Math.max(1000, cameraDistance * 5);
  camera.near = nearPlane;
  camera.far = farPlane;
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

  console.log("[OCCT] File size:", fileBuffer.length, "bytes");
  console.log("[OCCT] File name:", file.name);

  const startParse = performance.now();
  const result = occt.ReadStepFile(fileBuffer, null);
  const parseTime = performance.now() - startParse;

  // Check if result is valid
  if (!result) {
    throw new Error("OCCT returned null result");
  }

  // Debug: log result structure
  console.log("[OCCT] result:", result);
  console.log("[OCCT] result keys:", Object.keys(result));

  // Log full result for debugging colors
  if (result.meshes && result.meshes.length > 0) {
    console.log("[OCCT] First mesh full dump:", JSON.stringify(result.meshes[0], null, 2).slice(0, 2000));
  }

  // Check for success flag
  if (result.success === false) {
    console.error("[OCCT] ReadStepFile returned success: false");
    throw new Error("OCCT failed to parse STEP file (success: false)");
  }

  // Log color information with pretty printing
  if (result.meshes && result.meshes.length > 0) {
    console.log(`[OCCT] Total meshes: ${result.meshes.length}`);
    console.log("[OCCT] First mesh structure:", JSON.stringify(Object.keys(result.meshes[0]), null, 2));
    console.log("[OCCT] First mesh color:", JSON.stringify(result.meshes[0].color, null, 2));

    // Sample a few meshes to see their colors
    console.log("=== Sample mesh colors ===");
    for (let i = 0; i < Math.min(5, result.meshes.length); i++) {
      const m = result.meshes[i];
      console.log(`Mesh ${i}: color=${JSON.stringify(m.color)}, name=${m.name}, verts=${m.attributes?.position?.array?.length / 3}`);

      // Check brep_faces for color info
      if (m.brep_faces && m.brep_faces.length > 0) {
        console.log(`  brep_faces count: ${m.brep_faces.length}`);
        console.log(`  brep_faces[0] keys: ${JSON.stringify(Object.keys(m.brep_faces[0]))}`);
        console.log(`  brep_faces[0]: ${JSON.stringify(m.brep_faces[0])}`);
      }
    }

    // Count meshes with colors (at mesh level)
    let meshesWithColors = 0;
    const uniqueMeshColors = new Set<string>();
    for (const m of result.meshes) {
      if (m.color) {
        meshesWithColors++;
        uniqueMeshColors.add(JSON.stringify(m.color));
      }
    }
    console.log(`Meshes with mesh-level colors: ${meshesWithColors} / ${result.meshes.length}`);

    // Count per-face colors from brep_faces
    let facesWithColors = 0;
    let totalFaces = 0;
    const uniqueFaceColors = new Set<string>();
    for (const m of result.meshes) {
      if (m.brep_faces) {
        totalFaces += m.brep_faces.length;
        for (const face of m.brep_faces) {
          if (face.color) {
            facesWithColors++;
            uniqueFaceColors.add(JSON.stringify(face.color));
          }
        }
      }
    }
    console.log(`Faces with colors: ${facesWithColors} / ${totalFaces}`);
    console.log(`Unique mesh colors: ${uniqueMeshColors.size}`, Array.from(uniqueMeshColors).map(c => JSON.parse(c)));
    console.log(`Unique face colors: ${uniqueFaceColors.size}`, Array.from(uniqueFaceColors).slice(0, 20).map(c => JSON.parse(c)));
  }

  if (!result.meshes || !Array.isArray(result.meshes)) {
    throw new Error(`OCCT result.meshes is invalid: ${typeof result.meshes}`);
  }

  if (result.meshes.length === 0) {
    throw new Error("OCCT returned no meshes");
  }

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
