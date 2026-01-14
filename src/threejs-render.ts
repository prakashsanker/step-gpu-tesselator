import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type {Mesh} from "./step-parser";

export function createThreeMeshFromTesselation(mesh: Mesh): THREE.Mesh {
    // DEBUG: Check what we received
    console.log(`[Three.js] Received mesh: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} triangles`);

    // DEBUG: Check index range
    let maxIdx = 0;
    for (let i = 0; i < mesh.indices.length; i++) {
        if (mesh.indices[i] > maxIdx) maxIdx = mesh.indices[i];
    }
    console.log(`[Three.js] Index range: 0 to ${maxIdx}`);

    // Convert from Z-up (CAD convention) to Y-up (Three.js convention)
    // Rotate -90 degrees around X axis: (x, y, z) -> (x, z, -y)
    const convertedPositions = new Float32Array(mesh.positions.length);
    for (let i = 0; i < mesh.positions.length; i += 3) {
        convertedPositions[i] = mesh.positions[i];          // x stays the same
        convertedPositions[i + 1] = mesh.positions[i + 2];  // y = old z
        convertedPositions[i + 2] = -mesh.positions[i + 1]; // z = -old y
    }

    // Also convert normals if present
    let convertedNormals: Float32Array | undefined;
    if (mesh.normals) {
        convertedNormals = new Float32Array(mesh.normals.length);
        for (let i = 0; i < mesh.normals.length; i += 3) {
            convertedNormals[i] = mesh.normals[i];          // x stays the same
            convertedNormals[i + 1] = mesh.normals[i + 2];  // y = old z
            convertedNormals[i + 2] = -mesh.normals[i + 1]; // z = -old y
        }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(convertedPositions, 3)
    );

    // Use provided normals for smooth shading (C7.3), otherwise compute them
    if (convertedNormals) {
        geometry.setAttribute(
            "normal",
            new THREE.BufferAttribute(convertedNormals, 3)
        );
        console.log(`[Three.js] Using provided normals (${convertedNormals.length / 3} normals)`);
    } else {
        // Fall back to computed normals
        console.log(`[Three.js] No normals provided, computing vertex normals...`);
        geometry.computeVertexNormals();
    }

    geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));

    // DEBUG: Verify geometry
    console.log(`[Three.js] Geometry index count: ${geometry.index?.count}`);
    console.log(`[Three.js] Geometry position count: ${geometry.attributes.position.count}`);

    // Check if we have per-vertex colors
    const hasVertexColors = mesh.vertexColors && mesh.vertexColors.length > 0;
    console.log(`[Three.js] Has vertex colors: ${hasVertexColors}, length: ${mesh.vertexColors?.length || 0}`);

    if (hasVertexColors && mesh.vertexColors) {
        // Add vertex colors to geometry
        geometry.setAttribute(
            "color",
            new THREE.BufferAttribute(mesh.vertexColors, 3)
        );
        console.log(`[Three.js] Added vertex colors attribute`);
    }

    // Create material - use vertex colors if available, otherwise fallback to single color
    let material: THREE.MeshStandardMaterial;

    if (hasVertexColors) {
        // Use vertex colors
        material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            metalness: 0.2,
            roughness: 0.5,
            side: THREE.DoubleSide,
            flatShading: false,
        });
        console.log(`[Three.js] Using vertex colors material`);
    } else {
        // Fallback: Use single color from STYLED_ITEM if available
        let materialColor: THREE.Color | number = 0x6699ff; // Default: bright blue
        if (mesh.color) {
            materialColor = new THREE.Color(mesh.color.r, mesh.color.g, mesh.color.b);
        }
        material = new THREE.MeshStandardMaterial({
            color: materialColor,
            metalness: 0.2,
            roughness: 0.5,
            side: THREE.DoubleSide,
            flatShading: false,
        });
        console.log(`[Three.js] Using single color material`);
    }

    return new THREE.Mesh(geometry, material);
}

export function render(threeMesh: THREE.Mesh) {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;

  // Use logarithmic depth buffer to reduce z-fighting in large scenes
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    logarithmicDepthBuffer: true
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x404040);  // Medium gray for better contrast

  // Compute bounding box to center and frame the mesh
  threeMesh.geometry.computeBoundingBox();
  const boundingBox = threeMesh.geometry.boundingBox!;
  const center = new THREE.Vector3();
  boundingBox.getCenter(center);
  const size = new THREE.Vector3();
  boundingBox.getSize(size);

  // Calculate appropriate camera distance based on mesh size
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = 45;
  const cameraDistance = maxDim / (2 * Math.tan((fov * Math.PI) / 360)) * 1.5;

  // Use better near/far ratio to reduce z-fighting
  // Near plane should be as large as possible while still showing the model
  const nearPlane = Math.max(0.1, maxDim * 0.001);
  const farPlane = Math.max(1000, cameraDistance * 5);

  const camera = new THREE.PerspectiveCamera(
    fov,
    window.innerWidth / window.innerHeight,
    nearPlane,
    farPlane
  );

  // Position camera to look at center from an angle
  camera.position.set(
    center.x + cameraDistance * 0.7,
    center.y + cameraDistance * 0.7,
    center.z + cameraDistance * 0.7
  );
  camera.lookAt(center);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.copy(center); // Orbit around the mesh center

  // Lights - brighter for better visibility
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x888888, 1.5);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
  dirLight.position.set(
    center.x + cameraDistance,
    center.y + cameraDistance * 1.5,
    center.z + cameraDistance
  );
  scene.add(dirLight);

  // Add fill light from opposite side for better shape definition
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
  fillLight.position.set(
    center.x - cameraDistance,
    center.y + cameraDistance * 0.5,
    center.z - cameraDistance
  );
  scene.add(fillLight);
  scene.add(threeMesh);

  // Add grid scaled to mesh size
  const gridSize = Math.ceil(maxDim * 2);
  const grid = new THREE.GridHelper(gridSize, Math.min(gridSize, 20));
  grid.position.y = boundingBox.min.y; // Place grid at bottom of mesh
  scene.add(grid);

  // handle resize
  window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });

  // render loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}
